"""Leakage-aware per-drug elastic-net training and probability calibration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from inspect import signature
from itertools import product
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # StratifiedGroupKFold is available in supported modern sklearn releases.
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility path for older environments
    StratifiedGroupKFold = None  # type: ignore[assignment,misc]

from .evaluation import binary_metrics, calibration_metrics
from .schemas import DecisionThresholds


class TrainingDataError(ValueError):
    """Raised when data cannot support leakage-safe fitting and calibration."""


class ModelArtifactError(ValueError):
    """Raised when a serialized artifact is not a Genome Firewall model bundle."""


@dataclass(frozen=True, slots=True)
class ThresholdSelectionConfig:
    max_called_error_rate: float = 0.10
    max_false_susceptible_rate: float = 0.05
    min_calls_per_side: int = 5
    grid_size: int = 51

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_called_error_rate <= 1.0:
            raise ValueError("max_called_error_rate must lie in [0, 1]")
        if not 0.0 <= self.max_false_susceptible_rate <= 1.0:
            raise ValueError("max_false_susceptible_rate must lie in [0, 1]")
        if self.min_calls_per_side < 1:
            raise ValueError("min_calls_per_side must be positive")
        if self.grid_size < 5:
            raise ValueError("grid_size must be at least 5")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    calibration_fraction: float = 0.20
    split_attempts: int = 200
    min_calibration_samples: int = 20
    min_calibration_groups: int = 2
    cv_splits: int = 5
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    l1_ratios: tuple[float, ...] = (0.0, 0.5, 1.0)
    class_weight: Literal["balanced"] | None = "balanced"
    max_iter: int = 5_000
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    min_isotonic_samples: int = 200
    random_state: int = 42
    threshold_selection: ThresholdSelectionConfig = ThresholdSelectionConfig()

    def __post_init__(self) -> None:
        if not 0.05 <= self.calibration_fraction <= 0.5:
            raise ValueError("calibration_fraction must lie in [0.05, 0.5]")
        if self.split_attempts < 1:
            raise ValueError("split_attempts must be positive")
        if self.min_calibration_samples < 2 or self.min_calibration_groups < 1:
            raise ValueError("calibration minimums must be positive")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")
        if not self.c_values or any(value <= 0 for value in self.c_values):
            raise ValueError("c_values must contain positive values")
        if not self.l1_ratios or any(not 0.0 <= value <= 1.0 for value in self.l1_ratios):
            raise ValueError("l1_ratios must lie in [0, 1]")
        if self.max_iter < 100:
            raise ValueError("max_iter must be at least 100")


@dataclass(frozen=True, slots=True)
class ThresholdSelectionResult:
    thresholds: DecisionThresholds
    constraint_satisfied: bool
    coverage: float
    called_error_rate: float | None
    false_susceptible_rate_among_work_calls: float | None
    n_work_calls: int
    n_fail_calls: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.model_dump(mode="json"),
            "constraint_satisfied": self.constraint_satisfied,
            "coverage": self.coverage,
            "called_error_rate": self.called_error_rate,
            "false_susceptible_rate_among_work_calls": (
                self.false_susceptible_rate_among_work_calls
            ),
            "n_work_calls": self.n_work_calls,
            "n_fail_calls": self.n_fail_calls,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ModelBundle:
    """Serializable estimator, calibrator, thresholds, and audit metadata."""

    drug: str
    species: str
    model_version: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    estimator: Pipeline
    calibrator: BaseEstimator
    calibration_method: str
    thresholds: DecisionThresholds
    training_summary: dict[str, Any]
    feature_center: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    feature_distance_threshold: float | None = None
    artifact_format_version: int = 1

    def unexpected_nonzero_features(
        self,
        features: Mapping[str, float] | pd.DataFrame | np.ndarray,
    ) -> tuple[str, ...]:
        """Return unseen named features carrying a nonzero signal."""

        known = set(self.feature_names)
        if isinstance(features, Mapping):
            extras: list[str] = []
            for name, value in features.items():
                if name in known:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"feature {name!r} is not numeric") from exc
                if not np.isfinite(numeric):
                    raise ValueError(f"feature {name!r} is not finite")
                if numeric != 0.0:
                    extras.append(str(name))
            return tuple(sorted(extras))
        if isinstance(features, pd.DataFrame):
            extra_columns = [str(name) for name in features.columns if name not in known]
            if not extra_columns:
                return ()
            try:
                extra_values = features[extra_columns].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("unexpected feature columns must be numeric") from exc
            if not np.all(np.isfinite(extra_values)):
                raise ValueError("unexpected feature columns must be finite")
            return tuple(
                name
                for index, name in enumerate(extra_columns)
                if np.any(extra_values[:, index] != 0.0)
            )
        return ()

    def _matrix(
        self,
        features: Mapping[str, float] | pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        unexpected = self.unexpected_nonzero_features(features)
        if unexpected:
            raise ModelArtifactError(
                "input contains unexpected nonzero features: " + ", ".join(unexpected[:10])
            )
        if isinstance(features, Mapping):
            values = [features.get(name, 0.0) for name in self.feature_names]
            matrix = np.asarray([values], dtype=float)
        elif isinstance(features, pd.DataFrame):
            if features.empty:
                raise ValueError("feature frame must contain at least one row")
            try:
                matrix = (
                    features.reindex(columns=self.feature_names, fill_value=0.0)
                    .apply(pd.to_numeric, errors="raise")
                    .to_numpy(dtype=float)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("all model features must be numeric") from exc
        else:
            matrix = np.asarray(features, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.ndim != 2:
                raise ValueError("features must be a one- or two-dimensional matrix")
            if matrix.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"expected {len(self.feature_names)} features, got {matrix.shape[1]}"
                )
        if np.any(np.isinf(matrix)):
            raise ValueError("features cannot contain infinity")
        return matrix

    def decision_function(
        self,
        features: Mapping[str, float] | pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        matrix = self._matrix(features)
        return np.asarray(self.estimator.decision_function(matrix), dtype=float).reshape(-1)

    def predict_proba(
        self,
        features: Mapping[str, float] | pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        scores = self.decision_function(features)
        if self.calibration_method == "sigmoid":
            probabilities = self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        elif self.calibration_method == "isotonic":
            probabilities = self.calibrator.predict(scores)
        else:  # A corrupt artifact must fail rather than silently use raw scores.
            raise ModelArtifactError(f"unknown calibration method: {self.calibration_method}")
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        if not np.all(np.isfinite(probabilities)):
            raise ModelArtifactError("calibrator produced a non-finite probability")
        return np.clip(probabilities, 0.0, 1.0)

    def feature_profile_distance(
        self,
        features: Mapping[str, float] | pd.DataFrame | np.ndarray,
    ) -> np.ndarray | None:
        if self.feature_center is None or self.feature_scale is None:
            return None
        matrix = self._matrix(features)
        imputed = np.asarray(self.estimator.named_steps["imputer"].transform(matrix), dtype=float)
        standardized = (imputed - self.feature_center) / self.feature_scale
        return np.sqrt(np.mean(np.square(standardized), axis=1))

    def is_feature_profile_novel(
        self,
        features: Mapping[str, float] | pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        distances = self.feature_profile_distance(features)
        if distances is None or self.feature_distance_threshold is None:
            return np.zeros(self._matrix(features).shape[0], dtype=bool)
        return distances > self.feature_distance_threshold

    def metadata(self) -> dict[str, Any]:
        return {
            "artifact_format_version": self.artifact_format_version,
            "drug": self.drug,
            "species": self.species,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "calibration_method": self.calibration_method,
            "thresholds": self.thresholds.model_dump(mode="json"),
            "training_summary": self.training_summary,
            "feature_distance_threshold": self.feature_distance_threshold,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise ModelArtifactError("artifact is not a Genome Firewall ModelBundle")
        if loaded.artifact_format_version != 1:
            raise ModelArtifactError(
                f"unsupported artifact format {loaded.artifact_format_version}"
            )
        return loaded


def _normalize_labels(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise TrainingDataError("labels must be a non-empty one-dimensional array")
    result: list[int] = []
    for value in array.tolist():
        if isinstance(value, (bool, np.bool_)):
            result.append(int(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric) and numeric in {0.0, 1.0}:
                result.append(int(numeric))
                continue
        token = str(value).strip().casefold()
        if token in {"r", "resistant", "true", "1"}:
            result.append(1)
        elif token in {"s", "susceptible", "false", "0"}:
            result.append(0)
        else:
            raise TrainingDataError(
                f"unsupported label {value!r}; intermediate/unknown labels must be excluded"
            )
    normalized = np.asarray(result, dtype=np.int8)
    if np.unique(normalized).size != 2:
        raise TrainingDataError("training requires both resistant and susceptible labels")
    return normalized


def _prepare_features(
    features: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(features, pd.DataFrame):
        if features.empty or features.shape[1] == 0:
            raise TrainingDataError("feature frame must be non-empty")
        names = tuple(str(column) for column in features.columns)
        if feature_names is not None and tuple(feature_names) != names:
            raise TrainingDataError("feature_names must match DataFrame columns in order")
        try:
            matrix = features.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TrainingDataError("all features must be numeric") from exc
    else:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise TrainingDataError("features must be a non-empty two-dimensional matrix")
        if feature_names is None:
            names = tuple(f"feature_{index:05d}" for index in range(matrix.shape[1]))
        else:
            names = tuple(str(name) for name in feature_names)
    if len(names) != matrix.shape[1]:
        raise TrainingDataError("feature name count does not match feature matrix width")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise TrainingDataError("feature names must be non-empty and unique")
    if np.any(np.isinf(matrix)):
        raise TrainingDataError("features cannot contain infinity")
    if np.any(np.all(np.isnan(matrix), axis=0)):
        raise TrainingDataError("features cannot contain an entirely missing column")
    return matrix, names


def _prepare_groups(groups: Sequence[Any] | np.ndarray, n: int, y: np.ndarray) -> np.ndarray:
    array = np.asarray(groups, dtype=object)
    if array.ndim != 1 or array.size != n:
        raise TrainingDataError("groups must be one-dimensional and match sample count")
    if any(pd.isna(value) for value in array.tolist()):
        raise TrainingDataError("groups cannot contain missing values")
    normalized = np.asarray([str(value) for value in array.tolist()], dtype=object)
    unique = np.unique(normalized)
    if unique.size < 4:
        raise TrainingDataError("at least four independent groups are required")
    for label, name in ((0, "susceptible"), (1, "resistant")):
        if np.unique(normalized[y == label]).size < 2:
            raise TrainingDataError(f"{name} samples must occur in at least two independent groups")
    return normalized


def _group_disjoint_split(
    y: np.ndarray,
    groups: np.ndarray,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=config.split_attempts,
        test_size=config.calibration_fraction,
        random_state=config.random_state,
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    dummy = np.zeros((y.size, 1), dtype=np.int8)
    prevalence = float(y.mean())
    for fit_indices, calibration_indices in splitter.split(dummy, y, groups):
        if np.unique(y[fit_indices]).size < 2 or np.unique(y[calibration_indices]).size < 2:
            continue
        if calibration_indices.size < config.min_calibration_samples:
            continue
        if np.unique(groups[calibration_indices]).size < config.min_calibration_groups:
            continue
        score = abs(calibration_indices.size / y.size - config.calibration_fraction)
        score += abs(float(y[calibration_indices].mean()) - prevalence)
        if best is None or score < best[0]:
            best = (score, fit_indices, calibration_indices)
    if best is None:
        raise TrainingDataError(
            "could not form group-disjoint fit/calibration partitions with both classes; "
            "add groups or lower explicit calibration minimums"
        )
    fit_indices, calibration_indices = best[1], best[2]
    overlap = set(groups[fit_indices]).intersection(groups[calibration_indices])
    if overlap:  # Defensive assertion against future splitter changes.
        raise RuntimeError("group leakage detected between fit and calibration partitions")
    return fit_indices, calibration_indices


def _sample_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    return weights / weights.mean()


def _estimator(config: TrainingConfig, c_value: float, l1_ratio: float) -> Pipeline:
    classifier_arguments: dict[str, Any] = {
        "solver": "saga",
        "C": c_value,
        "l1_ratio": l1_ratio,
        "class_weight": config.class_weight,
        "max_iter": config.max_iter,
        "random_state": config.random_state,
    }
    # sklearn 1.8+ derives the penalty from l1_ratio and deprecates the explicit
    # argument.  Supported older releases still require penalty="elasticnet".
    if signature(LogisticRegression).parameters["penalty"].default != "deprecated":
        classifier_arguments["penalty"] = "elasticnet"
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                LogisticRegression(**classifier_arguments),
            ),
        ]
    )


def _balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    resistant_recall = float(np.mean(predicted[y == 1] == 1))
    susceptible_recall = float(np.mean(predicted[y == 0] == 0))
    return (resistant_recall + susceptible_recall) / 2.0


def _select_hyperparameters(
    matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: TrainingConfig,
) -> tuple[float, float, dict[str, Any]]:
    resistant_groups = np.unique(groups[y == 1]).size
    susceptible_groups = np.unique(groups[y == 0]).size
    splits = min(
        config.cv_splits,
        np.unique(groups).size,
        resistant_groups,
        susceptible_groups,
    )
    if splits < 2:
        raise TrainingDataError("fit partition cannot support group-disjoint cross-validation")
    if StratifiedGroupKFold is not None:
        splitter: Any = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=config.random_state
        )
    else:  # pragma: no cover
        splitter = GroupKFold(n_splits=splits)

    split_indices = list(splitter.split(matrix, y, groups))
    best: tuple[float, float, float, float, list[float]] | None = None
    for c_value, l1_ratio in product(config.c_values, config.l1_ratios):
        fold_scores: list[float] = []
        for train_indices, validation_indices in split_indices:
            if np.unique(y[train_indices]).size < 2 or np.unique(y[validation_indices]).size < 2:
                continue
            model = _estimator(config, c_value, l1_ratio)
            model.fit(
                matrix[train_indices],
                y[train_indices],
                classifier__sample_weight=_sample_weights(groups[train_indices]),
            )
            predicted = model.predict(matrix[validation_indices]).astype(np.int8)
            fold_scores.append(_balanced_accuracy(y[validation_indices], predicted))
        if not fold_scores:
            continue
        mean = float(np.mean(fold_scores))
        std = float(np.std(fold_scores))
        # Prefer higher mean, then lower variance, then simpler regularization.
        candidate = (mean, -std, -c_value, -abs(l1_ratio - 0.5), fold_scores)
        if best is None or candidate[:4] > best[:4]:
            best = candidate
            best_c = c_value
            best_l1 = l1_ratio
    if best is None:
        raise TrainingDataError("no hyperparameter candidate produced valid grouped folds")
    return (
        best_c,
        best_l1,
        {
            "folds": len(best[4]),
            "balanced_accuracy_mean": float(np.mean(best[4])),
            "balanced_accuracy_std": float(np.std(best[4])),
            "fold_balanced_accuracy": [float(value) for value in best[4]],
            "selected_c": best_c,
            "selected_l1_ratio": best_l1,
        },
    )


def _fit_calibrator(
    scores: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> BaseEstimator:
    if config.calibration_method == "sigmoid":
        calibrator = LogisticRegression(C=1_000.0, solver="lbfgs", max_iter=2_000)
        calibrator.fit(scores.reshape(-1, 1), y)
        return calibrator
    if y.size < config.min_isotonic_samples or min(int(y.sum()), int((y == 0).sum())) < 20:
        raise TrainingDataError(
            "isotonic calibration requires the configured sample floor and at least "
            "20 samples from each class"
        )
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(scores, y)
    return calibrator


def _calibrated_probabilities(
    calibrator: BaseEstimator,
    scores: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "sigmoid":
        result = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]  # type: ignore[attr-defined]
    else:
        result = calibrator.predict(scores)
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


def select_decision_thresholds(
    y_true: Sequence[Any] | np.ndarray,
    p_resistant: Sequence[float] | np.ndarray,
    *,
    calibration_groups: int,
    config: ThresholdSelectionConfig | None = None,
) -> ThresholdSelectionResult:
    """Maximize calibration coverage subject to explicit error ceilings."""

    policy = config or ThresholdSelectionConfig()
    y = _normalize_labels(y_true)
    p = np.asarray(p_resistant, dtype=float)
    if p.ndim != 1 or p.size != y.size or not np.all(np.isfinite(p)):
        raise TrainingDataError("calibration probabilities must be finite and match labels")
    if np.any((p < 0.0) | (p > 1.0)):
        raise TrainingDataError("calibration probabilities must lie in [0, 1]")

    work_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.499_999, policy.grid_size),
                p[p < 0.5],
            ]
        )
    )
    fail_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.500_001, 1.0, policy.grid_size),
                p[p > 0.5],
            ]
        )
    )
    best: tuple[tuple[float, float, float, float], ThresholdSelectionResult] | None = None
    for work_max in work_grid:
        work = p <= work_max
        n_work = int(work.sum())
        if n_work < policy.min_calls_per_side:
            continue
        false_susceptible = int(np.sum(work & (y == 1)))
        false_susceptible_rate = false_susceptible / n_work
        if false_susceptible_rate > policy.max_false_susceptible_rate:
            continue
        for fail_min in fail_grid:
            if work_max >= fail_min:
                continue
            fail = p >= fail_min
            n_fail = int(fail.sum())
            if n_fail < policy.min_calls_per_side:
                continue
            called = work | fail
            n_called = int(called.sum())
            errors = false_susceptible + int(np.sum(fail & (y == 0)))
            called_error = errors / n_called
            if called_error > policy.max_called_error_rate:
                continue
            coverage = n_called / y.size
            thresholds = DecisionThresholds(
                work_max=float(work_max),
                fail_min=float(fail_min),
                supported=True,
                calibration_samples=int(y.size),
                calibration_groups=calibration_groups,
                max_called_error_rate=policy.max_called_error_rate,
                max_false_susceptible_rate=policy.max_false_susceptible_rate,
            )
            result = ThresholdSelectionResult(
                thresholds=thresholds,
                constraint_satisfied=True,
                coverage=coverage,
                called_error_rate=called_error,
                false_susceptible_rate_among_work_calls=false_susceptible_rate,
                n_work_calls=n_work,
                n_fail_calls=n_fail,
            )
            # On equal coverage prefer lower dangerous error, lower total error,
            # then a wider abstention interval.
            rank = (
                coverage,
                -false_susceptible_rate,
                -called_error,
                float(fail_min - work_max),
            )
            if best is None or rank > best[0]:
                best = (rank, result)

    if best is not None:
        return best[1]
    return ThresholdSelectionResult(
        thresholds=DecisionThresholds(
            work_max=0.0,
            fail_min=1.0,
            supported=False,
            calibration_samples=int(y.size),
            calibration_groups=calibration_groups,
            max_called_error_rate=policy.max_called_error_rate,
            max_false_susceptible_rate=policy.max_false_susceptible_rate,
        ),
        constraint_satisfied=False,
        coverage=0.0,
        called_error_rate=None,
        false_susceptible_rate_among_work_calls=None,
        n_work_calls=0,
        n_fail_calls=0,
        reason="No threshold pair met the declared error and minimum-call constraints.",
    )


def _group_hashes(groups: np.ndarray) -> list[str]:
    return sorted(
        sha256(str(group).encode("utf-8")).hexdigest()[:16] for group in np.unique(groups)
    )


def train_drug_model(
    features: pd.DataFrame | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    drug: str,
    *,
    species: str = "Escherichia coli",
    feature_names: Sequence[str] | None = None,
    model_version: str = "0.1.0",
    feature_schema_version: str = "1",
    config: TrainingConfig | None = None,
) -> ModelBundle:
    """Train one drug model with group-disjoint tuning and calibration.

    No held-out-test metric is produced here.  The returned summary clearly
    labels grouped-CV and calibration-partition measurements; final evaluation
    must be run separately on untouched groups with :mod:`genome_firewall.evaluation`.
    """

    if not drug.strip() or not species.strip():
        raise TrainingDataError("drug and species must be non-empty")
    training_config = config or TrainingConfig()
    matrix, names = _prepare_features(features, feature_names)
    y = _normalize_labels(labels)
    if matrix.shape[0] != y.size:
        raise TrainingDataError("feature and label sample counts do not match")
    group_array = _prepare_groups(groups, y.size, y)
    fit_indices, calibration_indices = _group_disjoint_split(y, group_array, training_config)

    best_c, best_l1, cv_summary = _select_hyperparameters(
        matrix[fit_indices],
        y[fit_indices],
        group_array[fit_indices],
        training_config,
    )
    estimator = _estimator(training_config, best_c, best_l1)
    estimator.fit(
        matrix[fit_indices],
        y[fit_indices],
        classifier__sample_weight=_sample_weights(group_array[fit_indices]),
    )
    calibration_scores = np.asarray(
        estimator.decision_function(matrix[calibration_indices]), dtype=float
    )
    calibrator = _fit_calibrator(calibration_scores, y[calibration_indices], training_config)
    calibration_probabilities = _calibrated_probabilities(
        calibrator, calibration_scores, training_config.calibration_method
    )
    calibration_group_count = int(np.unique(group_array[calibration_indices]).size)
    threshold_result = select_decision_thresholds(
        y[calibration_indices],
        calibration_probabilities,
        calibration_groups=calibration_group_count,
        config=training_config.threshold_selection,
    )

    imputed_fit = np.asarray(
        estimator.named_steps["imputer"].transform(matrix[fit_indices]), dtype=float
    )
    feature_center = np.mean(imputed_fit, axis=0)
    feature_scale = np.std(imputed_fit, axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    fit_distances = np.sqrt(
        np.mean(np.square((imputed_fit - feature_center) / feature_scale), axis=1)
    )
    # This is feature-profile novelty, not a substitute for whole-genome OOD.
    distance_threshold = float(np.quantile(fit_distances, 0.995))

    calibration_binary = binary_metrics(
        y[calibration_indices], calibration_probabilities
    ).model_dump(mode="json")
    calibration_quality = calibration_metrics(
        y[calibration_indices], calibration_probabilities
    ).model_dump(mode="json")
    training_summary: dict[str, Any] = {
        "metric_scope": (
            "Grouped cross-validation and untouched calibration partition only; "
            "not organizer test performance."
        ),
        "trained_at": datetime.now(UTC).isoformat(),
        "random_seed": training_config.random_state,
        "total_samples": int(y.size),
        "fit_samples": int(fit_indices.size),
        "calibration_samples": int(calibration_indices.size),
        "fit_groups": int(np.unique(group_array[fit_indices]).size),
        "calibration_groups": calibration_group_count,
        "fit_group_hashes": _group_hashes(group_array[fit_indices]),
        "calibration_group_hashes": _group_hashes(group_array[calibration_indices]),
        "fit_class_counts": {
            "susceptible": int(np.sum(y[fit_indices] == 0)),
            "resistant": int(np.sum(y[fit_indices] == 1)),
        },
        "calibration_class_counts": {
            "susceptible": int(np.sum(y[calibration_indices] == 0)),
            "resistant": int(np.sum(y[calibration_indices] == 1)),
        },
        "group_disjoint": True,
        "group_weighted_fit": True,
        "cross_validation": cv_summary,
        "calibration_binary_metrics": calibration_binary,
        "calibration_quality": calibration_quality,
        "threshold_selection": threshold_result.to_dict(),
        "feature_profile_novelty": {
            "method": "standardized RMS distance on fit-partition feature vectors",
            "quantile": 0.995,
            "threshold": distance_threshold,
        },
    }
    return ModelBundle(
        drug=drug.strip(),
        species=species.strip(),
        model_version=model_version,
        feature_schema_version=feature_schema_version,
        feature_names=names,
        estimator=estimator,
        calibrator=calibrator,
        calibration_method=training_config.calibration_method,
        thresholds=threshold_result.thresholds,
        training_summary=training_summary,
        feature_center=feature_center,
        feature_scale=feature_scale,
        feature_distance_threshold=distance_threshold,
    )


def load_model_bundle(path: str | Path) -> ModelBundle:
    return ModelBundle.load(path)


__all__ = [
    "ModelArtifactError",
    "ModelBundle",
    "ThresholdSelectionConfig",
    "ThresholdSelectionResult",
    "TrainingConfig",
    "TrainingDataError",
    "load_model_bundle",
    "select_decision_thresholds",
    "train_drug_model",
]
