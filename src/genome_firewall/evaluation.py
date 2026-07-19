"""Data-derived evaluation metrics for binary probabilities and abstaining calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .schemas import PredictionCall


class MetricsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BinaryMetrics(MetricsModel):
    n_samples: int = Field(ge=0)
    n_resistant: int = Field(ge=0)
    n_susceptible: int = Field(ge=0)
    prevalence_resistant: float | None = Field(default=None, ge=0.0, le=1.0)
    balanced_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    resistant_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    susceptible_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    resistant_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    pr_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_threshold: float = Field(ge=0.0, le=1.0)


class DecisionMetrics(MetricsModel):
    n_samples: int = Field(ge=0)
    n_called: int = Field(ge=0)
    n_no_call: int = Field(ge=0)
    n_work_calls: int = Field(ge=0)
    n_fail_calls: int = Field(ge=0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    no_call_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    called_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    called_balanced_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    system_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    false_susceptible_count: int = Field(ge=0)
    false_susceptible_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_susceptible_rate_among_work_calls: float | None = Field(default=None, ge=0.0, le=1.0)
    false_resistant_count: int = Field(ge=0)


class CalibrationBin(MetricsModel):
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    n_samples: int = Field(ge=0)
    mean_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_resistant_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


class CalibrationMetrics(MetricsModel):
    n_samples: int = Field(ge=0)
    expected_calibration_error: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_slope: float | None = None
    calibration_intercept: float | None = None
    bins: list[CalibrationBin]


class GroupMetrics(MetricsModel):
    n_groups: int = Field(ge=0)
    group_macro_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    worst_group_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    group_macro_balanced_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    groups_with_both_classes: int = Field(ge=0)


class RiskCoveragePoint(MetricsModel):
    minimum_confidence: float = Field(ge=0.5, le=1.0)
    n_called: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    error_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class EvaluationReport(MetricsModel):
    binary: BinaryMetrics
    calibration: CalibrationMetrics
    decisions: DecisionMetrics | None = None
    groups: GroupMetrics | None = None
    risk_coverage: list[RiskCoveragePoint]


def _as_binary(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("binary labels must be one-dimensional")
    if array.size == 0:
        return array.astype(np.int8)
    if array.dtype.kind in "biuf":
        numeric = array.astype(float)
        if not np.all(np.isfinite(numeric)) or not np.all(np.isin(numeric, [0.0, 1.0])):
            raise ValueError("binary labels must contain only 0 and 1")
        return numeric.astype(np.int8)
    normalized: list[int] = []
    for value in array.tolist():
        token = str(value).strip().casefold()
        if token in {"1", "true", "r", "resistant"}:
            normalized.append(1)
        elif token in {"0", "false", "s", "susceptible"}:
            normalized.append(0)
        else:
            raise ValueError(f"unsupported binary label: {value!r}")
    return np.asarray(normalized, dtype=np.int8)


def _as_probabilities(values: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or probabilities.size != n:
        raise ValueError("probabilities must be one-dimensional and match label length")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return probabilities


def binary_metrics(
    y_true: Sequence[Any] | np.ndarray,
    p_resistant: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute per-drug binary metrics without inventing undefined values."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    y = _as_binary(y_true)
    p = _as_probabilities(p_resistant, y.size)
    n = int(y.size)
    positives = int(y.sum())
    negatives = n - positives
    predicted = (p >= threshold).astype(np.int8)
    tp = int(np.sum((y == 1) & (predicted == 1)))
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    resistant_recall = tp / positives if positives else None
    susceptible_recall = tn / negatives if negatives else None
    balanced = (
        (resistant_recall + susceptible_recall) / 2.0
        if resistant_recall is not None and susceptible_recall is not None
        else None
    )
    f1_denominator = 2 * tp + fp + fn
    resistant_f1 = 2 * tp / f1_denominator if positives and f1_denominator else None
    auroc = float(roc_auc_score(y, p)) if positives and negatives else None
    pr_auc = float(average_precision_score(y, p)) if positives else None
    brier = float(brier_score_loss(y, p)) if n else None
    return BinaryMetrics(
        n_samples=n,
        n_resistant=positives,
        n_susceptible=negatives,
        prevalence_resistant=positives / n if n else None,
        balanced_accuracy=balanced,
        resistant_recall=resistant_recall,
        susceptible_recall=susceptible_recall,
        resistant_f1=resistant_f1,
        auroc=auroc,
        pr_auc=pr_auc,
        brier_score=brier,
        probability_threshold=threshold,
    )


def _normalize_calls(calls: Sequence[PredictionCall | str], n: int) -> np.ndarray:
    if len(calls) != n:
        raise ValueError("calls must match label length")
    normalized: list[str] = []
    for call in calls:
        try:
            normalized.append(PredictionCall(call).value)
        except ValueError as exc:
            raise ValueError(f"unsupported prediction call: {call!r}") from exc
    return np.asarray(normalized, dtype=object)


def decision_metrics(
    y_true: Sequence[Any] | np.ndarray,
    calls: Sequence[PredictionCall | str],
) -> DecisionMetrics:
    """Evaluate work/fail/no-call decisions, including dangerous false work calls."""

    y = _as_binary(y_true)
    normalized = _normalize_calls(calls, y.size)
    work = normalized == PredictionCall.LIKELY_TO_WORK.value
    fail = normalized == PredictionCall.LIKELY_TO_FAIL.value
    called = work | fail
    no_call = ~called
    correct = ((y == 0) & work) | ((y == 1) & fail)
    false_susceptible = (y == 1) & work
    false_resistant = (y == 0) & fail
    n = int(y.size)
    n_called = int(called.sum())
    n_work = int(work.sum())
    n_resistant = int((y == 1).sum())

    called_resistant = int(((y == 1) & called).sum())
    called_susceptible = int(((y == 0) & called).sum())
    called_resistant_recall = (
        int(((y == 1) & fail).sum()) / called_resistant if called_resistant else None
    )
    called_susceptible_recall = (
        int(((y == 0) & work).sum()) / called_susceptible if called_susceptible else None
    )
    called_balanced = (
        (called_resistant_recall + called_susceptible_recall) / 2.0
        if called_resistant_recall is not None and called_susceptible_recall is not None
        else None
    )
    false_susceptible_count = int(false_susceptible.sum())
    return DecisionMetrics(
        n_samples=n,
        n_called=n_called,
        n_no_call=int(no_call.sum()),
        n_work_calls=n_work,
        n_fail_calls=int(fail.sum()),
        coverage=n_called / n if n else None,
        no_call_rate=int(no_call.sum()) / n if n else None,
        called_accuracy=int(correct.sum()) / n_called if n_called else None,
        called_balanced_accuracy=called_balanced,
        system_accuracy=int(correct.sum()) / n if n else None,
        false_susceptible_count=false_susceptible_count,
        false_susceptible_rate=(false_susceptible_count / n_resistant if n_resistant else None),
        false_susceptible_rate_among_work_calls=(
            false_susceptible_count / n_work if n_work else None
        ),
        false_resistant_count=int(false_resistant.sum()),
    )


def calibration_metrics(
    y_true: Sequence[Any] | np.ndarray,
    p_resistant: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y = _as_binary(y_true)
    p = _as_probabilities(p_resistant, y.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    weighted_gap = 0.0
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        mask = (p >= lower) & (p < upper if index < n_bins - 1 else p <= upper)
        count = int(mask.sum())
        mean_probability = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        if count and mean_probability is not None and observed is not None and y.size:
            weighted_gap += count / y.size * abs(mean_probability - observed)
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                n_samples=count,
                mean_probability=mean_probability,
                observed_resistant_fraction=observed,
            )
        )

    slope: float | None = None
    intercept: float | None = None
    if y.size and np.unique(y).size == 2:
        clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        calibration_model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=2_000)
        calibration_model.fit(logits, y)
        slope = float(calibration_model.coef_[0, 0])
        intercept = float(calibration_model.intercept_[0])
    return CalibrationMetrics(
        n_samples=int(y.size),
        expected_calibration_error=weighted_gap if y.size else None,
        calibration_slope=slope,
        calibration_intercept=intercept,
        bins=bins,
    )


def group_metrics(
    y_true: Sequence[Any] | np.ndarray,
    p_resistant: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> GroupMetrics:
    y = _as_binary(y_true)
    p = _as_probabilities(p_resistant, y.size)
    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1 or group_array.size != y.size:
        raise ValueError("groups must be one-dimensional and match label length")
    predicted = p >= threshold
    accuracies: list[float] = []
    balanced: list[float] = []
    for group in np.unique(group_array):
        mask = group_array == group
        group_y = y[mask]
        group_predicted = predicted[mask]
        accuracies.append(float(np.mean(group_y == group_predicted)))
        if np.unique(group_y).size == 2:
            resistant_recall = float(np.mean(group_predicted[group_y == 1]))
            susceptible_recall = float(np.mean(~group_predicted[group_y == 0]))
            balanced.append((resistant_recall + susceptible_recall) / 2.0)
    return GroupMetrics(
        n_groups=len(accuracies),
        group_macro_accuracy=float(np.mean(accuracies)) if accuracies else None,
        worst_group_accuracy=float(np.min(accuracies)) if accuracies else None,
        group_macro_balanced_accuracy=float(np.mean(balanced)) if balanced else None,
        groups_with_both_classes=len(balanced),
    )


def risk_coverage_curve(
    y_true: Sequence[Any] | np.ndarray,
    p_resistant: Sequence[float] | np.ndarray,
    *,
    points: int = 20,
) -> list[RiskCoveragePoint]:
    if points < 2:
        raise ValueError("points must be at least 2")
    y = _as_binary(y_true)
    p = _as_probabilities(p_resistant, y.size)
    if not y.size:
        return []
    confidence = np.maximum(p, 1.0 - p)
    candidate_thresholds = np.unique(
        np.quantile(confidence, np.linspace(0.0, 1.0, min(points, y.size)))
    )
    result: list[RiskCoveragePoint] = []
    predicted = p >= 0.5
    for minimum in candidate_thresholds:
        called = confidence >= minimum
        n_called = int(called.sum())
        error = float(np.mean(predicted[called] != y[called])) if n_called else None
        result.append(
            RiskCoveragePoint(
                minimum_confidence=float(minimum),
                n_called=n_called,
                coverage=n_called / y.size,
                error_rate=error,
            )
        )
    return result


def evaluate_predictions(
    y_true: Sequence[Any] | np.ndarray,
    p_resistant: Sequence[float] | np.ndarray,
    *,
    calls: Sequence[PredictionCall | str] | None = None,
    groups: Sequence[Any] | np.ndarray | None = None,
    probability_threshold: float = 0.5,
    n_bins: int = 10,
) -> EvaluationReport:
    """Build a complete, explicitly sized per-drug evaluation report."""

    return EvaluationReport(
        binary=binary_metrics(y_true, p_resistant, threshold=probability_threshold),
        calibration=calibration_metrics(y_true, p_resistant, n_bins=n_bins),
        decisions=decision_metrics(y_true, calls) if calls is not None else None,
        groups=(
            group_metrics(y_true, p_resistant, groups, threshold=probability_threshold)
            if groups is not None
            else None
        ),
        risk_coverage=risk_coverage_curve(y_true, p_resistant),
    )


__all__ = [
    "BinaryMetrics",
    "CalibrationBin",
    "CalibrationMetrics",
    "DecisionMetrics",
    "EvaluationReport",
    "GroupMetrics",
    "RiskCoveragePoint",
    "binary_metrics",
    "calibration_metrics",
    "decision_metrics",
    "evaluate_predictions",
    "group_metrics",
    "risk_coverage_curve",
]
