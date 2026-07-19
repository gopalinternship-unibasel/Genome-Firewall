"""Versioned model registry and end-to-end structured prediction assembly."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decision import DecisionPolicy
from .schemas import (
    AnalysisMode,
    DataStatus,
    DecisionInput,
    DecisionThresholds,
    EvidenceItem,
    FastaQC,
    PredictionReport,
    TargetStatus,
)
from .training import ModelArtifactError, ModelBundle


def _key(species: str, drug: str) -> tuple[str, str]:
    return species.strip().casefold(), drug.strip().casefold()


def _artifact_name(bundle: ModelBundle) -> str:
    readable = re.sub(r"[^a-z0-9]+", "-", bundle.drug.casefold()).strip("-") or "drug"
    identity = f"{bundle.species}\0{bundle.drug}\0{bundle.model_version}"
    suffix = sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{suffix}.joblib"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coerce_evidence(values: Sequence[EvidenceItem | Mapping[str, Any]]) -> list[EvidenceItem]:
    return [
        value if isinstance(value, EvidenceItem) else EvidenceItem.model_validate(value)
        for value in values
    ]


def _require_single_sample(features: Mapping[str, float] | pd.DataFrame | np.ndarray) -> None:
    if isinstance(features, Mapping):
        return
    if isinstance(features, pd.DataFrame):
        rows = len(features)
    else:
        array = np.asarray(features)
        if array.ndim == 1:
            rows = 1
        elif array.ndim == 2:
            rows = int(array.shape[0])
        else:
            raise ValueError("inference features must be one- or two-dimensional")
    if rows != 1:
        raise ValueError("ModelRegistry.predict requires exactly one sample")


class ModelRegistry:
    """Collection of one independently calibrated model per species/drug pair."""

    manifest_format_version = 1

    def __init__(
        self,
        bundles: Sequence[ModelBundle] | None = None,
        *,
        registry_version: str = "0.1.0",
    ) -> None:
        self.registry_version = registry_version
        self._bundles: dict[tuple[str, str], ModelBundle] = {}
        for bundle in bundles or ():
            self.register(bundle)

    def __len__(self) -> int:
        return len(self._bundles)

    def register(self, bundle: ModelBundle, *, replace: bool = False) -> None:
        if not isinstance(bundle, ModelBundle):
            raise TypeError("bundle must be a ModelBundle")
        key = _key(bundle.species, bundle.drug)
        if key in self._bundles and not replace:
            raise ValueError(
                f"a model is already registered for {bundle.species!r}/{bundle.drug!r}"
            )
        self._bundles[key] = bundle

    def get(self, species: str, drug: str) -> ModelBundle | None:
        return self._bundles.get(_key(species, drug))

    def supported_species(self) -> list[str]:
        return sorted({bundle.species for bundle in self._bundles.values()})

    def supported_drugs(self, species: str) -> list[str]:
        species_key = species.strip().casefold()
        return sorted(
            bundle.drug
            for (registered_species, _), bundle in self._bundles.items()
            if registered_species == species_key
        )

    def predict(
        self,
        *,
        sample_id: str,
        species: str,
        features: Mapping[str, float] | pd.DataFrame | np.ndarray,
        qc: FastaQC,
        target_status_by_drug: Mapping[str, TargetStatus | str],
        evidence_by_drug: Mapping[str, Sequence[EvidenceItem | Mapping[str, Any]]] | None = None,
        requested_drugs: Sequence[str] | None = None,
        is_ood: bool = False,
        ood_by_drug: Mapping[str, bool] | None = None,
        feature_profile_novel_by_drug: Mapping[str, bool] | None = None,
        validated_absent_target_rule_by_drug: Mapping[str, bool] | None = None,
        annotation_ok: bool = True,
        mode: AnalysisMode = AnalysisMode.LIVE,
        data_status: DataStatus | None = None,
        demo_disclaimer: str | None = None,
    ) -> PredictionReport:
        """Predict requested drugs and apply all gates; model errors become no-calls."""

        if not isinstance(qc, FastaQC):
            raise TypeError("qc must be a FastaQC result from analyze_fasta")
        _require_single_sample(features)
        species_drugs = self.supported_drugs(species)
        if requested_drugs is None:
            drugs = species_drugs or sorted({bundle.drug for bundle in self._bundles.values()})
        else:
            drugs = [str(drug).strip() for drug in requested_drugs]
        if not drugs or any(not drug for drug in drugs):
            raise ValueError("at least one non-empty requested drug is required")
        if len({drug.casefold() for drug in drugs}) != len(drugs):
            raise ValueError("requested drugs must be unique")

        target_lookup = {
            key.casefold(): TargetStatus(value) for key, value in target_status_by_drug.items()
        }
        evidence_lookup = {
            key.casefold(): _coerce_evidence(value)
            for key, value in (evidence_by_drug or {}).items()
        }
        ood_lookup = {key.casefold(): value for key, value in (ood_by_drug or {}).items()}
        novelty_lookup = {
            key.casefold(): value for key, value in (feature_profile_novel_by_drug or {}).items()
        }
        absent_rule_lookup = {
            key.casefold(): value
            for key, value in (validated_absent_target_rule_by_drug or {}).items()
        }
        species_supported = bool(species_drugs)
        policy = DecisionPolicy()
        predictions = []
        model_versions: dict[str, str] = {}

        for drug in drugs:
            folded_drug = drug.casefold()
            bundle = self.get(species, drug)
            target_status = target_lookup.get(folded_drug, TargetStatus.NOT_ASSESSED)
            evidence = evidence_lookup.get(folded_drug, [])
            p_resistant: float | None = None
            feature_novel = novelty_lookup.get(folded_drug, False)
            thresholds = DecisionThresholds(
                work_max=0.0,
                fail_min=1.0,
                supported=False,
                calibration_samples=0,
                calibration_groups=0,
            )
            model_version: str | None = None

            if bundle is not None:
                thresholds = bundle.thresholds
                model_version = bundle.model_version
                model_versions[drug] = bundle.model_version
                try:
                    unexpected = bundle.unexpected_nonzero_features(features)
                    if unexpected:
                        feature_novel = True
                    else:
                        p_resistant = float(bundle.predict_proba(features)[0])
                        if folded_drug not in novelty_lookup:
                            feature_novel = bool(bundle.is_feature_profile_novel(features)[0])
                except (ValueError, TypeError, ModelArtifactError, IndexError):
                    # A scientific pipeline must fail closed.  p_resistant remains
                    # None and the deterministic policy returns MODEL_ERROR.
                    p_resistant = None
                    feature_novel = feature_novel or False

            request = DecisionInput(
                drug=drug,
                p_resistant=p_resistant,
                target_status=target_status,
                evidence=evidence,
                qc_passed=qc.passes_qc,
                annotation_ok=annotation_ok,
                species_supported=species_supported,
                drug_supported=bundle is not None,
                is_ood=is_ood or ood_lookup.get(folded_drug, False),
                feature_profile_novel=feature_novel,
                validated_absent_target_rule=absent_rule_lookup.get(folded_drug, False),
            )
            predictions.append(policy.decide(request, thresholds, model_version=model_version))

        effective_status = data_status or (
            DataStatus.DEMONSTRATION_FIXTURE
            if mode == AnalysisMode.DEMO
            else DataStatus.MODEL_DERIVED
        )
        return PredictionReport(
            sample_id=sample_id,
            supported_species=species,
            model_version=self.registry_version,
            predictions=predictions,
            mode=mode,
            data_status=effective_status,
            demo_disclaimer=demo_disclaimer,
            qc=qc,
            provenance={
                "registry_version": self.registry_version,
                "feature_sha256": qc.sha256,
                "model_versions": model_versions,
                "raw_genome_shared_with_explainer": False,
            },
        )

    def save(self, directory: str | Path) -> Path:
        """Save all bundles and a hash-verified JSON manifest."""

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for bundle in sorted(
            self._bundles.values(), key=lambda item: (item.species.casefold(), item.drug.casefold())
        ):
            filename = _artifact_name(bundle)
            artifact_path = bundle.save(root / filename)
            entries.append(
                {
                    "species": bundle.species,
                    "drug": bundle.drug,
                    "model_version": bundle.model_version,
                    "file": filename,
                    "sha256": _file_sha256(artifact_path),
                    "metadata": bundle.metadata(),
                }
            )
        manifest = {
            "manifest_format_version": self.manifest_format_version,
            "registry_version": self.registry_version,
            "created_at": datetime.now(UTC).isoformat(),
            "models": entries,
        }
        destination = root / "manifest.json"
        temporary = root / ".manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> ModelRegistry:
        """Load trusted local artifacts after manifest path and hash validation."""

        root = Path(directory).resolve()
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError("could not read a valid registry manifest") from exc
        if manifest.get("manifest_format_version") != cls.manifest_format_version:
            raise ModelArtifactError("unsupported registry manifest format")
        entries = manifest.get("models")
        if not isinstance(entries, list):
            raise ModelArtifactError("registry manifest models must be a list")
        registry = cls(registry_version=str(manifest.get("registry_version", "unknown")))
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ModelArtifactError("registry manifest contains an invalid model entry")
            artifact_path = (root / entry["file"]).resolve()
            if root not in artifact_path.parents:
                raise ModelArtifactError("registry artifact path escapes its directory")
            if not artifact_path.is_file():
                raise ModelArtifactError(f"missing model artifact: {entry['file']}")
            expected_hash = entry.get("sha256")
            if not isinstance(expected_hash, str) or _file_sha256(artifact_path) != expected_hash:
                raise ModelArtifactError(f"model artifact hash mismatch: {entry['file']}")
            bundle = ModelBundle.load(artifact_path)
            if bundle.species != entry.get("species") or bundle.drug != entry.get("drug"):
                raise ModelArtifactError("model artifact identity does not match manifest")
            registry.register(bundle)
        return registry


__all__ = ["ModelRegistry"]
