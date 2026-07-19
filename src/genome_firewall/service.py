"""Application service joining QC, annotation, artifacts, and fail-closed reports."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .amrfinder import (
    AMRFINDER_FEATURE_SCHEMA_VERSION,
    AMRHit,
    AnnotationResult,
    run_amrfinder,
)
from .fasta import analyze_fasta
from .predictor import ModelRegistry
from .release import ReleaseEvaluationError, load_release_evaluation
from .schemas import (
    AnalysisMode,
    DataStatus,
    DrugPrediction,
    EvidenceCategory,
    EvidenceDirection,
    EvidenceItem,
    EvidenceKind,
    NoCallReason,
    PredictionCall,
    PredictionReport,
    TargetStatus,
)
from .targets import assess_target, load_drug_catalog

DEFAULT_SPECIES = "Escherichia coli"
DEFAULT_DRUGS = ("Ciprofloxacin", "Ceftriaxone", "Gentamicin")


def unavailable_report(
    *,
    sample_id: str,
    species: str,
    drugs: Sequence[str],
    qc: Any,
    reason: NoCallReason,
    detail: str,
    provenance: Mapping[str, Any] | None = None,
) -> PredictionReport:
    return PredictionReport(
        sample_id=sample_id,
        supported_species=species,
        model_version="unavailable",
        mode=AnalysisMode.LIVE,
        data_status=DataStatus.UNAVAILABLE,
        qc=qc,
        predictions=[
            DrugPrediction(
                drug=drug,
                call=PredictionCall.NO_CALL,
                p_resistant=None,
                displayed_confidence=None,
                target_status=TargetStatus.NOT_ASSESSED,
                evidence_category=EvidenceCategory.NONE,
                evidence=[],
                no_call_reason=reason,
                decision_reasons=[detail],
                model_version=None,
            )
            for drug in drugs
        ],
        provenance={
            "raw_genome_shared_with_explainer": False,
            "failure_detail": detail,
            **dict(provenance or {}),
        },
    )


def _catalog_context(
    catalog_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, tuple[str, dict[str, Any]]]]:
    if catalog_path is None:
        return {}, {}
    try:
        catalog = load_drug_catalog(catalog_path)
        drugs = catalog["species"]["escherichia_coli"]["drugs"]
        return catalog, {
            str(value["display_name"]).casefold(): (str(key), value) for key, value in drugs.items()
        }
    except (OSError, KeyError, TypeError, ValueError):
        return {}, {}


def _marker_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _hit_quality_status(hit: AMRHit, *, mutation_match: bool) -> str:
    """Classify allowlisted hits using a conservative, explicit policy."""

    method = (hit.method or "").casefold()
    if "exact" in method:
        return "CONFIRMED"
    if mutation_match and "point" in method and (hit.identity_pct or 0.0) >= 90.0:
        return "CONFIRMED"
    if (
        hit.identity_pct is not None
        and hit.coverage_pct is not None
        and hit.identity_pct >= 90.0
        and hit.coverage_pct >= 90.0
    ):
        return "CONFIRMED"
    return "AMBIGUOUS"


def evidence_from_hits(
    hits: Sequence[AMRHit],
    *,
    drugs: Sequence[str],
    catalog_path: str | Path | None = None,
) -> dict[str, list[EvidenceItem]]:
    """Map detected determinants only through a reviewed/configured allowlist."""

    catalog, catalog_drugs = _catalog_context(catalog_path)
    species_reviewed = (
        catalog.get("catalog_status") == "validated"
        and catalog.get("species", {}).get("escherichia_coli", {}).get("status") == "validated"
    )
    result: dict[str, list[EvidenceItem]] = {drug: [] for drug in drugs}
    for drug in drugs:
        _, entry = catalog_drugs.get(drug.casefold(), ("", {}))
        allowlist = entry.get("evidence_allowlist", {}) if isinstance(entry, dict) else {}
        genes = [_marker_key(str(value)) for value in allowlist.get("genes", [])]
        mutations = [_marker_key(str(value)) for value in allowlist.get("mutations", [])]
        reviewed = species_reviewed and entry.get("status") == "validated"
        for hit_index, hit in enumerate(hits):
            normalized = _marker_key(hit.element_symbol)
            gene_match = any(normalized.startswith(value) for value in genes)
            mutation_match = any(value in normalized for value in mutations)
            if not (gene_match or mutation_match):
                continue
            quality_status = _hit_quality_status(hit, mutation_match=mutation_match)
            kind = (
                EvidenceKind.RESISTANCE_MUTATION if mutation_match else EvidenceKind.RESISTANCE_GENE
            )
            result[drug].append(
                EvidenceItem(
                    id=f"amrfinder::{_marker_key(drug)}::{normalized}::{hit_index}",
                    kind=kind,
                    name=hit.element_symbol,
                    direction=EvidenceDirection.RESISTANCE,
                    curated=reviewed and quality_status == "CONFIRMED",
                    source="Local AMRFinderPlus output",
                    description=hit.element_name,
                    details={
                        "method": hit.method,
                        "identity_pct": hit.identity_pct,
                        "coverage_pct": hit.coverage_pct,
                        "quality_status": quality_status,
                    },
                )
            )
    return result


def _target_evidence_and_status(
    *,
    drugs: Sequence[str],
    catalog_path: str | Path | None,
    target_hits_by_drug: Mapping[str, Mapping[str, TargetStatus | str]] | None,
) -> tuple[dict[str, TargetStatus], dict[str, list[EvidenceItem]], dict[str, Any]]:
    """Recompute target gates from observations; never trust a supplied verdict."""

    catalog, catalog_drugs = _catalog_context(catalog_path)
    target_lookup = {key.casefold(): value for key, value in (target_hits_by_drug or {}).items()}
    statuses = {drug: TargetStatus.NOT_ASSESSED for drug in drugs}
    evidence: dict[str, list[EvidenceItem]] = {drug: [] for drug in drugs}
    audit: dict[str, Any] = {
        "catalog_status": catalog.get("catalog_status", "missing"),
        "target_observations_supplied": bool(target_hits_by_drug),
        "target_assessments": {},
    }
    if not catalog:
        audit["target_error"] = "A readable target catalog was not available"
        return statuses, evidence, audit

    for drug in drugs:
        catalog_item = catalog_drugs.get(drug.casefold())
        supplied = target_lookup.get(drug.casefold())
        if catalog_item is None or not isinstance(supplied, Mapping):
            audit["target_assessments"][drug] = "NOT_ASSESSED"
            continue
        drug_key, _ = catalog_item
        try:
            assessment = assess_target(
                catalog,
                species_key="escherichia_coli",
                drug_key=drug_key,
                hits=supplied,
            )
        except (TypeError, ValueError) as exc:
            audit["target_assessments"][drug] = f"INVALID_OBSERVATIONS: {exc}"
            continue
        statuses[drug] = assessment.status
        audit["target_assessments"][drug] = assessment.status.value
        normalized_supplied = {_marker_key(str(key)): value for key, value in supplied.items()}
        for target in assessment.required:
            raw_status = normalized_supplied.get(_marker_key(target), TargetStatus.NOT_ASSESSED)
            try:
                observed = (
                    raw_status
                    if isinstance(raw_status, TargetStatus)
                    else TargetStatus(str(raw_status).upper())
                )
            except ValueError:
                observed = TargetStatus.NOT_ASSESSED
            evidence[drug].append(
                EvidenceItem(
                    id=f"target::{_marker_key(drug)}::{_marker_key(target)}",
                    kind=EvidenceKind.TARGET_DETECTION,
                    name=target,
                    direction=EvidenceDirection.NEUTRAL,
                    curated=assessment.catalog_reviewed,
                    source="Pinned target catalog and precomputed target-search observations",
                    description="Target observation used by the deterministic molecular-target gate.",
                    details={"observed_status": observed.value},
                )
            )
    return statuses, evidence, audit


def _compatibility_error(
    registry: ModelRegistry,
    *,
    species: str,
    drugs: Sequence[str],
    annotation: AnnotationResult,
) -> str | None:
    if not annotation.amrfinder_version:
        return "The live AMRFinderPlus version could not be verified."
    if not annotation.database_version or annotation.database_version.startswith("unverified:"):
        return "The live AMRFinderPlus database version/hash could not be verified."
    for drug in drugs:
        bundle = registry.get(species, drug)
        if bundle is None:
            continue
        if bundle.feature_schema_version != AMRFINDER_FEATURE_SCHEMA_VERSION:
            return f"{drug}: model feature schema is incompatible with the live extractor."
        summary = bundle.training_summary
        expected_schema = summary.get("annotation_feature_schema_version")
        expected_tool = summary.get("amrfinder_version")
        expected_database = summary.get("amrfinder_database_version")
        if not all((expected_schema, expected_tool, expected_database)):
            return f"{drug}: model artifact lacks pinned annotation provenance."
        if str(expected_schema) != AMRFINDER_FEATURE_SCHEMA_VERSION:
            return f"{drug}: pinned annotation feature schema is incompatible."
        if str(expected_tool) != annotation.amrfinder_version:
            return f"{drug}: AMRFinderPlus version differs from the training pipeline."
        if str(expected_database) != annotation.database_version:
            return f"{drug}: AMRFinderPlus database differs from the training pipeline."
    return None


def _annotation_timeout() -> int:
    try:
        value = int(os.getenv("AMRFINDER_TIMEOUT_SECONDS", "600"))
    except ValueError:
        value = 600
    return max(30, min(value, 3_600))


def analyze_upload(
    fasta: bytes | str,
    *,
    sample_id: str = "uploaded-sample",
    species: str = DEFAULT_SPECIES,
    requested_drugs: Sequence[str] = DEFAULT_DRUGS,
    artifact_dir: str | Path | None = None,
    target_hits_by_drug: Mapping[str, Mapping[str, TargetStatus | str]] | None = None,
    catalog_path: str | Path | None = None,
) -> PredictionReport:
    """Analyze an authorized assembled FASTA without ever substituting demo data."""

    drugs = tuple(requested_drugs)
    qc = analyze_fasta(fasta)
    if not qc.passes_qc:
        return unavailable_report(
            sample_id=sample_id,
            species=species,
            drugs=drugs,
            qc=qc,
            reason=NoCallReason.ANNOTATION_OR_QC_FAILURE,
            detail="Uploaded FASTA did not pass the configured structural/assembly QC policy.",
        )

    root = Path(artifact_dir or os.getenv("GENOME_FIREWALL_ARTIFACT_DIR", "artifacts"))
    try:
        release_evaluation = load_release_evaluation(
            root,
            species=species,
            requested_drugs=drugs,
        )
    except ReleaseEvaluationError as exc:
        detail = f"Release evaluation compatibility check failed: {exc}"
        return unavailable_report(
            sample_id=sample_id,
            species=species,
            drugs=drugs,
            qc=qc,
            reason=NoCallReason.MODEL_ERROR,
            detail=detail,
            provenance={"release_evaluation_error": str(exc)[:500]},
        )

    allow_annotation = (
        os.getenv("GENOME_FIREWALL_ALLOW_LIVE_ANNOTATION", "false").casefold() == "true"
    )
    annotation = (
        run_amrfinder(
            fasta,
            organism=os.getenv("AMRFINDER_ORGANISM", "Escherichia"),
            timeout_seconds=_annotation_timeout(),
        )
        if allow_annotation
        else AnnotationResult(ok=False, error="Live annotation is disabled by configuration")
    )
    if not annotation.ok:
        return unavailable_report(
            sample_id=sample_id,
            species=species,
            drugs=drugs,
            qc=qc,
            reason=NoCallReason.ANNOTATION_OR_QC_FAILURE,
            detail=annotation.error or "Annotation unavailable",
            provenance={
                "annotation_ok": False,
                "annotation_error": annotation.error or "Annotation unavailable",
            },
        )

    try:
        release_evaluation.verify_annotation(
            feature_schema_version=AMRFINDER_FEATURE_SCHEMA_VERSION,
            tool_version=annotation.amrfinder_version,
            database_version=annotation.database_version,
        )
    except ReleaseEvaluationError as exc:
        detail = f"Release evaluation compatibility check failed: {exc}"
        return unavailable_report(
            sample_id=sample_id,
            species=species,
            drugs=drugs,
            qc=qc,
            reason=NoCallReason.MODEL_ERROR,
            detail=detail,
            provenance={
                "release_evaluation_error": str(exc)[:500],
                "amrfinder_version": annotation.amrfinder_version,
                "amrfinder_database_version": annotation.database_version,
                "annotation_feature_schema_version": AMRFINDER_FEATURE_SCHEMA_VERSION,
            },
        )

    try:
        registry = ModelRegistry.load(root)
    except Exception as exc:
        return unavailable_report(
            sample_id=sample_id,
            species=species,
            drugs=drugs,
            qc=qc,
            reason=NoCallReason.MODEL_ERROR,
            detail="Compatible trained model artifacts are not installed; no prediction was fabricated.",
            provenance={"artifact_error": str(exc)[:500]},
        )

    incompatibility = _compatibility_error(
        registry,
        species=species,
        drugs=drugs,
        annotation=annotation,
    )
    if incompatibility:
        return unavailable_report(
            sample_id=sample_id,
            species=species,
            drugs=drugs,
            qc=qc,
            reason=NoCallReason.MODEL_ERROR,
            detail=f"Annotation/model compatibility check failed: {incompatibility}",
            provenance={
                "amrfinder_version": annotation.amrfinder_version,
                "amrfinder_database_version": annotation.database_version,
                "annotation_feature_schema_version": AMRFINDER_FEATURE_SCHEMA_VERSION,
            },
        )

    evidence = evidence_from_hits(
        annotation.hits,
        drugs=drugs,
        catalog_path=catalog_path,
    )
    targets, target_evidence, target_audit = _target_evidence_and_status(
        drugs=drugs,
        catalog_path=catalog_path,
        target_hits_by_drug=target_hits_by_drug,
    )
    for drug in drugs:
        evidence[drug].extend(target_evidence[drug])

    try:
        release_evaluation.verify_files_unchanged()
    except ReleaseEvaluationError as exc:
        detail = f"Release evaluation compatibility check failed: {exc}"
        return unavailable_report(
            sample_id=sample_id,
            species=species,
            drugs=drugs,
            qc=qc,
            reason=NoCallReason.MODEL_ERROR,
            detail=detail,
            provenance={"release_evaluation_error": str(exc)[:500]},
        )

    report = registry.predict(
        sample_id=sample_id,
        species=species,
        features=annotation.features,
        qc=qc,
        target_status_by_drug=targets,
        evidence_by_drug=evidence,
        requested_drugs=drugs,
        annotation_ok=True,
        mode=AnalysisMode.LIVE,
        data_status=DataStatus.MODEL_DERIVED,
    )
    return report.model_copy(
        update={
            "provenance": {
                **report.provenance,
                "annotation_ok": True,
                "amrfinder_version": annotation.amrfinder_version,
                "amrfinder_database_version": annotation.database_version,
                "annotation_feature_schema_version": AMRFINDER_FEATURE_SCHEMA_VERSION,
                "release_evaluation_sha256": release_evaluation.release_sha256,
                "release_manifest_sha256": release_evaluation.manifest_sha256,
                "release_dataset_sha256": release_evaluation.dataset_sha256,
                "release_split_sha256": release_evaluation.split_sha256,
                "release_reviewer": release_evaluation.reviewer,
                "release_reviewed_at": release_evaluation.reviewed_at,
                **target_audit,
            }
        }
    )


__all__ = [
    "DEFAULT_DRUGS",
    "DEFAULT_SPECIES",
    "analyze_upload",
    "evidence_from_hits",
    "unavailable_report",
]
