"""Transparent demonstration fixtures for the hackathon interface.

These objects exist only so the product can be judged before the organizer's
fixed dataset and trained artifacts are installed.  They are intentionally
impossible to confuse with model-derived results: every report is in DEMO mode,
uses the DEMONSTRATION_FIXTURE data status, and carries a prominent disclaimer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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

DEMO_DISCLAIMER = (
    "ILLUSTRATIVE DEMO FIXTURE - NOT A MODEL OR BENCHMARK RESULT. Values are "
    "synthetic and exist only to demonstrate the interface and safety logic."
)


def _target(name: str, accession: str) -> EvidenceItem:
    return EvidenceItem(
        id=f"target::{name}",
        kind=EvidenceKind.TARGET_DETECTION,
        name=name,
        direction=EvidenceDirection.NEUTRAL,
        curated=True,
        source=f"Illustrative target-panel reference {accession}",
        description="Illustrative target-presence evidence for UI demonstration only.",
    )


def _mutation(name: str) -> EvidenceItem:
    return EvidenceItem(
        id=f"mutation::{name}",
        kind=EvidenceKind.RESISTANCE_MUTATION,
        name=name,
        direction=EvidenceDirection.RESISTANCE,
        curated=True,
        source="Illustrative AMRFinderPlus-style fixture",
        description="Curated-determinant style evidence in the demonstration fixture.",
    )


def _gene(name: str) -> EvidenceItem:
    return EvidenceItem(
        id=f"gene::{name}",
        kind=EvidenceKind.RESISTANCE_GENE,
        name=name,
        direction=EvidenceDirection.RESISTANCE,
        curated=True,
        source="Illustrative AMRFinderPlus-style fixture",
        description="Curated-determinant style evidence in the demonstration fixture.",
    )


def _association(name: str) -> EvidenceItem:
    return EvidenceItem(
        id=f"feature::{name}",
        kind=EvidenceKind.MODEL_FEATURE,
        name=name,
        direction=EvidenceDirection.RESISTANCE,
        curated=False,
        source="Illustrative elastic-net coefficient",
        description=(
            "Statistical association only. This fixture does not claim biological causation."
        ),
    )


def _report(sample_id: str, predictions: list[DrugPrediction]) -> PredictionReport:
    return PredictionReport(
        sample_id=sample_id,
        supported_species="Escherichia coli",
        model_version="demo-fixture-1.0",
        predictions=predictions,
        mode=AnalysisMode.DEMO,
        data_status=DataStatus.DEMONSTRATION_FIXTURE,
        demo_disclaimer=DEMO_DISCLAIMER,
        provenance={
            "fixture": True,
            "benchmark_valid": False,
            "amrfinder_version": "illustrative-only",
            "dataset": "No organizer dataset installed",
        },
    )


_METRICS: dict[str, Any] = {
    "status": "UNAVAILABLE_NO_ORGANIZER_EVALUATION",
    "disclaimer": (
        "No numeric performance fixture is bundled. Install the organizer dataset, train the "
        "models, and load an untouched group-held-out evaluation artifact to populate metrics."
    ),
    "per_drug": [],
    "reliability": [],
    "risk_coverage": [],
}


def _marker_fail_case() -> dict[str, Any]:
    report = _report(
        "DEMO-EC-001",
        [
            DrugPrediction(
                drug="Ciprofloxacin",
                call=PredictionCall.LIKELY_TO_FAIL,
                p_resistant=0.96,
                displayed_confidence=0.96,
                target_status=TargetStatus.PRESENT,
                evidence_category=EvidenceCategory.CURATED_DETERMINANT,
                evidence=[
                    _target("DNA gyrase / topoisomerase target panel", "DEMO-TGT-001"),
                    _mutation("gyrA S83L"),
                    _mutation("parC S80I"),
                ],
                decision_reasons=[
                    "Required molecular-target panel is present.",
                    "Calibrated probability is above the illustrative fail threshold.",
                    "Curated resistance-determinant style evidence is present.",
                ],
                model_version="demo-fixture-1.0",
            ),
            DrugPrediction(
                drug="Ceftriaxone",
                call=PredictionCall.LIKELY_TO_WORK,
                p_resistant=0.08,
                displayed_confidence=0.92,
                target_status=TargetStatus.PRESENT,
                evidence_category=EvidenceCategory.NONE,
                evidence=[_target("Penicillin-binding target panel", "DEMO-TGT-002")],
                decision_reasons=[
                    "Required molecular-target panel is present.",
                    "No curated resistance signal is present in this illustrative fixture.",
                    "Calibrated probability is below the illustrative work threshold.",
                ],
                model_version="demo-fixture-1.0",
            ),
            DrugPrediction(
                drug="Gentamicin",
                call=PredictionCall.NO_CALL,
                p_resistant=0.58,
                displayed_confidence=None,
                target_status=TargetStatus.PRESENT,
                evidence_category=EvidenceCategory.STATISTICAL_ASSOCIATION,
                evidence=[
                    _target("30S ribosomal target panel", "DEMO-TGT-003"),
                    _association("gene-family profile 17"),
                ],
                no_call_reason=NoCallReason.LOW_CONFIDENCE,
                decision_reasons=[
                    "Probability lies inside the abstention interval.",
                    "Statistical association is not treated as biological proof.",
                ],
                model_version="demo-fixture-1.0",
            ),
        ],
    )
    return {
        "case_id": "marker_fail",
        "title": "Evidence-first mixed result",
        "subtitle": "Shows fail, work, and no-call in one transparent report.",
        "report": report.model_dump(mode="json"),
        "metrics": deepcopy(_METRICS),
        "decision_context": {},
    }


def _target_work_case() -> dict[str, Any]:
    report = _report(
        "DEMO-EC-002",
        [
            DrugPrediction(
                drug="Ciprofloxacin",
                call=PredictionCall.LIKELY_TO_WORK,
                p_resistant=0.12,
                displayed_confidence=0.88,
                target_status=TargetStatus.PRESENT,
                evidence_category=EvidenceCategory.NONE,
                evidence=[_target("DNA gyrase / topoisomerase target panel", "DEMO-TGT-001")],
                decision_reasons=[
                    "Target confirmed; illustrative probability is below the work threshold."
                ],
                model_version="demo-fixture-1.0",
            ),
            DrugPrediction(
                drug="Ceftriaxone",
                call=PredictionCall.LIKELY_TO_WORK,
                p_resistant=0.07,
                displayed_confidence=0.93,
                target_status=TargetStatus.PRESENT,
                evidence_category=EvidenceCategory.NONE,
                evidence=[_target("Penicillin-binding target panel", "DEMO-TGT-002")],
                decision_reasons=[
                    "Target confirmed; illustrative probability is below the work threshold."
                ],
                model_version="demo-fixture-1.0",
            ),
            DrugPrediction(
                drug="Gentamicin",
                call=PredictionCall.NO_CALL,
                p_resistant=0.19,
                displayed_confidence=None,
                target_status=TargetStatus.AMBIGUOUS,
                evidence_category=EvidenceCategory.NONE,
                evidence=[],
                no_call_reason=NoCallReason.TARGET_AMBIGUOUS,
                decision_reasons=["A work call is prohibited while target evidence is ambiguous."],
                model_version="demo-fixture-1.0",
            ),
        ],
    )
    return {
        "case_id": "target_work",
        "title": "Target-gated likely-work case",
        "subtitle": "Demonstrates that low resistance probability is not enough without target confirmation.",
        "report": report.model_dump(mode="json"),
        "metrics": deepcopy(_METRICS),
        "decision_context": {},
    }


def _honest_no_call_case() -> dict[str, Any]:
    report = _report(
        "DEMO-EC-003",
        [
            DrugPrediction(
                drug="Ciprofloxacin",
                call=PredictionCall.NO_CALL,
                p_resistant=0.94,
                displayed_confidence=None,
                target_status=TargetStatus.PRESENT,
                evidence_category=EvidenceCategory.CURATED_DETERMINANT,
                evidence=[_mutation("gyrA S83L")],
                no_call_reason=NoCallReason.GENOME_OUT_OF_DISTRIBUTION,
                decision_reasons=["High model score is overridden by the genome novelty gate."],
                model_version="demo-fixture-1.0",
            ),
            DrugPrediction(
                drug="Ceftriaxone",
                call=PredictionCall.NO_CALL,
                p_resistant=0.06,
                displayed_confidence=None,
                target_status=TargetStatus.PRESENT,
                evidence_category=EvidenceCategory.STATISTICAL_ASSOCIATION,
                evidence=[_association("unseen AMR feature profile")],
                no_call_reason=NoCallReason.FEATURE_PROFILE_NOVEL,
                decision_reasons=["A low model score cannot bypass the feature-novelty gate."],
                model_version="demo-fixture-1.0",
            ),
            DrugPrediction(
                drug="Gentamicin",
                call=PredictionCall.NO_CALL,
                p_resistant=0.51,
                displayed_confidence=None,
                target_status=TargetStatus.AMBIGUOUS,
                evidence_category=EvidenceCategory.NONE,
                evidence=[],
                no_call_reason=NoCallReason.TARGET_AMBIGUOUS,
                decision_reasons=["Fragmented target evidence triggers abstention."],
                model_version="demo-fixture-1.0",
            ),
        ],
    )
    return {
        "case_id": "honest_no_call",
        "title": "Honest abstention under novelty",
        "subtitle": "Shows that safety gates override even apparently confident model scores.",
        "report": report.model_dump(mode="json"),
        "metrics": deepcopy(_METRICS),
        "decision_context": {},
    }


_LOADERS = {
    "marker_fail": _marker_fail_case,
    "target_work": _target_work_case,
    "honest_no_call": _honest_no_call_case,
}


def available_demo_cases() -> list[dict[str, str]]:
    """Return chooser metadata without constructing the full reports."""

    return [
        {"case_id": case_id, "title": loader()["title"], "subtitle": loader()["subtitle"]}
        for case_id, loader in _LOADERS.items()
    ]


def load_demo_case(case_id: str = "marker_fail") -> dict[str, Any]:
    """Load one validated, explicitly illustrative demo fixture."""

    try:
        return _LOADERS[case_id]()
    except KeyError as exc:
        choices = ", ".join(sorted(_LOADERS))
        raise ValueError(f"Unknown demo case {case_id!r}; choose one of: {choices}") from exc


__all__ = ["DEMO_DISCLAIMER", "available_demo_cases", "load_demo_case"]
