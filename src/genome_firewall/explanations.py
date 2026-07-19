"""Grounded, removable explanation layer.

The scientific pipeline owns every call, probability, target status, and item of
evidence.  This module can narrate those facts, but it cannot create or modify
them.  A deterministic template is always available and is the default.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import RESEARCH_WARNING, EvidenceKind, PredictionReport


class ExplanationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1_500)
    evidence: list[str] = Field(default_factory=list, max_length=12)
    uncertainty: str = Field(min_length=1, max_length=1_500)
    limitations: str = Field(min_length=1, max_length=1_500)
    confirmation_required: bool = True
    source_ids: list[str] = Field(default_factory=list, max_length=32)
    generated_by: Literal["template", "openai"] = "template"


class ExplanationPlan(BaseModel):
    """Constrained plan: the model may prioritize existing facts, not write claims."""

    model_config = ConfigDict(extra="forbid")

    focus: Literal["balanced", "uncertainty", "curated_evidence"] = "balanced"
    drug_order: list[str] = Field(default_factory=list, max_length=12)
    source_ids: list[str] = Field(default_factory=list, max_length=32)


def _as_report(report: PredictionReport | dict[str, Any]) -> PredictionReport:
    if isinstance(report, PredictionReport):
        return report
    if "report" in report and isinstance(report["report"], dict):
        return PredictionReport.model_validate(report["report"])
    return PredictionReport.model_validate(report)


def _label(call: str) -> str:
    return {
        "LIKELY_TO_FAIL": "likely to fail",
        "LIKELY_TO_WORK": "likely to work",
        "NO_CALL": "no-call",
    }.get(call, call.lower().replace("_", " "))


def _evidence_qualifier(kind: EvidenceKind, curated: bool) -> str:
    if kind == EvidenceKind.TARGET_DETECTION:
        return "reviewed target evidence" if curated else "provisional target observation"
    if kind in {EvidenceKind.RESISTANCE_GENE, EvidenceKind.RESISTANCE_MUTATION}:
        return "curated determinant" if curated else "provisional determinant match"
    return "statistical association"


def template_explanation(
    report: PredictionReport | dict[str, Any],
    audience: Literal["laboratory", "plain_language"] = "plain_language",
) -> ExplanationOutput:
    """Generate a deterministic explanation from the validated report only."""

    validated = _as_report(report)
    calls = [f"{item.drug}: {_label(item.call.value)}" for item in validated.predictions]
    evidence: list[str] = []
    source_ids: list[str] = []
    no_calls: list[str] = []

    for prediction in validated.predictions:
        if prediction.no_call_reason is not None:
            no_calls.append(
                f"{prediction.drug} abstained because {prediction.no_call_reason.value.lower().replace('_', ' ')}."
            )
        for item in prediction.evidence:
            qualifier = _evidence_qualifier(item.kind, item.curated)
            evidence.append(f"{prediction.drug}: {item.name} ({qualifier}).")
            if item.id:
                source_ids.append(item.id)

    if audience == "laboratory":
        prefix = "The genomic decision-support report returned"
        limitations = (
            "Calls are based on the installed model bundle, target gate, and detected genomic "
            "evidence. Genotype does not prove phenotype, and statistical associations are not "
            "biological causation."
        )
    else:
        prefix = "This research-only genome screen returned"
        limitations = (
            "A genome can provide early clues, but it cannot replace laboratory susceptibility "
            "testing or professional review."
        )

    uncertainty = (
        " ".join(no_calls)
        if no_calls
        else "No drug was abstained in this report, but every call still requires laboratory confirmation."
    )
    return ExplanationOutput(
        summary=f"{prefix}: " + "; ".join(calls) + ".",
        evidence=evidence[:12],
        uncertainty=uncertainty,
        limitations=limitations,
        confirmation_required=True,
        source_ids=list(dict.fromkeys(source_ids)),
        generated_by="template",
    )


def _grounded_payload(report: PredictionReport) -> dict[str, Any]:
    """Return the only facts that may be sent to the language model.

    Raw FASTA, patient data, QC sequence content, and model internals are excluded.
    """

    return {
        "supported_species": report.supported_species,
        "mode": report.mode.value,
        "data_status": report.data_status.value,
        "demo_disclaimer": report.demo_disclaimer,
        "warning": report.warning,
        "predictions": [
            {
                "drug": prediction.drug,
                "call": prediction.call.value,
                "target_status": prediction.target_status.value,
                "evidence_category": prediction.evidence_category.value,
                "no_call_reason": (
                    prediction.no_call_reason.value if prediction.no_call_reason else None
                ),
                "evidence": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "kind": item.kind.value,
                        "curated": item.curated,
                    }
                    for item in prediction.evidence
                ],
            }
            for prediction in report.predictions
        ],
    }


def openai_explanation(
    report: PredictionReport | dict[str, Any],
    audience: Literal["laboratory", "plain_language"] = "plain_language",
    *,
    model: str | None = None,
) -> ExplanationOutput:
    """Use OpenAI only to prioritize existing evidence for deterministic copy.

    The model returns enums and allowlisted IDs, never free-form medical prose.
    Rendering remains deterministic. If the API is unavailable or its plan is
    invalid, the standard template is returned.
    """

    validated = _as_report(report)
    fallback = template_explanation(validated, audience)
    if not os.getenv("OPENAI_API_KEY"):
        return fallback

    try:
        from openai import OpenAI

        timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
        client = OpenAI(timeout=max(1.0, min(timeout, 120.0)))
        payload = _grounded_payload(validated)
        response = client.responses.parse(
            model=model or os.getenv("OPENAI_MODEL", "gpt-5.6-terra"),
            instructions=(
                "Plan the ordering of facts in a research-only AMR explanation. Return only one "
                "allowed focus enum, drug names copied exactly from the input, and evidence source "
                "IDs copied exactly from the input. Do not write prose. Do not rank treatment, "
                "recommend a drug, or introduce any new fact. Prioritize uncertainty and curated "
                f"evidence appropriately for this audience: {audience}."
            ),
            input=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            text_format=ExplanationPlan,
            max_output_tokens=300,
        )
        plan = response.output_parsed
        if plan is None:
            return fallback
        allowed_ids = {
            item.id
            for prediction in validated.predictions
            for item in prediction.evidence
            if item.id
        }
        allowed_drugs = {prediction.drug for prediction in validated.predictions}
        if not set(plan.source_ids).issubset(allowed_ids):
            return fallback
        if not set(plan.drug_order).issubset(allowed_drugs):
            return fallback

        evidence_by_id: dict[str, str] = {}
        for prediction in validated.predictions:
            for item in prediction.evidence:
                if item.id:
                    qualifier = _evidence_qualifier(item.kind, item.curated)
                    evidence_by_id[item.id] = f"{prediction.drug}: {item.name} ({qualifier})."
        ordered = [evidence_by_id[source_id] for source_id in plan.source_ids]
        ordered.extend(item for item in fallback.evidence if item not in ordered)
        return fallback.model_copy(
            update={
                "evidence": ordered[:12],
                "source_ids": plan.source_ids or fallback.source_ids,
                "generated_by": "openai",
            }
        )
    except Exception:
        return fallback


def explain_report(
    report: PredictionReport | dict[str, Any],
    audience: Literal["laboratory", "plain_language"] = "plain_language",
    *,
    use_openai: bool | None = None,
) -> dict[str, Any]:
    """UI-friendly facade returning a JSON-safe dictionary."""

    enabled = (
        (
            os.getenv(
                "GENOME_FIREWALL_ENABLE_OPENAI",
                os.getenv("GENOME_FIREWALL_USE_OPENAI", "false"),
            ).casefold()
            == "true"
        )
        if use_openai is None
        else use_openai
    )
    result = (
        openai_explanation(report, audience) if enabled else template_explanation(report, audience)
    )
    return result.model_dump(mode="json")


def guarded_question(question: str) -> str:
    """Deterministically enforce the treatment-advice boundary in the demo."""

    normalized = question.casefold()
    prohibited = (
        "prescribe",
        "which drug",
        "which antibiotic",
        "recommend",
        "dose",
        "dosage",
        "treatment",
        "should i use",
    )
    if any(term in normalized for term in prohibited):
        return (
            "I cannot select, recommend, or dose a treatment. Genome Firewall is research "
            "decision support only. " + RESEARCH_WARNING
        )
    return (
        "I can explain only the calls, evidence categories, target status, uncertainty, and "
        "limitations already present in the validated report."
    )


__all__ = [
    "ExplanationOutput",
    "ExplanationPlan",
    "explain_report",
    "guarded_question",
    "openai_explanation",
    "template_explanation",
]
