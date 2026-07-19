"""Reviewed target-catalog loading and deterministic presence assessment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas import TargetStatus


class TargetAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: TargetStatus
    required: list[str] = Field(default_factory=list)
    detected: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    ambiguous: list[str] = Field(default_factory=list)
    unassessed: list[str] = Field(default_factory=list)
    catalog_reviewed: bool = False
    reason: str


def load_drug_catalog(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, dict) or "species" not in catalog:
        raise ValueError("Drug catalog must contain a species mapping")
    return catalog


def assess_target(
    catalog: Mapping[str, Any],
    *,
    species_key: str,
    drug_key: str,
    hits: Mapping[str, TargetStatus | str],
) -> TargetAssessment:
    """Assess a precomputed target-hit panel without inferring susceptibility.

    `hits` must be supplied by a separately validated target-search workflow.
    Provisional catalog entries are deliberately ambiguous in trained mode.
    """

    try:
        species = catalog["species"][species_key]
        drug = species["drugs"][drug_key]
        gate = drug["target_gate"]
    except (KeyError, TypeError):
        return TargetAssessment(
            status=TargetStatus.NOT_ASSESSED,
            reason=f"No target rule for {species_key}/{drug_key}",
        )

    required = [str(value) for value in gate.get("targets", [])]
    normalized_hits = {
        str(key).casefold(): (
            value if isinstance(value, TargetStatus) else TargetStatus(str(value).upper())
        )
        for key, value in hits.items()
    }
    detected = [
        name for name in required if normalized_hits.get(name.casefold()) == TargetStatus.PRESENT
    ]
    ambiguous = [
        name for name in required if normalized_hits.get(name.casefold()) == TargetStatus.AMBIGUOUS
    ]
    missing = [
        name
        for name in required
        if normalized_hits.get(name.casefold(), TargetStatus.NOT_ASSESSED) == TargetStatus.ABSENT
    ]
    unassessed = [
        name
        for name in required
        if normalized_hits.get(name.casefold(), TargetStatus.NOT_ASSESSED)
        == TargetStatus.NOT_ASSESSED
    ]
    reviewed = (
        species.get("status") == "validated"
        and drug.get("status") == "validated"
        and catalog.get("catalog_status") == "validated"
    )
    logic = gate.get("logic", "all_of")

    if not reviewed:
        status = TargetStatus.AMBIGUOUS
        reason = "Target catalog entry is provisional and cannot enable a trained likely-work call."
    elif ambiguous:
        status = TargetStatus.AMBIGUOUS
        reason = "At least one required target hit is partial or ambiguous."
    elif not required:
        status = TargetStatus.NOT_ASSESSED
        reason = "The reviewed target rule contains no required targets."
    elif len(unassessed) == len(required):
        status = TargetStatus.NOT_ASSESSED
        reason = "No required target has an explicit observation."
    elif logic == "any_of" and detected:
        status = TargetStatus.PRESENT
        reason = "At least one reviewed target is confirmed present."
    elif logic in {"all_of", "reviewed_panel"} and not missing and not unassessed:
        status = TargetStatus.PRESENT
        reason = "All required targets in the reviewed panel are confirmed present."
    elif len(missing) == len(required):
        status = TargetStatus.ABSENT
        reason = "No required target was detected by the validated panel."
    else:
        status = TargetStatus.AMBIGUOUS
        reason = "Target panel is incomplete for this genome."

    return TargetAssessment(
        status=status,
        required=required,
        detected=detected,
        missing=missing,
        ambiguous=ambiguous,
        unassessed=unassessed,
        catalog_reviewed=reviewed,
        reason=reason,
    )


__all__ = ["TargetAssessment", "assess_target", "load_drug_catalog"]
