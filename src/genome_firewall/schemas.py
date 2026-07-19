"""Validated public contracts for Genome Firewall.

The schemas in this module are deliberately small and JSON-safe.  They are the
only objects the product layer needs to render; raw model objects and raw FASTA
data never need to cross that boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

RESEARCH_WARNING = (
    "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE. This report provides genomic "
    "decision support only and does not select treatment. Every antibiotic-response "
    "result must be confirmed with standard laboratory susceptibility testing and "
    "reviewed by a trained healthcare or laboratory professional."
)


class StrictModel(BaseModel):
    """Base class that rejects silently misspelled contract fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PredictionCall(StrEnum):
    LIKELY_TO_WORK = "LIKELY_TO_WORK"
    LIKELY_TO_FAIL = "LIKELY_TO_FAIL"
    NO_CALL = "NO_CALL"


class TargetStatus(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_ASSESSED = "NOT_ASSESSED"


class EvidenceCategory(StrEnum):
    CURATED_DETERMINANT = "CURATED_DETERMINANT"
    STATISTICAL_ASSOCIATION = "STATISTICAL_ASSOCIATION"
    MIXED = "MIXED"
    NONE = "NONE"


class EvidenceKind(StrEnum):
    RESISTANCE_GENE = "RESISTANCE_GENE"
    RESISTANCE_MUTATION = "RESISTANCE_MUTATION"
    TARGET_DETECTION = "TARGET_DETECTION"
    MODEL_FEATURE = "MODEL_FEATURE"
    OTHER = "OTHER"


class EvidenceDirection(StrEnum):
    RESISTANCE = "RESISTANCE"
    SUSCEPTIBILITY = "SUSCEPTIBILITY"
    NEUTRAL = "NEUTRAL"


class NoCallReason(StrEnum):
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TARGET_AMBIGUOUS = "TARGET_AMBIGUOUS"
    TARGET_NOT_ASSESSED = "TARGET_NOT_ASSESSED"
    TARGET_ABSENT_UNVALIDATED = "TARGET_ABSENT_UNVALIDATED"
    KNOWN_MARKER_CONFLICT = "KNOWN_MARKER_CONFLICT"
    EVIDENCE_AMBIGUOUS = "EVIDENCE_AMBIGUOUS"
    GENOME_OUT_OF_DISTRIBUTION = "GENOME_OUT_OF_DISTRIBUTION"
    FEATURE_PROFILE_NOVEL = "FEATURE_PROFILE_NOVEL"
    INSUFFICIENT_CALIBRATION_SUPPORT = "INSUFFICIENT_CALIBRATION_SUPPORT"
    UNSUPPORTED_SPECIES = "UNSUPPORTED_SPECIES"
    UNSUPPORTED_DRUG = "UNSUPPORTED_DRUG"
    ANNOTATION_OR_QC_FAILURE = "ANNOTATION_OR_QC_FAILURE"
    MODEL_ERROR = "MODEL_ERROR"


class AnalysisMode(StrEnum):
    LIVE = "LIVE"
    DEMO = "DEMO"


class DataStatus(StrEnum):
    MODEL_DERIVED = "MODEL_DERIVED"
    DEMONSTRATION_FIXTURE = "DEMONSTRATION_FIXTURE"
    UNAVAILABLE = "UNAVAILABLE"


class FastaLimits(StrictModel):
    """Assembly-level QC limits.

    The defaults are intentionally conservative for a bacterial assembly.  Unit
    tests and non-bacterial research can supply a different policy explicitly.
    """

    max_file_bytes: int = Field(default=50_000_000, gt=0)
    min_total_bases: int = Field(default=100_000, ge=1)
    max_total_bases: int = Field(default=15_000_000, ge=1)
    max_contigs: int = Field(default=1_000, ge=1)
    min_n50: int = Field(default=1_000, ge=1)
    max_ambiguous_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    max_duplicate_headers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> FastaLimits:
        if self.min_total_bases > self.max_total_bases:
            raise ValueError("min_total_bases must not exceed max_total_bases")
        return self


class FastaQC(StrictModel):
    """Structural and assembly QC result for one FASTA payload."""

    valid_fasta: bool
    passes_qc: bool
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_bytes: int = Field(ge=0)
    sequence_count: int = Field(ge=0)
    total_bases: int = Field(ge=0)
    n50: int = Field(ge=0)
    l50: int = Field(ge=0)
    shortest_contig: int = Field(ge=0)
    longest_contig: int = Field(ge=0)
    gc_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    ambiguous_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    n_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    invalid_character_count: int = Field(ge=0)
    duplicate_header_count: int = Field(ge=0)
    empty_sequence_count: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidenceItem(StrictModel):
    """One auditable item; model associations are never labeled as curated."""

    id: str | None = None
    kind: EvidenceKind
    name: str = Field(min_length=1, max_length=256)
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    curated: bool = False
    source: str | None = Field(default=None, max_length=512)
    description: str | None = Field(default=None, max_length=2_000)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def protect_curated_label(self) -> EvidenceItem:
        if self.kind == EvidenceKind.MODEL_FEATURE and self.curated:
            raise ValueError("MODEL_FEATURE evidence cannot be labeled curated")
        return self


class DecisionThresholds(StrictModel):
    """Drug-specific abstention thresholds learned on calibration groups."""

    work_max: float = Field(ge=0.0, le=1.0)
    fail_min: float = Field(ge=0.0, le=1.0)
    supported: bool = True
    calibration_samples: int = Field(default=0, ge=0)
    calibration_groups: int = Field(default=0, ge=0)
    max_called_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_false_susceptible_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_gap(self) -> DecisionThresholds:
        if self.work_max >= self.fail_min:
            raise ValueError("work_max must be strictly lower than fail_min")
        if self.supported and (self.calibration_samples <= 0 or self.calibration_groups <= 0):
            raise ValueError("supported thresholds require calibration samples and groups")
        return self


class DecisionInput(StrictModel):
    """Complete deterministic-policy input for one species/drug pair."""

    drug: str = Field(min_length=1, max_length=256)
    p_resistant: float | None = Field(default=None, ge=0.0, le=1.0)
    target_status: TargetStatus = TargetStatus.NOT_ASSESSED
    evidence: list[EvidenceItem] = Field(default_factory=list)
    qc_passed: bool = True
    annotation_ok: bool = True
    species_supported: bool = True
    drug_supported: bool = True
    is_ood: bool = False
    feature_profile_novel: bool = False
    validated_absent_target_rule: bool = False


class DrugPrediction(StrictModel):
    drug: str = Field(min_length=1, max_length=256)
    call: PredictionCall
    p_resistant: float | None = Field(default=None, ge=0.0, le=1.0)
    displayed_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    target_status: TargetStatus
    evidence_category: EvidenceCategory
    evidence: list[EvidenceItem] = Field(default_factory=list)
    no_call_reason: NoCallReason | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    model_version: str | None = None
    work_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fail_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_samples: int | None = Field(default=None, ge=1)
    calibration_groups: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_call_contract(self) -> DrugPrediction:
        if self.call == PredictionCall.NO_CALL:
            if self.no_call_reason is None:
                raise ValueError("NO_CALL requires no_call_reason")
            if self.displayed_confidence is not None:
                raise ValueError("NO_CALL must not display a confidence")
        elif self.no_call_reason is not None:
            raise ValueError("called predictions cannot contain no_call_reason")
        if (
            self.call == PredictionCall.LIKELY_TO_WORK
            and self.target_status != TargetStatus.PRESENT
        ):
            raise ValueError("LIKELY_TO_WORK requires a PRESENT target status")
        if (self.work_threshold is None) != (self.fail_threshold is None):
            raise ValueError("work and fail thresholds must be exposed together")
        if (self.calibration_samples is None) != (self.calibration_groups is None):
            raise ValueError("calibration sample and group counts must be exposed together")
        return self


class PredictionReport(StrictModel):
    sample_id: str = Field(min_length=1, max_length=256)
    supported_species: str = Field(min_length=1, max_length=256)
    model_version: str = Field(min_length=1, max_length=128)
    predictions: list[DrugPrediction]
    mode: AnalysisMode = AnalysisMode.LIVE
    data_status: DataStatus = DataStatus.MODEL_DERIVED
    demo_disclaimer: str | None = Field(default=None, max_length=2_000)
    warning: Literal[
        "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE. This report provides genomic decision support only and does not select treatment. Every antibiotic-response result must be confirmed with standard laboratory susceptibility testing and reviewed by a trained healthcare or laboratory professional."
    ] = RESEARCH_WARNING
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    qc: FastaQC | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_demo_and_drugs(self) -> PredictionReport:
        if self.mode == AnalysisMode.DEMO and not self.demo_disclaimer:
            raise ValueError("DEMO mode requires demo_disclaimer")
        if self.mode == AnalysisMode.DEMO and self.data_status != DataStatus.DEMONSTRATION_FIXTURE:
            raise ValueError("DEMO mode requires DEMONSTRATION_FIXTURE data status")
        if self.mode == AnalysisMode.LIVE and self.data_status == DataStatus.DEMONSTRATION_FIXTURE:
            raise ValueError("LIVE mode cannot use DEMONSTRATION_FIXTURE data status")
        if self.data_status == DataStatus.UNAVAILABLE and any(
            prediction.call != PredictionCall.NO_CALL for prediction in self.predictions
        ):
            raise ValueError("UNAVAILABLE reports may contain only NO_CALL predictions")
        drugs = [prediction.drug.casefold() for prediction in self.predictions]
        if len(drugs) != len(set(drugs)):
            raise ValueError("predictions must contain unique drug names")
        source_ids = [
            item.id
            for prediction in self.predictions
            for item in prediction.evidence
            if item.id is not None
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("evidence IDs must be globally unique within a report")
        return self


__all__ = [
    "AnalysisMode",
    "DataStatus",
    "DecisionInput",
    "DecisionThresholds",
    "DrugPrediction",
    "EvidenceCategory",
    "EvidenceDirection",
    "EvidenceItem",
    "EvidenceKind",
    "FastaLimits",
    "FastaQC",
    "NoCallReason",
    "PredictionCall",
    "PredictionReport",
    "RESEARCH_WARNING",
    "TargetStatus",
]
