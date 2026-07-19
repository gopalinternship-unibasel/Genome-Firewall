"""Deterministic, fail-closed three-way decision policy."""

from __future__ import annotations

from math import isfinite

from .schemas import (
    DecisionInput,
    DecisionThresholds,
    DrugPrediction,
    EvidenceCategory,
    EvidenceDirection,
    EvidenceItem,
    EvidenceKind,
    NoCallReason,
    PredictionCall,
    TargetStatus,
)


def evidence_category(evidence: list[EvidenceItem]) -> EvidenceCategory:
    """Separate curated biological determinants from statistical features."""

    curated = any(
        item.curated
        and item.kind in {EvidenceKind.RESISTANCE_GENE, EvidenceKind.RESISTANCE_MUTATION}
        for item in evidence
    )
    statistical = any(item.kind == EvidenceKind.MODEL_FEATURE for item in evidence)
    if curated and statistical:
        return EvidenceCategory.MIXED
    if curated:
        return EvidenceCategory.CURATED_DETERMINANT
    if statistical:
        return EvidenceCategory.STATISTICAL_ASSOCIATION
    return EvidenceCategory.NONE


def _known_direction(evidence: list[EvidenceItem], direction: EvidenceDirection) -> bool:
    return any(
        item.curated
        and item.direction == direction
        and item.kind in {EvidenceKind.RESISTANCE_GENE, EvidenceKind.RESISTANCE_MUTATION}
        for item in evidence
    )


class DecisionPolicy:
    """Apply safety gates in a fixed, auditable order.

    The class contains no learned state.  Probabilities and thresholds are inputs,
    which makes the exact same policy usable in training evaluation and inference.
    """

    def decide(
        self,
        request: DecisionInput,
        thresholds: DecisionThresholds,
        *,
        model_version: str | None = None,
    ) -> DrugPrediction:
        category = evidence_category(request.evidence)
        threshold_context = (
            {
                "work_threshold": thresholds.work_max,
                "fail_threshold": thresholds.fail_min,
                "calibration_samples": thresholds.calibration_samples,
                "calibration_groups": thresholds.calibration_groups,
            }
            if thresholds.supported
            else {}
        )

        def no_call(reason: NoCallReason, explanation: str) -> DrugPrediction:
            return DrugPrediction(
                drug=request.drug,
                call=PredictionCall.NO_CALL,
                p_resistant=request.p_resistant,
                displayed_confidence=None,
                target_status=request.target_status,
                evidence_category=category,
                evidence=request.evidence,
                no_call_reason=reason,
                decision_reasons=[explanation],
                model_version=model_version,
                **threshold_context,
            )

        def called(
            call: PredictionCall,
            confidence: float | None,
            explanation: str,
        ) -> DrugPrediction:
            return DrugPrediction(
                drug=request.drug,
                call=call,
                p_resistant=request.p_resistant,
                displayed_confidence=confidence,
                target_status=request.target_status,
                evidence_category=category,
                evidence=request.evidence,
                no_call_reason=None,
                decision_reasons=[explanation],
                model_version=model_version,
                **threshold_context,
            )

        # 1. Input and annotation quality gates.
        if not request.qc_passed or not request.annotation_ok:
            return no_call(
                NoCallReason.ANNOTATION_OR_QC_FAILURE,
                "Input QC or resistance annotation did not complete successfully.",
            )

        # 2. Explicit support and distribution gates.
        if not request.species_supported:
            return no_call(
                NoCallReason.UNSUPPORTED_SPECIES,
                "No validated model is registered for this species.",
            )
        if not request.drug_supported:
            return no_call(
                NoCallReason.UNSUPPORTED_DRUG,
                "No validated model is registered for this drug.",
            )
        if request.is_ood:
            return no_call(
                NoCallReason.GENOME_OUT_OF_DISTRIBUTION,
                "The genome is outside the validated model distribution.",
            )
        if request.feature_profile_novel:
            return no_call(
                NoCallReason.FEATURE_PROFILE_NOVEL,
                "The resistance-feature profile is too novel for a supported call.",
            )
        if any(
            item.kind in {EvidenceKind.RESISTANCE_GENE, EvidenceKind.RESISTANCE_MUTATION}
            and str(item.details.get("quality_status", "")).upper() == "AMBIGUOUS"
            for item in request.evidence
        ):
            return no_call(
                NoCallReason.EVIDENCE_AMBIGUOUS,
                "A possible allowlisted resistance determinant was detected with ambiguous quality.",
            )

        # 3. A usable target assessment is required for this narrow prototype.
        if request.target_status == TargetStatus.AMBIGUOUS:
            return no_call(
                NoCallReason.TARGET_AMBIGUOUS,
                "The molecular target could not be detected unambiguously.",
            )
        if request.target_status == TargetStatus.NOT_ASSESSED:
            return no_call(
                NoCallReason.TARGET_NOT_ASSESSED,
                "The molecular target was not assessed.",
            )

        probability = request.p_resistant
        usable_probability = probability is not None and isfinite(probability)
        model_work = (
            thresholds.supported and usable_probability and probability <= thresholds.work_max
        )
        model_fail = (
            thresholds.supported and usable_probability and probability >= thresholds.fail_min
        )
        known_resistance = _known_direction(request.evidence, EvidenceDirection.RESISTANCE)
        known_susceptibility = _known_direction(request.evidence, EvidenceDirection.SUSCEPTIBILITY)

        # 4. Known curated evidence must not be silently overruled by the model.
        if (model_work and known_resistance) or (model_fail and known_susceptibility):
            return no_call(
                NoCallReason.KNOWN_MARKER_CONFLICT,
                "Curated determinant evidence conflicts with the calibrated model.",
            )

        # 5. Target absence can only force failure under an explicitly validated rule.
        if request.target_status == TargetStatus.ABSENT:
            if request.validated_absent_target_rule:
                return called(
                    PredictionCall.LIKELY_TO_FAIL,
                    None,
                    "A validated rule detected absence of the required molecular target.",
                )
            return no_call(
                NoCallReason.TARGET_ABSENT_UNVALIDATED,
                "Target absence has not been validated as a response rule for this model.",
            )

        # All probability-backed calls require calibration support.
        if not thresholds.supported:
            return no_call(
                NoCallReason.INSUFFICIENT_CALIBRATION_SUPPORT,
                "Calibration data did not support safe work/fail thresholds.",
            )
        if not usable_probability:
            return no_call(
                NoCallReason.MODEL_ERROR,
                "The model did not produce a finite calibrated probability.",
            )

        # 6. High calibrated probability can produce a likely-fail call.
        if model_fail:
            return called(
                PredictionCall.LIKELY_TO_FAIL,
                probability,
                f"Calibrated resistance probability met the fail threshold "
                f"({probability:.3f} >= {thresholds.fail_min:.3f}).",
            )

        # 7. Low probability may produce a likely-work call only with target present.
        if model_work and request.target_status == TargetStatus.PRESENT:
            return called(
                PredictionCall.LIKELY_TO_WORK,
                1.0 - probability,
                f"Calibrated resistance probability met the work threshold "
                f"({probability:.3f} <= {thresholds.work_max:.3f}) and the target is present.",
            )

        # 8. Everything between thresholds is an intentional abstention.
        return no_call(
            NoCallReason.LOW_CONFIDENCE,
            "The calibrated probability falls inside the abstention interval.",
        )


def decide(
    request: DecisionInput,
    thresholds: DecisionThresholds,
    *,
    model_version: str | None = None,
) -> DrugPrediction:
    """Functional convenience wrapper around :class:`DecisionPolicy`."""

    return DecisionPolicy().decide(request, thresholds, model_version=model_version)


__all__ = ["DecisionPolicy", "decide", "evidence_category"]
