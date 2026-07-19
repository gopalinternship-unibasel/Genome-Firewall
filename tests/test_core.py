from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from genome_firewall.decision import DecisionPolicy
from genome_firewall.evaluation import binary_metrics, decision_metrics, evaluate_predictions
from genome_firewall.fasta import FastaFormatError, analyze_fasta, parse_fasta
from genome_firewall.predictor import ModelRegistry
from genome_firewall.schemas import (
    RESEARCH_WARNING,
    AnalysisMode,
    DataStatus,
    DecisionInput,
    DecisionThresholds,
    DrugPrediction,
    EvidenceCategory,
    EvidenceDirection,
    EvidenceItem,
    EvidenceKind,
    FastaLimits,
    NoCallReason,
    PredictionCall,
    PredictionReport,
    TargetStatus,
)
from genome_firewall.training import (
    ModelBundle,
    ThresholdSelectionConfig,
    TrainingConfig,
    TrainingDataError,
    select_decision_thresholds,
    train_drug_model,
)


def lenient_fasta_limits() -> FastaLimits:
    return FastaLimits(
        max_file_bytes=10_000,
        min_total_bases=1,
        max_total_bases=10_000,
        max_contigs=20,
        min_n50=1,
        max_ambiguous_fraction=0.50,
        max_duplicate_headers=0,
    )


def supported_thresholds() -> DecisionThresholds:
    return DecisionThresholds(
        work_max=0.20,
        fail_min=0.80,
        supported=True,
        calibration_samples=100,
        calibration_groups=10,
    )


def test_fasta_qc_reports_structural_and_assembly_statistics() -> None:
    fasta = ">contig-1 description\nACGTNNNN\n>contig-2\nGGGG\n"
    qc = analyze_fasta(fasta, lenient_fasta_limits())

    assert qc.valid_fasta is True
    assert qc.passes_qc is True
    assert qc.sequence_count == 2
    assert qc.total_bases == 12
    assert (qc.n50, qc.l50) == (8, 1)
    assert qc.gc_fraction == pytest.approx(0.75)
    assert qc.ambiguous_fraction == pytest.approx(4 / 12)
    assert len(qc.sha256) == 64
    assert [record.identifier for record in parse_fasta(fasta)] == ["contig-1", "contig-2"]


def test_fasta_qc_fails_closed_and_strict_parser_raises() -> None:
    malformed = ">same\nACGTZ\n>same\n\n"
    qc = analyze_fasta(malformed, lenient_fasta_limits())

    assert qc.valid_fasta is False
    assert qc.passes_qc is False
    assert qc.invalid_character_count == 1
    assert qc.duplicate_header_count == 1
    assert qc.empty_sequence_count == 1
    with pytest.raises(FastaFormatError):
        parse_fasta(malformed)


def test_decision_policy_covers_fail_work_and_uncertainty() -> None:
    policy = DecisionPolicy()
    thresholds = supported_thresholds()

    fail = policy.decide(
        DecisionInput(drug="Drug A", p_resistant=0.91, target_status=TargetStatus.PRESENT),
        thresholds,
    )
    work = policy.decide(
        DecisionInput(drug="Drug A", p_resistant=0.08, target_status=TargetStatus.PRESENT),
        thresholds,
    )
    uncertain = policy.decide(
        DecisionInput(drug="Drug A", p_resistant=0.50, target_status=TargetStatus.PRESENT),
        thresholds,
    )

    assert fail.call == PredictionCall.LIKELY_TO_FAIL
    assert fail.displayed_confidence == pytest.approx(0.91)
    assert work.call == PredictionCall.LIKELY_TO_WORK
    assert work.displayed_confidence == pytest.approx(0.92)
    assert uncertain.call == PredictionCall.NO_CALL
    assert uncertain.no_call_reason == NoCallReason.LOW_CONFIDENCE
    assert uncertain.displayed_confidence is None


@pytest.mark.parametrize(
    ("updates", "expected_reason"),
    [
        ({"qc_passed": False}, NoCallReason.ANNOTATION_OR_QC_FAILURE),
        ({"species_supported": False}, NoCallReason.UNSUPPORTED_SPECIES),
        ({"drug_supported": False}, NoCallReason.UNSUPPORTED_DRUG),
        ({"is_ood": True}, NoCallReason.GENOME_OUT_OF_DISTRIBUTION),
        ({"feature_profile_novel": True}, NoCallReason.FEATURE_PROFILE_NOVEL),
        ({"target_status": TargetStatus.AMBIGUOUS}, NoCallReason.TARGET_AMBIGUOUS),
        ({"target_status": TargetStatus.NOT_ASSESSED}, NoCallReason.TARGET_NOT_ASSESSED),
        ({"target_status": TargetStatus.ABSENT}, NoCallReason.TARGET_ABSENT_UNVALIDATED),
    ],
)
def test_decision_safety_gates(updates: dict[str, object], expected_reason: NoCallReason) -> None:
    values: dict[str, object] = {
        "drug": "Drug A",
        "p_resistant": 0.05,
        "target_status": TargetStatus.PRESENT,
    }
    values.update(updates)
    prediction = DecisionPolicy().decide(DecisionInput(**values), supported_thresholds())
    assert prediction.call == PredictionCall.NO_CALL
    assert prediction.no_call_reason == expected_reason


def test_curated_marker_conflict_blocks_likely_work() -> None:
    marker = EvidenceItem(
        kind=EvidenceKind.RESISTANCE_GENE,
        name="example resistance gene",
        direction=EvidenceDirection.RESISTANCE,
        curated=True,
    )
    prediction = DecisionPolicy().decide(
        DecisionInput(
            drug="Drug A",
            p_resistant=0.05,
            target_status=TargetStatus.PRESENT,
            evidence=[marker],
        ),
        supported_thresholds(),
    )
    assert prediction.call == PredictionCall.NO_CALL
    assert prediction.no_call_reason == NoCallReason.KNOWN_MARKER_CONFLICT


def test_ambiguous_allowlisted_hit_forces_no_call() -> None:
    marker = EvidenceItem(
        kind=EvidenceKind.RESISTANCE_GENE,
        name="possible partial determinant",
        direction=EvidenceDirection.RESISTANCE,
        curated=False,
        details={"quality_status": "AMBIGUOUS"},
    )
    prediction = DecisionPolicy().decide(
        DecisionInput(
            drug="Drug A",
            p_resistant=0.95,
            target_status=TargetStatus.PRESENT,
            evidence=[marker],
        ),
        supported_thresholds(),
    )
    assert prediction.call == PredictionCall.NO_CALL
    assert prediction.no_call_reason == NoCallReason.EVIDENCE_AMBIGUOUS


def test_validated_absent_target_rule_can_fail_without_fabricated_confidence() -> None:
    prediction = DecisionPolicy().decide(
        DecisionInput(
            drug="Drug A",
            p_resistant=0.20,
            target_status=TargetStatus.ABSENT,
            validated_absent_target_rule=True,
        ),
        supported_thresholds(),
    )
    assert prediction.call == PredictionCall.LIKELY_TO_FAIL
    assert prediction.displayed_confidence is None


def test_demo_report_requires_disclaimer_and_warning_is_fixed() -> None:
    with pytest.raises(ValidationError):
        PredictionReport(
            sample_id="sample",
            supported_species="Escherichia coli",
            model_version="demo",
            predictions=[],
            mode=AnalysisMode.DEMO,
        )
    report = PredictionReport(
        sample_id="sample",
        supported_species="Escherichia coli",
        model_version="demo",
        predictions=[],
        mode=AnalysisMode.DEMO,
        data_status=DataStatus.DEMONSTRATION_FIXTURE,
        demo_disclaimer="Synthetic interface fixture; not a benchmark.",
    )
    assert report.warning == RESEARCH_WARNING


def test_schema_rejects_unsafe_mode_status_and_work_target_combinations() -> None:
    with pytest.raises(ValidationError, match="LIKELY_TO_WORK"):
        DrugPrediction(
            drug="Drug A",
            call=PredictionCall.LIKELY_TO_WORK,
            p_resistant=0.05,
            displayed_confidence=0.95,
            target_status=TargetStatus.AMBIGUOUS,
            evidence_category=EvidenceCategory.NONE,
        )
    with pytest.raises(ValidationError, match="DEMONSTRATION_FIXTURE"):
        PredictionReport(
            sample_id="sample",
            supported_species="Escherichia coli",
            model_version="model",
            predictions=[],
            mode=AnalysisMode.LIVE,
            data_status=DataStatus.DEMONSTRATION_FIXTURE,
        )


def test_metrics_keep_undefined_values_explicit() -> None:
    metrics = binary_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert metrics.n_samples == 3
    assert metrics.auroc is None
    assert metrics.pr_auc is None
    assert metrics.resistant_recall is None
    assert metrics.brier_score is not None

    calls = [PredictionCall.LIKELY_TO_WORK, PredictionCall.NO_CALL, PredictionCall.LIKELY_TO_FAIL]
    abstaining = decision_metrics([0, 1, 1], calls)
    assert abstaining.n_called == 2
    assert abstaining.coverage == pytest.approx(2 / 3)
    assert abstaining.system_accuracy == pytest.approx(2 / 3)


def test_threshold_selection_is_measured_and_can_fail_closed() -> None:
    y = np.asarray([0] * 10 + [1] * 10)
    clean_p = np.asarray([0.05] * 10 + [0.95] * 10)
    result = select_decision_thresholds(
        y,
        clean_p,
        calibration_groups=4,
        config=ThresholdSelectionConfig(min_calls_per_side=3),
    )
    assert result.constraint_satisfied is True
    assert result.coverage == 1.0
    assert result.called_error_rate == 0.0
    assert result.thresholds.calibration_samples == 20

    ambiguous_p = np.linspace(0.40, 0.60, 20)
    unsupported = select_decision_thresholds(
        y,
        ambiguous_p,
        calibration_groups=4,
        config=ThresholdSelectionConfig(
            max_called_error_rate=0.0,
            max_false_susceptible_rate=0.0,
            min_calls_per_side=11,
        ),
    )
    assert unsupported.constraint_satisfied is False
    assert unsupported.thresholds.supported is False
    assert unsupported.coverage == 0.0


def synthetic_grouped_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    groups = np.repeat([f"group-{index:02d}" for index in range(30)], 6)
    y = np.tile(np.asarray([0, 0, 0, 1, 1, 1]), 30)
    frame = pd.DataFrame(
        {
            "gene::signal": y.astype(float),
            "continuous::signal": y + rng.normal(0.0, 0.08, y.size),
            "noise::one": rng.normal(size=y.size),
            "noise::two": rng.binomial(1, 0.2, size=y.size),
        }
    )
    return frame, y, groups


def test_training_is_group_disjoint_calibrated_and_serializable(tmp_path) -> None:
    frame, y, groups = synthetic_grouped_data()
    bundle = train_drug_model(
        frame,
        y,
        groups,
        "Drug A",
        config=TrainingConfig(
            c_values=(0.3, 1.0),
            l1_ratios=(0.0, 0.5),
            cv_splits=3,
            threshold_selection=ThresholdSelectionConfig(min_calls_per_side=3),
        ),
    )

    assert bundle.thresholds.supported is True
    fit_hashes = set(bundle.training_summary["fit_group_hashes"])
    calibration_hashes = set(bundle.training_summary["calibration_group_hashes"])
    assert fit_hashes.isdisjoint(calibration_hashes)
    assert bundle.training_summary["metric_scope"].endswith("not organizer test performance.")
    probabilities = bundle.predict_proba(frame.iloc[:3])
    assert probabilities.shape == (3,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    artifact = bundle.save(tmp_path / "drug-a.joblib")
    loaded = ModelBundle.load(artifact)
    assert loaded.drug == bundle.drug
    np.testing.assert_allclose(loaded.predict_proba(frame.iloc[:3]), probabilities)

    registry = ModelRegistry([loaded], registry_version="test-registry")
    manifest = registry.save(tmp_path / "registry")
    assert manifest.name == "manifest.json"
    reloaded_registry = ModelRegistry.load(manifest.parent)
    qc = analyze_fasta(">contig\nACGTACGT\n", lenient_fasta_limits())
    report = reloaded_registry.predict(
        sample_id="sample-1",
        species="Escherichia coli",
        features=frame.iloc[0].to_dict(),
        qc=qc,
        target_status_by_drug={"Drug A": TargetStatus.PRESENT},
        requested_drugs=["Drug A"],
        feature_profile_novel_by_drug={"Drug A": False},
    )
    assert report.predictions[0].p_resistant is not None
    assert report.warning == RESEARCH_WARNING
    assert report.provenance["raw_genome_shared_with_explainer"] is False

    novel_features = frame.iloc[0].to_dict()
    novel_features["gene::previously_unseen"] = 1.0
    novel_report = reloaded_registry.predict(
        sample_id="sample-novel",
        species="Escherichia coli",
        features=novel_features,
        qc=qc,
        target_status_by_drug={"Drug A": TargetStatus.PRESENT},
        requested_drugs=["Drug A"],
    )
    assert novel_report.predictions[0].call == PredictionCall.NO_CALL
    assert novel_report.predictions[0].no_call_reason == NoCallReason.FEATURE_PROFILE_NOVEL

    with pytest.raises(ValueError, match="exactly one sample"):
        reloaded_registry.predict(
            sample_id="batch",
            species="Escherichia coli",
            features=frame.iloc[:2],
            qc=qc,
            target_status_by_drug={"Drug A": TargetStatus.PRESENT},
            requested_drugs=["Drug A"],
        )


def test_training_rejects_intermediate_labels() -> None:
    frame, y, groups = synthetic_grouped_data()
    labels = y.astype(object)
    labels[0] = "intermediate"
    with pytest.raises(TrainingDataError, match="intermediate/unknown"):
        train_drug_model(frame, labels, groups, "Drug A")


def test_evaluation_report_contains_only_observed_sample_sizes() -> None:
    report = evaluate_predictions(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        calls=[
            PredictionCall.LIKELY_TO_WORK,
            PredictionCall.NO_CALL,
            PredictionCall.LIKELY_TO_FAIL,
            PredictionCall.LIKELY_TO_FAIL,
        ],
        groups=["a", "b", "c", "d"],
        n_bins=4,
    )
    assert report.binary.n_samples == 4
    assert report.calibration.n_samples == 4
    assert report.decisions is not None and report.decisions.n_samples == 4
    assert sum(bin_.n_samples for bin_ in report.calibration.bins) == 4
