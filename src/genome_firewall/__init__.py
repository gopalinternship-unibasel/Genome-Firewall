"""Genome Firewall scientific core.

This package is a research prototype and is not a treatment-selection system.
"""

from .decision import DecisionPolicy, decide, evidence_category
from .evaluation import (
    BinaryMetrics,
    CalibrationMetrics,
    DecisionMetrics,
    EvaluationReport,
    binary_metrics,
    calibration_metrics,
    decision_metrics,
    evaluate_predictions,
    group_metrics,
    risk_coverage_curve,
)
from .fasta import FastaFormatError, FastaRecord, analyze_fasta, parse_fasta
from .predictor import ModelRegistry
from .schemas import (
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
    FastaQC,
    NoCallReason,
    PredictionCall,
    PredictionReport,
    TargetStatus,
)
from .training import (
    ModelArtifactError,
    ModelBundle,
    ThresholdSelectionConfig,
    ThresholdSelectionResult,
    TrainingConfig,
    TrainingDataError,
    load_model_bundle,
    select_decision_thresholds,
    train_drug_model,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisMode",
    "BinaryMetrics",
    "CalibrationMetrics",
    "DataStatus",
    "DecisionInput",
    "DecisionMetrics",
    "DecisionPolicy",
    "DecisionThresholds",
    "DrugPrediction",
    "EvaluationReport",
    "EvidenceCategory",
    "EvidenceDirection",
    "EvidenceItem",
    "EvidenceKind",
    "FastaFormatError",
    "FastaLimits",
    "FastaQC",
    "FastaRecord",
    "ModelArtifactError",
    "ModelBundle",
    "ModelRegistry",
    "NoCallReason",
    "PredictionCall",
    "PredictionReport",
    "RESEARCH_WARNING",
    "TargetStatus",
    "ThresholdSelectionConfig",
    "ThresholdSelectionResult",
    "TrainingConfig",
    "TrainingDataError",
    "analyze_fasta",
    "binary_metrics",
    "calibration_metrics",
    "decide",
    "decision_metrics",
    "evaluate_predictions",
    "evidence_category",
    "group_metrics",
    "load_model_bundle",
    "parse_fasta",
    "risk_coverage_curve",
    "select_decision_thresholds",
    "train_drug_model",
]
