from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from genome_firewall.amrfinder import AMRHit, features_from_hits, parse_amrfinder_tsv
from genome_firewall.demo import DEMO_DISCLAIMER, available_demo_cases, load_demo_case
from genome_firewall.explanations import explain_report, guarded_question
from genome_firewall.schemas import PredictionReport, TargetStatus
from genome_firewall.service import analyze_upload, evidence_from_hits
from genome_firewall.targets import assess_target, load_drug_catalog


def test_all_demo_cases_are_validated_and_unmistakably_illustrative() -> None:
    metadata = available_demo_cases()
    assert {item["case_id"] for item in metadata} == {
        "marker_fail",
        "target_work",
        "honest_no_call",
    }
    for item in metadata:
        case = load_demo_case(item["case_id"])
        report = PredictionReport.model_validate(case["report"])
        assert report.mode.value == "DEMO"
        assert report.data_status.value == "DEMONSTRATION_FIXTURE"
        assert report.demo_disclaimer == DEMO_DISCLAIMER
        assert case["metrics"]["status"] == "UNAVAILABLE_NO_ORGANIZER_EVALUATION"
        assert case["metrics"]["per_drug"] == []
        for prediction in report.predictions:
            if prediction.call.value == "NO_CALL":
                assert prediction.displayed_confidence is None


def test_template_explanation_and_treatment_boundary(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    explanation = explain_report(load_demo_case("marker_fail"), use_openai=True)
    assert explanation["generated_by"] == "template"
    assert explanation["confirmation_required"] is True
    refusal = guarded_question("Which antibiotic should I prescribe?")
    assert "cannot select" in refusal
    assert "NOT FOR CLINICAL USE" in refusal


def test_explanation_distinguishes_target_evidence_from_determinants() -> None:
    explanation = explain_report(load_demo_case("target_work"), use_openai=False)
    assert any("reviewed target evidence" in item for item in explanation["evidence"])


def test_amrfinder_parser_normalizes_gene_and_mutation_features(tmp_path: Path) -> None:
    output = tmp_path / "amr.tsv"
    output.write_text(
        "Element symbol\tElement name\tType\tMethod\t% Identity to reference sequence\n"
        "blaCTX-M-15\tESBL\tAMR\tEXACT\t100\n"
        "gyrA_S83L\tquinolone mutation\tPOINT\tPOINTX\t100\n",
        encoding="utf-8",
    )
    hits = parse_amrfinder_tsv(output)
    features = features_from_hits(hits)
    assert features["gene::blaCTX-M-15"] == 1.0
    assert features["mutation::gyrA_S83L"] == 1.0


def test_amrfinder_parser_rejects_unknown_output_schema(tmp_path: Path) -> None:
    output = tmp_path / "drifted.tsv"
    output.write_text("Unknown ID\tDescription\nabc\tchanged schema\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recognized"):
        parse_amrfinder_tsv(output)


def test_provisional_target_catalog_fails_closed() -> None:
    project = Path(__file__).resolve().parents[1]
    catalog = load_drug_catalog(project / "config" / "drug_catalog.yaml")
    result = assess_target(
        catalog,
        species_key="escherichia_coli",
        drug_key="ciprofloxacin",
        hits={"gyrA": TargetStatus.PRESENT, "parC": TargetStatus.PRESENT},
    )
    assert result.status == TargetStatus.AMBIGUOUS
    assert result.catalog_reviewed is False


def test_unassessed_targets_never_become_absent() -> None:
    project = Path(__file__).resolve().parents[1]
    catalog = deepcopy(load_drug_catalog(project / "config" / "drug_catalog.yaml"))
    catalog["catalog_status"] = "validated"
    species = catalog["species"]["escherichia_coli"]
    species["status"] = "validated"
    species["drugs"]["ciprofloxacin"]["status"] = "validated"

    empty = assess_target(
        catalog,
        species_key="escherichia_coli",
        drug_key="ciprofloxacin",
        hits={},
    )
    explicit_absence = assess_target(
        catalog,
        species_key="escherichia_coli",
        drug_key="ciprofloxacin",
        hits={"gyrA": TargetStatus.ABSENT, "parC": TargetStatus.ABSENT},
    )
    assert empty.status == TargetStatus.NOT_ASSESSED
    assert explicit_absence.status == TargetStatus.ABSENT


def test_species_provisional_catalog_cannot_enable_reviewed_target() -> None:
    project = Path(__file__).resolve().parents[1]
    catalog = deepcopy(load_drug_catalog(project / "config" / "drug_catalog.yaml"))
    catalog["catalog_status"] = "validated"
    catalog["species"]["escherichia_coli"]["drugs"]["ciprofloxacin"]["status"] = "validated"
    result = assess_target(
        catalog,
        species_key="escherichia_coli",
        drug_key="ciprofloxacin",
        hits={"gyrA": TargetStatus.PRESENT, "parC": TargetStatus.PRESENT},
    )
    assert result.status == TargetStatus.AMBIGUOUS
    assert result.catalog_reviewed is False


def test_provisional_allowlist_match_is_not_mislabeled_curated() -> None:
    project = Path(__file__).resolve().parents[1]
    hits = [
        AMRHit(
            element_symbol="blaCTX-M-15",
            element_name="ESBL",
            method="EXACT",
            identity_pct=100,
            coverage_pct=100,
        ),
        AMRHit(
            element_symbol="blaCMY-2",
            element_name="low-quality partial hit",
            method="PARTIAL",
            identity_pct=60,
            coverage_pct=40,
        ),
    ]
    evidence = evidence_from_hits(
        hits,
        drugs=["Ceftriaxone"],
        catalog_path=project / "config" / "drug_catalog.yaml",
    )["Ceftriaxone"]
    assert [item.name for item in evidence] == ["blaCTX-M-15", "blaCMY-2"]
    assert evidence[0].curated is False
    assert evidence[1].details["quality_status"] == "AMBIGUOUS"


def test_point_mutation_method_has_explicit_confirmed_policy() -> None:
    project = Path(__file__).resolve().parents[1]
    evidence = evidence_from_hits(
        [
            AMRHit(
                element_symbol="gyrA_S83L",
                element_name="quinolone point mutation",
                method="POINTX",
                identity_pct=100,
            )
        ],
        drugs=["Ciprofloxacin"],
        catalog_path=project / "config" / "drug_catalog.yaml",
    )["Ciprofloxacin"]
    assert len(evidence) == 1
    assert evidence[0].details["quality_status"] == "CONFIRMED"
    assert evidence[0].curated is False


def test_species_provisional_determinant_cannot_be_curated(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    catalog = deepcopy(load_drug_catalog(project / "config" / "drug_catalog.yaml"))
    catalog["catalog_status"] = "validated"
    catalog["species"]["escherichia_coli"]["drugs"]["ceftriaxone"]["status"] = "validated"
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
    evidence = evidence_from_hits(
        [AMRHit(element_symbol="blaCTX-M-15", method="EXACT")],
        drugs=["Ceftriaxone"],
        catalog_path=catalog_path,
    )["Ceftriaxone"]
    assert len(evidence) == 1
    assert evidence[0].curated is False


def test_ui_separates_provisional_determinants_and_hashes_upload_ids(monkeypatch) -> None:
    monkeypatch.setenv("GENOME_FIREWALL_ARTIFACT_DIR", "missing-test-artifacts")
    from app import _analyze_upload, _evidence_buckets

    targets, curated, provisional, associations = _evidence_buckets(
        [
            {"kind": "TARGET_DETECTION", "curated": False},
            {"kind": "RESISTANCE_GENE", "curated": False},
            {"kind": "RESISTANCE_MUTATION", "curated": True},
            {"kind": "MODEL_FEATURE", "curated": False},
        ]
    )
    assert tuple(map(len, (targets, curated, provisional, associations))) == (1, 1, 1, 1)

    fasta = b">assembled_contig\n" + (b"ACGT" * 30_000) + b"\n"
    _, report, _ = _analyze_upload(fasta)
    assert report["sample_id"] == f"upload-{sha256(fasta).hexdigest()[:12]}"


def test_report_rejects_duplicate_evidence_source_ids() -> None:
    case = load_demo_case("target_work")
    predictions = case["report"]["predictions"]
    predictions[1]["evidence"][0]["id"] = predictions[0]["evidence"][0]["id"]
    try:
        PredictionReport.model_validate(case["report"])
    except ValueError as exc:
        assert "globally unique" in str(exc)
    else:  # pragma: no cover - contract failure is the assertion
        raise AssertionError("duplicate evidence IDs were accepted")


def test_arbitrary_valid_upload_never_falls_back_to_demo(tmp_path: Path) -> None:
    fasta = b">assembled_contig\n" + (b"ACGT" * 30_000) + b"\n"
    report = analyze_upload(fasta, artifact_dir=tmp_path / "missing-artifacts")
    assert report.mode.value == "LIVE"
    assert report.data_status.value == "UNAVAILABLE"
    assert all(prediction.call.value == "NO_CALL" for prediction in report.predictions)
    assert all(prediction.p_resistant is None for prediction in report.predictions)
    assert report.demo_disclaimer is None
