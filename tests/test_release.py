from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

import genome_firewall.service as service
from genome_firewall.release import (
    RELEASE_CHECKSUM_FILENAME,
    RELEASE_EVALUATION_FILENAME,
    ReleaseEvaluationError,
    load_release_evaluation,
)


def _write_manifest(root: Path, *, drugs: tuple[str, ...] = ("Drug A",)) -> str:
    manifest = {
        "manifest_format_version": 1,
        "registry_version": "test-only",
        "models": [
            {"species": "Escherichia coli", "drug": drug, "file": f"{index}.joblib"}
            for index, drug in enumerate(drugs)
        ],
    }
    encoded = json.dumps(manifest, sort_keys=True).encode()
    (root / "manifest.json").write_bytes(encoded)
    return sha256(encoded).hexdigest()


def _valid_payload(manifest_sha256: str, *, drugs: tuple[str, ...] = ("Drug A",)) -> dict:
    return {
        "schema_version": 1,
        "release_eligible": True,
        "group_disjoint_verified": True,
        "target_workflow_validated": True,
        "dataset_sha256": "1" * 64,
        "split_sha256": "2" * 64,
        "reviewer": "Independent test reviewer",
        "reviewed_at": "2026-07-19T00:00:00Z",
        "manifest_sha256": manifest_sha256,
        "annotation": {
            "feature_schema_version": "1",
            "tool_version": "test-tool-1",
            "database_version": "test-db-1",
        },
        "drug_evaluations": [
            {
                "species": "Escherichia coli",
                "drug": drug,
                "supported": True,
                "policy_constraints_satisfied": True,
                "metrics": {
                    "n_samples": 8,
                    "n_resistant": 4,
                    "n_susceptible": 4,
                    "n_groups": 4,
                    "coverage": 0.75,
                    "called_accuracy": 0.875,
                    "false_susceptible_rate": 0.0,
                },
            }
            for drug in drugs
        ],
    }


def _write_release(root: Path, payload: dict, *, checksum_override: str | None = None) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode()
    (root / RELEASE_EVALUATION_FILENAME).write_bytes(encoded)
    checksum = checksum_override or sha256(encoded).hexdigest()
    (root / RELEASE_CHECKSUM_FILENAME).write_text(
        f"{checksum}  {RELEASE_EVALUATION_FILENAME}\n",
        encoding="ascii",
    )


def _valid_gate(root: Path, *, drugs: tuple[str, ...] = ("Drug A",)):
    manifest_sha256 = _write_manifest(root, drugs=drugs)
    _write_release(root, _valid_payload(manifest_sha256, drugs=drugs))
    return load_release_evaluation(
        root,
        species="Escherichia coli",
        requested_drugs=drugs,
    )


def test_valid_release_is_bound_to_manifest_and_exact_annotation_versions(tmp_path: Path) -> None:
    gate = _valid_gate(tmp_path)

    gate.verify_annotation(
        feature_schema_version="1",
        tool_version="test-tool-1",
        database_version="test-db-1",
    )

    assert gate.dataset_sha256 == "1" * 64
    assert gate.split_sha256 == "2" * 64
    assert gate.evaluated_pairs == frozenset({("escherichia coli", "drug a")})


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("release_eligible", False, "release_eligible"),
        ("group_disjoint_verified", False, "group_disjoint_verified"),
        ("target_workflow_validated", False, "target_workflow_validated"),
        ("dataset_sha256", "", "dataset_sha256"),
        ("split_sha256", "not-a-hash", "split_sha256"),
        ("reviewer", "  ", "reviewer"),
        ("reviewed_at", "", "reviewed_at"),
    ],
)
def test_release_rejects_missing_or_invalid_required_attestations(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    manifest_sha256 = _write_manifest(tmp_path)
    payload = _valid_payload(manifest_sha256)
    payload[field] = invalid_value
    _write_release(tmp_path, payload)

    with pytest.raises(ReleaseEvaluationError, match=message):
        load_release_evaluation(
            tmp_path,
            species="Escherichia coli",
            requested_drugs=["Drug A"],
        )


def test_release_rejects_detached_checksum_or_manifest_binding_mismatch(tmp_path: Path) -> None:
    manifest_sha256 = _write_manifest(tmp_path)
    _write_release(tmp_path, _valid_payload(manifest_sha256), checksum_override="0" * 64)
    with pytest.raises(ReleaseEvaluationError, match="detached checksum mismatch"):
        load_release_evaluation(
            tmp_path,
            species="Escherichia coli",
            requested_drugs=["Drug A"],
        )

    payload = _valid_payload("f" * 64)
    _write_release(tmp_path, payload)
    with pytest.raises(ReleaseEvaluationError, match="does not match manifest"):
        load_release_evaluation(
            tmp_path,
            species="Escherichia coli",
            requested_drugs=["Drug A"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(drug_evaluations=[]),
        lambda payload: payload["drug_evaluations"][0].update(supported=False),
        lambda payload: payload["drug_evaluations"][0].update(policy_constraints_satisfied=False),
        lambda payload: payload["drug_evaluations"][0].update(metrics={}),
    ],
)
def test_release_requires_positive_metrics_for_every_requested_supported_drug(
    tmp_path: Path,
    mutation,
) -> None:
    manifest_sha256 = _write_manifest(tmp_path, drugs=("Drug A", "Drug B"))
    payload = _valid_payload(manifest_sha256, drugs=("Drug A", "Drug B"))
    mutation(payload)
    _write_release(tmp_path, payload)

    with pytest.raises(ReleaseEvaluationError):
        load_release_evaluation(
            tmp_path,
            species="Escherichia coli",
            requested_drugs=["Drug A", "Drug B"],
        )


@pytest.mark.parametrize(
    ("field", "live_value"),
    [
        ("feature_schema_version", "2"),
        ("tool_version", "different-tool"),
        ("database_version", "different-db"),
    ],
)
def test_release_rejects_any_annotation_version_mismatch(
    tmp_path: Path,
    field: str,
    live_value: str,
) -> None:
    gate = _valid_gate(tmp_path)
    versions = {
        "feature_schema_version": "1",
        "tool_version": "test-tool-1",
        "database_version": "test-db-1",
    }
    versions[field] = live_value

    with pytest.raises(ReleaseEvaluationError, match="version mismatch"):
        gate.verify_annotation(**versions)


def test_service_rejects_missing_gate_before_loading_or_calling_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_model_load(*_args, **_kwargs):
        pytest.fail("model registry was touched before the release gate passed")

    monkeypatch.setattr(service.ModelRegistry, "load", unexpected_model_load)
    fasta = b">assembled_contig\n" + (b"ACGT" * 30_000) + b"\n"

    report = service.analyze_upload(fasta, artifact_dir=tmp_path)

    assert report.data_status.value == "UNAVAILABLE"
    assert all(prediction.call.value == "NO_CALL" for prediction in report.predictions)
    assert all(
        "Release evaluation compatibility check failed" in prediction.decision_reasons[0]
        for prediction in report.predictions
    )
    assert "release_evaluation.json is required" in report.provenance["release_evaluation_error"]


def test_service_rejects_annotation_mismatch_before_loading_or_calling_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _valid_gate(tmp_path)

    def unexpected_model_load(*_args, **_kwargs):
        pytest.fail("model registry was touched before exact annotation compatibility passed")

    monkeypatch.setenv("GENOME_FIREWALL_ALLOW_LIVE_ANNOTATION", "true")
    monkeypatch.setattr(service.ModelRegistry, "load", unexpected_model_load)
    monkeypatch.setattr(
        service,
        "run_amrfinder",
        lambda *_args, **_kwargs: service.AnnotationResult(
            ok=True,
            amrfinder_version="different-tool",
            database_version="test-db-1",
        ),
    )
    fasta = b">assembled_contig\n" + (b"ACGT" * 30_000) + b"\n"

    report = service.analyze_upload(
        fasta,
        artifact_dir=tmp_path,
        requested_drugs=["Drug A"],
    )

    assert report.data_status.value == "UNAVAILABLE"
    assert "version mismatch" in report.provenance["release_evaluation_error"]
