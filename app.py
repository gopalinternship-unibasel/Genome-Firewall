"""Genome Firewall Streamlit experience.

The interface is deliberately abstention-first. Demo fixtures are visibly marked,
and uploaded genomes never receive synthetic predictions: when a real inference
entry point or trained artifacts are unavailable, the app returns a setup/no-call
state while still reporting deterministic FASTA quality checks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import html
import importlib
import inspect
import json
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
STYLESHEET = ROOT / "assets" / "styles.css"
load_dotenv(ROOT / ".env", override=False)

PRODUCT_NAME = "Genome Firewall"
PRODUCT_TAGLINE = "Selective genomic resistance intelligence"
SUPPORTED_SPECIES = "Escherichia coli"
SUPPORTED_DRUGS = ("Ciprofloxacin", "Ceftriaxone", "Gentamicin")
RESEARCH_WARNING = (
    "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE. This report provides genomic "
    "decision support only and does not select treatment. Every antibiotic-response "
    "result must be confirmed with standard laboratory susceptibility testing and "
    "reviewed by a trained healthcare or laboratory professional."
)

DEMO_CASES: dict[str, dict[str, str]] = {
    "marker_fail": {
        "label": "Determinant detected",
        "eyebrow": "LIKELY-TO-FAIL PATH",
        "description": "A curated resistance determinant supports an elevated-risk call.",
    },
    "target_work": {
        "label": "Target-confirmed signal",
        "eyebrow": "LIKELY-TO-WORK PATH",
        "description": "The molecular target gate is satisfied and calibrated risk is low.",
    },
    "honest_no_call": {
        "label": "Conflicting evidence",
        "eyebrow": "HONEST NO-CALL PATH",
        "description": "The system abstains because the available signals do not agree.",
    },
}

CALL_META = {
    "LIKELY_TO_FAIL": {
        "label": "Likely to fail",
        "short": "Fail signal",
        "css": "fail",
        "glyph": "↑",
        "priority": 0,
    },
    "NO_CALL": {
        "label": "No-call",
        "short": "No-call",
        "css": "nocall",
        "glyph": "—",
        "priority": 1,
    },
    "LIKELY_TO_WORK": {
        "label": "Likely to work",
        "short": "Work signal",
        "css": "work",
        "glyph": "✓",
        "priority": 2,
    },
}


st.set_page_config(
    page_title="Genome Firewall · Research prototype",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "Genome Firewall is an abstention-first genomics research prototype.",
    },
)


def _inject_styles() -> None:
    if STYLESHEET.exists():
        st.markdown(
            f"<style>{STYLESHEET.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
        )


def _plain(value: Any) -> Any:
    """Convert Pydantic/dataclass/enum-rich objects to plain Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and value.__class__.__module__ == "enum":
        return _plain(value.value)
    if hasattr(value, "model_dump"):
        try:
            return _plain(value.model_dump(mode="json"))
        except TypeError:
            return _plain(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _plain(value.dict())
        except Exception:
            pass
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "value"):
        return _plain(value.value)
    return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    plain = _plain(value)
    return dict(plain) if isinstance(plain, Mapping) else {}


def _text(value: Any, default: str = "Not available") -> str:
    if value is None or value == "":
        return default
    return str(value).replace("_", " ").strip()


def _esc(value: Any, default: str = "Not available") -> str:
    return html.escape(_text(value, default))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _percent(value: Any, digits: int = 0, default: str = "—") -> str:
    number = _number(value)
    if number is None:
        return default
    if number > 1.0:
        number = number / 100.0
    return f"{number:.{digits}%}"


def _canonical_call(value: Any) -> str:
    normalized = str(getattr(value, "value", value) or "NO_CALL").upper().strip()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "FAIL": "LIKELY_TO_FAIL",
        "LIKELY_FAIL": "LIKELY_TO_FAIL",
        "RESISTANT": "LIKELY_TO_FAIL",
        "WORK": "LIKELY_TO_WORK",
        "LIKELY_WORK": "LIKELY_TO_WORK",
        "SUSCEPTIBLE": "LIKELY_TO_WORK",
        "NOCALL": "NO_CALL",
        "ABSTAIN": "NO_CALL",
        "UNSUPPORTED": "NO_CALL",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in CALL_META else "NO_CALL"


def _normalize_report(raw: Any) -> dict[str, Any]:
    outer = _as_mapping(raw)
    nested = outer.get("report")
    if isinstance(nested, Mapping):
        report = _as_mapping(nested)
        report["_explanation_payload"] = outer
        # Keep presentation metadata supplied by a demo/evaluation wrapper while
        # making the validated report the canonical object used by the UI.
        for key in (
            "case_id",
            "title",
            "subtitle",
            "metrics",
            "validation",
            "decision_context",
            "disclaimer",
        ):
            if key in outer and key not in report:
                report[key] = outer[key]
    else:
        report = outer
    predictions = report.get("predictions") or report.get("results") or []
    if isinstance(predictions, Mapping):
        predictions = [dict(value, drug=key) for key, value in predictions.items()]
    report["predictions"] = [_as_mapping(item) for item in predictions]
    report["sample_id"] = report.get("sample_id") or report.get("case_id") or "sample"
    report["supported_species"] = (
        report.get("supported_species") or report.get("species") or "Not verified"
    )
    report["model_version"] = report.get("model_version") or "Not reported"
    report["warning"] = report.get("warning") or RESEARCH_WARNING
    report["qc"] = _as_mapping(report.get("qc"))
    return report


def _prediction_confidence(prediction: Mapping[str, Any]) -> float | None:
    explicit = _number(prediction.get("displayed_confidence"))
    if explicit is not None:
        return explicit
    p_resistant = _number(prediction.get("p_resistant"))
    if p_resistant is None:
        return None
    call = _canonical_call(prediction.get("call"))
    if call == "LIKELY_TO_FAIL":
        return p_resistant
    if call == "LIKELY_TO_WORK":
        return 1.0 - p_resistant
    return None


def _init_state() -> None:
    defaults = {
        "report": None,
        "report_source": None,
        "source_label": None,
        "pipeline_state": "idle",
        "qc": None,
        "explanation": None,
        "explanation_sample": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_demo_case(case_id: str) -> dict[str, Any]:
    try:
        module = importlib.import_module("genome_firewall.demo")
        loader = module.load_demo_case
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Demo fixtures are not installed. Run the project setup or restore the demo module."
        ) from exc

    raw = loader(case_id)
    report = _normalize_report(raw)
    report["_demo_case_id"] = case_id
    report["_is_demo_fixture"] = True
    return report


def _basic_fasta_inspection(data: bytes) -> dict[str, Any]:
    """A deterministic structural fallback; this never performs prediction."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return {
            "valid": False,
            "passed": False,
            "issues": [f"Input is not UTF-8 text: {exc}"],
            "inspection_source": "basic structural fallback",
        }

    sequences: list[str] = []
    current: list[str] = []
    headers = 0
    issues: list[str] = []
    allowed = set("ACGTURYSWKMBDHVNX-.acgturyswkmbdhvnx")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            headers += 1
            if len(line) == 1:
                issues.append(f"Header on line {line_number} has no identifier.")
            continue
        if headers == 0:
            issues.append("Sequence data appears before the first FASTA header.")
            break
        illegal = sorted(set(line) - allowed)
        if illegal:
            issues.append(f"Unsupported character(s) on line {line_number}: {''.join(illegal[:8])}")
            break
        current.append(line.replace("-", "").replace(".", ""))
    if current:
        sequences.append("".join(current))

    lengths = sorted((len(sequence) for sequence in sequences), reverse=True)
    total_bases = sum(lengths)
    n_count = sum(sequence.upper().count("N") for sequence in sequences)
    n50 = 0
    running = 0
    for length in lengths:
        running += length
        if running >= total_bases / 2:
            n50 = length
            break
    if not headers:
        issues.append("No FASTA header was found.")
    if headers != len(sequences):
        issues.append("At least one FASTA record has no sequence.")
    if total_bases == 0:
        issues.append("No nucleotide sequence was found.")

    valid = not issues
    return {
        "valid": valid,
        "passed": valid,
        "contig_count": len(sequences),
        "sequence_count": len(sequences),
        "total_bases": total_bases,
        "n50": n50,
        "n_fraction": (n_count / total_bases) if total_bases else None,
        "sha256": hashlib.sha256(data).hexdigest(),
        "issues": issues,
        "inspection_source": "basic structural fallback",
    }


def _inspect_fasta(data: bytes) -> dict[str, Any]:
    try:
        module = importlib.import_module("genome_firewall.fasta")
        analyzer = module.analyze_fasta
    except (ImportError, AttributeError):
        return _basic_fasta_inspection(data)

    try:
        qc = _as_mapping(analyzer(data))
        qc["inspection_source"] = "genome_firewall.fasta.analyze_fasta"
        return qc
    except Exception as exc:
        fallback = _basic_fasta_inspection(data)
        fallback["valid"] = False
        fallback["passed"] = False
        fallback.setdefault("issues", []).insert(0, f"Project FASTA validation failed: {exc}")
        fallback["inspection_source"] = "project validator error + structural fallback"
        return fallback


def _qc_passed(qc: Mapping[str, Any]) -> bool:
    if "passes_qc" in qc:
        return bool(qc.get("passes_qc"))
    if "passed" in qc:
        return bool(qc.get("passed"))
    if "valid_fasta" in qc and not bool(qc.get("valid_fasta")):
        return False
    if "valid" in qc:
        return bool(qc.get("valid"))
    status = str(qc.get("status") or "").upper()
    if status in {"PASS", "PASSED", "VALID", "OK"}:
        return True
    return not bool(qc.get("issues") or qc.get("errors"))


def _call_real_inference(
    data: bytes,
    sample_id: str,
    target_hits_by_drug: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Call only an explicit FASTA inference entry point; never substitute fixtures."""
    candidates = (
        ("genome_firewall.service", "analyze_upload"),
        ("genome_firewall.inference", "predict_fasta"),
        ("genome_firewall.pipeline", "predict_fasta"),
    )
    function = None
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            break
        except (ImportError, AttributeError):
            continue
    if function is None:
        return None

    text = data.decode("utf-8-sig")
    signature = inspect.signature(function)
    kwargs: dict[str, Any] = {}
    if "sample_id" in signature.parameters:
        kwargs["sample_id"] = sample_id
    if "catalog_path" in signature.parameters:
        kwargs["catalog_path"] = ROOT / "config" / "drug_catalog.yaml"
    if "target_hits_by_drug" in signature.parameters:
        kwargs["target_hits_by_drug"] = target_hits_by_drug
    raw = function(text, **kwargs)
    return _normalize_report(raw)


def _setup_no_call_report(
    sample_id: str,
    qc: Mapping[str, Any],
    reason: str,
    detail: str,
) -> dict[str, Any]:
    predictions = []
    for drug in SUPPORTED_DRUGS:
        predictions.append(
            {
                "drug": drug,
                "call": "NO_CALL",
                "p_resistant": None,
                "displayed_confidence": None,
                "target_status": "NOT_ASSESSED",
                "evidence_category": "NONE",
                "evidence": [],
                "no_call_reason": reason,
                "decision_reasons": [detail],
                "model_version": "not loaded",
            }
        )
    return _normalize_report(
        {
            "sample_id": sample_id,
            "supported_species": f"{SUPPORTED_SPECIES} (user-declared; not genomically verified)",
            "model_version": "setup required",
            "generated_at": datetime.now(UTC).isoformat(),
            "predictions": predictions,
            "qc": dict(qc),
            "warning": RESEARCH_WARNING,
            "setup_state": True,
            "setup_message": detail,
        }
    )


def _analyze_upload(
    data: bytes,
    target_hits_by_drug: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    qc = _inspect_fasta(data)
    sample_id = f"upload-{hashlib.sha256(data).hexdigest()[:12]}"
    if not _qc_passed(qc):
        report = _setup_no_call_report(
            sample_id,
            qc,
            "ANNOTATION_OR_QC_FAILURE",
            "FASTA inspection did not pass. Prediction was not attempted.",
        )
        return qc, report, "qc_failed"

    try:
        real_report = _call_real_inference(data, sample_id, target_hits_by_drug)
    except Exception as exc:
        report = _setup_no_call_report(
            sample_id,
            qc,
            "ANNOTATION_OR_QC_FAILURE",
            f"The real inference pipeline did not complete: {exc}",
        )
        return qc, report, "inference_error"

    if real_report is None:
        report = _setup_no_call_report(
            sample_id,
            qc,
            "MODEL_ERROR",
            "FASTA inspection passed, but no trained inference artifacts are installed. No prediction was made.",
        )
        return qc, report, "setup_required"

    real_report["qc"] = real_report.get("qc") or qc
    real_report["_is_demo_fixture"] = False
    data_status = str(real_report.get("data_status") or "").upper()
    model_version = str(real_report.get("model_version") or "").lower()
    state = (
        "setup_required"
        if data_status in {"UNAVAILABLE", "SETUP_REQUIRED"}
        or model_version in {"unavailable", "setup required"}
        else "complete"
    )
    return qc, real_report, state


def _render_warning() -> None:
    st.markdown(
        f"""
        <div class="gf-warning" role="alert">
          <span class="gf-warning-icon" aria-hidden="true">!</span>
          <div><strong>Research prototype · not for clinical use</strong><br>
          <span>{_esc(RESEARCH_WARNING)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <section class="gf-hero">
          <div class="gf-eyebrow">SELECTIVE GENOMIC RESISTANCE INTELLIGENCE</div>
          <h1>Fast enough to help.<br><span>Honest enough to abstain.</span></h1>
          <p>Genome Firewall turns an assembled bacterial genome into calibrated,
          traceable research signals—with molecular target gates and a designed
          no-call when evidence is not strong enough.</p>
          <div class="gf-chip-row" aria-label="Project scope">
            <span class="gf-chip">E. coli edition</span>
            <span class="gf-chip">3 focused antibiotics</span>
            <span class="gf-chip">Group-aware calibration</span>
            <span class="gf-chip">Lab confirmation required</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_value_strip() -> None:
    st.markdown(
        """
        <div class="gf-value-grid">
          <div class="gf-value-item">
            <span class="gf-value-index">01</span>
            <div><strong>Calibrated</strong><p>Release requires measured, group-disjoint probability checks.</p></div>
          </div>
          <div class="gf-value-item">
            <span class="gf-value-index">02</span>
            <div><strong>Selective</strong><p>A no-call is a safety decision, never a hidden failure.</p></div>
          </div>
          <div class="gf-value-item">
            <span class="gf-value-index">03</span>
            <div><strong>Traceable</strong><p>Every call exposes target status, evidence and policy logic.</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_demo_selector() -> None:
    st.markdown("### Choose a guided case")
    st.caption(
        "Every option below is a disclosed demonstration fixture. It is selected to show a decision path; "
        "it is not a fresh or clinical analysis."
    )
    case_id = st.radio(
        "Demo fixture",
        options=list(DEMO_CASES),
        format_func=lambda item: DEMO_CASES[item]["label"],
        horizontal=True,
        label_visibility="collapsed",
        key="demo_case_picker",
    )
    case = DEMO_CASES[case_id]
    st.markdown(
        f"""
        <div class="gf-demo-preview">
          <div>
            <span class="gf-fixture-badge">DEMO FIXTURE</span>
            <span class="gf-eyebrow-inline">{_esc(case["eyebrow"])}</span>
          </div>
          <strong>{_esc(case["label"])}</strong>
          <p>{_esc(case["description"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Run selected demo", type="primary", width="stretch"):
        try:
            report = _load_demo_case(case_id)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state.report = report
            st.session_state.report_source = "demo"
            st.session_state.source_label = f"Demo fixture · {case['label']}"
            st.session_state.pipeline_state = "complete"
            st.session_state.qc = report.get("qc")
            st.session_state.explanation = None
            st.session_state.explanation_sample = None


def _render_upload_selector() -> None:
    st.markdown("### Inspect an assembled genome")
    st.caption(
        "Uploads use the real FASTA inspector. They receive predictions only when a trained inference "
        "pipeline is installed—demo fixtures are never substituted. A trained likely-to-work call "
        "also requires the optional target-observation JSON from a validated upstream search."
    )
    uploaded = st.file_uploader(
        "FASTA assembly",
        type=["fasta", "fa", "fna", "fas"],
        accept_multiple_files=False,
        help="UTF-8 nucleotide FASTA only. Raw reads and mixed samples are outside this prototype's scope.",
    )
    target_file = st.file_uploader(
        "Target observations (optional JSON)",
        type=["json"],
        accept_multiple_files=False,
        key="target_observations_upload",
        help=(
            "Per-drug, per-target observations from a separately validated target-search workflow. "
            "Genome Firewall recomputes the final gate; a supplied verdict is never trusted."
        ),
    )
    confirmed = st.checkbox(
        "I confirm this is a quality-checked, assembled E. coli research genome and contains no patient identifiers.",
        key="upload_confirmation",
    )
    can_run = uploaded is not None and confirmed
    if st.button(
        "Inspect and analyze",
        type="primary",
        width="stretch",
        disabled=not can_run,
        key="analyze_upload_button",
    ):
        data = uploaded.getvalue()
        try:
            max_upload_mb = max(1, min(int(os.getenv("GENOME_FIREWALL_MAX_UPLOAD_MB", "20")), 100))
        except ValueError:
            max_upload_mb = 20
        if len(data) > max_upload_mb * 1024 * 1024:
            st.error(f"The upload exceeds the {max_upload_mb} MB prototype limit.")
            return
        target_hits: Mapping[str, Mapping[str, str]] | None = None
        if target_file is not None:
            try:
                candidate = json.loads(target_file.getvalue().decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f"Target observation JSON is not valid: {exc}")
                return
            if not isinstance(candidate, dict) or any(
                not isinstance(value, dict) for value in candidate.values()
            ):
                st.error("Target observations must map each drug to a per-target JSON object.")
                return
            target_hits = candidate
        with st.spinner("Inspecting FASTA and checking for a real inference pipeline…"):
            qc, report, state = _analyze_upload(data, target_hits)
        st.session_state.qc = qc
        st.session_state.report = report
        st.session_state.report_source = "upload"
        st.session_state.source_label = (
            f"Uploaded FASTA · {report.get('sample_id', 'content hash')}"
        )
        st.session_state.pipeline_state = state
        st.session_state.explanation = None
        st.session_state.explanation_sample = None


def _render_source_selector() -> None:
    st.markdown(
        """
        <div class="gf-section-heading">
          <span>Start an analysis</span>
          <p>Explore a disclosed fixture or inspect your own assembled FASTA.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    demo_tab, upload_tab = st.tabs(["Guided demo fixtures", "Upload FASTA"])
    with demo_tab:
        _render_demo_selector()
    with upload_tab:
        _render_upload_selector()


def _pipeline_statuses(state: str) -> list[str]:
    if state == "complete":
        return ["done"] * 5
    if state == "setup_required":
        return ["done", "locked", "locked", "locked", "done"]
    if state == "qc_failed":
        return ["failed", "locked", "locked", "locked", "done"]
    if state == "inference_error":
        return ["done", "failed", "locked", "locked", "done"]
    return ["idle"] * 5


def _render_pipeline(state: str) -> None:
    stages = (
        ("01", "Input integrity", "FASTA + assembly QC"),
        ("02", "AMR annotation", "Curated determinant scan"),
        ("03", "Target gate", "Required target presence"),
        ("04", "Calibrated model", "Per-drug probability"),
        ("05", "Safety policy", "Conflict + no-call rules"),
    )
    statuses = _pipeline_statuses(state)
    stage_html = []
    for (index, title, detail), status in zip(stages, statuses, strict=True):
        symbol = {"done": "✓", "failed": "!", "locked": "·", "idle": "·"}[status]
        stage_html.append(
            f'<div class="gf-stage {status}">'
            f'<span class="gf-stage-status" aria-hidden="true">{symbol}</span>'
            f'<span class="gf-stage-index">{index}</span>'
            f"<strong>{html.escape(title)}</strong>"
            f"<small>{html.escape(detail)}</small>"
            "</div>"
        )
    st.markdown(
        '<div class="gf-section-heading compact">'
        "<span>Transparent analysis path</span>"
        "<p>Each result is downstream of deterministic quality and safety gates.</p>"
        "</div>"
        '<div class="gf-pipeline" role="list" aria-label="Analysis stages">'
        + "".join(stage_html)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_qc(qc: Mapping[str, Any]) -> None:
    if not qc:
        return
    with st.expander("Input integrity and provenance", expanded=False):
        count = qc.get("contig_count", qc.get("sequence_count"))
        bases = qc.get("total_bases", qc.get("total_length"))
        n50 = qc.get("n50", qc.get("N50"))
        n_fraction = qc.get("n_fraction", qc.get("ambiguous_fraction"))
        cols = st.columns(4)
        cols[0].metric("Inspection", "Passed" if _qc_passed(qc) else "Needs attention")
        cols[1].metric("Contigs", f"{int(count):,}" if _number(count) is not None else "—")
        cols[2].metric("Assembly size", f"{int(bases):,} bp" if _number(bases) is not None else "—")
        cols[3].metric("N50", f"{int(n50):,} bp" if _number(n50) is not None else "—")
        if _number(n_fraction) is not None:
            st.caption(f"Ambiguous-base fraction: {_percent(n_fraction, 2)}")
        issues: list[Any] = []
        for key in ("errors", "issues", "warnings"):
            values = qc.get(key) or []
            issues.extend([values] if isinstance(values, str) else values)
        issues = list(dict.fromkeys(str(item) for item in issues))
        if issues:
            st.warning(" · ".join(str(item) for item in issues))
        sha = qc.get("sha256") or qc.get("file_hash")
        source = qc.get("inspection_source")
        provenance = [
            part
            for part in (
                f"Inspector: {source}" if source else None,
                f"SHA-256: {sha}" if sha else None,
            )
            if part
        ]
        if provenance:
            st.code("\n".join(provenance), language=None)


def _result_card(prediction: Mapping[str, Any]) -> str:
    call = _canonical_call(prediction.get("call"))
    meta = CALL_META[call]
    drug = prediction.get("drug") or prediction.get("antibiotic") or "Unknown drug"
    confidence = _prediction_confidence(prediction)
    p_resistant = _number(prediction.get("p_resistant"))
    target = _text(prediction.get("target_status"), "Not assessed").title()
    evidence = _text(prediction.get("evidence_category"), "None").title()
    reason = prediction.get("no_call_reason")
    if not reason:
        reasons = prediction.get("decision_reasons") or []
        reason = reasons[0] if reasons else "Decision policy satisfied"
    return f"""
    <article class="gf-result-card {meta["css"]}">
      <div class="gf-result-topline">
        <span class="gf-status {meta["css"]}"><b aria-hidden="true">{meta["glyph"]}</b> {_esc(meta["label"])}</span>
        <span class="gf-confidence">{_percent(confidence)} confidence</span>
      </div>
      <h3>{_esc(drug)}</h3>
      <p>{_esc(reason)}</p>
      <div class="gf-result-meta">
        <span><small>P(resistant)</small><strong>{_percent(p_resistant)}</strong></span>
        <span><small>Target</small><strong>{_esc(target)}</strong></span>
        <span><small>Evidence</small><strong>{_esc(evidence)}</strong></span>
      </div>
    </article>
    """


def _result_rows(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        call = _canonical_call(prediction.get("call"))
        no_call_reason = prediction.get("no_call_reason")
        reasons = prediction.get("decision_reasons") or []
        short_reason = no_call_reason or (reasons[0] if reasons else "Policy thresholds satisfied")
        rows.append(
            {
                "Antibiotic": _text(
                    prediction.get("drug") or prediction.get("antibiotic"), "Unknown"
                ),
                "Research signal": CALL_META[call]["label"],
                "Confidence": _prediction_confidence(prediction),
                "P(resistant)": _number(prediction.get("p_resistant")),
                "Target": _text(prediction.get("target_status"), "Not assessed").title(),
                "Evidence": _text(prediction.get("evidence_category"), "None").title(),
                "Why": _text(short_reason),
            }
        )
    return rows


def _predictions_with_context(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Attach optional threshold/calibration context to each displayed drug."""
    context_by_drug = _as_mapping(report.get("decision_context"))
    normalized_context = {
        str(key).casefold(): _as_mapping(value) for key, value in context_by_drug.items()
    }
    predictions: list[dict[str, Any]] = []
    for raw_prediction in report.get("predictions") or []:
        prediction = _as_mapping(raw_prediction)
        drug = str(prediction.get("drug") or prediction.get("antibiotic") or "")
        context = normalized_context.get(drug.casefold(), {})
        for key in ("work_threshold", "fail_threshold"):
            if prediction.get(key) is None and context.get(key) is not None:
                prediction[key] = context[key]
        if not prediction.get("calibration_context") and context:
            prediction["calibration_context"] = {
                "bin_label": context.get("bin_label"),
                "observed_accuracy": context.get("observed_accuracy"),
                "sample_count": context.get("sample_count"),
            }
        predictions.append(prediction)
    return predictions


def _render_result_table(predictions: list[dict[str, Any]]) -> None:
    rows = _result_rows(predictions)
    try:
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="percent"
                ),
                "P(resistant)": st.column_config.NumberColumn(
                    "P(resistant)", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )
    except Exception:
        st.table(rows)


def _render_report_header(report: Mapping[str, Any], source: str) -> None:
    is_demo = bool(report.get("_is_demo_fixture")) or source == "demo"
    badge = "DEMO FIXTURE · NOT LIVE ANALYSIS" if is_demo else "UPLOADED FASTA · REAL PIPELINE ONLY"
    badge_class = "fixture" if is_demo else "upload"
    generated = report.get("generated_at") or report.get("timestamp") or "Not reported"
    st.markdown(
        f"""
        <div class="gf-report-header">
          <div>
            <span class="gf-source-badge {badge_class}">{_esc(badge)}</span>
            <h2>{_esc(report.get("sample_id"), "Sample")}</h2>
            <p>{_esc(report.get("supported_species"))}</p>
          </div>
          <div class="gf-report-provenance">
            <span><small>MODEL VERSION</small>{_esc(report.get("model_version"))}</span>
            <span><small>GENERATED</small>{_esc(generated)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _evidence_buckets(
    evidence: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    targets: list[dict[str, Any]] = []
    curated: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    for item in evidence:
        kind = str(
            item.get("category") or item.get("type") or item.get("evidence_type") or ""
        ).upper()
        kind = str(item.get("kind") or kind).upper()
        if "TARGET" in kind:
            targets.append(item)
            continue
        if bool(item.get("curated")):
            curated.append(item)
        elif any(token in kind for token in ("DETERMINANT", "GENE", "MUTATION")):
            provisional.append(item)
        else:
            associations.append(item)
    return targets, curated, provisional, associations


def _render_evidence_items(items: list[dict[str, Any]], empty_message: str) -> None:
    if not items:
        st.markdown(
            f"<div class='gf-empty-inline'>{_esc(empty_message)}</div>", unsafe_allow_html=True
        )
        return
    for item in items:
        label = (
            item.get("label")
            or item.get("name")
            or item.get("gene")
            or item.get("mutation")
            or item.get("feature")
            or "Evidence item"
        )
        detail = (
            item.get("description")
            or item.get("detail")
            or item.get("finding")
            or item.get("interpretation")
            or "No additional annotation was provided."
        )
        source = item.get("source") or item.get("reference") or item.get("quality")
        source_html = f"<small>{_esc(source)}</small>" if source else ""
        st.markdown(
            f"""
            <div class="gf-evidence-row">
              <span class="gf-evidence-dot" aria-hidden="true"></span>
              <div><strong>{_esc(label)}</strong><p>{_esc(detail)}</p>{source_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_probability_context(prediction: Mapping[str, Any]) -> None:
    p_resistant = _number(prediction.get("p_resistant"))
    work_threshold = _number(
        prediction.get("work_threshold") or _as_mapping(prediction.get("thresholds")).get("work")
    )
    fail_threshold = _number(
        prediction.get("fail_threshold") or _as_mapping(prediction.get("thresholds")).get("fail")
    )
    st.markdown("#### Decision context")
    if p_resistant is None:
        st.info("No calibrated probability was produced for this no-call.")
    else:
        st.metric("Calibrated P(resistant)", _percent(p_resistant, 1))
        st.progress(min(1.0, max(0.0, p_resistant)))
    if work_threshold is not None and fail_threshold is not None:
        st.caption(
            f"Work threshold ≤ {_percent(work_threshold)} · No-call band between thresholds · "
            f"Fail threshold ≥ {_percent(fail_threshold)}"
        )
    else:
        st.caption("Decision thresholds were not exposed by this report artifact.")
    calibration_samples = _number(prediction.get("calibration_samples"))
    calibration_groups = _number(prediction.get("calibration_groups"))
    if calibration_samples is not None and calibration_groups is not None:
        st.caption(
            f"Threshold support: {int(calibration_samples):,} calibration isolates across "
            f"{int(calibration_groups):,} disjoint groups."
        )

    reason = prediction.get("no_call_reason")
    if reason:
        st.markdown(
            f"<div class='gf-policy-note'><strong>No-call trigger</strong><span>{_esc(reason)}</span></div>",
            unsafe_allow_html=True,
        )
    reasons = prediction.get("decision_reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    if reasons:
        st.markdown("**Policy trace**")
        for index, reason_text in enumerate(reasons, start=1):
            st.markdown(f"{index}. {_text(reason_text)}")


def _render_calibration_context(prediction: Mapping[str, Any]) -> None:
    context = _as_mapping(
        prediction.get("calibration_context")
        or prediction.get("reliability_context")
        or prediction.get("calibration_bin")
    )
    st.markdown("#### How reliable is this confidence?")
    if not context:
        st.info("Calibration-bin context is not included in this report artifact.")
        return
    observed = context.get("observed_accuracy") or context.get("observed_rate")
    sample_count = context.get("sample_count") or context.get("n")
    label = context.get("bin_label") or context.get("range") or "Matched confidence bin"
    cols = st.columns(3)
    cols[0].metric("Confidence bin", _text(label))
    cols[1].metric("Observed correctness", _percent(observed, 1))
    cols[2].metric("Held-out isolates", f"{int(sample_count):,}" if _number(sample_count) else "—")
    st.caption(
        "Observed on group-held-out calibration data; it is not a guarantee for an individual genome."
    )


def _call_explainer(report: Mapping[str, Any], audience: str) -> Any:
    module = importlib.import_module("genome_firewall.explanations")
    helper = module.explain_report
    # Only the validated report is provided. The raw FASTA never enters this call.
    audience_value = "laboratory" if audience.startswith("Laboratory") else "plain_language"
    payload = report.get("_explanation_payload")
    if not isinstance(payload, Mapping):
        allowed = {
            "sample_id",
            "supported_species",
            "model_version",
            "predictions",
            "warning",
            "generated_at",
            "qc",
            "mode",
            "data_status",
            "demo_disclaimer",
            "provenance",
        }
        payload = {key: value for key, value in report.items() if key in allowed}
        if payload.get("qc") == {}:
            payload["qc"] = None
    signature = inspect.signature(helper)
    kwargs = {"audience": audience_value} if "audience" in signature.parameters else {}
    return _plain(helper(dict(payload), **kwargs))


def _render_explanation(report: Mapping[str, Any]) -> None:
    with st.expander("Grounded explanation · optional", expanded=False):
        st.markdown(
            "The explanation layer can restate validated report fields. It cannot change a call, "
            "invent evidence, recommend an antibiotic, or receive the raw genome."
        )
        audience = st.selectbox(
            "Audience",
            ["Laboratory professional", "Public-health reader"],
            key="explanation_audience",
        )
        if st.button("Generate grounded explanation", key="explain_report_button"):
            try:
                result = _call_explainer(report, audience)
            except Exception as exc:
                st.error(f"The explanation helper is unavailable: {exc}")
            else:
                st.session_state.explanation = result
                st.session_state.explanation_sample = report.get("sample_id")

        if st.session_state.explanation_sample != report.get("sample_id"):
            return
        explanation = st.session_state.explanation
        if explanation is None:
            return
        if isinstance(explanation, Mapping):
            audience_key = (
                "laboratory_professional" if audience.startswith("Laboratory") else "public_health"
            )
            body = (
                explanation.get(audience_key)
                or explanation.get(
                    "lab_professional"
                    if audience.startswith("Laboratory")
                    else "public_health_reader"
                )
                or explanation.get("explanation")
                or explanation.get("text")
                or explanation.get("summary")
                or json.dumps(explanation, indent=2)
            )
            source = (
                explanation.get("generated_by")
                or explanation.get("source")
                or explanation.get("mode")
                or "validated report fields"
            )
        else:
            body = str(explanation)
            source = "validated report fields"
        st.markdown(
            f"""
            <div class="gf-explanation">
              <span>GROUNDED OUTPUT · {_esc(source).upper()}</span>
              <p>{_esc(body)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if isinstance(explanation, Mapping):
            evidence_lines = explanation.get("evidence")
            if isinstance(evidence_lines, list) and evidence_lines:
                st.markdown("**Evidence priority**")
                for item in evidence_lines:
                    st.markdown(f"- {_esc(item)}")
            uncertainty = explanation.get("uncertainty")
            if uncertainty:
                st.markdown(f"**Uncertainty:** {_esc(uncertainty)}")
            limitations = explanation.get("limitations")
            if limitations:
                st.markdown(f"**Limitations:** {_esc(limitations)}")
            source_ids = explanation.get("source_ids")
            if isinstance(source_ids, list) and source_ids:
                st.caption("Traceable source IDs: " + ", ".join(_text(item) for item in source_ids))
        st.caption(
            "This explanation is research context only. It is not a treatment recommendation."
        )


def _render_evidence(report: Mapping[str, Any], predictions: list[dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="gf-section-heading compact">
          <span>Evidence drill-down</span>
          <p>Separate curated biology from non-causal model associations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    names = [_text(item.get("drug") or item.get("antibiotic"), "Unknown") for item in predictions]
    selected = st.selectbox("Antibiotic", names, key="evidence_drug")
    prediction = predictions[names.index(selected)]
    call = _canonical_call(prediction.get("call"))
    meta = CALL_META[call]
    target = _text(prediction.get("target_status"), "Not assessed").title()
    st.markdown(
        f"""
        <div class="gf-selected-result {meta["css"]}">
          <div><span class="gf-status {meta["css"]}"><b>{meta["glyph"]}</b> {_esc(meta["label"])}</span><h3>{_esc(selected)}</h3></div>
          <div><small>MOLECULAR TARGET</small><strong>{_esc(target)}</strong></div>
          <div><small>EVIDENCE CLASS</small><strong>{_esc(_text(prediction.get("evidence_category"), "None").title())}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        evidence = [_as_mapping(item) for item in (prediction.get("evidence") or [])]
        targets, curated, provisional, associations = _evidence_buckets(evidence)
        st.markdown("#### Target-detection evidence")
        _render_evidence_items(targets, "No target-detection detail was included in this report.")
        st.markdown("#### Curated determinant evidence")
        _render_evidence_items(curated, "No curated determinant was reported for this drug.")
        st.markdown("#### Provisional determinant matches")
        st.caption(
            "Unreviewed or ambiguous matches can trigger abstention; they are not curated facts."
        )
        _render_evidence_items(provisional, "No provisional determinant match was reported.")
        st.markdown("#### Statistical associations")
        st.caption("Model associations are predictive signals, not proof of biological causation.")
        _render_evidence_items(associations, "No statistical association details were reported.")
    with right:
        _render_probability_context(prediction)
        _render_calibration_context(prediction)

    _render_explanation(report)


def _render_results(report: Mapping[str, Any], source: str) -> None:
    predictions = _predictions_with_context(report)
    predictions.sort(key=lambda item: CALL_META[_canonical_call(item.get("call"))]["priority"])
    _render_report_header(report, source)

    if report.get("setup_state"):
        st.warning(_text(report.get("setup_message"), "The inference pipeline requires setup."))

    if not predictions:
        st.info("This report does not contain per-drug predictions.")
        return

    counts = {key: 0 for key in CALL_META}
    for prediction in predictions:
        counts[_canonical_call(prediction.get("call"))] += 1
    metric_cols = st.columns(4)
    metric_cols[0].metric("Antibiotics assessed", len(predictions))
    metric_cols[1].metric("Likely-to-fail signals", counts["LIKELY_TO_FAIL"])
    metric_cols[2].metric("No-calls", counts["NO_CALL"])
    metric_cols[3].metric("Likely-to-work signals", counts["LIKELY_TO_WORK"])

    card_cols = st.columns(min(3, len(predictions)), gap="medium")
    for index, prediction in enumerate(predictions):
        with card_cols[index % len(card_cols)]:
            st.markdown(_result_card(prediction), unsafe_allow_html=True)

    summary_tab, evidence_tab = st.tabs(["Decision table", "Evidence & confidence"])
    with summary_tab:
        _render_result_table(predictions)
        st.caption(
            "Likely-to-work means the target gate and calibrated policy were satisfied. It does not mean treatment is recommended."
        )
    with evidence_tab:
        _render_evidence(report, predictions)


def _illustrative_validation() -> dict[str, Any]:
    return {
        "is_fixture": True,
        "label": "Awaiting organizer held-out evaluation",
        "metrics": [],
        "reliability": [],
        "risk_coverage": [],
    }


def _load_validation(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report:
        embedded = report.get("validation") or report.get("evaluation")
        if isinstance(embedded, Mapping):
            validation = _as_mapping(embedded)
            validation.setdefault("is_fixture", bool(report.get("_is_demo_fixture")))
            return validation
        metrics_bundle = report.get("metrics")
        if isinstance(metrics_bundle, Mapping) and any(
            key in metrics_bundle for key in ("per_drug", "reliability", "risk_coverage")
        ):
            reliability = []
            for item in metrics_bundle.get("reliability") or []:
                row = _as_mapping(item)
                row.setdefault("drug", "All demo drugs")
                if "count" in row and "n" not in row:
                    row["n"] = row["count"]
                reliability.append(row)
            risk_coverage = []
            for item in metrics_bundle.get("risk_coverage") or []:
                row = _as_mapping(item)
                row.setdefault("drug", "All demo drugs")
                risk_coverage.append(row)
            return {
                "is_fixture": bool(report.get("_is_demo_fixture"))
                or "ILLUSTRATIVE" in str(metrics_bundle.get("status") or "").upper(),
                "metrics": metrics_bundle.get("per_drug") or [],
                "reliability": reliability,
                "risk_coverage": risk_coverage,
            }
    try:
        module = importlib.import_module("genome_firewall.demo")
        loader = module.load_validation_fixture
        validation = _as_mapping(loader())
        validation.setdefault("is_fixture", True)
        return validation
    except (ImportError, AttributeError, FileNotFoundError):
        return _illustrative_validation()
    except Exception:
        return _illustrative_validation()


def _render_reliability_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No reliability-bin data is present in this evaluation artifact.")
        return
    try:
        import altair as alt
        import pandas as pd

        frame = pd.DataFrame(rows).rename(
            columns={
                "predicted": "predicted_probability",
                "observed": "observed_rate",
                "antibiotic": "drug",
            }
        )
        frame["predicted_probability"] = frame["predicted_probability"].astype(float)
        frame["observed_rate"] = frame["observed_rate"].astype(float)
        actual = (
            alt.Chart(frame)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=56), strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "predicted_probability:Q",
                    title="Predicted resistance probability",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y(
                    "observed_rate:Q",
                    title="Observed resistant fraction",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color("drug:N", title="Antibiotic"),
                tooltip=[
                    alt.Tooltip("drug:N", title="Antibiotic"),
                    alt.Tooltip("predicted_probability:Q", title="Predicted", format=".0%"),
                    alt.Tooltip("observed_rate:Q", title="Observed", format=".0%"),
                    alt.Tooltip("n:Q", title="Isolates"),
                ],
            )
        )
        ideal_frame = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        ideal = (
            alt.Chart(ideal_frame)
            .mark_line(strokeDash=[6, 5], strokeWidth=1.5, opacity=0.55)
            .encode(x="x:Q", y="y:Q")
        )
        st.altair_chart((ideal + actual).properties(height=360), width="stretch", theme="streamlit")
    except Exception:
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_risk_coverage_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No risk-versus-coverage data is present in this evaluation artifact.")
        return
    try:
        import altair as alt
        import pandas as pd

        frame = pd.DataFrame(rows).rename(
            columns={
                "risk": "called_error_rate",
                "error": "called_error_rate",
                "error_rate": "called_error_rate",
                "antibiotic": "drug",
            }
        )
        chart = (
            alt.Chart(frame)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=56), strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "coverage:Q",
                    title="Call coverage",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y(
                    "called_error_rate:Q",
                    title="Error among called samples",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color("drug:N", title="Antibiotic"),
                tooltip=[
                    alt.Tooltip("drug:N", title="Antibiotic"),
                    alt.Tooltip("coverage:Q", title="Coverage", format=".0%"),
                    alt.Tooltip("called_error_rate:Q", title="Called error", format=".1%"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, width="stretch", theme="streamlit")
    except Exception:
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_metrics_table(metrics: Any) -> None:
    if not metrics:
        st.info("No per-drug metric table is present in this evaluation artifact.")
        return
    if isinstance(metrics, Mapping):
        rows = []
        for drug, values in metrics.items():
            row = _as_mapping(values)
            row.setdefault("Antibiotic", drug)
            rows.append(row)
    else:
        rows = [_as_mapping(item) for item in metrics]
    label_map = {
        "drug": "Antibiotic",
        "antibiotic": "Antibiotic",
        "balanced_accuracy": "Balanced accuracy",
        "resistant_recall": "Resistant recall",
        "susceptible_recall": "Susceptible recall",
        "false_susceptible_rate": "False-susceptible rate",
        "coverage": "Coverage",
        "f1": "Resistant F1",
        "auroc": "AUROC",
        "pr_auc": "PR-AUC",
        "brier": "Brier score",
        "brier_score": "Brier score",
    }
    percentage_fields = {
        "Balanced accuracy",
        "Resistant recall",
        "Susceptible recall",
        "False-susceptible rate",
        "Coverage",
    }
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        display: dict[str, Any] = {}
        for key, value in row.items():
            label = label_map.get(str(key), str(key).replace("_", " ").title())
            if label in percentage_fields and _number(value) is not None:
                display[label] = _percent(value, 1)
            elif (
                label in {"Brier score", "AUROC", "PR-AUC", "Resistant F1"}
                and _number(value) is not None
            ):
                display[label] = f"{float(value):.3f}"
            else:
                display[label] = value
        display_rows.append(display)
    try:
        st.dataframe(display_rows, width="stretch", hide_index=True)
    except Exception:
        st.table(display_rows)


def _render_split_contract(validation: Mapping[str, Any]) -> None:
    split = _as_mapping(validation.get("split") or validation.get("group_split"))
    verified = bool(split.get("verified") or validation.get("group_disjoint_verified"))
    status = "VERIFIED ARTIFACT" if verified else "REQUIRED EVALUATION CONTRACT"
    audit_message = (
        "This artifact reports that the audit passed."
        if verified
        else "Load a trained evaluation artifact to verify the audit."
    )
    st.markdown(
        f"""
        <div class="gf-split-wrap">
          <div class="gf-split-head"><strong>Group-disjoint evaluation</strong><span>{_esc(status)}</span></div>
          <div class="gf-split-track">
            <div class="fit"><b>MODEL FIT</b><small>Whole genetic groups</small></div>
            <div class="tune"><b>TUNING</b><small>Separate groups</small></div>
            <div class="cal"><b>CALIBRATION</b><small>Untouched groups</small></div>
            <div class="test"><b>HELD-OUT TEST</b><small>Final groups</small></div>
          </div>
          <p>Near-identical genomes and their related groups must remain in one partition. {_esc(audit_message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_validation(report: Mapping[str, Any] | None) -> None:
    st.markdown(
        """
        <div class="gf-page-intro">
          <span class="gf-eyebrow">VALIDATION</span>
          <h1>Trust is a measured property.</h1>
          <p>Calibration shows whether confidence matches reality. Risk-versus-coverage shows the price—and value—of abstention.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    validation = _load_validation(report)
    if validation.get("is_fixture", True):
        st.markdown(
            """
            <div class="gf-fixture-callout" role="note">
              <strong>NO MEASURED PERFORMANCE LOADED</strong>
              <span>This project refuses to display invented benchmark numbers. Install the organizer dataset and an untouched group-held-out evaluation artifact to populate this page.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("Loaded measured held-out evaluation artifact.")

    _render_split_contract(validation)
    st.markdown("### Per-drug evaluation")
    _render_metrics_table(validation.get("metrics"))
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Reliability")
        st.caption(
            "Perfect calibration follows the diagonal: predicted risk equals observed resistance."
        )
        _render_reliability_chart(
            [_as_mapping(item) for item in (validation.get("reliability") or [])]
        )
    with right:
        st.markdown("### Risk versus coverage")
        st.caption("Moving left means the system abstains more; the called error rate should fall.")
        _render_risk_coverage_chart(
            [_as_mapping(item) for item in (validation.get("risk_coverage") or [])]
        )
    st.markdown(
        """
        <div class="gf-method-note">
          <strong>What judges should look for</strong>
          <span>Per-drug resistant recall, susceptible recall, false-susceptible risk, calibration, group diversity and the disclosed no-call rate—not one pooled accuracy headline.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _artifact_status() -> tuple[str, str]:
    manifests = (
        ROOT / "artifacts" / "manifest.json",
        ROOT / "artifacts" / "model_manifest.json",
        ROOT / "models" / "manifest.json",
    )
    if any(path.exists() for path in manifests):
        return "Artifact manifest detected", "ready"
    try:
        module = importlib.import_module("genome_firewall.inference")
        _predict_fasta = module.predict_fasta
        return "Inference entry point detected", "ready"
    except (ImportError, AttributeError):
        return "Training artifacts not installed", "setup"


def _render_model_card(report: Mapping[str, Any] | None) -> None:
    status, status_class = _artifact_status()
    report = report or {}
    st.markdown(
        f"""
        <div class="gf-page-intro model-card-intro">
          <span class="gf-eyebrow">MODEL CARD · RESEARCH PROTOTYPE</span>
          <h1>Know the boundary before the result.</h1>
          <p>A compact disclosure of intended use, scientific assumptions, supported scope and failure modes.</p>
          <span class="gf-artifact-status {status_class}">{_esc(status)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview, boundary = st.columns([1.05, 0.95], gap="large")
    with overview:
        st.markdown("### Reference system")
        st.markdown(
            """
            - **Task:** selective binary antimicrobial-response prediction with an explicit no-call.
            - **Species scope:** *Escherichia coli* only; species identity must be established upstream.
            - **Drug scope:** ciprofloxacin, ceftriaxone and gentamicin, only after each drug passes data and calibration gates.
            - **Model:** one sparse, calibrated logistic model per drug with deterministic target and conflict gates.
            - **Input:** quality-checked assembled nucleotide FASTA; no raw reads, mixed samples or patient metadata.
            """
        )
    with boundary:
        st.markdown("### Decision boundary")
        st.markdown(
            """
            <div class="gf-boundary-list">
              <div><span>CAN</span><p>Expose genomic evidence, calibrated risk, target status and no-call reasons.</p></div>
              <div><span>CANNOT</span><p>Select treatment, dose a drug, replace culture-based testing or infer patient outcome.</p></div>
              <div><span>REQUIRES</span><p>Standard laboratory susceptibility confirmation and trained professional review.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    limitations_tab, governance_tab, provenance_tab = st.tabs(
        ["Limitations & failure modes", "Safety & governance", "Versions & provenance"]
    )
    with limitations_tab:
        st.markdown(
            """
            - Genotype does not guarantee phenotype; expression, regulation and unmodeled mechanisms can change response.
            - Novel lineages, novel resistance mechanisms and out-of-distribution feature profiles may trigger a no-call.
            - Fragmented assemblies can obscure targets and determinants; ambiguous target evidence blocks a likely-to-work call.
            - Population prevalence and laboratory methods can shift calibration. Local prospective validation is required.
            - A low resistance probability is not a treatment recommendation and cannot account for patient-level factors.
            """
        )
    with governance_tab:
        st.markdown(
            """
            - The language layer receives validated report fields only—never raw genome sequence.
            - The language layer cannot change calls, confidences, evidence or thresholds.
            - Uploaded sequences are processed for the current session; deployments must add an explicit retention policy.
            - Known biological determinants are displayed separately from statistical, non-causal associations.
            - Every exported or on-screen report must preserve the research-only warning.
            """
        )
    with provenance_tab:
        provenance = _as_mapping(report.get("provenance"))
        provenance_rows = [
            {
                "Field": "Active report model",
                "Value": _text(report.get("model_version"), "No report loaded"),
            },
            {
                "Field": "Active report species",
                "Value": _text(report.get("supported_species"), "No report loaded"),
            },
            {
                "Field": "Feature schema",
                "Value": _text(
                    report.get("feature_schema_version")
                    or provenance.get("annotation_feature_schema_version"),
                    "Must be supplied by artifact",
                ),
            },
            {
                "Field": "AMRFinderPlus",
                "Value": _text(
                    report.get("amrfinder_version") or provenance.get("amrfinder_version"),
                    "Must be supplied by artifact",
                ),
            },
            {
                "Field": "AMR database",
                "Value": _text(
                    report.get("amrfinder_database_version")
                    or provenance.get("amrfinder_database_version"),
                    "Must be supplied by artifact",
                ),
            },
            {"Field": "Generated", "Value": _text(report.get("generated_at"), "No report loaded")},
        ]
        st.dataframe(provenance_rows, width="stretch", hide_index=True)
        st.caption(
            "Missing versions are shown as missing; the interface never invents artifact provenance."
        )

    st.markdown(
        f"<div class='gf-warning model-card-warning'><strong>{_esc(RESEARCH_WARNING)}</strong></div>",
        unsafe_allow_html=True,
    )


def _render_empty_report() -> None:
    st.markdown(
        """
        <div class="gf-empty-state">
          <span aria-hidden="true">⌁</span>
          <h3>No analysis loaded yet</h3>
          <p>Choose a disclosed demo fixture or inspect an assembled FASTA above. Results will appear here with their evidence and no-call logic.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_analyze() -> None:
    _render_hero()
    _render_value_strip()
    _render_source_selector()
    report = st.session_state.report
    if not report:
        _render_empty_report()
        return
    report = _normalize_report(report)
    report["_is_demo_fixture"] = st.session_state.report_source == "demo"
    _render_pipeline(st.session_state.pipeline_state)
    _render_qc(_as_mapping(report.get("qc") or st.session_state.qc))
    _render_results(report, st.session_state.report_source or "unknown")


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="gf-brand">
              <div class="gf-brand-mark" aria-hidden="true">GF</div>
              <div><strong>Genome Firewall</strong><span>Research workspace</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        navigation = st.radio(
            "Workspace",
            ["Analyze", "Validation", "Model card"],
            label_visibility="collapsed",
            key="navigation",
        )
        st.markdown("<div class='gf-sidebar-spacer'></div>", unsafe_allow_html=True)
        source_label = st.session_state.source_label
        if source_label:
            source_kind = (
                "DEMO FIXTURE" if st.session_state.report_source == "demo" else "ACTIVE INPUT"
            )
            st.markdown(
                f"""
                <div class="gf-sidebar-source">
                  <small>{_esc(source_kind)}</small>
                  <strong>{_esc(source_label)}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="gf-sidebar-safety">
              <strong>Research only</strong>
              <span>Every result requires standard laboratory confirmation.</span>
            </div>
            <div class="gf-build-label">GENOME FIREWALL · E. COLI EDITION</div>
            """,
            unsafe_allow_html=True,
        )
    return navigation


def main() -> None:
    _inject_styles()
    _init_state()
    navigation = _render_sidebar()
    _render_warning()
    if navigation == "Analyze":
        _render_analyze()
    elif navigation == "Validation":
        report = _normalize_report(st.session_state.report) if st.session_state.report else None
        if report:
            report["_is_demo_fixture"] = st.session_state.report_source == "demo"
        _render_validation(report)
    else:
        report = _normalize_report(st.session_state.report) if st.session_state.report else None
        _render_model_card(report)


if __name__ == "__main__":
    main()
