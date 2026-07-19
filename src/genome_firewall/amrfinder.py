"""Fail-safe AMRFinderPlus adapter and normalized feature extraction."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Bump this whenever ``features_from_hits`` changes meaning or naming. Trained
# artifacts pin this value and live inference refuses incompatible features.
AMRFINDER_FEATURE_SCHEMA_VERSION = "1"


class AMRHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_symbol: str
    element_name: str | None = None
    scope: str | None = None
    type: str | None = None
    subtype: str | None = None
    drug_class: str | None = None
    subclass: str | None = None
    method: str | None = None
    coverage_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    identity_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    contig_id: str | None = None
    start: int | None = None
    stop: int | None = None
    closest_accession: str | None = None
    hmm_accession: str | None = None
    raw: dict[str, str] = Field(default_factory=dict)


class AnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    hits: list[AMRHit] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    amrfinder_version: str | None = None
    database_version: str | None = None
    error: str | None = None


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _first(row: dict[str, str], *names: str) -> str | None:
    normalized = {_key(key): value.strip() for key, value in row.items() if key}
    for name in names:
        value = normalized.get(_key(name))
        if value and value not in {"-", "NA", "N/A"}:
            return value
    return None


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.rstrip("%"))
    except ValueError:
        return None


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_amrfinder_tsv(path: str | Path) -> list[AMRHit]:
    """Parse current and common historical AMRFinderPlus TSV headers."""

    hits: list[AMRHit] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("AMRFinderPlus output has no header")
        accepted_symbol_headers = {
            _key(value) for value in ("Element symbol", "Gene symbol", "Gene")
        }
        observed_headers = {_key(value) for value in reader.fieldnames if value}
        if observed_headers.isdisjoint(accepted_symbol_headers):
            raise ValueError(
                "AMRFinderPlus output does not contain a recognized element/gene symbol header"
            )
        for row in reader:
            symbol = _first(row, "Element symbol", "Gene symbol", "Gene")
            if not symbol:
                continue
            hits.append(
                AMRHit(
                    element_symbol=symbol,
                    element_name=_first(row, "Element name", "Sequence name", "Protein name"),
                    scope=_first(row, "Scope"),
                    type=_first(row, "Type", "Element type"),
                    subtype=_first(row, "Subtype", "Element subtype"),
                    drug_class=_first(row, "Class"),
                    subclass=_first(row, "Subclass"),
                    method=_first(row, "Method"),
                    coverage_pct=_float(
                        _first(row, "% Coverage of reference sequence", "Coverage")
                    ),
                    identity_pct=_float(
                        _first(row, "% Identity to reference sequence", "Identity")
                    ),
                    contig_id=_first(row, "Contig id", "Contig"),
                    start=_int(_first(row, "Start")),
                    stop=_int(_first(row, "Stop")),
                    closest_accession=_first(
                        row, "Closest reference accession", "Accession of closest sequence"
                    ),
                    hmm_accession=_first(row, "HMM id", "HMM accession"),
                    raw={key: value for key, value in row.items() if key is not None},
                )
            )
    return hits


def features_from_hits(hits: list[AMRHit]) -> dict[str, float]:
    """Create a deterministic sparse feature mapping from normalized hits."""

    features: dict[str, float] = {}
    for hit in hits:
        description = " ".join(filter(None, (hit.type, hit.subtype, hit.element_name))).casefold()
        prefix = "mutation" if "mutation" in description or ":" in hit.element_symbol else "gene"
        canonical = re.sub(r"\s+", "_", hit.element_symbol.strip())
        features[f"{prefix}::{canonical}"] = 1.0
        if hit.method:
            quality = _key(hit.method)
            features[f"hit_quality::{canonical}::{quality}"] = 1.0
    return features


def _version(binary: str) -> str | None:
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        value = (completed.stdout or completed.stderr).strip()
        return value[:256] or None
    except (OSError, subprocess.SubprocessError):
        return None


def _database_identifier(database_path: str | None) -> str:
    if not database_path:
        return "unverified:default-local-database"
    root = Path(database_path)
    if not root.exists():
        return f"unverified:missing:{root.name}"
    candidates = [
        root / "database_version.txt",
        root / "version.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            raw = candidate.read_bytes()
            value = raw.decode("utf-8", errors="replace").strip().replace("\n", " ")[:120]
            digest = hashlib.sha256(raw).hexdigest()[:12]
            return f"{value or candidate.name};sha256={digest}"
    return f"unverified:path:{root.resolve()}"


def run_amrfinder(
    fasta: bytes | str,
    *,
    organism: str = "Escherichia",
    binary: str | None = None,
    database: str | None = None,
    timeout_seconds: int = 600,
) -> AnnotationResult:
    """Run AMRFinderPlus locally and fail closed on any setup/runtime error.

    No sequence content leaves the machine.  The subprocess receives explicit
    argument tokens and writes only inside a temporary directory.
    """

    executable = binary or os.getenv("AMRFINDER_BIN", "amrfinder")
    database_path = database or os.getenv("AMRFINDER_DB")
    version = _version(executable)
    if version is None:
        return AnnotationResult(
            ok=False, error=f"AMRFinderPlus executable not available: {executable}"
        )

    payload = fasta.encode("utf-8") if isinstance(fasta, str) else fasta
    try:
        with tempfile.TemporaryDirectory(prefix="genome-firewall-amr-") as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.fasta"
            output_path = temp / "amrfinder.tsv"
            input_path.write_bytes(payload)
            command = [executable, "-n", str(input_path), "-O", organism, "-o", str(output_path)]
            if database_path:
                command.extend(["-d", database_path])
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if completed.returncode != 0 or not output_path.exists():
                detail = (
                    completed.stderr or completed.stdout or "unknown AMRFinderPlus error"
                ).strip()
                return AnnotationResult(ok=False, amrfinder_version=version, error=detail[:2_000])
            hits = parse_amrfinder_tsv(output_path)
            return AnnotationResult(
                ok=True,
                hits=hits,
                features=features_from_hits(hits),
                amrfinder_version=version,
                database_version=_database_identifier(database_path),
            )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return AnnotationResult(ok=False, amrfinder_version=version, error=str(exc)[:2_000])


__all__ = [
    "AMRFINDER_FEATURE_SCHEMA_VERSION",
    "AMRHit",
    "AnnotationResult",
    "features_from_hits",
    "parse_amrfinder_tsv",
    "run_amrfinder",
]
