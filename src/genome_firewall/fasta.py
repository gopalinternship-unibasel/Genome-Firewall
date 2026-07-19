"""Strict, dependency-free FASTA parsing and assembly quality checks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from os import PathLike
from pathlib import Path

from .schemas import FastaLimits, FastaQC

DNA_IUPAC = frozenset("ACGTRYSWKMBDHVN")
UNAMBIGUOUS_DNA = frozenset("ACGT")


class FastaFormatError(ValueError):
    """Raised by :func:`parse_fasta` when strict parsing fails."""


@dataclass(frozen=True, slots=True)
class FastaRecord:
    identifier: str
    description: str
    sequence: str


def _read_source(source: str | bytes | PathLike[str]) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, PathLike) and not isinstance(source, str):
        return Path(source).read_bytes()
    if not isinstance(source, str):
        raise TypeError("source must be FASTA text, bytes, or a filesystem path")

    # Multiline strings and strings beginning with a FASTA header are content.
    # A short existing single-line string may be used as a convenience path;
    # pathlib.Path remains the unambiguous path interface.
    if "\n" in source or "\r" in source or source.lstrip().startswith(">"):
        return source.encode("utf-8")
    try:
        candidate = Path(source)
        if len(source) <= 1_024 and candidate.is_file():
            return candidate.read_bytes()
    except (OSError, ValueError):
        pass
    return source.encode("utf-8")


def _parse_text(text: str) -> tuple[list[FastaRecord], list[str], int, int, int]:
    records: list[FastaRecord] = []
    structural_errors: list[str] = []
    seen_identifiers: set[str] = set()
    duplicate_headers = 0
    empty_sequences = 0
    invalid_characters = 0

    current_identifier: str | None = None
    current_description = ""
    chunks: list[str] = []
    sequence_before_header_reported = False

    def finish_record() -> None:
        nonlocal empty_sequences
        if current_identifier is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            empty_sequences += 1
            structural_errors.append(f"Sequence '{current_identifier}' is empty")
        records.append(
            FastaRecord(
                identifier=current_identifier,
                description=current_description,
                sequence=sequence,
            )
        )

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish_record()
            chunks = []
            header = line[1:].strip()
            if not header:
                structural_errors.append(f"Header on line {line_number} has no identifier")
                current_identifier = f"__missing_identifier_{line_number}"
                current_description = ""
                continue
            parts = header.split(maxsplit=1)
            current_identifier = parts[0]
            current_description = parts[1] if len(parts) == 2 else ""
            if current_identifier in seen_identifiers:
                duplicate_headers += 1
            seen_identifiers.add(current_identifier)
            continue

        if current_identifier is None:
            if not sequence_before_header_reported:
                structural_errors.append(
                    f"Sequence data appears before the first FASTA header (line {line_number})"
                )
                sequence_before_header_reported = True
            continue

        sequence_line = "".join(line.split()).upper()
        invalid_characters += sum(character not in DNA_IUPAC for character in sequence_line)
        chunks.append(sequence_line)

    finish_record()
    if not records:
        structural_errors.append("No FASTA records were found")
    if invalid_characters:
        structural_errors.append(
            f"Found {invalid_characters} character(s) outside the DNA IUPAC alphabet"
        )
    return records, structural_errors, duplicate_headers, empty_sequences, invalid_characters


def parse_fasta(source: str | bytes | PathLike[str]) -> list[FastaRecord]:
    """Parse FASTA records and raise on structural or character errors.

    This strict parser is useful when downstream annotation must never receive a
    malformed payload.  For user-facing validation, :func:`analyze_fasta`
    returns all detected issues in one structured result instead.
    """

    raw = _read_source(source)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FastaFormatError("FASTA must be UTF-8 text") from exc
    records, errors, _, _, _ = _parse_text(text)
    if errors:
        raise FastaFormatError("; ".join(errors))
    return records


def _n50(lengths: Iterable[int], total: int) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    running = 0
    for index, length in enumerate(sorted(lengths, reverse=True), start=1):
        running += length
        if running * 2 >= total:
            return length, index
    return 0, 0


def analyze_fasta(
    source: str | bytes | PathLike[str],
    limits: FastaLimits | None = None,
) -> FastaQC:
    """Return deterministic structural and assembly QC for a FASTA payload."""

    policy = limits or FastaLimits()
    raw = _read_source(source)
    digest = sha256(raw).hexdigest()
    errors: list[str] = []
    warnings: list[str] = []

    try:
        text = raw.decode("utf-8")
        decode_ok = True
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        decode_ok = False
        errors.append("FASTA must be UTF-8 text")

    records, structural_errors, duplicates, empty, invalid = _parse_text(text)
    errors.extend(structural_errors)
    lengths = [len(record.sequence) for record in records]
    total = sum(lengths)
    n50, l50 = _n50(lengths, total)
    combined = "".join(record.sequence for record in records)
    unambiguous_count = sum(base in UNAMBIGUOUS_DNA for base in combined)
    gc_count = combined.count("G") + combined.count("C")
    ambiguous_count = sum(base in DNA_IUPAC and base not in UNAMBIGUOUS_DNA for base in combined)
    n_count = combined.count("N")

    gc_fraction = gc_count / unambiguous_count if unambiguous_count else None
    ambiguous_fraction = ambiguous_count / total if total else None
    n_fraction = n_count / total if total else None

    if len(raw) > policy.max_file_bytes:
        errors.append(
            f"File size {len(raw):,} bytes exceeds the {policy.max_file_bytes:,}-byte limit"
        )
    if total < policy.min_total_bases:
        errors.append(f"Assembly has {total:,} bases; minimum is {policy.min_total_bases:,}")
    if total > policy.max_total_bases:
        errors.append(f"Assembly has {total:,} bases; maximum is {policy.max_total_bases:,}")
    if len(records) > policy.max_contigs:
        errors.append(f"Assembly has {len(records):,} contigs; maximum is {policy.max_contigs:,}")
    if total and n50 < policy.min_n50:
        errors.append(f"Assembly N50 is {n50:,}; minimum is {policy.min_n50:,}")
    if ambiguous_fraction is not None and ambiguous_fraction > policy.max_ambiguous_fraction:
        errors.append(
            f"Ambiguous-base fraction {ambiguous_fraction:.3%} exceeds "
            f"{policy.max_ambiguous_fraction:.3%}"
        )
    if duplicates > policy.max_duplicate_headers:
        errors.append(
            f"Found {duplicates} duplicate FASTA identifier(s); maximum is "
            f"{policy.max_duplicate_headers}"
        )
    elif duplicates:
        warnings.append(f"Found {duplicates} duplicate FASTA identifier(s)")
    if ambiguous_fraction and ambiguous_fraction > 0:
        warnings.append(f"Assembly contains {ambiguous_fraction:.3%} ambiguous bases")

    valid_fasta = decode_ok and not structural_errors
    return FastaQC(
        valid_fasta=valid_fasta,
        passes_qc=valid_fasta and not errors,
        sha256=digest,
        file_bytes=len(raw),
        sequence_count=len(records),
        total_bases=total,
        n50=n50,
        l50=l50,
        shortest_contig=min(lengths, default=0),
        longest_contig=max(lengths, default=0),
        gc_fraction=gc_fraction,
        ambiguous_fraction=ambiguous_fraction,
        n_fraction=n_fraction,
        invalid_character_count=invalid,
        duplicate_header_count=duplicates,
        empty_sequence_count=empty,
        errors=errors,
        warnings=warnings,
    )


__all__ = [
    "DNA_IUPAC",
    "FastaFormatError",
    "FastaRecord",
    "analyze_fasta",
    "parse_fasta",
]
