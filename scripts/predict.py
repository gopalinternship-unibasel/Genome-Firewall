"""Run fail-closed FASTA inference and emit validated JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genome_firewall.service import DEFAULT_DRUGS, analyze_upload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--species", default="Escherichia coli")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--catalog", type=Path, default=Path("config/drug_catalog.yaml"))
    parser.add_argument("--drugs", nargs="+", default=list(DEFAULT_DRUGS))
    parser.add_argument(
        "--target-hits-json",
        type=Path,
        default=None,
        help=(
            "JSON mapping of drug -> target -> PRESENT/ABSENT/AMBIGUOUS from a "
            "separately validated target-search workflow. Final target verdicts are recomputed."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    targets = None
    if args.target_hits_json:
        targets = json.loads(args.target_hits_json.read_text(encoding="utf-8"))
        if not isinstance(targets, dict):
            parser.error("--target-hits-json must contain a JSON object")
        if any(not isinstance(value, dict) for value in targets.values()):
            parser.error("--target-hits-json values must be per-target JSON objects")

    report = analyze_upload(
        args.fasta.read_bytes(),
        sample_id=args.sample_id or args.fasta.stem,
        species=args.species,
        requested_drugs=args.drugs,
        artifact_dir=args.artifact_dir,
        target_hits_by_drug=targets,
        catalog_path=args.catalog,
    )
    rendered = report.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
