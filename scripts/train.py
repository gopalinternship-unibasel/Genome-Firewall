"""Train one calibrated model per drug from fixed, laboratory-measured data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from genome_firewall.amrfinder import AMRFINDER_FEATURE_SCHEMA_VERSION
from genome_firewall.predictor import ModelRegistry
from genome_firewall.training import TrainingConfig, train_drug_model

REQUIRED_LABEL_COLUMNS = {"sample_id", "species", "drug", "phenotype", "group_id"}
PROHIBITED_FEATURE_TOKENS = {
    "assembly",
    "breakpoint",
    "drug",
    "filename",
    "group",
    "label",
    "outcome",
    "partition",
    "path",
    "patient",
    "phenotype",
    "site",
    "species",
    "split",
    "year",
}


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise SystemExit(
                "Parquet input requires the optional dependency: "
                "python -m pip install 'genome-firewall[parquet]'"
            ) from exc
    return pd.read_csv(path)


def _prohibited_feature_columns(columns: list[str]) -> list[str]:
    blocked: list[str] = []
    for column in columns:
        tokens = {value for value in re.split(r"[^a-z0-9]+", column.casefold()) if value}
        if tokens & PROHIBITED_FEATURE_TOKENS:
            blocked.append(column)
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--species", default="Escherichia coli")
    parser.add_argument("--drugs", nargs="*", default=None)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--feature-schema-version", default=AMRFINDER_FEATURE_SCHEMA_VERSION)
    parser.add_argument(
        "--amrfinder-version",
        required=True,
        help="Exact AMRFinderPlus version used to create the feature table.",
    )
    parser.add_argument(
        "--amrfinder-database-version",
        required=True,
        help="Pinned AMRFinderPlus database version/hash used to create the feature table.",
    )
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.feature_schema_version != AMRFINDER_FEATURE_SCHEMA_VERSION:
        parser.error(
            "--feature-schema-version is incompatible with this code; regenerate features "
            f"with schema {AMRFINDER_FEATURE_SCHEMA_VERSION}"
        )
    if args.amrfinder_version.casefold().startswith("unverified"):
        parser.error("--amrfinder-version must be a verified, pinned version")
    if args.amrfinder_database_version.casefold().startswith("unverified"):
        parser.error("--amrfinder-database-version must be a verified version/hash")

    features = _read(args.features)
    labels = _read(args.labels)
    if "sample_id" not in features.columns:
        parser.error("feature table requires sample_id")
    missing = REQUIRED_LABEL_COLUMNS - set(labels.columns)
    if missing:
        parser.error(f"label table missing columns: {', '.join(sorted(missing))}")
    if features["sample_id"].duplicated().any():
        parser.error("feature table contains duplicate sample_id values")
    duplicate_rows = labels.duplicated(["sample_id", "drug"], keep=False)
    if duplicate_rows.any():
        parser.error("label table contains duplicate sample/drug rows; adjudicate them first")

    feature_columns = [column for column in features.columns if column != "sample_id"]
    if not feature_columns:
        parser.error("feature table has no model feature columns")
    prohibited = _prohibited_feature_columns(feature_columns)
    if prohibited:
        parser.error(
            "feature table contains prohibited metadata/outcome columns: "
            + ", ".join(sorted(prohibited))
        )
    numeric = features[feature_columns].apply(pd.to_numeric, errors="raise")
    feature_table = pd.concat([features[["sample_id"]], numeric], axis=1)

    scoped = labels.loc[
        labels["species"].astype(str).str.casefold() == args.species.casefold()
    ].copy()
    scoped["phenotype"] = scoped["phenotype"].astype(str).str.upper()
    scoped = scoped.loc[scoped["phenotype"].isin(["R", "S"])]
    drugs = args.drugs or sorted(scoped["drug"].dropna().astype(str).unique())
    registry = ModelRegistry(registry_version=args.model_version)
    skipped: dict[str, str] = {}

    for drug in drugs:
        drug_rows = scoped.loc[scoped["drug"].astype(str).str.casefold() == drug.casefold()]
        merged = drug_rows.merge(feature_table, on="sample_id", how="inner", validate="one_to_one")
        counts = merged["phenotype"].value_counts()
        if len(merged) < args.min_samples or any(
            counts.get(label, 0) < args.min_per_class for label in ("R", "S")
        ):
            skipped[drug] = f"insufficient support: n={len(merged)}, classes={counts.to_dict()}"
            continue
        bundle = train_drug_model(
            merged[feature_columns],
            merged["phenotype"],
            merged["group_id"],
            drug,
            species=args.species,
            model_version=args.model_version,
            feature_schema_version=args.feature_schema_version,
            config=TrainingConfig(),
        )
        bundle.training_summary.update(
            {
                "annotation_pipeline": "AMRFinderPlus",
                "annotation_feature_schema_version": AMRFINDER_FEATURE_SCHEMA_VERSION,
                "amrfinder_version": args.amrfinder_version,
                "amrfinder_database_version": args.amrfinder_database_version,
            }
        )
        registry.register(bundle)

    if len(registry) == 0:
        parser.error("no drug passed the configured eligibility and training gates")
    manifest = registry.save(args.output_dir)
    summary = {
        "manifest": str(manifest),
        "species": args.species,
        "trained_drugs": registry.supported_drugs(args.species),
        "skipped": skipped,
        "held_out_test_metrics": None,
        "note": "Training/calibration summaries are not final held-out benchmark metrics.",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
