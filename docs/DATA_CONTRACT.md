# Genome Firewall data contract

## Status

This repository does not bundle organizer data, patient data, a trained production model, or measured benchmark results. The synthetic demo fixtures exercise the product contract only. This document defines what must be supplied before truthful trained evaluation is possible.

## 1. Scope freeze

Before inspecting held-out outcomes, freeze:

- one organism scope and accepted taxonomy method;
- the supported drug list;
- phenotype interpretation and breakpoint version;
- AMRFinderPlus executable and database versions;
- feature-schema version;
- target-search workflow and reviewed target catalog;
- group definition and split seed;
- primary metrics and abstention constraints.

Changing any frozen item creates a new experiment and artifact version.

## 2. Label table

Accepted format: CSV or Parquet, one row per unique sample/drug pair.

| Field | Required | Contract |
|---|---:|---|
| `sample_id` | yes | Stable, de-identified join key; unique with `drug` |
| `species` | yes | Canonical organism name established upstream |
| `drug` | yes | Canonical antibiotic name |
| `phenotype` | yes | Laboratory-measured `R` or `S`; missing is never changed to `S` |
| `group_id` | yes | Genetic/homology group kept wholly within one partition |
| `assembly_path` | for annotation | Authorized assembled nucleotide FASTA |
| `source_dataset` | recommended | Dataset snapshot or accession |
| `ast_method` | recommended | Laboratory AST method |
| `breakpoint_standard` | recommended | EUCAST/CLSI/organizer-defined source |
| `breakpoint_version` | recommended | Pinned interpretation version |
| `site` / `year` | audit only | Drift and stratification fields, not default features |

Intermediate, susceptible-dose-dependent, unknown, contaminated, mixed, or conflicting labels must follow a predeclared exclusion/adjudication rule. Do not silently coerce them.

## 3. Feature table

Accepted format: CSV, or Parquet with the optional `genome-firewall[parquet]` dependency.

- Exactly one row per `sample_id`.
- One `sample_id` column plus numeric feature columns.
- No phenotype, breakpoint, filename, split, site, outcome-derived, or patient field may be a model feature.
- Missing feature values must follow the frozen preprocessing policy.
- Feature names and order are stored in every `ModelBundle`.
- The feature-generation job must record the exact AMRFinderPlus version, database identifier/hash, and `AMRFINDER_FEATURE_SCHEMA_VERSION`.

The training CLI rejects duplicate IDs, nonnumeric feature values, duplicate sample/drug labels, insufficient class support, and an incompatible feature-schema version.

## 4. Target observation input

Target presence is a separate safety gate, not a model feature and not something inferred from marker absence. The input is a JSON mapping of drug to target-level observations:

```json
{
  "Ciprofloxacin": {
    "gyrA": "PRESENT",
    "parC": "AMBIGUOUS"
  }
}
```

Allowed observation values are `PRESENT`, `ABSENT`, `AMBIGUOUS`, and `NOT_ASSESSED`. They must come from a separately validated target-search workflow with its own completeness and partial-hit policy. Genome Firewall recomputes the final target status from the pinned catalog; callers cannot supply a final verdict directly.

The bundled catalog is provisional. Its observations remain ambiguous for trained likely-to-work gating until both the catalog and per-drug entries are explicitly validated by a qualified reviewer.

## 5. Partition contract

Related genomes must never cross partitions.

1. Derive or receive `group_id` without using phenotype.
2. Reserve an untouched group-disjoint evaluation set.
3. Split the remaining groups into model-fit and calibration partitions.
4. Perform model selection only within the fit partition using group-aware folds.
5. Fit the calibrator and abstention thresholds on calibration groups only.
6. Open the held-out evaluation labels once for the frozen candidate.

At minimum, persist the sample IDs, group IDs, partition assignment, split seed, group-overlap audit, and a cryptographic hash of the split table. If temporal or site generalization matters, add an explicit external or forward-time evaluation rather than claiming it from a random split.

## 6. Evaluation artifact

No UI metric may be populated from training examples or demo fixtures. A release evaluation artifact should contain, per drug:

- support counts by class and group;
- AUROC and PR-AUC when defined;
- Brier score and reliability bins;
- resistant and susceptible recall;
- false-susceptible rate;
- call coverage, called accuracy, and risk-versus-coverage points;
- work/fail/no-call counts and no-call reasons;
- uncertainty intervals where the sample size supports them;
- dataset, split, model, feature, AMRFinderPlus, database, and policy versions.

Undefined metrics remain `null`; they are not converted to zero. Results from drugs that fail minimum support or threshold constraints remain unsupported/no-call.

## 7. Artifact compatibility

Every serialized model pins:

- organism and drug;
- model and artifact-format versions;
- ordered feature names and feature-schema version;
- estimator, calibrator, and drug-specific thresholds;
- training/calibration support summaries;
- AMRFinderPlus version;
- AMRFinderPlus database version/hash.

The registry verifies artifact hashes when loading. Live inference then requires exact feature-schema, tool, and database compatibility. Missing or unverified provenance returns `NO_CALL / MODEL_ERROR` before a model score is displayed.

## 8. Handoff checklist

- [ ] Data use and redistribution are authorized.
- [ ] `sample_id` is de-identified and contains no patient information.
- [ ] Species and AST methods are documented.
- [ ] Conflicting duplicates are adjudicated.
- [ ] Group definitions are phenotype-independent.
- [ ] Held-out groups have not been used for feature, model, or threshold selection.
- [ ] AMRFinderPlus and database identifiers are pinned.
- [ ] Target-search observations and catalog entries are reviewed.
- [ ] Artifact hashes and split hash are saved.
- [ ] Model card and UI contain only measured evaluation results.
- [ ] Research-only warning and AST confirmation requirement remain visible.
