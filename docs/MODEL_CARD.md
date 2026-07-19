# Genome Firewall Model Card

- **Card status:** prototype template
- **Model status:** not clinically validated
- **Version:** 0.1.0
- **Last updated:** 2026-07-19

> **RESEARCH PROTOTYPE - NOT FOR CLINICAL USE.** This report provides genomic decision support only and does not select treatment. Every antibiotic-response result must be confirmed with standard laboratory susceptibility testing and reviewed by a trained healthcare or laboratory professional.

## Summary

Genome Firewall is a selective AMR research prototype. For each explicitly supported species/drug pair, it combines AMRFinderPlus-derived numeric features, a separately calibrated statistical classifier, a molecular-target gate, known-evidence conflict checks, and drug-specific abstention thresholds. It returns one of:

- `LIKELY_TO_FAIL`
- `LIKELY_TO_WORK`
- `NO_CALL`

The canonical score is `p_resistant`, the model's calibrated estimate for the resistant class under the training/evaluation definition. The displayed label is a research signal, not a clinical susceptibility category. Every result requires standard laboratory AST confirmation and trained human review.

No empirical benchmark metrics are claimed in this repository unless they are produced from the organizer-provided dataset under the documented group-disjoint protocol and written into a versioned artifact. Demo fixtures are not a benchmark.

## Model details

| Item | Prototype design |
|---|---|
| Unit of prediction | One assembled bacterial genome and one supported drug |
| Statistical model | One sparse scikit-learn classifier per drug |
| Calibration | Separate calibration groups; sigmoid by default |
| Inputs | Versioned numeric features derived from authorized annotation data |
| Positive class | Resistant (`R`) |
| Decision policy | Separate work/fail thresholds with an abstention interval |
| Biological safety layer | Target presence/ambiguity and curated-evidence conflict gates |
| Artifact format | `manifest.json` plus one joblib `ModelBundle` per drug |
| Interface | Streamlit research report |
| Explanation | Deterministic template; optional OpenAI evidence-ordering plan downstream |

The baseline intentionally favors an interpretable, calibratable sparse model over a large end-to-end sequence model. Model coefficients are statistical associations and must not be presented as mechanistic or causal proof.

## Intended use

The prototype is intended for authorized retrospective research, educational demonstrations, and hackathon evaluation of:

- Leakage-aware model development on genomic groups.
- Probability calibration and selective prediction.
- Explicit no-call policies.
- Transparent separation of known biological evidence from statistical association.

It is not intended to diagnose infection, infer a phenotype without lab confirmation, recommend or rank antibiotics, specify dosing, replace AST, trigger public-health action, or guide an individual patient's care. See [SAFETY.md](SAFETY.md).

## Supported scope

The provisional demo catalog lists *Escherichia coli* and three illustrative drugs so the interface and contracts can be exercised. Those entries are not automatically model-supported. A species/drug pair becomes supported in trained mode only when its compatible `ModelBundle`, reviewed target rule, calibration thresholds, data provenance, and measured evaluation are present.

The application must show the support status that is actually loaded:

| Mode | Scientific meaning |
|---|---|
| `demo` | Deterministic fixtures showing expected behavior; no performance claim |
| `trained` | Locally loaded, version-checked artifacts trained on documented data |

Trained mode must fail closed on missing or incompatible artifacts. It must never silently substitute a fixture result.

## Training-data contract

One phenotype row represents one measured sample/drug observation. The recommended ingestion table is UTF-8 CSV or Parquet with the following fields:

| Field | Type | Required | Use |
|---|---:|---:|---|
| `sample_id` | string | yes | Stable de-identified join key |
| `species` | string | yes | Support filtering; never silently inferred from filename |
| `drug` | string | yes | Canonical drug identifier |
| `phenotype` | enum `R`/`S` | yes | Measured AST label |
| `group_id` | string | yes | Genetic/homology group kept within one split |
| `assembly_path` | string | yes for annotation | Path to an authorized assembled nucleotide FASTA |
| `source_dataset` | string | recommended | Provenance/audit only |
| `ast_method` | string | recommended | Provenance/stratified audit only |
| `breakpoint_standard` | string | recommended | Label provenance |
| `breakpoint_version` | string | recommended | Label provenance |
| `collection_year` | integer | optional | Shift audit only, not a predictive shortcut |
| `site_id` | string | optional | Group/shift audit only, not a predictive shortcut |

Rules:

- Only organizer-pinned or otherwise authorized, laboratory-measured labels are eligible.
- Intermediate, susceptible-dose-dependent, unknown, and conflicting labels are excluded unless a written protocol defines a different mapping before evaluation.
- Exact and near-duplicate assemblies are grouped before splitting.
- Conflicting phenotype rows within an effectively duplicate cluster are removed for that drug or adjudicated under a documented rule.
- Site, year, AST method, lineage label, filename, and phenotype-derived fields cannot be predictive features.
- Missing values are never silently converted to susceptible.

The numeric model matrix uses a `sample_id` index and stable feature names. Feature order is stored in each bundle and strictly reindexed at inference. Examples include AMR gene-family indicators, curated mutation indicators, and explicitly encoded hit-quality fields. Assembly QC, target status, group IDs, and provenance are policy/audit inputs unless a reviewed design says otherwise.

## Annotation and feature provenance

Every feature record should retain:

```json
{
  "sample_id": "deidentified-sample-id",
  "feature_schema_version": "1.0",
  "amrfinder_version": "recorded-at-runtime",
  "amrfinder_database_version": "recorded-at-runtime",
  "genes": [],
  "mutations": [],
  "hit_quality": {},
  "assembly_qc": {},
  "target_status": {}
}
```

Inference must reject incompatible schema, AMRFinderPlus/database, species, and model versions. Precomputed annotations used in demo mode must be labeled as fixtures and retain the versions that produced them.

## Split, fitting, and calibration protocol

1. Reserve the organizer test set untouched when one is provided.
2. Build exact/near-duplicate and genetic/homology groups using only allowed data.
3. Partition groups, not rows, into model-fit, tuning, and calibration data.
4. Fit all preprocessing and feature filtering on fit groups only.
5. Tune regularization with grouped cross-validation.
6. Fit probability calibration on untouched calibration groups.
7. Select separate work and fail thresholds on calibration groups to maximize useful coverage under a declared error ceiling.
8. Evaluate exactly once on group-disjoint held-out data.
9. Record counts, prevalence, groups, seed, hashes, versions, and failures alongside metrics.

Thresholds are part of the trained artifact. The provisional YAML catalog deliberately contains `null` thresholds so it cannot masquerade as a trained decision policy.

## Evaluation requirements

Metrics must be reported separately for every species/drug pair:

- Samples, genetic groups, resistant prevalence, and number in each class.
- Balanced accuracy, resistant recall, susceptible recall, and resistant-class F1.
- AUROC, PR-AUC, and Brier score when both classes are present.
- Calibration slope/intercept and a reliability diagram.
- False-susceptible rate among `LIKELY_TO_WORK` calls.
- Coverage, no-call rate, performance among called samples, and a risk-versus-coverage curve.
- Group-macro performance and performance on novel/OOD groups.
- Group-bootstrap confidence intervals when sample support permits.

### Current measured results

**Not yet evaluated on the organizer-provided dataset.** No numerical performance claim should appear in the UI, pitch, README, or export until the evaluation protocol above has been run and its artifact is available. When results exist, add the dataset/version, split hash, support counts, point estimates, and uncertainty here; do not copy metrics from demo fixtures.

## Decision logic

The decision policy is ordered and deterministic:

1. Invalid/low-quality input → reject or `NO_CALL`.
2. Unsupported species/drug or strong OOD signal → `NO_CALL`.
3. Ambiguous/partial target → `NO_CALL`.
4. Curated determinant conflicts with model → `NO_CALL`.
5. `p_resistant` at or above the validated fail threshold → `LIKELY_TO_FAIL`.
6. `p_resistant` at or below the validated work threshold, target confirmed, and no conflict → `LIKELY_TO_WORK`.
7. Otherwise → `NO_CALL`.

Probabilities are not displayed as certainty for `NO_CALL`. Likely-to-work requires more than the absence of a known marker.

## Limitations

Expected limitations include database coverage, changing resistance mechanisms, assembly quality, contamination/mixed cultures, species error, gene expression and copy-number effects, regulatory and permeability mechanisms, epistasis, heteroresistance, laboratory label noise, breakpoint changes, unrepresented lineages, geographic/temporal shift, and probability drift under a different prevalence.

The provisional molecular-target mappings require biological review. An AMRFinderPlus hit is genotypic evidence and is not by itself a phenotype. Statistical features may capture lineage or dataset correlations even after grouped splitting.

## Human oversight and output interpretation

Every report must display:

- Application mode (`demo` or `trained`).
- Species/drug and support status.
- Call and explicit no-call reason.
- `p_resistant`, thresholds, target status, and evidence category when valid.
- Model, feature schema, AMRFinderPlus, and database versions.
- Training/calibration support counts from the artifact.
- The persistent research-only warning.

A trained laboratory or healthcare professional must interpret the report together with standard AST and relevant clinical context. The application does not ingest clinical context because it does not make treatment decisions.

## OpenAI-generated explanations

The optional OpenAI Responses API layer receives only redacted enums, drug names, and evidence IDs, never raw sequence, sample IDs, or free-form evidence text. It returns a constrained ordering plan rather than medical prose. It cannot create or change calls, probabilities, evidence, thresholds, or target status; unrecognized names or IDs fall back to deterministic ordering and copy. The default configured model is `gpt-5.6-terra`, overridable with `OPENAI_MODEL`.

## Reproducibility checklist

- [ ] Dataset source, license/authorization, snapshot date, and hashes recorded.
- [ ] Split assignment and group/duplicate method recorded.
- [ ] Random seed and software versions recorded.
- [ ] AMRFinderPlus executable and database versions recorded.
- [ ] Feature schema and target/evidence catalog versions recorded.
- [ ] Model and calibrator hyperparameters recorded.
- [ ] Threshold selection objective and calibration partition recorded.
- [ ] Metrics and uncertainty stored only after genuine evaluation.
- [ ] Known limitations and failed eligibility gates disclosed.
- [ ] Model artifact and model card reviewed together.
