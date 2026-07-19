# Safety and Responsible-Use Policy

## Non-negotiable boundary

> **RESEARCH PROTOTYPE - NOT FOR CLINICAL USE.** This report provides genomic decision support only and does not select treatment. Every antibiotic-response result must be confirmed with standard laboratory susceptibility testing and reviewed by a trained healthcare or laboratory professional.

Genome Firewall is a hackathon research prototype for exploring selective antimicrobial-resistance (AMR) prediction from an already assembled bacterial genome. It is not a medical device, diagnostic, laboratory-developed test, treatment recommender, or substitute for phenotypic antimicrobial susceptibility testing (AST).

The safest result is often `NO_CALL`. The application must fail closed whenever its input, species, drug, annotation, molecular target, feature schema, model compatibility, or uncertainty checks are not satisfied.

## Intended and prohibited uses

Intended research uses:

- Demonstrating an auditable FASTA-to-research-report pipeline.
- Comparing group-disjoint modeling and calibration methods on authorized data.
- Exploring how explicit abstention changes risk and coverage.
- Supporting retrospective research, education, and hackathon evaluation.

Prohibited uses:

- Selecting, starting, stopping, dosing, or ranking a patient's treatment.
- Reporting a result as a confirmed phenotype, diagnosis, or clinical susceptibility result.
- Bypassing trained laboratory or healthcare review.
- Using raw clinical samples, mixed samples, unassembled reads, human sequence, or an unsupported organism.
- Designing, optimizing, or modifying organisms, resistance determinants, virulence, or transmissibility.
- Public-health action, isolation decisions, surveillance alerts, or automated reporting without independent validation and appropriate authority.
- Training on data without documented authorization, provenance, and permitted use.

## Safety gates

Calls are evaluated in a conservative order. Any earlier failure prevents a later confidence threshold from overriding it.

1. Reject malformed, oversized, non-nucleotide, or obviously low-quality FASTA.
2. Return `NO_CALL` for unsupported or uncertain species.
3. Return `NO_CALL` when annotation fails, versions do not match, or live tooling times out.
4. Return `NO_CALL` for a genome or feature profile outside declared model support.
5. Return `NO_CALL` when a required molecular target is missing, partial, or ambiguous unless a separately validated rule explicitly supports another action.
6. Return `NO_CALL` when curated determinant evidence conflicts with the statistical model.
7. Permit `LIKELY_TO_WORK` only below a calibrated drug-specific threshold, with target confirmation and no conflicting determinant.
8. Permit `LIKELY_TO_FAIL` only above a calibrated drug-specific threshold or under a separately reviewed deterministic rule.
9. Return `NO_CALL` in the uncertainty interval.

Absence of a resistance marker is never treated as proof of susceptibility. A feature weight, model explanation, or correlation is never labeled as a causal biological mechanism.

## Demo mode versus trained mode

Disclosed demo cases are deterministic, clearly labeled fixtures designed to show the three possible product paths. They are not evidence of performance and must never be described as held-out clinical results unless that provenance has actually been established.

The uploaded/trained path may produce calls only when all of the following are present:

- A versioned registry manifest and one serialized `ModelBundle` per supported drug.
- Exact feature-schema, species, drug, and software/database version compatibility.
- Documented group-disjoint fit, calibration, and evaluation partitions.
- Drug-specific thresholds learned from calibration groups, never test labels.
- A reviewed target gate and evidence allowlist.
- Measured evaluation results recorded in the model artifact and model card.

If any requirement is absent, trained mode must stop or return `NO_CALL`; it must not fall back to demo fixtures without a prominent mode change.

## AMRFinderPlus boundary

AMRFinderPlus output is genotypic evidence, not a phenotype guarantee. The adapter must:

- Invoke the executable without a shell and with an argument list.
- Use a private temporary directory and a timeout.
- Record executable and database versions with every annotation.
- Preserve a hash and audit metadata, but not silently persist uploaded sequence.
- Parse only expected columns and reject incompatible formats.
- Treat partial, low-quality, or ambiguous hits separately from curated high-confidence evidence.
- Never interpolate unrecognized gene or mutation names into a clinical claim.

Precomputed annotations are permitted for the hackathon demo only when they are prominently disclosed and their source fixture, software version, and database version are retained.

## OpenAI explanation boundary

The OpenAI Responses API is optional and downstream of the scientific pipeline. It may prioritize allowlisted evidence IDs for presentation; it may not write the displayed medical claims or perform annotation, prediction, thresholding, evidence discovery, clinical interpretation, or treatment selection. Displayed prose is deterministic.

Enforced controls:

- Never send raw FASTA, contigs, genomic subsequences, direct identifiers, or unreviewed free text.
- Send only a redacted payload of validated enums, drug names, and an allowlist of evidence IDs. Never send sample IDs or free-form evidence descriptions.
- Use Structured Outputs with a small schema and reject invalid output.
- Use the model configured by `OPENAI_MODEL` (default `gpt-5.6-terra`); never hard-code a secret or model into source.
- Validate that every mentioned drug, call, probability, target, gene, mutation, and no-call reason exists in the source JSON.
- Disallow dosage, treatment recommendations, prescriptions, comparative drug ranking, and new scientific claims.
- Keep the deterministic template as the primary fallback; API failure must not block the report.
- Never let generated text mutate the canonical prediction object.

The API key belongs in `.env` or the deployment secret store. `.env` and Streamlit secrets are excluded from version control.

## Privacy and data handling

- Use only de-identified, authorized bacterial assemblies.
- Do not include names, medical-record numbers, dates of birth, accession notes containing identifiers, or location fields in uploads.
- Hash input bytes for cache/provenance; do not log the sequence or the full upload filename.
- Default `GENOME_FIREWALL_STORE_UPLOADS=false` and delete temporary files after processing.
- Do not send sequence or sample metadata to third-party services.
- Keep raw, interim, and trained artifacts outside Git unless they are small, reviewed, redistributable fixtures.
- Document dataset licenses, consent/authorization boundaries, and retention periods before any deployment.

## Known failure modes

- Resistance mechanisms absent from the AMRFinderPlus database or selected feature schema.
- Phenotypes driven by gene expression, copy number, permeability, regulation, epistasis, heteroresistance, or laboratory conditions.
- Assembly fragmentation, contamination, mixed cultures, plasmid loss, and sequencing error.
- Species misidentification, novel lineages, near-clone leakage, temporal/site shift, and dataset-specific shortcuts.
- Changing breakpoints, phenotype label error, intermediate/SDD interpretations, and AST-method variation.
- Probability miscalibration when deployment prevalence differs from the calibration set.
- Target detection that is incomplete or biologically oversimplified.

## Pre-demo safety checklist

- [ ] Persistent non-clinical warning is visible before and after analysis.
- [ ] Demo/trained mode is visible on every result and export.
- [ ] Unsupported input returns an explicit error or `NO_CALL`.
- [ ] Three fixtures demonstrate fail, work, and no-call without implying benchmark performance.
- [ ] A model/marker conflict produces `NO_CALL`.
- [ ] A target-ambiguous case cannot produce `LIKELY_TO_WORK`.
- [ ] API disabled, invalid key, timeout, and schema failure all use deterministic explanation text.
- [ ] A prompt asking what to prescribe is refused.
- [ ] No raw FASTA, identifiers, secrets, or invented evidence appear in logs or API requests.
- [ ] Exported reports repeat the mandatory warning and model/data versions.

## Reporting a safety issue

Stop the demo and preserve only non-sensitive diagnostic information if a result bypasses a gate, changes after explanation generation, exposes sequence/identifiers, or presents treatment advice. Record the input category, mode, artifact versions, expected gate, actual behavior, and remediation. Do not share the underlying genome outside the authorized team.
