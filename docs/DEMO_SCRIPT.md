# Genome Firewall — three-minute judge demo

## The one-line pitch

Genome Firewall is an abstention-first genomic AMR research prototype: it provides an early, auditable signal when the evidence is supported and returns an explicit no-call when it is not.

The demonstration fixtures are synthetic product examples. They are not patient data, a clinical validation, or a benchmark.

## Before the judges arrive

1. Run `python -m pytest -q` and keep the passing result available.
2. Start `python -m streamlit run app.py` and open `http://localhost:8501`.
3. Leave OpenAI disabled for the most resilient demo; the deterministic explanation works offline.
4. Start on **Analyze** with no result loaded.
5. Do not promise numeric accuracy. The repository intentionally has no organizer-data benchmark artifact.

## 0:00–0:20 — hook

Say:

> “Most prediction demos are rewarded for always answering. In antimicrobial resistance, an unjustified answer is the dangerous failure mode. Genome Firewall treats uncertainty as a first-class product outcome.”

Point to the persistent research-only warning and the three outcomes: likely to fail, likely to work, and no-call.

## 0:20–0:55 — evidence-first fail path

1. Choose **Determinant detected**.
2. Click **Run selected demo**.
3. Open **Evidence & confidence**.
4. Select ciprofloxacin.

Say:

> “The report separates a disclosed determinant from statistical model associations. The validated policy—not the interface and not the language model—owns the call.”

Call out the target state, determinant evidence, calibrated probability context, and traceable evidence identifiers. Remind the judges that the displayed values belong to a synthetic UI fixture.

## 0:55–1:25 — target-gated work path

1. Return to the source selector.
2. Choose **Target-confirmed signal**.
3. Run it and inspect a likely-to-work card.

Say:

> “A low resistance score is not enough. A likely-to-work result also requires a reviewed molecular-target gate. Missing, ambiguous, or provisional target evidence blocks this path.”

Do not call this a treatment recommendation. It is an early research signal that requires standard laboratory confirmation.

## 1:25–1:55 — the trust moment

1. Choose **Conflicting evidence**.
2. Run it.
3. Open the decision table and evidence panel.

Say:

> “Here the score may look tempting, but novelty, target ambiguity, or conflicting evidence wins over confidence. The system abstains and states exactly why.”

This is the differentiating moment: show that no-call has no displayed confidence and is not quietly converted into susceptible.

## 1:55–2:20 — grounded explanation boundary

1. Expand **Grounded explanation**.
2. Generate the deterministic explanation.
3. Show the evidence priority, uncertainty, limitations, and source IDs.

Say:

> “OpenAI is optional and off by default. If enabled, it receives no sequence, sample identifier, free text, or model internals. It can only order allowlisted drug and evidence IDs; all prose is deterministic. It cannot change a result or recommend treatment.”

## 2:20–2:40 — honest validation

Open **Validation**.

Say:

> “We did not receive or bundle a measured organizer evaluation artifact, so this screen refuses to invent performance. Training enforces group-disjoint fitting and calibration; a separately audited held-out artifact must be hash-bound to the model manifest before trained calls are eligible.”

An empty, explicit validation state is stronger than an attractive fake number.

## 2:40–3:00 — close

Open **Model card** and briefly show versions, limitations, and provenance.

Close with:

> “Genome Firewall is not trying to replace AST. It is a fast, transparent research layer that knows when its evidence is good enough to speak—and when it must stay silent.”

## Architecture answer in 20 seconds

An assembled FASTA passes strict structural and assembly QC, then local AMRFinderPlus produces a pinned feature schema. One calibrated sparse model runs per drug. A deterministic policy applies support, built-in feature-profile novelty, externally supplied OOD when available, target, evidence-quality, conflict, and calibration gates. The result is a strictly validated report consumed by the UI and optional grounded explanation layer.

## Judge Q&A

### “Is it clinically validated?”

No. It is a research prototype. Clinical validity and organizer-dataset performance are not claimed; every result requires phenotypic AST and professional review.

### “Why does the repository include likely-to-work examples?”

They demonstrate the intended product contract using synthetic fixtures. The bundled target catalog is deliberately provisional and cannot enable a trained likely-to-work call until domain review and target-search validation are completed.

### “Where is the AI?”

The scientific classifier is a calibrated, interpretable machine-learning model. The optional OpenAI layer is tightly constrained to evidence ordering and is removable. Scientific calls never depend on it.

### “Why a sparse logistic model instead of a deep model?”

For a small hackathon dataset, group leakage and calibration are usually bigger risks than model capacity. The baseline is auditable, regularized, fast, and easier to validate. The registry can later host a stronger model only if it wins on untouched grouped evaluation.

### “How do you prevent leakage?”

`group_id` keeps related genomes apart during fitting and calibration. The data contract requires the external held-out workflow to prove that its group hashes do not overlap and to bind that audit to the release artifact. The training CLI blocks common metadata/outcome columns from model features.

### “What happens when AMRFinderPlus or a model is missing?”

The upload returns a live, non-demo no-call. It never substitutes one of the successful fixtures.

### “How do you protect genomic data?”

Annotation runs locally. The optional explanation payload excludes FASTA, contigs, sample IDs, descriptions, reasons, and raw scores. Deployments still need an explicit access, retention, and incident policy before real-world use.

## Failure-safe backup

If live annotation or the network is unavailable, continue with the bundled disclosed fixtures. They require neither AMRFinderPlus nor OpenAI. If the UI cannot start, show the model card, `docs/SAFETY.md`, the validated report schema, and the passing tests; do not fabricate a live result.
