# Refund judge

A local LLM decides whether a credit card purchase-dispute complaint should be closed **with monetary relief** (refund, credit, reversal) or **without** it. Its decisions are scored against what the company actually recorded in the US CFPB Consumer Complaint Database, then the judge is fixed where the two disagree, attacked with rewrites that should not change the outcome, and turned into an automate-or-route-to-human rule with a measured trade-off between how much is automated and how often it agrees with the human decision.

Everything runs on one laptop with a local model, every model response is cached, and every number below is printed by code from those caches (`make reproduce` fails if a single number in this README differs from what the code regenerates).

## Read this first: limitations

- **Agreement, not correctness.** The label is the company's recorded response to the complaint. It says what the company did, not what was right. A judge that agrees more with companies is not a fairer judge, and nothing here measures fairness or legal correctness.
- **One side of the story.** The judge sees only the consumer's narrative. The company's evidence (transaction records, merchant rebuttal, earlier dispute outcomes) is not in the data, so a large part of the outcome is unpredictable from the text by construction.
- **Self-selected data.** These are complaints that reached the CFPB, from consumers who consented to publishing their narrative. The CFPB states the database is not a statistical sample of consumer experiences.
- **Narratives come from an archived snapshot.** The CFPB removed consumer narratives from the live database in September 2026 (see the API release notes). Narratives here come from a byte-identical mirror of the official bulk export as of 2026-08-11 (SHA-256 checked, see `data/provenance.json`). Product, issue, company and company response of every complaint were re-checked against the live official API; the narrative text itself can no longer be checked against an official source.
- **Small samples.** DEV and TEST were sized to fit a few hours of local compute; read the confidence intervals, not the point estimates.
- **DEV was used for iteration** (error analysis, prompt changes, the fact model, the automation threshold), so DEV numbers after iteration are optimistic. TEST was run once, after everything was frozen.
- **One model, one machine.** Temperature 0 and a fixed seed reduce but do not guarantee run-to-run determinism; the cache, not the model, is what makes this report reproducible.
- **The gaming test is home-made.** Attacks and defence were written by the same author; a defence tuned to known attacks overstates robustness. One attack was written specifically to get past the defence's input filter.
- **Error categories were assigned by a single reader**, without a second annotator, so there is no inter-annotator agreement for them.

## Data

US CFPB Consumer Complaint Database (public domain, CC0). One decision type, chosen because it is closest to a payments dispute: product **Credit card**, issue **Problem with a purchase shown on your statement**, received by the CFPB in calendar years 2023 to 2025, narrative present. Label: `1` = "Closed with monetary relief", `0` = "Closed with explanation" or "Closed with non-monetary relief"; every other status is dropped. Near-duplicate narratives (re-filed or templated complaints) are collapsed to their earliest record so the same text cannot appear in two splits.

<!-- BEGIN data -->
| Step | Complaints |
|---|---|
| Rows in the archived official bulk export | 17,021,062 |
| Credit card / purchase-on-statement complaints received in the window | 39,044 |
| ... with a consumer narrative | 22,044 |
| ... dropped: status not mapped (in progress / untimely) | 10 |
| ... dropped: metadata differs from the live API | 4 |
| ... dropped: not found in the live API | 0 |
| ... removed as near-duplicates (re-filed or templated text) | 512 |
| Sampling frame | 21,518 |

| Recorded company response | Complaints with narrative | Label |
|---|---|---|
| Closed with explanation | 13,995 | 0 |
| Closed with monetary relief | 6,576 | 1 |
| Closed with non-monetary relief | 1,463 | 0 |
| Untimely response | 10 | dropped |
<!-- END data -->

Base rates after the stratified split (stratified by label, seeded; IDs frozen in `data/splits.json`):

<!-- BEGIN base_rates -->
| Split | Complaints | Share closed with monetary relief |
|---|---|---|
| TRAIN | 8,000 | 30.1% |
| DEV | 200 | 30.0% |
| TEST | 250 | 30.0% |
| DEBUG | 12 | 33.3% |
<!-- END base_rates -->

## Pre-registered analysis plan

Written and committed before the first DEV or TEST model call; any later change is visible in the git history. Numbers that parameterise the plan are in the settings table, which is generated from `src/judge/config.py`.

1. **Unit and outcome.** One complaint; the binary label above.
2. **Primary metric: Cohen's kappa** between the decision and the recorded outcome. Accuracy rewards always answering "no relief", so it is secondary, together with precision and recall for each class.
3. **Uncertainty.** Percentile bootstrap over complaints, 95% intervals. Version-to-version differences use a paired bootstrap on the same complaints. Flip rates (small counts, often zero) use Wilson score intervals.
4. **Splits.** TRAIN is used only by the classic baseline. DEV is used for everything that involves choices: error analysis, prompt changes, the fact model, thresholds. TEST is run exactly once, with the versions frozen in `results/test_run.json` before the run starts; the script refuses to run it twice.
5. **Baselines.** Majority class; a company rule (predict relief when the company's TRAIN relief rate is above one half); TF-IDF + logistic regression on the same text the judge sees, regularisation and class weighting chosen by cross-validation on TRAIN only. Cost: CPU time per item for the classic model against wall time per call for the judge.
6. **Judge v1 then v2.** v1 is a plain rubric prompt. Every DEV disagreement of v1 is read and given one primary category. v2 addresses the largest categories (structured fact extraction before the decision is planned). v1 against v2 on DEV with paired intervals.
7. **Gaming.** A seeded, label-stratified subset of DEV; each narrative gets appended rewrites that add no verifiable fact (claimed promise of a refund, emotional pressure, legal threat, an instruction addressed to the AI reviewer, a fake internal case note). Metrics: flip rate (decision differs from the unmodified narrative under the same judge), flips toward relief, mean change in the probability of relief. Then a defence: a programmatic filter that removes instruction-like text and routes the complaint to a human, and a decision computed from the extracted facts by a small logistic model instead of by the LLM. Re-measured on the same rewrites.
8. **Automation.** Confidence is the token probability of the judge's decision (primary) or its stated probability (secondary). Operating point: the lowest confidence threshold whose DEV agreement on automated cases reaches the target with a minimum number of automated cases; applied unchanged on TEST. The same rule is applied to the logistic-regression baseline.
9. **Estimates.** Realistic: TEST at the threshold chosen on DEV. Ceiling: the best threshold chosen on TEST itself (optimistic by construction), and the agreement of recorded outcomes among near-identical complaints against the same company (a label-noise proxy; a true ceiling would need several independent human decisions per complaint, which this data does not have).
10. **Invalid outputs** (not valid JSON or no boolean decision) are counted, scored as "no relief", and never automated.
11. Anything decided after looking at DEV results is labelled as such below.

<!-- BEGIN settings -->
| Setting | Value |
|---|---|
| Model | `qwen3.8:27b` via Ollama (local) |
| Ollama version / model digest | 0.40.15 / `47951e4a666180bc2aedfc93c169472f2de783696fe196efbf063f5730ecf163` |
| Decoding | temperature 0, seed 42, thinking disabled, JSON schema enforced, num_ctx 8192 |
| Narrative cap | 6000 characters (same text for every model) |
| Split sizes (DEBUG / DEV / TEST / TRAIN) | 12 / 200 / 250 / 8000 |
| Split seed | 20261109 |
| Bootstrap | 10000 resamples, seed 0, percentile 95% CI |
| Automation target | agreement >= 0.90 on automated DEV cases, at least 20 automated |
| Gaming subset | 30 DEV complaints x 5 rewrites, seed 7 |
| Near-duplicate rule | word-trigram TF-IDF cosine >= 0.5 (collapsed); ceiling proxy uses same-company pairs >= 0.9 |
<!-- END settings -->

## Results

### Baselines and judge versions on DEV

<!-- BEGIN dev_results -->
_Not run yet._
<!-- END dev_results -->

### Error analysis of judge v1 on DEV

<!-- BEGIN error_analysis -->
_Not run yet._
<!-- END error_analysis -->

### Gaming test

<!-- BEGIN gaming -->
_Not run yet._
<!-- END gaming -->

### Automation against agreement

<!-- BEGIN automation -->
_Not run yet._
<!-- END automation -->

### Held-out TEST (run once)

<!-- BEGIN test_results -->
_Not run yet._
<!-- END test_results -->

### Realistic estimate and ceiling

<!-- BEGIN estimates -->
_Not run yet._
<!-- END estimates -->

### Cost and runtime

<!-- BEGIN cost -->
_Not run yet._
<!-- END cost -->

## Reproduce

```bash
make reproduce
```

Creates `.venv`, installs pinned dependencies, runs the tests, recomputes every metric from `data/` and `results/cache/responses.jsonl` without calling any model, and fails if `README.md`, `results/metrics.json` or `results/report.md` differ from the regenerated output.

Other targets: `make run` repeats the model calls that are missing from the cache (needs Ollama with the model in the settings table); `make data` rebuilds `data/` from the archived bulk export (a large download) and the live CFPB API.

## Layout

- `src/judge/`: `data.py` (download, checks, splits), `prompts.py` (judge versions), `llm.py` (Ollama client and response cache), `run.py` (resumable runs, the single TEST run), `baselines.py`, `gaming.py` (rewrites and defence), `metrics.py`, `analysis.py`, `report.py`.
- `data/`: frozen records and split IDs, live-API metadata used for the label check, provenance.
- `results/`: cached raw responses, run log, error-analysis annotations, `metrics.json`, `report.md`, figure.
- `tests/`: metric and pipeline tests (`pytest`).
