# Refund decisions on disputed card charges: an LLM judge scored against the recorded outcome

<!-- BEGIN first_screen -->
**The data.** Complaints to the US regulator (the CFPB) about disputed credit-card charges, mostly filed after the card issuer had handled the dispute itself: 73% of the 8,000 training complaints are filed as 'company isn't resolving a dispute', and judge v2 reads a prior refusal in 175 of 200 development complaints. The label is how the company closed the complaint with the regulator, with money back (goodwill credits included) or without, so every figure here is agreement with what the company did, not correctness.

- **Always answering "no money back" matches the record in 70.0% of the 250 held-out complaints; the best system, the guarded pipeline, matches it in 73.6%, a difference of +3.6 points [+0.4, +7.2].**
- **A model applying the written refund rules to the consumer's account (judge v1) would pay 176 of the 250; the companies paid 75.** It errs towards paying (117 payments the company did not make, 16 refusals where it paid); judge v2 and the guarded pipeline err towards refusing (18 / 54; 5 / 61).
- **The record follows which issuer answered** *(exploratory)*. A lookup of each issuer's past payout rate, which never reads the complaint, agrees with the record about as well as the best text model (kappa 0.354 against 0.320, difference -0.034 [-0.176, +0.109]); hiding issuer names costs the text model ranking quality (AUC -0.051 [-0.096, -0.007]); and among complaints about the same issuer no system that reads the text ranks paid above unpaid clearly better than a coin toss.
- **Decision: automate nothing.** No system is shown to reach the planned 90% agreement on the cases it would decide alone (TF-IDF: 29/250 at 93.1%, Wilson [78.0%, 98.1%]), and in a test on development complaints a fake internal note pasted into the text turned 27 of 28 of judge v2's refusals into refunds.
<!-- END first_screen -->

**What was done.** A local LLM decided refund or no refund on these complaints from the consumer's text, with two prompts and a guarded pipeline; six simpler baselines were scored the same way; everything was tuned on a development split (DEV), then the method and every claim were frozen and a held-out split was scored once. **Authorship:** the code, the prompts, the gaming rewrites, the defence and the first reading of the disagreements were written or drafted by an AI coding assistant working on this repository; the reviews named in the deviations were separate AI-agent sessions, not people; the blind human labels are the author's own and are in progress. There is no independent second author. Where a code comment says "by hand" (for example the issuer alias list in `src/judge/baselines.py`), it means a fixed list written into the code by that assistant.

## Held-out result and the frozen claims

<!-- BEGIN headline -->
**Held-out run**: 250 complaints, scored once from commit `822811d`. That commit froze the method; it precedes the run in this repository's history, and a GitHub gist created before the run holds its hashes (see the freeze section below). **Claims frozen before the run**, each with the outcome its rule in `results/frozen.json` dictates (the outcome words were chosen after the run):

- *v2 replaces v1* (only if the paired kappa difference lies entirely above zero): **not met**, +0.120 [-0.013, +0.249].
- *Each judge against TF-IDF, the issuer relief rate and the company rule* (6 paired kappa differences): **decided:** judge v1 is below the TF-IDF model (-0.234 [-0.370, -0.096]); judge v1 is below the issuer relief rate (-0.268 [-0.407, -0.124]); both also exclude zero at the 99.17% level that corrects for 6 comparisons *(exploratory)*. **Undetermined**, the interval includes zero: judge v2 against the TF-IDF model (-0.114 [-0.277, +0.048]); judge v2 against the issuer relief rate (-0.148 [-0.328, +0.039]); judge v1 against the company rule (-0.083 [-0.221, +0.049]); judge v2 against the company rule (+0.037 [-0.141, +0.211]).
- *TF-IDF orders complaints better than the judge within one issuer*: **undetermined** against judge v1 (AUC difference -0.014 [-0.144, +0.107]); **undetermined** against judge v2 (AUC difference -0.031 [-0.135, +0.085]).
- *The TF-IDF threshold frozen on DEV meets the 90% agreement target on held-out cases*: **undetermined**, not shown to meet the target (29/250 automated, 93.1%, Wilson [78.0%, 98.1%]; always answering no relief agrees on 75.9% of the same cases).
- Two frozen items cannot fail and are listed for completeness: the LLM configurations automate nothing, because none had a DEV threshold; and the noise pass was done (judge v1 repeated 248/250 decisions, judge v2 repeated 250/250 decisions).

| System | Reads | Matches the record, held-out (n=250) | Would pay, of 250 (where the company did not / refusals where it paid); the companies paid 75 | Held-out kappa [95% CI] | DEV kappa [95% CI], n=200 | Automated on held-out at the frozen threshold | Time per 1,000 decisions (DEV timings) |
|---|---|---|---|---|---|---|---|
| Always no relief | nothing | 70.0% | 0 (0 / 75) | 0.000 (constant answer) | 0.000 (constant answer) | not assessed | a table lookup |
| Keyword rule (added later) | narrative | 68.8% | 33 (18 / 60) | 0.116 [-0.002, 0.238] | 0.076 [-0.053, 0.213] | not assessed | 0.1 s (CPU) |
| Company rule (planned) | issuer only | 73.2% | 14 (3 / 64) | 0.169 [0.069, 0.278] | 0.157 [0.033, 0.279] | not assessed | a table lookup |
| Issuer relief rate (added later) | issuer only | 71.2% | 91 (44 / 28) | 0.354 [0.230, 0.473] | 0.318 [0.190, 0.442] | not assessed | a table lookup |
| TF-IDF, issuer names masked (added later) | text, issuer names masked | 68.4% | 64 (34 / 45) | 0.215 [0.085, 0.344] | 0.295 [0.153, 0.433] | not assessed | 1.4 s (CPU) |
| TF-IDF + logistic regression | text | 72.4% | 66 (30 / 39) | 0.320 [0.192, 0.445] | 0.383 [0.245, 0.513] | 29/250, agreement 93.1% (Wilson [78.0%, 98.1%]) | 1.4 s (CPU) |
| LLM judge v1 (plain rubric) | text | 46.8% | 176 (117 / 16) | 0.085 [0.000, 0.172] | 0.120 [0.024, 0.221] | none (no DEV threshold) | about 3 h (one laptop GPU) |
| LLM judge v2 (predicts the record) | text | 71.2% | 39 (18 / 54) | 0.205 [0.077, 0.332] | 0.070 [-0.051, 0.197] | none (no DEV threshold) | about 4 h (one laptop GPU) |
| Guarded pipeline (v2 facts -> fact model) | text | 73.6% | 19 (5 / 61) | 0.201 [0.092, 0.316] | 0.009 [-0.070, 0.103] | none (no DEV threshold) | about 4 h (one laptop GPU) |

Kappa is agreement beyond what the base rate gives (0 for a constant answer). Relief base rate 30.0% held-out, 30.0% DEV. Text: the product and issue fields and the consumer's narrative; issuer only: the issuer's relief rate on the 8,000 TRAIN complaints. DEV was used for every choice after the first prompt, so the DEV column is optimistic.

Blind human labels are being collected; nothing is reported until the set is complete and locked.
<!-- END headline -->

**Limits.**

- **Agreement, not correctness.** The label is what the company told the regulator. A policy that agrees more with companies is not a fairer one, and nothing here measures fairness or legal correctness.
- **One side of the story.** Every text system sees only the consumer's narrative and the fields the consumer selected; the bank's records, the merchant's response and card-scheme rules are not in the data and drive the real decision.
- **Small and self-selected; read the intervals, not the point estimates.** The CFPB itself says the database is not a statistical sample of consumer experiences. The split is random, not by date. DEV was used for every choice after the first prompt, so later DEV numbers are optimistic.
- **Not deterministic on a busy machine.** On DEV the same request sometimes gave a different answer at temperature zero with a fixed seed; on the held-out noise pass, run on a quieter machine, the decision rarely changed (see Repeatability). The report is reproducible from the cached answers, not by asking again.
- **Home-made.** A defence tuned to known attacks overstates robustness, and the attacks were written by the same assistant that wrote the defence. The narratives come from a third-party copy of the official bulk export that cannot be checked against an official source.

## What this means for automating a decision like this

<!-- BEGIN practice -->
Only the claims above were frozen before the run. Everything in this section is exploratory: the analyses and their wording were chosen after the held-out results were seen, and the bullets that restate a frozen claim say so.

- **Which way the errors go.** Payments the company did not make / refusals where it paid, held-out: judge v1 117 / 16; judge v2 18 / 54; guarded pipeline 5 / 61; TF-IDF model 30 / 39; issuer lookup 44 / 28. No money figure is given: the records hold company, company_response, complaint_id, date_received, issue, label, narrative, product, sub_issue, sub_product and no amount, and the dollar figures in the narratives are the consumer's own, unverified and not the refunded amount.
- **The issuer.** The issuer lookup's held-out kappa 0.354 [0.230, 0.473] is the highest point estimate, but TF-IDF cannot be told apart from it (kappa difference -0.034 [-0.176, +0.109], AUC difference +0.017 [-0.055, +0.097]), and its threshold (the TRAIN base rate) was chosen after DEV; the planned rule, which thresholds the same rates at one half, gets 0.169. Masking issuer names lowers TF-IDF's AUC (-0.051 [-0.096, -0.007]). Within one issuer the text systems reach AUC 0.58 to 0.61; the highest lower bound is 0.500, for judge v2 (0.497 with 10,000 resamples instead of 2,000), so no text system is clearly better than chance within an issuer on held-out; on DEV, TF-IDF's within-issuer AUC was 0.688 [0.573, 0.791], clear of 0.5. The lookup encodes each issuer's past habit: it shows what the label measures and is not a policy to deploy; inside one company it is a constant.
- **What the LLM adds over a lookup or TF-IDF** (restates the frozen claims): nothing this sample can show. In 6 paired comparisons the judges are below in 2 and not distinguishable in the rest. This is a negative result for these two prompts on one local model, not for LLMs on this task.
- **The automation slice is mostly an issuer lookup.** TF-IDF at its frozen threshold automated 29/250 at 93.1% (Wilson [78.0%, 98.1%]; frozen claim: not shown to meet the target). In 27 of those 29 the decision is the one the issuer lookup makes; 25 of the 29 narratives name an issuer; the proposed refusals come from issuers that paid 21% of their TRAIN complaints on average, the proposed refunds from issuers that paid 54%. The masked model makes the same decision in 29 of the 29, so issuer habit shows in the wording as well as in the names. The DEV operating point qualified on the boundary (20 cases, the minimum, at 90.0%, the target) and 19 held-out cases lie within 0.02 of the threshold. Inside a single company the issuer is fixed, so this slice would not exist in this form, and within an issuer no text system is clearly above chance. Routing as measured: decide alone, none; for a person to approve, 24 proposed refusals (the company paid 2) and 5 proposed refunds (it paid 5); escalate, 221. Whether pre-filled proposals save a reviewer time was not measured.
- **Gaming the judges** (a fixed set of 5 appended English sentences, one template each, on 30 DEV complaints, plus two controls; a smoke test, not a red team). The fake internal note turned 27 of 28 of judge v2's refusals and 6 of 6 of judge v1's into refunds (the neutral sentence: 2 of 6 for v1); with 6 refusals the test cannot tell v1 from the control: it shows that v2 is moved and cannot show whether v1 is. Of the 10 rewrite-against-control tests for the two judges, 1 survives a Holm correction (v2 fake note). The input filter is seven English patterns: it caught the one wording it was written against (30/30 copies), 0 of 10 other wordings of the same instruction, 0 of 8 obfuscated wordings written after the review, 0 of 30 fake notes, and it flags 3 of 3 harmless sentences, so its flag is a routing lever the complainant holds. The guarded pipeline flipped 5/30 on the note, but it pays 0 of these 30 and 9 of 200 DEV complaints, and a rule that always refuses would show no flips at all, so a low flip rate alone shows little. The note did change the facts v2 extracted (issuer error false to true in 18 of 30, issuer status changed in 23), and the fact model happens to weight 'credited' and 'pending' towards no relief; the low flip rate is not shown robustness.
- **Gaming TF-IDF** (DEV only, no model call). A sentence naming four issuers whose names TF-IDF weights towards relief turned 18 of 21 of its refusals on the same 30 complaints into refunds (65% to 95%, Wilson), and 106 of 127 on all 200 DEV complaints; the neutral sentence turned 0, the five rewrites 0 to 1, and naming four issuers weighted towards no relief turned 6 of 9 refunds into refusals. The sentences were written with the published feature list in view.
- **Not tested by the gaming test**: fabricated merchant correspondence; fake prior approvals with reference numbers; claims written to match the published rubric, cues and fact-model weights; a merchant dispute reframed as fraud through the sub-issue the consumer selects (every system reads it); other languages and obfuscated text sent to the model; the attack at the start, in the middle or inside a quoted email, and padding that pushes real content past the narrative cut; attacks aimed at the fact-extraction step, named as such; using the filter's flag to choose routing; refiling with trivial edits until the answer changes; an attack set written by someone other than the defence author and held out from tuning. The prompts, cues and fact-model weights are public in this repository. Text written by the claimant is evidence to check against the system of record, not an instruction.
- **DEV against held-out.** Judge v2 moved from 0.070 to 0.205: +0.136 [-0.044, +0.310] as two independent samples, within sampling variation (the change in v2 minus v1 is +0.170 [-0.033, +0.367]); no DEV-to-held-out test was registered. The guarded pipeline moved by +0.192 [+0.044, +0.333], which is not ordinary noise; its DEV figure is out-of-fold (ten fact models, each fitted on nine tenths of DEV) while the held-out figure uses one model fitted on all of DEV, a different estimator.
- **Repeatability** (the noise pass was frozen; this reading was not). Asked twice, the judges gave the same decision in 248/250 and 250/250 held-out cases and 99/100 and 196/200 DEV cases (v1, v2). On held-out, judge v2's answers were byte-identical in 250/250, against 191/200 on DEV, where its confidence moved by up to 0.85 and the laptop ran with up to 17 GB in swap: the held-out pass measured a quiet machine. The confidence score is not stable enough to set a threshold on.
- **Cost** (compute on one Apple M4 Max laptop with 36 GB of memory and a local model, DEV timings; no money cost or handle time was measured): a table lookup for the issuer rows; TF-IDF 1.4 s of CPU per 1,000; judge v1 3.1 h, judge v2 3.7 h (on the held-out run v1 2.4 h and v2 3.8 h); a person: not measured until the blind labels are locked.
- **Realistic against optimistic** (plan item 9; full table in the report's Estimates): TF-IDF automates 11.6% at 93.1% on held-out at the frozen threshold, 10.0% at 90.0% on DEV where the threshold was chosen, and 14.0% at 91.4% with the best threshold chosen on the held-out split itself; the LLM configurations automate nothing in any of the three.
- **Not tried**: few-shot examples, giving the LLM the issuer's payout rate, a larger or hosted model, a label written by trained agents under a written policy. **Not measured**: handle time, money, customer outcomes, appeal rate, chargeback reason codes and scheme deadlines, billing-error time limits, who is accountable, whether a reviewer can use the output.
<!-- END practice -->

## A person's blind labels

<!-- BEGIN human -->
Sample: 100 TRAIN complaints drawn by a hash of the ID (`data/label_sample.json`), which no model had answered and the annotator had not seen; labelled blind by the author against `docs/labelling_rubric.md`, which gives the person the rules the judge v1 prompt was given. Person against v1 therefore measures whether the model applies a written policy the way a person reading the same text does; person against v2 compares answers to different questions. The labels are one non-expert's reading through a machine translation, not ground truth. The analysis was registered before any label existed (`docs/analysis_plan_human_labels.md`, 2026-11-10 01:16 UTC, amended 2026-11-10 02:31 UTC). The Russian reading aid beside the English original was made by a separate model, not the judged one (Anthropic's `claude-fable-5-2`, run as sub-agents of the AI coding assistant that wrote this repository, so those sub-agents read the 100 narratives), never part of a judged input (0 translations fail a completeness check, 0 are missing).

_No locked set of labels yet._ Nothing is reported from a partial set. Model answers for the sampled complaints so far: 0 (the judges run on them only after the lock, with `make judge-labelled`).
<!-- END human -->

What `make judge-labelled` followed by `make report` prints once the labels are locked: the person against the recorded decision next to the issuer lookup and every other system on the same decided cases; each judge version and TF-IDF against the person, with the direction of disagreement; the paired differences the plan registered; time per label; the calibration and evaluation halves; whether a v3 prompt exists and, if so, its result under the registered rule on the evaluation half only; and, case by case, the author's own reading of the sorted disagreements from the second labelling screen. Rows marked as not registered were added after labelling had begun, without any label being read, and are descriptive only.

## Freeze and what it proves

<!-- BEGIN freeze_record -->
Gist: https://gist.github.com/dmitri-sulga/dc5fd5e1dd67532a64698ce11b0d3ea5 (secret, that is unlisted: readable by anyone with the link). Revisions, with the commit time GitHub's API reports for each:

| Revision committed (GitHub server time) | Frozen commit it names | SHA-256 of results/frozen.json |
|---|---|---|
| 2026-11-10 04:13:35 UTC | `97dd167d94cd` | `2ddde4cd2f613116...` |
| 2026-11-10 08:16:37 UTC | `822811d125e8` | `cd8950249f3e1997...` |

The run marker (`results/test_run.json`) records the start of the held-out run at 2026-11-10 08:18:54 UTC, from commit `822811d125e8`; the frozen file now on disk has SHA-256 `cd8950249f3e1997...`. Both match the gist's last revision (`make check` compares them with the hashes copied from it). The first revision names an earlier freeze, replaced before any held-out complaint was sent to the model.

What this shows: by the time of the last revision, 2 minutes 17 seconds before the run marker, the holder of the author's GitHub account had the frozen commit hash and the hash of the frozen file, and a commit hash fixes every file of that commit. What it does not show: that no held-out complaint had been sent to the model before then (that rests on the request journal and the caches in this repository), or anything about the times of other commits. A gist is controlled by its owner and is not a notarised record; its revision times are GitHub's, not an independent timestamping service's.

Answer files at the run commit `81f37e5`, recorded after the run (they were not in the run marker): `results/cache/responses.jsonl` 2,526,410 bytes, SHA-256 `2c1364119eedf0b5...`; `results/cache/replays.jsonl` 1,617,642 bytes, SHA-256 `f96418534474e774...`; `results/cache/sent.jsonl` 78,604 bytes, SHA-256 `630f3bb19cd364fd...`. `make check` fails if these bytes change; they anchor later changes only once that commit is public.
<!-- END freeze_record -->

## Where the detail is

Everything is generated into `results/report.md` by the same code, from the same caches:

- **Held-out split**: all metrics with intervals, AUC within and across issuers, paired differences, the operating points applied unchanged, the noise pass ([Test results](results/report.md#test-results)); realistic, optimistic and ceiling estimates, including the label-noise proxy from near-identical narratives ([Estimates](results/report.md#estimates)); the frozen method, thresholds and claim rules ([Frozen](results/report.md#frozen)).
- **Exploratory, added after the held-out run** (`src/judge/posthoc.py`): accuracy against always answering no, errors by direction, DEV against held-out, the corrected intervals, the automation slice against the issuer lookup, gaming the TF-IDF model, the filter on further wordings, and relief by sub-issue ([Exploratory](results/report.md#exploratory)).
- **DEV**: the wide table with AUC columns ([Summary](results/report.md#summary)); full metrics, confusion counts, paired differences and the learning curve ([Dev results](results/report.md#dev-results)); the issuer analysis ([V1v2](results/report.md#v1v2), [Issuer](results/report.md#issuer)); earlier and later halves ([Date halves](results/report.md#date-halves)).
- **Reading of the disagreements**, drafted by the AI coding assistant and not yet confirmed by the author ([Error groups](results/report.md#error-groups), [Error analysis](results/report.md#error-analysis)).
- **Gaming test** per rewrite and control ([Gaming summary](results/report.md#gaming-summary), [Gaming](results/report.md#gaming)); **automation against agreement**, split-half checks and the DEV rehearsal of the held-out procedure ([Automation summary](results/report.md#automation-summary), [Automation](results/report.md#automation), [Rehearsal](results/report.md#rehearsal), `results/automation_curve.png`); **test-retest** ([Retest](results/report.md#retest)); **cost and timing** ([Cost](results/report.md#cost)); settings and the data funnel ([Settings](results/report.md#settings), [Data](results/report.md#data)).

## What every system reads

The LLM and the TF-IDF model get the same text: the product and sub-product, the issue and sub-issue the consumer selected, and the consumer's narrative. The keyword rule reads the narrative only. All of it was written by the consumer when filing, before the company answered. The two issuer rows read the company field instead and nothing else; no text system is given that field, but consumers usually name the company in their text, and only a system trained on labelled complaints can make use of a name. Nothing else is in the frozen records except the complaint ID, the date received and the recorded response; the company's public response, the timeliness flag, state, ZIP code and tags exist only in the label cross-check file and are never read by any system. There is no amount field.

<!-- BEGIN inputs -->
Narratives are cut at 6,000 characters for every system. Model: `qwen3.8:27b`, temperature 0, seed 42.
<!-- END inputs -->

## Method

1. **Data.** One decision type close to a payment dispute: credit card complaints about a purchase shown on the statement, with a narrative. Label: closed with monetary relief against closed with explanation or with non-monetary relief; other statuses dropped. Near-duplicate narratives are collapsed before a seeded, label-stratified split into TRAIN, DEV and held-out TEST (`data/splits.json`).
2. **Plan first.** Metrics, splits, baselines and the automation rule were committed before the first LLM call on DEV (`docs/analysis_plan.md`, verbatim from the first commit; the timestamps are local). The size of TRAIN had been chosen with a DEV learning curve of the classic baseline in view.
3. **Baselines**: majority class; the planned company rule; TF-IDF with logistic regression fitted on TRAIN. Added later: a keyword rule (cues from the DEV reading, choice among them on TRAIN), the issuer's TRAIN relief rate, and TF-IDF with issuer names masked.
4. **Judge v1** is a plain rubric: what should happen. **Judge v2** was written minutes after the v1 run from a machine pass over the disagreements: it asks a different question (predict the recorded answer), states the base rate and a list of cues, and extracts facts before deciding. Three changes at once, so nothing can be attributed to one of them. The diff is in `src/judge/prompts.py`. Here "judge" means the decision model itself, not a grader of other outputs.
5. **Gaming test.** A fixed set of five sentences appended at the end of the narrative, one template each (claimed promise, emotional pressure, legal threat, an instruction to the reviewer, a fake internal note), plus two controls. Defence: a filter of seven English patterns that removes instruction-like text and sends the complaint to a person, and a decision computed from the extracted facts by a small logistic model instead of the LLM's verdict (the guarded pipeline).
6. **Automation.** Confidence is the token probability of the decision; the operating point is the lowest threshold whose DEV agreement on automated cases reaches the target.
7. **Held-out TEST** was run once, by one command, after the method and every claim about it were frozen in `results/frozen.json`.

## Reproduce

```bash
make reproduce   # tests, then every table from data/ and results/cache/, no model needed
```

What the check covers: it regenerates every generated block of this README (the parts between `BEGIN` and `END` markers), all of `results/report.md` and `results/metrics.json`, and fails if any of them differs; it fails if this README has a result-like number written in digits outside those blocks; and it fails if a file hashed in `results/frozen.json` changed without a record in `results/post_run_changes.json`, or if the answers recorded at the run commit changed. Hand-written text outside the blocks, numbers written as words and `docs/deviations.md` are not checked. `make run` repeats missing model calls (Ollama with the model named above), `make label` opens the blind labelling page (local only), `make heldout` performed the single held-out run and now refuses to run again. Raw numbers: `results/metrics.json`.

## Deviations, withdrawn claims and additions after the plan

The full list is in [`docs/deviations.md`](docs/deviations.md). In short: two statements of the first commit about the data source were withdrawn; the baseline grid was widened after the first commit to follow the plan text; judge v2 does not do what the plan said it would; the reading of the disagreements was drafted by the AI coding assistant, not by a person; several analyses were added after DEV results were seen; and twice after the held-out run the presentation changed without any computation of a frozen number changing, the second time adding the exploratory analyses above.

<!-- BEGIN deviation_facts -->
Numbers behind the deviations list: the first commit's grid would have chosen C = 16.0 without class weighting, DEV kappa 0.272, against 0.383 for the grid that follows the plan text. Label cross-check against the live API: 4 of 22,044 complaints with a narrative differed and were dropped, 0 were missing.
<!-- END deviation_facts -->
