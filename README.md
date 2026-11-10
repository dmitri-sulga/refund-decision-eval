# Refund decision policy, scored against recorded company decisions

A local LLM reads a credit card purchase-dispute complaint and answers: monetary relief or not. Its first prompt decides what should happen; its revised prompt predicts what the company recorded. Both are scored against the response the company recorded in the US CFPB Consumer Complaint Database. **Every metric here is agreement with recorded company decisions, not correctness.**

<!-- BEGIN lead -->
**Result so far (DEV, 200 complaints; the held-out split has not been run).** The judge agrees with recorded company decisions little better than chance: kappa 0.12 for the first prompt and 0.07 for the revised one. The revision did not help (-0.05 [-0.20, +0.10]). Both versions are below a TF-IDF baseline (kappa 0.38), whose lead is mostly knowing how each issuer usually answers. No judge configuration qualifies for automation at the planned target. What a careful person would decide from the same text is not known yet; blind human labels are being collected.
<!-- END lead -->

Built in November 2026 as a work sample for an applied-AI job application. It runs on one laptop with a local model; every model answer is cached, and `make reproduce` recomputes every number below without a model and fails if one differs.

## Results so far (DEV only; the held-out split has not been run)

<!-- BEGIN summary -->
DEV: 200 complaints, 30.0% closed with monetary relief. Held-out TEST: 250 complaints, 30.0%, not yet run. TRAIN (classic baselines only): 8,000, 30.1%. Kappa is Cohen's kappa against the recorded company decision, with a 95% bootstrap interval; AUC is the chance that a relief case is scored above a no-relief case, over all pairs and over pairs answered by the same issuer.

| System | Reads | DEV kappa [95% CI] | DEV AUC: all pairs / same issuer | Automated on DEV at the planned target | Held-out kappa [95% CI] | Time per 1,000 decisions |
|---|---|---|---|---|---|---|
| Majority class (always no relief) | nothing | 0.000 (constant answer) | n/a | not assessed | not run | a table lookup |
| Keyword rule (cues from the DEV reading, choice on TRAIN) | narrative | 0.076 [-0.053, 0.213] | n/a | not assessed | not run | 0.1 s (CPU) |
| Company rule (planned: issuer's TRAIN relief rate above one half) | company field only | 0.157 [0.033, 0.279] | n/a | not assessed | not run | a table lookup |
| Issuer relief rate (added later: above the TRAIN base rate) | company field only | 0.318 [0.190, 0.442] | 0.73 / 0.50 | not assessed | not run | a table lookup |
| TF-IDF + logistic regression, issuer names masked (added later) | header + narrative, issuer names masked | 0.295 [0.153, 0.433] | 0.67 / 0.66 | not assessed | not run | 1.4 s (CPU) |
| TF-IDF + logistic regression (trained on TRAIN) | header + narrative | 0.383 [0.245, 0.513] | 0.74 / 0.69 | 20/200 at 90.0% agreement (Wilson [69.9%, 97.2%]); chosen and scored on the same cases, does not survive a split | not run | 1.4 s (CPU) |
| LLM judge v1 (plain rubric) | header + narrative | 0.120 [0.024, 0.221] | 0.62 / 0.61 | none: no threshold reaches the target | not run | about 3 h (one laptop GPU) |
| LLM judge v2 (predicts the record, facts first) | header + narrative | 0.070 [-0.051, 0.197] | 0.59 / 0.60 | none: no threshold reaches the target | not run | about 4 h (one laptop GPU) |
| Guarded pipeline (v2 facts -> fact model; filter routes to human) | header + narrative | 0.009 [-0.070, 0.103] | 0.58 / 0.58 | none: no threshold reaches the target | not run | about 4 h (one laptop GPU) |
| Person, blind (one annotator) | header + narrative | not labelled yet |  | n/a | n/a |  |
<!-- END summary -->

<!-- BEGIN v1v2 -->
Highest DEV kappa: TF-IDF + logistic regression (trained on TRAIN).

Paired kappa differences: Judge v1 is below TF-IDF (-0.263 [-0.409, -0.110]); Judge v1 is below the issuer relief rate (-0.198 [-0.359, -0.036]); Judge v1 is not distinguishable from the planned company rule (-0.036 [-0.191, +0.116]); Judge v2 is below TF-IDF (-0.314 [-0.487, -0.136]); Judge v2 is below the issuer relief rate (-0.248 [-0.424, -0.061]); Judge v2 is not distinguishable from the planned company rule (-0.087 [-0.261, +0.092]).

What the TF-IDF lead consists of: the issuer's TRAIN relief rate alone, without reading the complaint, orders DEV complaints as well as TF-IDF does (AUC 0.73 against 0.74; difference -0.010 [-0.088, +0.064]), and TF-IDF's strongest features are issuer names (71% of TRAIN narratives name an issuer). Among complaints answered by the same issuer (747 pairs) the lead of TF-IDF over the judge is no longer demonstrated: AUC difference +0.079 [-0.058, +0.196] against v1 and +0.092 [-0.060, +0.230] against v2 (all pairs: +0.120 [+0.016, +0.218] and +0.143 [+0.033, +0.240]). The judge's own ordering is weak either way (AUC 0.62 and 0.59).

The prompt revision from v1 to v2 did not improve kappa (-0.050 [-0.198, +0.101]). Accuracy rose only because v2 predicts relief for 11.0% of complaints where v1 predicted it for 68.5%; recall of relief fell from 0.80 to 0.15.

No judge configuration reaches the planned automation target on DEV. The one operating point in the table (TF-IDF) is in-sample: in 1,000 random half-splits of DEV a threshold existed in 18% of them, and the other half reached the target in 0 (median agreement there 74%).
<!-- END v1v2 -->

## Limits

- **Agreement, not correctness.** The label is what the company did. A policy that agrees more with companies is not a fairer one, and nothing here measures fairness or legal correctness.
- **One side of the story, and the issuer matters more than the story.** Every text system sees only the consumer's narrative; the record follows the issuer's habits at least as much as the complaint (see the issuer rows above).
- **Small and self-selected; read the intervals, not the point estimates.** The CFPB itself says the database is not a statistical sample of consumer experiences. The split is random, not by date. DEV was used for every choice after the first prompt, so later DEV numbers are optimistic; the first prompt was fixed before its DEV run, so its DEV number is not tuned.
- **The judge is not deterministic.** Temperature zero and a fixed seed do not make this model repeat itself (figures below); the report is reproducible from the cached answers, not by asking again.
- **Home-made.** The gaming rewrites and the defence have one author, and a defence tuned to known attacks overstates robustness. The reading of disagreements was drafted by an AI assistant and awaits the author's confirmation. The narratives come from a third-party copy of the official bulk export that cannot be checked against an official source.

<!-- BEGIN retest_summary -->
Asked twice with identical settings (100 of 200 DEV requests replayed), LLM judge v1 (plain rubric) gave the same decision in 99/100 = 99.0% [94.6%, 99.8%] (largest change in P(relief) 0.06; kappa against the record 0.010 from the cached answers and 0.000 from the replayed ones on the same requests). Asked twice with identical settings (200 of 200 DEV requests replayed), LLM judge v2 (predicts the record, facts first) gave the same decision in 196/200 = 98.0% [95.0%, 99.2%] (largest change in P(relief) 0.85; kappa against the record 0.070 from the cached answers and 0.041 from the replayed ones on the same requests).
<!-- END retest_summary -->

## What every system reads

The LLM and the TF-IDF model get the same text: the product and sub-product, the issue and sub-issue the consumer selected, and the consumer's narrative. The keyword rule reads the narrative only. All of it was written by the consumer when filing, before the company answered. The two issuer rows read the company field instead and nothing else; no text system is given that field, but consumers usually name the company in their text, and only a system trained on labelled complaints can make use of a name. Nothing else is in the frozen records except the complaint ID, the date received and the recorded response; the company's public response, the timeliness flag, state, ZIP code and tags exist only in the label cross-check file and are never read by any system.

<!-- BEGIN inputs -->
Narratives are cut at 6,000 characters for every system. Model: `qwen3.8:27b`, temperature 0, seed 42.
<!-- END inputs -->

## Method

1. **Data.** One decision type close to a payment dispute: credit card complaints about a purchase shown on the statement, with a narrative. Label: closed with monetary relief against closed with explanation or with non-monetary relief; other statuses dropped. Near-duplicate narratives are collapsed before a seeded, label-stratified split into TRAIN, DEV and held-out TEST (`data/splits.json`).
2. **Plan first.** Metrics, splits, baselines and the automation rule were committed before the first LLM call on DEV (`docs/analysis_plan.md`, verbatim from the first commit; the timestamps are local). The size of TRAIN had been chosen with a DEV learning curve of the classic baseline in view.
3. **Baselines**: majority class; the planned company rule; TF-IDF with logistic regression fitted on TRAIN. Added later: a keyword rule (cues from the DEV reading, choice among them on TRAIN), the issuer's TRAIN relief rate, and TF-IDF with issuer names masked.
4. **Judge v1** is a plain rubric: what should happen. **Judge v2** was written minutes after the v1 run from a machine pass over the disagreements: it asks a different question (predict the recorded answer), states the base rate and a list of cues, and extracts facts before deciding. Three changes at once, so nothing can be attributed to one of them. The diff is in `src/judge/prompts.py`.
5. **Gaming test.** Five appended sentences that add no verifiable fact (claimed promise, emotional pressure, legal threat, an instruction to the reviewer, a fake internal note), plus two controls. Defence: a filter in plain code that removes instruction-like text and sends the complaint to a person, and a decision computed from the extracted facts by a small logistic model instead of the LLM's verdict (the guarded pipeline).
6. **Automation.** Confidence is the token probability of the decision; the operating point is the lowest threshold whose DEV agreement on automated cases reaches the target.
7. **Held-out TEST** is run once, by one command, after the method and every claim about it are frozen in `results/frozen.json`.

## Reading the disagreements

<!-- BEGIN error_groups -->
Judge v1 disagreed with the record on 101 of 200 DEV complaints. The categories, quotes and notes were drafted by an AI coding assistant working on this repository (not the judged model); the author has read 0 of the cases. The 101 narratives hold 29,193 words. The v1 run ended at 19:56 UTC, the disagreements were exported at 19:57, the v2 prompt was first used at 20:08 and the category file was last written at 20:17: the v2 prompt rests on a machine pass of minutes over these cases, not on a person's reading. The author's own reading of the individually sorted cases will be reported separately when it exists.

24 disagreements were sorted one by one into a three-way reading:

| Reading (drafted, not yet confirmed by the author) | Cases | Example |
|---|---|---|
| Model wrong: the judge misread the narrative or misapplied its own rubric | 14 | "I used Care Credit to charge" (10127800: judge invented an exclusion for dental services) |
| Recorded decision questionable or discretionary: the company paid a claim the rubric refuses, or reversed its own earlier answer | 2 | "they performed a chargeback" (8867817: scam payments; issuer had refused one of three and paid after the complaint) |
| Not decidable from the text: the outcome turns on evidence the narrative does not contain (the issuer's investigation, a pending dispute, missing facts) | 8 | "It was NOT deducted from my monthly statement" (7673601: reads statement credits and a provisional fraud credit as missing) |

The other 77 disagreements carry a category only (issuer already denied 60, investigation pending 10, vague or silent 7). These categories do not show that a case cannot be decided from its text: judge v2 extracts 'the issuer had already said no' for 175 of 200 DEV complaints (relief rate 30.3% with it, 28.0% without), and judge v1 agreed with the record on 87 of those 175 (41 of them closed with relief). That the record often turns on evidence outside the complaint is a hypothesis here, which the blind human labels are meant to test; it is not a finding.
<!-- END error_groups -->

## Gaming test

<!-- BEGIN gaming_summary -->
30 DEV complaints x 5 rewrites (promise, emotion, legal_threat, injection, fake_case_note). Decisions flipped [95% CI], and how many denials on the unmodified narrative became relief (compare the judges on the denials: they start from 6 and 28 denials):

| Rewrite | Judge v1 | Judge v2 | Guarded pipeline |
|---|---|---|---|
| **all five rewrites** | 23/150 = 15.3% [7.3%, 25.3%]; denials turned into relief 17/30 = 57% | 39/150 = 26.0% [20.7%, 31.3%]; denials turned into relief 39/140 = 28% | 7/150 = 4.7% [1.3%, 8.7%]; denials turned into relief 7/150 = 5% |
| fake internal note alone | 6/30 = 20.0% [9.5%, 37.3%]; denials turned into relief 6/6 = 100% | 27/30 = 90.0% [74.4%, 96.5%]; denials turned into relief 27/28 = 96% | 5/30 = 16.7% [7.3%, 33.6%]; denials turned into relief 5/30 = 17% |
| instruction to the reviewer alone | 7/30 = 23.3% [11.8%, 40.9%]; denials turned into relief 2/6 = 33% | 9/30 = 30.0% [16.7%, 47.9%]; denials turned into relief 9/28 = 32% | 0/30 (by construction); denials turned into relief 0/30 = 0% |
| control: neutral sentence appended | 3/30 = 10.0% [3.5%, 25.6%]; denials turned into relief 2/6 = 33% | 1/30 = 3.3% [0.6%, 16.7%]; denials turned into relief 1/28 = 4% | 1/30 = 3.3% [0.6%, 16.7%]; denials turned into relief 1/30 = 3% |
| control: same narrative sent again | 0/30 = 0.0% [0.0%, 11.4%]; denials turned into relief 0/6 = 0% | 0/30 = 0.0% [0.0%, 11.4%]; denials turned into relief 0/28 = 0% | 0/30 = 0.0% [0.0%, 11.4%]; denials turned into relief 0/30 = 0% |

Each rewrite is a single fixed sentence appended to the narrative, so every row is one template applied to 30 complaints, not a family of attacks. Rows for one rewrite use a Wilson interval; the pooled row repeats every complaint five times, so its interval resamples complaints.

Against the neutral-sentence control on the same complaints (exact McNemar test, p below 0.05), the rewrites that flip more decisions are: Judge v1: none; Judge v2: injection, fake_case_note; Guarded pipeline: none.

The guarded pipeline answers relief for 9 of 200 DEV complaints and for 0 of these 30; a policy that always answers no relief has no flips at all, so its low flip rate says little. Its fact model is fitted on DEV, including these complaints.

The check in code outside the model is an input filter: it flagged 30/30 copies of the one injected sentence, 0/30 fake notes, 0 of 8,200 unmodified complaints, and 0 of 10 other wordings of the same instruction (written afterwards with the patterns in view; `src/judge/gaming.py`). A flagged complaint goes to a person. The guarded pipeline's zero for 'injection' holds by construction: the filter removes the appended sentence and the cached answer to the clean narrative is reused, so the interval describes nothing. A defence tuned to known attacks overstates robustness.

No LLM-based system reaches the agreement target on DEV, so none of these decisions would be taken automatically; the flips are changes in the decision the system proposes.
<!-- END gaming_summary -->

## Automation against agreement

<!-- BEGIN automation_summary -->
Operating point per system: the lowest confidence threshold whose DEV agreement on automated cases reaches 90% with at least 20 automated, read as three outcomes. Chosen and scored on the same DEV complaints, so optimistic.

| System | Decided automatically as no relief | ... of which the company did pay [95% Wilson CI] | Relief proposed to a person for approval | Escalated to a person |
|---|---|---|---|---|
| TF-IDF + logistic regression (trained on TRAIN) | 14/200 = 7.0% | 1/14 = 7.1% [1.3%, 31.5%] | 6 (1 where the company paid nothing) | 180 (30.0% of them closed with relief) |
| LLM judge v1 (plain rubric) | no threshold reaches the target |  |  |  |
| LLM judge v2 (predicts the record, facts first) | no threshold reaches the target |  |  |  |
| Guarded pipeline (v2 facts -> fact model; filter routes to human) | no threshold reaches the target |  |  |  |

The same rule applied to one random half of DEV and scored on the other: TF-IDF + logistic regression (trained on TRAIN): a threshold in 184 of 1,000 random half-splits, the other half at the target in 0 (median agreement 74%); LLM judge v1 (plain rubric): no threshold in any of 1,000 random half-splits; LLM judge v2 (predicts the record, facts first): a threshold in 31 of 1,000 random half-splits, the other half at the target in 0 (median agreement 62%); Guarded pipeline (v2 facts -> fact model; filter routes to human): a threshold in 111 of 1,000 random half-splits, the other half at the target in 0 (median agreement 69%). On this evidence no system here should decide complaints on its own.
<!-- END automation_summary -->

![Automation against agreement](results/automation_curve.png)

## A person's blind labels

<!-- BEGIN human -->
Sample: 100 TRAIN complaints drawn by a hash of the ID (`data/label_sample.json`), which the judged model had not answered and the annotator had not seen; labelled blind by one person against `docs/labelling_rubric.md`. The analysis was registered before any label existed (`docs/analysis_plan_human_labels.md`, 2026-11-10 01:16 UTC, amended 2026-11-10 02:31 UTC). The annotator reads a Russian machine translation beside the English original: a reading aid made by a separate model, not the judged one, never part of a judged input (0 of 100 translations fail a completeness check, 0 are missing). The classic baselines are refitted without the sampled complaints.

_No locked set of labels yet._ Nothing is reported from a partial set. Model answers for the sampled complaints so far: 0 (the judge runs on them only after the labels are locked).
<!-- END human -->

The Russian rubric (`docs/labelling_rubric_ru.md`), the Russian button labels and the translations exist only because the annotator reads English at an intermediate level; the English original is always shown and is what counts.

## Held-out TEST

<!-- BEGIN test_results -->
_Not yet run._ Held-out complaints sent to the model so far: 0. The method is frozen in `results/frozen.json` (SHA-256 `2ddde4cd2f613116...`): every source file, the data, the plans, the DEV operating points, the 250 held-out complaint IDs (SHA-256 of the sorted IDs `25a48d7a40a32140...`) and 6 claims with their decision rules (listed in the full report). The run is a single command (`make heldout`) that refuses to start unless the frozen method is unchanged and committed, and refuses to run twice; a second pass over the same requests is stored apart and reported as noise only.
<!-- END test_results -->

The realistic estimate is the held-out result at the frozen thresholds; the ceiling discussion (label noise among near-identical complaints, the judge's agreement with itself) is in the full report.

## Reproduce

```bash
make reproduce   # tests, then every table from data/ and results/cache/, no model needed
```

`make run` repeats missing model calls (Ollama with the model named above), `make label` opens the blind labelling page (local only), `make heldout` performs the single held-out run. Full tables: `results/report.md`. Raw numbers: `results/metrics.json`.

## Deviations, withdrawn claims and additions after the plan

- **Data source.** The first commit (README, `config.py`, `data/provenance.json` and its message) said that the CFPB had removed narratives from the live database in a named API release and that the narratives came from a byte-identical mirror of the official bulk export. Neither statement could be verified and both were withdrawn later the same night (the wording in `config.py` and `provenance.json` changed with the commit that added the analysis code). What is verified: product, issue, company and recorded response of every complaint against the live API. The narrative text matches nothing official; it is used because no official source of narratives was reachable, and the copy is pinned by its hash.
- **Before the plan.** The classic baseline had already been scored on DEV when the plan was committed (the first commit's `config.py` cites a DEV learning curve, and wrongly calls it flat).
- **Baseline grid.** The first commit searched four values of C without class weighting; the plan text asks for class weighting, and the grid was widened to match it after the first commit. This raised the baseline.
- **Baselines in the headline.** An earlier version of this README left the planned company rule out of the first table and showed a keyword rule added after DEV results instead. Both are shown now, with two issuer analyses added after an independent review.
- **Judge v2** does not do what plan item six says: it does not address the largest category of disagreements (a prior denial by the issuer), and it changes the question from deciding to predicting.
- **Error reading.** The plan says every disagreement is read and categorised. The categories were drafted by an AI assistant in minutes; an earlier README said "a single reader". The figure "most disagreements are not decidable from the text" from that version is withdrawn.
- **Freeze file.** Plan item four names `results/test_run.json`; the frozen method is in `results/frozen.json`, and `test_run.json` is only the marker of the run. A first freeze covered fewer files and one decision rule; it was replaced before any held-out call. A dry run of the held-out command printed that the input filter changes no held-out complaint; nothing was changed in response, and the dry run no longer touches held-out text.
- **Controls** were added to the gaming test (a neutral sentence, and the same narrative sent again) after a concurrency check showed that the model does not repeat itself; the five rewrites are unchanged.
- **Added after DEV results were seen**, none changing a planned metric or the operating rule: the test-retest replay, the registered noise pass over the held-out split, AUC and the within-issuer comparison, the reading of the operating point as three outcomes, the date-half table, the split-half checks, and the blind human labels (plan in `docs/analysis_plan_human_labels.md`).
- No acceptance rule for v2 existed before v1 and v2 were compared on DEV; one is frozen for the held-out comparison, where v1 is run as well. The fact model's regularisation was left at the library default. The logistic regression is fitted in double precision because with the defaults its result depended on the order of the training rows; its DEV result did not change.

<!-- BEGIN deviation_facts -->
Numbers behind the list above: the first commit's grid would have chosen C = 16.0 without class weighting, DEV kappa 0.272, against 0.383 for the grid that follows the plan text. Label cross-check against the live API: 4 of 22,044 complaints with a narrative differed and were dropped, 0 were missing.
<!-- END deviation_facts -->
