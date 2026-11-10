# Analysis plan

The section below is copied verbatim from the README of the first commit of this
repository (`37b32d1`), which was made before the first DEV or TEST model call.
A test (`tests/test_plan.py`) fails if it differs from that commit. Everything
that was added or done differently afterwards is listed under "Deviations and
additions" in the README.

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
