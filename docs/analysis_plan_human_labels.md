# Addendum to the analysis plan: blind human labels

Registered 2026-11-10T01:16:09+00:00 (UTC), before any human label existed
(`results/human_labels/` did not exist; the commit that adds this file is the
record). The analysis code refuses labels with an earlier timestamp. Nothing
below was run before registration; no third prompt version was drafted.

**Amended 2026-11-10T02:31:23+00:00 (UTC), still before any label existed: see Amendment 1 at
the end. It replaces the sample and the source of the translation. The text
below is kept as registered; the two passages it replaces are marked.**

## Why

The reading of judge v1's DEV disagreements put most of them in "not decidable
from the text": the recorded company decision turns on evidence the complaint
does not contain. Agreement with recorded decisions therefore says little about
whether the judge reads a complaint the way a person would. The human labels
are the measurement of that.

(Withdrawn later; note added 2026-11-10 before the held-out run, the rest of
this plan unchanged: the figure this motive rests on, that most disagreements
are not decidable from the text, came from a machine pass by an AI coding
assistant and was withdrawn; see "Error reading" in the README. Whether the
record turns on evidence outside the complaint is a hypothesis these labels
test, not a finding.)

## What a label is

One person (the author) decides each sampled complaint from the complaint text
alone, under the written rubric (`docs/labelling_rubric.md`): money back, no
money back, or cannot tell. It is his decision under the rubric, not a guess at
what the company did and not a statement of what is legally right.

- One annotator, no second reader, so no inter-annotator agreement.
- He is not a lawyer and has no disputes-handling training.
- (Replaced by Amendment 1: the translation no longer comes from the judged
  model.) He reads English at an intermediate level and reads a Russian machine
  translation next to the English original. The translation comes from the same
  local model that is being judged (separate prompt, separate cache). A
  misreading by that model can therefore reach both the judge and the
  annotator, which can inflate judge-to-person agreement. Translations that fail
  simple checks (cut off, amounts differ, not Russian) are flagged on screen and
  counted in the report.
- The tool shows only what the judge sees: product, issue and narrative. No
  recorded outcome, no company field, no model output, no error category.
- (Replaced by Amendment 1: the sample is drawn from TRAIN.) The sample is the
  fixed draw in `data/label_sample.json`: 100 DEV complaints,
  in a fixed order. They are DEV cases, so no held-out complaint is read by
  anyone before the held-out run; the cost is that the prompts were iterated on
  these complaints (against recorded outcomes, not against these labels).
- Time per case is recorded. Answers are locked when the hundredth is given;
  nothing is reported from a partial or unlocked set.

## What will be reported

Cohen's kappa (primary) and accuracy, each with a 95% percentile bootstrap
interval over complaints, same settings as the main plan:

1. **Person against the recorded company decision.** The text-only human
   reference for this task: how far a person reading the complaint alone is from
   what the company did.
2. **Judge v1 and judge v2 against the person.** Agreement of the judge with a
   human label.
3. **TF-IDF + logistic regression against the person**, and the paired
   difference (judge minus TF-IDF) against the person with a paired bootstrap
   interval. The keyword rule, the majority class and the guarded pipeline are
   listed for completeness.

"Cannot tell" in each of them: such cases are left out of kappa and accuracy
(the person gave no decision to agree with) and reported as a count and share,
together with the recorded relief rate and each system's relief rate on those
cases. As a sensitivity line, every figure is also given with "cannot tell"
counted as "no money back" (without a decision nothing is paid). If the two
readings disagree in sign or the interval changes side of zero, the report says
so and draws no conclusion.

The same figures are given separately for the calibration and the evaluation
half (below). Systems v1, v2 and the baselines were fixed before any label
existed (cache timestamps), so their figures on all 100 cases are untuned with
respect to these labels.

## Calibration and evaluation halves, and a possible v3

The 100 sampled IDs are ordered by the SHA-256 of the ID string; the first 50
are the calibration half, the last 50 the evaluation half
(`judge.human.halves`, checked by a test).

- A third prompt version (v3) may be written only after the labels are locked,
  and only against the calibration half: its labels, notes and disagreements.
  Item-level labels and notes of the evaluation half are not read while v3 is
  drafted; an export command writes the calibration half only.
- v3 is frozen (its request template hash is recorded in the run log) before it
  is run on the evaluation half, and it is run there once.
- **Acceptance rule for v3**, on the evaluation half, decided cases only: the
  comparator is whichever of v1 and v2 has the higher kappa against the person
  on the calibration half. v3 is accepted as agreeing better with the person
  only if the 95% paired-bootstrap interval of kappa(v3, person) minus
  kappa(comparator, person) lies entirely above zero. Otherwise the report says
  that no improvement was demonstrated. v3's kappa against recorded decisions
  on all of DEV is reported next to it and is not part of the rule.
- **What to expect from n of about 50** (fewer after "cannot tell"): a 95%
  interval for kappa about 0.25 to 0.3 either side of the estimate, for accuracy
  about 0.12 to 0.14 either side, and wider for a paired difference. Only a
  large improvement can pass the rule; "no improvement demonstrated" is the
  likely outcome and will be reported as such.
- v3 is not part of the frozen held-out method. Adding it to the held-out run
  would require freezing the method again before that run starts; the held-out
  labels are recorded company decisions, not these human labels.

## After the blind pass

Once the labels are locked, and not before, a second screen shows the author
the 24 disagreement cases that were sorted individually in the v1 error
reading, with the recorded outcome, the judge's answer and the drafted reading,
and records whether he confirms or changes each one. Those answers are stored
separately and are not part of the blind analysis. Until they exist the README
states how the drafted readings were produced.

## Amendment 1 (2026-11-10T02:31:23+00:00 UTC, before any label existed)

An independent review of the repository found that the blindness of the labels
could not be established for the registered sample, and that the translation
came from the model being judged. Both are changed here, before the first
label. No label, and no draw of the new sample, existed when this was written.

**Why the DEV sample is withdrawn.** 55 of its 100 complaints are disagreements
of judge v1 that are listed, with a category and a quote, in
`results/error_analysis/v1_dev.csv`; a listed case together with the judge's
answer gives away the recorded outcome, and 11 of them are printed in the full
report. Removing those cases is not possible either: "was a disagreement" is
defined by the outcome, so the remaining DEV cases would be a sample selected
on the outcome.

**Options considered.** (a) Fewer DEV cases: rejected for the reason just given.
(b) Complaints from the part of the sampling frame that is in no split:
untouched by everything, but they are not in the frozen records file and
adding them means rebuilding the data. (c) TRAIN complaints: in the frozen
records, never sent to any model, never read one by one (TRAIN was used to fit
the classic baselines and for aggregate keyword counts only), and no prompt was
written against them. **Chosen: (c).**

**The new sample.** 100 TRAIN complaints: the TRAIN IDs are ordered by the
SHA-256 of the string `label-sample-2:` followed by the ID, and the first 100
are taken, in that order (`judge.human.sample_from`). It is not stratified on
anything. Stratifying on the recorded outcome would tell the annotator the
share of refunds in his sample; issuer or length strata would add little at
this size. The calibration and evaluation halves follow the registered hash
rule, applied to the new sample.

**What follows from using TRAIN.**
- The judge has no answer for these complaints when they are labelled. Judge v1
  and v2 are run on them, with the frozen prompts, only after the labels are
  locked; the runner refuses earlier. The cache timestamps then show that no
  model answer existed while the person was labelling.
- The classic baselines were fitted on TRAIN, including these 100. For every
  figure on the labelled sample they are refitted on the other 7,900 TRAIN
  complaints with the frozen hyperparameters (TF-IDF model, keyword-rule
  choice, issuer rates), so no baseline has seen the recorded outcome of a
  labelled case. The fact model of the guarded pipeline is fitted on DEV and is
  unaffected.
- The earlier consequence ("the prompts were iterated on these complaints") no
  longer applies. A sensitivity line for cases printed in the report is not
  needed: no sampled case appears in any annotation or report (checked by a
  test).

**The translation.** The Russian text beside the English original is produced
by a separate model, not the judged one, so a misreading by the judged model
cannot reach the annotator through the translation. The translations that had
already been made with the judged model were discarded unseen by the annotator.
The English original stays visible and counts where the two differ; the
completeness checks and the on-screen warning are unchanged.

**Two different questions.** The rubric asks what should happen, which is the
question judge v1 was asked. Judge v2 was asked to predict the recorded
decision. Person against v1 compares two answers to the same question; person
against v2 compares answers to different questions, and the report says so next
to the figure.

**Second screen.** Unchanged: the 24 individually sorted DEV disagreements are
shown only after the blind pass on the new sample is locked.
