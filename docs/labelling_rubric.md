# Labelling rubric for the blind human labels

One person labels a fixed sample of complaints without seeing what the company
did or what any model answered. Start with `make label` (a page opens in the
browser; `make label-terminal` does the same in the terminal); stop at any time
and run the same command to continue. A Russian version of this page is in
`docs/labelling_rubric_ru.md`; where the two differ, this one counts.

## What you see, and what you do not

You see what the models see: the product and issue the consumer selected, and
the complaint text (long texts are cut at the same point as for the models).
Beside the English original the tool shows a Russian machine translation. It is
a reading aid only, it can be wrong, and no judged system reads it; where the
two differ, the English original counts.
You do not see the company's answer, any model output, the complaint ID, the
date, or the company field (the consumer may name the company in the text).
Do not look a complaint up anywhere.

## The question

From this text alone: **should the card issuer give the cardholder money back**
(a refund, a statement credit, or the reversal of a charge, fee or interest)?

- `1` money back
- `2` no money back (an explanation, or an action that is not money)
- `3` cannot tell

You are not guessing what the company did. You decide what should happen if the
text is all there is, using the same rules the first judge prompt was given:

1. Unauthorized charges (the card was used without the cardholder's permission)
   normally qualify for money back.
2. Billing errors (a duplicate charge, a wrong amount, a merchant refund that
   was never credited) normally qualify.
3. Goods or services not received, or materially not as described, qualify when
   the cardholder tried to resolve it with the merchant and the merchant failed
   to fix it.
4. No money back when the cardholder (or someone they allowed) made the purchase
   and got what was agreed, when the complaint is only dissatisfaction with a
   disclosed price, policy or service, or when the problem has already been
   resolved.

## When to answer "cannot tell"

Use it when the text does not describe a transaction you can identify, or when
the decision turns entirely on evidence the text does not give and you have no
reason to lean either way. If the text leans one way, decide. An issuer having
already said no is a fact about the case, not an answer: decide on what the
text tells you about the charge.

## Working rules

- Decide each case on its own; do not try to hit a target share of refunds.
- One or two minutes per case is enough. If you change your mind about the
  case you just answered, undo it on the next screen.
- The note is optional. If you write one, name the fact that decided it.
- Answering the last case locks all answers; they cannot be changed afterwards.
  Do not open the results, the cache or the error analysis before that.

## How the labels are used

The report compares your labels with the recorded company decisions (how far a
careful reader of the text alone is from what companies did) and with each
system's decisions on the same cases. "Cannot tell" is counted and reported,
and left out of the agreement figures. One person is one annotator: there is
no second reader, and the report says so. The analysis was written down before
any label existed: `docs/analysis_plan_human_labels.md`.

## After the blind pass

Once all answers are locked, the page offers a second screen with the 24
disagreements between judge v1 and the recorded decision that were sorted one
by one. There the recorded decision, the judge's answer and the drafted reading
are visible; you confirm the reading or change it. Those answers are stored
separately.
