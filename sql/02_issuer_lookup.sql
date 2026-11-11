-- What does the issuer lookup decide for each held-out complaint, from each issuer's payout rate on TRAIN?
--
-- Reproduces: the 'issuer relief rate' baseline of src/judge/baselines.py (issuer_rate) and its decision in
-- src/judge/analysis.py (Systems.predict), exactly:
--   score    = (paid on TRAIN + 10 * base) / (complaints on TRAIN + 10), where base is the TRAIN relief rate;
--              10 is ISSUER_PSEUDO_COUNT, ten pretend complaints at the base rate, so that an issuer with
--              three complaints is not scored 0% or 100%
--   unseen   = an issuer with no TRAIN complaint gets score = base
--   decision = pay when score > base (strictly greater), so an unseen issuer is always refused
-- The issuer is the company field exactly as the study uses it; the study does not normalise names.
-- Printed for the first time (added later; descriptive): Result 1, each issuer's TRAIN count, paid count,
-- raw and smoothed rate; the report gives only the issuer facts that 07 reproduces.
--
-- This file creates three views that 03 and 07 use, then shows two results. Standard SQL; it also runs
-- on Postgres as written, with the same rows in the same order: issuers with equal TRAIN counts are
-- ordered by a number (their lowest TRAIN complaint ID), not by name, because the order of names
-- depends on how a database compares text (SQLite compares bytes; Postgres follows the collation of
-- its locale, which may ignore case).

-- One row: the TRAIN relief rate (the threshold, and the score of an unseen issuer).
CREATE VIEW train_base AS
SELECT CAST(SUM(label) AS DOUBLE PRECISION) / COUNT(*) AS base
FROM complaints
WHERE split = 'train';

-- One row per issuer seen in TRAIN: its complaints, how many were paid, and the smoothed rate.
-- lowest_complaint_id is there only to break ties in TRAIN volume with a number (see above).
CREATE VIEW issuer_train_rates AS
SELECT
    c.issuer,
    COUNT(*)                                             AS n_train,
    SUM(c.label)                                         AS paid_train,
    (SUM(c.label) + 10 * b.base) / (COUNT(*) + 10)       AS smoothed_rate,
    MIN(CAST(c.complaint_id AS INTEGER))                 AS lowest_complaint_id
FROM complaints AS c
CROSS JOIN train_base AS b                               -- attaches the single base row to every complaint
WHERE c.split = 'train'
GROUP BY c.issuer, b.base;                               -- b.base is one value; Postgres wants it named here

-- One row per DEV and held-out complaint: its issuer's score and the lookup's decision.
-- LEFT JOIN keeps a complaint whose issuer has no TRAIN row (r.* is then NULL); COALESCE gives it the base.
CREATE VIEW issuer_lookup AS
SELECT
    c.complaint_id,
    c.split,
    c.issuer,
    c.label,
    COALESCE(r.smoothed_rate, b.base)                            AS score,
    CASE WHEN COALESCE(r.smoothed_rate, b.base) > b.base THEN 1 ELSE 0 END AS decision,
    CASE WHEN r.issuer IS NULL THEN 1 ELSE 0 END                AS issuer_unseen_in_train
FROM complaints AS c
CROSS JOIN train_base AS b
LEFT JOIN issuer_train_rates AS r ON r.issuer = c.issuer
WHERE c.split IN ('dev', 'test');

-- Result 1: each issuer's payout rate on TRAIN, largest issuers first (ties: lowest complaint ID first).
-- The raw rate is printed beside the smoothed one; the lookup decides on the smoothed rate.
SELECT
    issuer,
    n_train,
    paid_train,
    CAST(paid_train AS DOUBLE PRECISION) / n_train       AS raw_rate,
    smoothed_rate,
    CASE WHEN smoothed_rate > (SELECT base FROM train_base) THEN 'pay' ELSE 'refuse' END AS lookup_says
FROM issuer_train_rates
ORDER BY n_train DESC, lowest_complaint_id;

-- Result 2: the lookup's decision for each held-out complaint, next to what the company recorded.
SELECT complaint_id, issuer, score, decision, label AS company_paid, issuer_unseen_in_train
FROM issuer_lookup
WHERE split = 'test'
ORDER BY CAST(complaint_id AS INTEGER);
