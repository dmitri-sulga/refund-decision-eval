-- Which issuers carry the TRAIN volume, what share of it do the largest ones hold together, and does the
-- lookup pay or refuse them?
--
-- Reproduces: the issuer facts of the full report (results/metrics.json, issuer): the number of issuers in
-- TRAIN, the issuer with the highest and the lowest smoothed rate among those with at least 100 TRAIN
-- complaints, and how many issuers with at least 20 TRAIN complaints paid more than half (the planned
-- company rule's threshold). Needs the views of 02. Printed for the first time (added later; descriptive):
-- Result 1, each issuer's share of TRAIN and the running share.
--
-- Window functions compute a value for each row from other rows, without collapsing them as GROUP BY does:
--   ROW_NUMBER() OVER (ORDER BY ...)   numbers the rows in that order
--   SUM(n) OVER ()                     the total over all rows, repeated on every row
--   SUM(n) OVER (ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)   a running total
--   RANK() OVER (ORDER BY ...)         rank, with ties sharing a rank
-- Standard SQL; window functions need SQLite 3.25 or later (Python's sqlite3 has had them since 2018).
-- Runs on Postgres as written, with the same rows in the same order: issuers with equal TRAIN counts
-- are ordered by their lowest TRAIN complaint ID (a number, from 02), not by name, whose order depends
-- on how the database compares text.

-- Result 1: issuers by TRAIN volume with a running share (ties: lowest complaint ID first).
SELECT
    ROW_NUMBER() OVER (ORDER BY n_train DESC, lowest_complaint_id)        AS volume_rank,
    issuer,
    n_train,
    CAST(n_train AS DOUBLE PRECISION) / SUM(n_train) OVER ()               AS share_of_train,
    CAST(SUM(n_train) OVER (ORDER BY n_train DESC, lowest_complaint_id
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS DOUBLE PRECISION)
        / SUM(n_train) OVER ()                                             AS running_share,
    smoothed_rate,
    CASE WHEN smoothed_rate > (SELECT base FROM train_base) THEN 'pay' ELSE 'refuse' END AS lookup_says
FROM issuer_train_rates
ORDER BY volume_rank;

-- Result 2: issuers with at least 100 TRAIN complaints, ranked by smoothed rate (highest first).
-- The window runs after WHERE, so the rank is among these issuers only.
SELECT
    RANK() OVER (ORDER BY smoothed_rate DESC) AS rate_rank,
    issuer,
    n_train,
    smoothed_rate
FROM issuer_train_rates
WHERE n_train >= 100
ORDER BY rate_rank;

-- Result 3: counts behind the issuer facts.
SELECT
    COUNT(*)                                                        AS issuers_in_train,
    SUM(n_train)                                                    AS train_complaints,
    SUM(CASE WHEN n_train >= 20
              AND CAST(paid_train AS DOUBLE PRECISION) / n_train > 0.5 THEN 1 ELSE 0 END) AS issuers_paying_over_half
FROM issuer_train_rates;
