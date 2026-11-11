-- When were the complaints received, and did the payout rate move between the earlier and the later
-- half of DEV?
--
-- Reproduces: the date ranges per split and the DEV date halves of the full report (results/metrics.json,
-- date_halves: from, median and to of each split; n, from, to and relief base rate of each half of DEV)
-- and the first and last date of the records (posthoc.data_window).
--
-- Window functions with PARTITION BY restart for each split, like a GROUP BY that keeps every row.
-- The median is the date at position n / 2 + 1 of the sorted dates (integer division, as in Python's
-- n // 2 on 0-based positions; it truncates in SQLite and Postgres, and every n here is even). For an
-- even n there are two middle dates and this takes the later one, as the Python does; Postgres's
-- PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY date_received) takes the earlier one and would print
-- 2025-01-05 for DEV. Standard SQL; runs on Postgres as written.

-- Result 1: first, middle and last date received per split (TRAIN, DEV and held-out; the 12 complaints
-- used only for smoke tests are left out, as in the report).
WITH ordered AS (
    SELECT
        split,
        date_received,
        ROW_NUMBER() OVER (PARTITION BY split ORDER BY date_received) AS pos,
        COUNT(*)     OVER (PARTITION BY split)                        AS n
    FROM complaints
    WHERE split IN ('train', 'dev', 'test')
)
SELECT
    split,
    MIN(n)                                               AS complaints,
    MIN(date_received)                                   AS first_date,
    MAX(CASE WHEN pos = n / 2 + 1 THEN date_received END) AS median_date,
    MAX(date_received)                                   AS last_date
FROM ordered
GROUP BY split
ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'dev' THEN 2 WHEN 'test' THEN 3 ELSE 4 END;

-- Result 2: DEV cut into an earlier and a later half by date received (ties broken by complaint ID, as in
-- src/judge/analysis.py, date_halves), with the relief rate of each half.
WITH dev AS (
    SELECT
        date_received,
        label,
        ROW_NUMBER() OVER (ORDER BY date_received, CAST(complaint_id AS INTEGER)) AS pos,
        COUNT(*)     OVER ()                                                      AS n
    FROM complaints
    WHERE split = 'dev'
),
halves AS (
    SELECT CASE WHEN pos <= n / 2 THEN 'earlier' ELSE 'later' END AS half, date_received, label
    FROM dev
)
SELECT
    half,
    COUNT(*)                                         AS complaints,
    MIN(date_received)                               AS first_date,
    MAX(date_received)                               AS last_date,
    CAST(SUM(label) AS DOUBLE PRECISION) / COUNT(*)  AS relief_base_rate
FROM halves
GROUP BY half
ORDER BY half;

-- Result 3: the date window of everything the study scores (TRAIN, DEV and held-out).
SELECT MIN(date_received) AS first_date, MAX(date_received) AS last_date, COUNT(*) AS complaints
FROM complaints
WHERE split IN ('train', 'dev', 'test');
