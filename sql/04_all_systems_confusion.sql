-- For every system on the held-out split: how often does it pay, how often against the record, and in
-- which direction does it err?
--
-- Reproduces: the README table's columns 'Matches the record', 'Pays (of these, the company did not
-- pay)', 'Refuses where the company paid' and 'Held-out kappa' (the point estimates), for every system
-- in one query (results/metrics.json, test.results.<system>.confusion, accuracy and kappa).
--
-- Conditional aggregation: SUM(CASE WHEN ... THEN 1 ELSE 0 END) counts the rows that meet a condition,
-- so one pass over decisions JOIN complaints, grouped by system, gives every cell of every 2x2 table.
-- Standard SQL; runs on Postgres as written.

WITH cells AS (
    SELECT
        d.system,
        COUNT(*)                                                       AS n,
        SUM(c.label)                                                   AS company_pays,
        SUM(d.decision)                                                AS system_pays,
        SUM(CASE WHEN d.decision = 1 AND c.label = 0 THEN 1 ELSE 0 END) AS pays_company_did_not,
        SUM(CASE WHEN d.decision = 0 AND c.label = 1 THEN 1 ELSE 0 END) AS refuses_company_paid,
        SUM(CASE WHEN d.decision = c.label THEN 1 ELSE 0 END)          AS matches
    FROM decisions AS d
    JOIN complaints AS c ON c.complaint_id = d.complaint_id
    WHERE c.split = 'test'
    GROUP BY d.system
),
rates AS (
    SELECT
        cells.*,
        CAST(matches AS DOUBLE PRECISION) / n      AS agreement,
        CAST(company_pays AS DOUBLE PRECISION) / n AS p_record_pays,
        CAST(system_pays AS DOUBLE PRECISION) / n  AS p_system_pays
    FROM cells
),
chance AS (                                      -- as in 03
    SELECT
        rates.*,
        p_record_pays * p_system_pays + (1 - p_record_pays) * (1 - p_system_pays) AS chance_agreement
    FROM rates
)
SELECT
    s.display_name AS system,
    c.n,
    c.agreement,
    c.system_pays,
    c.pays_company_did_not,
    c.refuses_company_paid,
    c.company_pays,
    CASE WHEN c.chance_agreement < 1
         THEN (c.agreement - c.chance_agreement) / (1 - c.chance_agreement) END AS kappa
FROM chance AS c
JOIN systems AS s ON s.system = c.system
ORDER BY s.position;
