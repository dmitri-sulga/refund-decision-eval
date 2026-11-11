-- How often does the issuer lookup agree with the company's record, and how much better than chance?
--
-- Reproduces: the issuer lookup's 2x2 confusion counts, agreement and Cohen's kappa, on DEV and on the
-- held-out split (results/metrics.json, dev.issuer_rate and test.results.issuer_rate; the README table
-- prints them in the row 'Issuer relief rate'). Needs the views of 02.
--
-- Kappa = (agreement - chance agreement) / (1 - chance agreement), where chance agreement is what two
-- independent raters with the same pay rates would reach: p_record * p_system + (1 - p_record) * (1 - p_system).
-- It is computed in the same order as src/judge/metrics.py, so the floats agree to the last digit.
-- Standard SQL (CTEs: WITH ... AS); runs on Postgres as written.

WITH cells AS (                                  -- the four cells of the 2x2 table, one row per split
    SELECT
        split,
        COUNT(*)                                                    AS n,
        SUM(CASE WHEN label = 1 AND decision = 1 THEN 1 ELSE 0 END) AS both_pay,
        SUM(CASE WHEN label = 0 AND decision = 1 THEN 1 ELSE 0 END) AS pays_company_did_not,
        SUM(CASE WHEN label = 1 AND decision = 0 THEN 1 ELSE 0 END) AS refuses_company_paid,
        SUM(CASE WHEN label = 0 AND decision = 0 THEN 1 ELSE 0 END) AS both_refuse
    FROM issuer_lookup
    GROUP BY split
),
rates AS (                                       -- the shares kappa needs
    SELECT
        cells.*,
        CAST(both_pay + both_refuse AS DOUBLE PRECISION) / n          AS agreement,
        CAST(both_pay + refuses_company_paid AS DOUBLE PRECISION) / n AS p_record_pays,
        CAST(both_pay + pays_company_did_not AS DOUBLE PRECISION) / n AS p_system_pays
    FROM cells
),
chance AS (
    SELECT
        rates.*,
        p_record_pays * p_system_pays + (1 - p_record_pays) * (1 - p_system_pays) AS chance_agreement
    FROM rates
)
SELECT
    split,
    n,
    both_pay,
    pays_company_did_not,
    refuses_company_paid,
    both_refuse,
    both_pay + pays_company_did_not AS system_pays,
    agreement,
    CASE WHEN chance_agreement < 1                -- undefined when the record and the system give the same
                                                  -- single answer on every case (chance agreement is then 1)
         THEN (agreement - chance_agreement) / (1 - chance_agreement) END AS kappa
FROM chance
ORDER BY split;
