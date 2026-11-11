-- Within one issuer, how often does each system score a paid complaint above an unpaid one?
--
-- Reproduces: the within-issuer AUC of every system with a score on the held-out split (results/metrics.json,
-- test.ranking.systems.<system>.within_issuer_auc, point estimate; the README gives the range over the
-- text systems). The interval around it comes from resampling and is not reproduced here.
--
-- AUC over the same-issuer pairs of 05: a pair counts 1 if the paid complaint has the higher score, 1/2 on
-- a tie, 0 otherwise, pooled over all issuers (src/judge/metrics.py, within_group_auc). A missing score
-- counts as 0.5, as in the frozen ranking (COALESCE); on the held-out split no scored system has a missing
-- score, so this rule is not exercised. systems.scored marks the systems that give a ranking score (the
-- loader fills it from SCORED in src/judge/analysis.py). Standard SQL; runs on Postgres as written.

WITH scored AS (
    SELECT d.system, c.complaint_id, c.issuer, c.label, COALESCE(d.score, 0.5) AS score
    FROM decisions AS d
    JOIN complaints AS c ON c.complaint_id = d.complaint_id
    JOIN systems AS s ON s.system = d.system
    WHERE c.split = 'test'
      AND s.scored = 1
),
pairs AS (                                       -- the self-join of 05, once per system
    SELECT
        p.system,
        CASE WHEN p.score > u.score THEN 1.0
             WHEN p.score = u.score THEN 0.5
             ELSE 0.0 END AS win
    FROM scored AS p
    JOIN scored AS u
      ON u.system = p.system AND u.issuer = p.issuer AND u.label = 0
    WHERE p.label = 1
)
SELECT s.display_name AS system, COUNT(*) AS pairs, SUM(win) AS wins, SUM(win) / COUNT(*) AS within_issuer_auc
FROM pairs
JOIN systems AS s ON s.system = pairs.system
GROUP BY s.display_name, s.position
ORDER BY s.position;
