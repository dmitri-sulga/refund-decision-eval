-- Within one issuer, how many (paid, unpaid) pairs of held-out complaints are there to compare?
--
-- Reproduces: the number of same-issuer pairs and of issuers that have one (results/metrics.json,
-- test.ranking.within_issuer_pairs and posthoc.within_issuer_by_issuer.issuers_with_pairs). The
-- within-issuer AUC (06) is computed over exactly these pairs. Printed for the first time (added later;
-- descriptive): Result 1, the pairs of each issuer.
--
-- Two ways to count the same thing: (a) per issuer, paid count times unpaid count (GROUP BY, then
-- multiply); (b) build every pair with a self-join and count the rows. The last query checks they agree.
-- Standard SQL; runs on Postgres as written.

-- Result 1: per issuer, (a).
-- HAVING filters groups after GROUP BY (WHERE filters rows before it): keep issuers with both kinds.
SELECT
    issuer,
    SUM(label)                   AS paid,
    SUM(1 - label)               AS unpaid,
    SUM(label) * SUM(1 - label)  AS pairs
FROM complaints
WHERE split = 'test'
GROUP BY issuer
HAVING SUM(label) > 0 AND SUM(1 - label) > 0
ORDER BY pairs DESC, issuer;

-- Result 2: the totals both ways, side by side.
WITH by_product AS (                             -- (a) product of grouped counts
    SELECT COUNT(*) AS issuers, SUM(pairs) AS pairs
    FROM (
        SELECT issuer, SUM(label) * SUM(1 - label) AS pairs
        FROM complaints
        WHERE split = 'test'
        GROUP BY issuer
        HAVING SUM(label) > 0 AND SUM(1 - label) > 0
    ) AS per_issuer
),
by_self_join AS (                                -- (b) one row per (paid, unpaid) pair of the same issuer
    SELECT COUNT(DISTINCT p.issuer) AS issuers, COUNT(*) AS pairs
    FROM complaints AS p
    JOIN complaints AS u
      ON u.issuer = p.issuer                     -- same issuer
     AND u.split = 'test' AND u.label = 0        -- the unpaid side
    WHERE p.split = 'test' AND p.label = 1       -- the paid side
)
SELECT
    a.issuers AS issuers_by_product,
    b.issuers AS issuers_by_self_join,
    a.pairs   AS pairs_by_product,
    b.pairs   AS pairs_by_self_join,
    CASE WHEN a.issuers = b.issuers AND a.pairs = b.pairs THEN 'agree' ELSE 'DIFFER' END AS check_result
FROM by_product AS a
CROSS JOIN by_self_join AS b;
