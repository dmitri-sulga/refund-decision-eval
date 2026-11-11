-- How often did the companies pay, in each split?
--
-- Reproduces: the relief base rate per split (results/metrics.json, base_rates), which the README table
-- prints under its held-out and DEV columns. 'test' is the held-out split.
--
-- Standard SQL. SUM(label) counts the paid complaints because label is 1 for paid and 0 otherwise.
-- The CAST makes the division a floating-point one: integer / integer truncates in SQLite and in
-- Postgres (the standard leaves it to each database; MySQL's / returns a decimal). DOUBLE PRECISION is
-- the Postgres name for a 64-bit float; SQLite reads any type name containing 'DOUB' as REAL, which is
-- the same thing.

SELECT
    split,
    COUNT(*)                                            AS complaints,
    SUM(label)                                          AS paid,
    CAST(SUM(label) AS DOUBLE PRECISION) / COUNT(*)     AS relief_base_rate
FROM complaints
GROUP BY split
ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'dev' THEN 2 WHEN 'test' THEN 3 ELSE 4 END;
