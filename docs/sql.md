# The baselines in SQL: a walk-through

The queries in `sql/` were added after the held-out run and after the blind human labels were locked. They decide nothing and feed no reported number. Most of what they print recomputes, in SQL, figures that this study already reports from Python (the README and `results/report.md`). The rest is a descriptive breakdown printed here for the first time, added later and descriptive: each issuer's TRAIN count, paid count, raw and smoothed rate (02 Result 1, 07 Result 1), each issuer's share of TRAIN and the running share (07 Result 1), and the same-issuer pairs per issuer (05 Result 1). Each section below says which is which. The two figures of that breakdown that the README quotes (the share of TRAIN the five largest issuers hold, and the pairs of the two issuers with the most) are computed in Python as well (`posthoc.issuer_volume` and `posthoc.within_issuer_by_issuer.pairs_by_issuer` in `results/metrics.json`).

The tests in `tests/test_sql.py` check every column of every result, row by row and in order, against the same figure computed in Python from the same data (by the frozen code where it computes one) and, where the study reports it, against `results/metrics.json`: counts to the last integer, rates, kappa and AUC within 1e-9 (1e-12 where the arithmetic is the same), and rounded to the six decimals of `results/metrics.json`. One more test compares the expected outputs on this page with what the queries print; that one is a snapshot of this page, not a check of the figures. Some deliberate mistakes in a query cannot be caught on this data because they change nothing in its output (for example `>= 100` instead of `> 100` when no issuer has exactly 100 TRAIN complaints); `docs/deviations.md` lists the ones that were tried.

The queries, the loader (`src/judge/sql.py`), the tests and this walk-through were chosen and written by the AI coding assistant that wrote the rest of the code, after the held-out run and after the human labels were locked. They exist because a review by a separate AI-agent session asked what the SQL for the issuer-rate baseline and for the same-issuer pairs would be, and the author then asked for an SQL section. The walk-through explains every query clause by clause, for a reader new to SQL. The record of this addition is in `docs/deviations.md` and `results/post_run_changes.json`.

## Running them

```bash
make sql                                  # load the data, run every file in sql/, print the results
PYTHONPATH=src .venv/bin/python -m judge.sql --rows 0             # every row, not only the first sixteen
PYTHONPATH=src .venv/bin/python -m judge.sql --db /tmp/judge.db   # keep the database to explore it
sqlite3 /tmp/judge.db                     # then, for example: SELECT * FROM issuer_lookup LIMIT 5;
```

The loader uses only Python's built-in `sqlite3` module. The database lives in memory; `--db` keeps a copy in a new file, and the loader refuses a path inside the repository or a file that already exists. The only file in the repository it writes is this page, and only with `--docs`, which rewrites the expected-output blocks below.

## The tables

- **`complaints`**: one row per complaint of the study (TRAIN, DEV, held-out and the small debug set). Columns: `complaint_id`, `split` (`train`, `dev`, `test` for held-out, `debug`), `issuer` (the company field exactly as the study uses it; the study does not normalise issuer names), `company_response`, `label` (1 when the company recorded monetary relief; the loader checks it against the study's own mapping in `src/judge/data.py`) and `date_received` (an ISO date string, the only date in the records).
- **`decisions`**: one row per system and held-out complaint. Columns: `system`, `complaint_id`, `decision` (1 = the system pays) and `score` (the score the frozen ranking uses; NULL when the system has none or the model gave no usable probability). The judges and the guarded pipeline come from the answer caches; the classic baselines have no cached answers, so the loader refits them on TRAIN with the frozen code, reusing the hyperparameters recorded at the freeze (the report reruns the cross-validated search on TRAIN, which picks the same values).
- **`systems`**: the system names as the README table prints them, their order, and `scored` (1 when the system gives a ranking score, as `SCORED` in `src/judge/analysis.py` lists them).

## SQLite and Postgres

Everything is standard SQL. The independent review of this change loaded the same three tables into PostgreSQL 14 and ran every file unchanged: every statement ran and every number agreed with SQLite (within 1e-12). One thing differed: in 02 Result 1 and 07 Result 1, issuers with equal TRAIN counts came out in another order, with other `volume_rank` and `running_share`, because those queries broke ties by the issuer's name and the two databases order text differently: SQLite compares bytes (capitals first, like Python's `sorted`), while a Postgres database in the `en_US.UTF-8` locale ignores case. Both queries now break such ties by a number, the issuer's lowest TRAIN complaint ID, so the order no longer depends on how text is compared, and no figure the README or the report gives changed (the five largest issuers have different counts). Run again on PostgreSQL 14 after that change, all eight files print the same rows in the same order as SQLite. Points where the two differ:

- `integer / integer` truncates in both (the SQL standard leaves it to each database; MySQL's `/` returns a decimal), so every rate casts first: `CAST(x AS DOUBLE PRECISION) / n`. `DOUBLE PRECISION` is the Postgres name of a 64-bit float; SQLite accepts it and treats it as `REAL`.
- Text order depends on the collation (see above), so no query here relies on the order of names to fix the order of rows.
- `GROUP BY c.issuer, b.base` in 02 names a column that is constant per group; SQLite would accept it without, Postgres would not.
- Window functions need SQLite 3.25 or later; Python has shipped a newer one for years.
- Dates are stored as text in ISO form, which sorts in date order. In Postgres you would use the `DATE` type.

## 01: How often did the companies pay, in each split?

Reproduces the relief base rate per split (`base_rates` in `results/metrics.json`; the report's base-rate section gives the same four rows, including the 12 complaints used only for smoke tests).

```sql
SELECT split, COUNT(*) AS complaints, SUM(label) AS paid,
       CAST(SUM(label) AS DOUBLE PRECISION) / COUNT(*) AS relief_base_rate
FROM complaints GROUP BY split ORDER BY ...;
```

- `FROM complaints`: read every row of the table.
- `GROUP BY split`: put the rows into one bucket per split; every aggregate below is computed per bucket.
- `COUNT(*)`: the number of rows in the bucket.
- `SUM(label)`: the number of paid complaints, because `label` is 1 for paid and 0 otherwise.
- `CAST(... AS DOUBLE PRECISION) / COUNT(*)`: the share paid, as a float (see the note above on integer division).
- `ORDER BY CASE split WHEN 'train' THEN 1 ... END`: a custom sort order, TRAIN first.

<!-- BEGIN output:01_base_rate_per_split.sql -->
```text
split  complaints  paid  relief_base_rate
-----  ----------  ----  ----------------
train  8000        2407  0.300875
dev    200         60    0.3
test   250         75    0.3
debug  12          4     0.333333
```
<!-- END output:01_base_rate_per_split.sql -->

## 02: What does the issuer lookup decide for each held-out complaint?

The issuer lookup ('Issuer relief rate (added later)' in the README) never reads the complaint: it pays when the issuer paid more often than average on TRAIN. The rule, from `issuer_rate` in `src/judge/baselines.py` and `Systems.predict` in `src/judge/analysis.py`:

- the issuer's score is `(paid on TRAIN + 10 × base) / (complaints on TRAIN + 10)`, where `base` is the TRAIN relief rate; the 10 is `ISSUER_PSEUDO_COUNT`, ten pretend complaints at the average rate, so that an issuer with three complaints is not scored 0% or 100%;
- an issuer with no TRAIN complaint gets the score `base`;
- the decision is "pay" when the score is strictly greater than `base`, so an unseen issuer is always refused.

The file first creates three **views** (a view is a saved query that later queries can read like a table), then shows two results.

- `CREATE VIEW train_base AS SELECT CAST(SUM(label) AS DOUBLE PRECISION) / COUNT(*) AS base ... WHERE split = 'train'`: one row, one column: the TRAIN relief rate.
- `CREATE VIEW issuer_train_rates AS SELECT c.issuer, COUNT(*), SUM(c.label), (SUM(c.label) + 10 * b.base) / (COUNT(*) + 10), MIN(CAST(c.complaint_id AS INTEGER))`: one row per issuer seen in TRAIN, with its smoothed rate and its lowest TRAIN complaint ID (used only to order issuers with equal counts).
- `FROM complaints AS c CROSS JOIN train_base AS b`: `CROSS JOIN` pairs every row on the left with every row on the right; the right side has one row, so it simply attaches `base` to every complaint.
- `WHERE c.split = 'train'`: only TRAIN complaints build the rates (DEV and held-out never do).
- `GROUP BY c.issuer, b.base`: one bucket per issuer.
- `CREATE VIEW issuer_lookup AS ... FROM complaints AS c CROSS JOIN train_base AS b LEFT JOIN issuer_train_rates AS r ON r.issuer = c.issuer`: every DEV and held-out complaint next to its issuer's TRAIN row.
- `LEFT JOIN`: keep every complaint on the left even when no TRAIN row matches; the right-hand columns are then NULL. An inner `JOIN` would silently drop those complaints.
- `COALESCE(r.smoothed_rate, b.base)`: the first value that is not NULL, so an unseen issuer gets `base`.
- `CASE WHEN ... > b.base THEN 1 ELSE 0 END`: the decision.
- `CASE WHEN r.issuer IS NULL THEN 1 ELSE 0 END AS issuer_unseen_in_train`: flags the complaints that used the unseen-issuer rule (`= NULL` would never be true; NULL is tested with `IS NULL`).
- Result 1 lists each issuer's TRAIN count, paid count, raw rate and smoothed rate, largest issuers first (`ORDER BY n_train DESC, lowest_complaint_id`; the second key breaks ties with a number, so the order is fixed and the same in every database). The report gives only the number of issuers, the lowest and the highest smoothed rate among issuers with at least 100 TRAIN complaints, and the count above one half (07); the table itself is printed here for the first time (added later; descriptive). The lookup decides on the smoothed rate; the raw rate is beside it so the two can be compared.
- Result 2 lists the decision for each held-out complaint, in complaint-ID order (`CAST(complaint_id AS INTEGER)`, because the IDs are stored as text and text sorts '10' before '9').

<!-- BEGIN output:02_issuer_lookup.sql -->
```text
issuer                                 n_train  paid_train  raw_rate   smoothed_rate  lookup_says
-------------------------------------  -------  ----------  ---------  -------------  -----------
CITIBANK, N.A.                         1126     548         0.486679   0.485043       pay
JPMORGAN CHASE & CO.                   850      165         0.194118   0.195359       refuse
CAPITAL ONE FINANCIAL CORPORATION      820      204         0.24878    0.249408       refuse
SYNCHRONY FINANCIAL                    616      188         0.305195   0.305126       pay
BANK OF AMERICA, NATIONAL ASSOCIATION  581      370         0.636833   0.631148       pay
GOLDMAN SACHS BANK USA                 553      255         0.461121   0.458275       pay
AMERICAN EXPRESS COMPANY               466      138         0.296137   0.296237       refuse
DISCOVER BANK                          380      48          0.126316   0.130792       refuse
WELLS FARGO & COMPANY                  355      71          0.2        0.202764       refuse
BARCLAYS BANK DELAWARE                 298      45          0.151007   0.155873       refuse
Bread Financial Holdings, Inc.         210      89          0.42381    0.418222       pay
Chime Financial Inc                    168      13          0.077381   0.0899368      refuse
U.S. BANCORP                           161      32          0.198758   0.20473        refuse
TD BANK US HOLDING COMPANY             135      50          0.37037    0.365578       pay
NAVY FEDERAL CREDIT UNION              120      28          0.233333   0.238529       refuse
Block, Inc.                            104      6           0.0576923  0.0790241      refuse
... 148 more rows

complaint_id  issuer                             score      decision  company_paid  issuer_unseen_in_train
------------  ---------------------------------  ---------  --------  ------------  ----------------------
7496236       AMERICAN EXPRESS COMPANY           0.296237   0         0             0
7520893       CAPITAL ONE FINANCIAL CORPORATION  0.249408   0         0             0
7543478       WELLS FARGO & COMPANY              0.202764   0         0             0
7548348       CAPITAL ONE FINANCIAL CORPORATION  0.249408   0         0             0
7549514       GOLDMAN SACHS BANK USA             0.458275   1         0             0
7553346       Chime Financial Inc                0.0899368  0         1             0
7559174       SYNCHRONY FINANCIAL                0.305126   1         1             0
7586825       SYNCHRONY FINANCIAL                0.305126   1         0             0
7621510       CITIBANK, N.A.                     0.485043   1         0             0
7635207       DISCOVER BANK                      0.130792   0         0             0
7691084       CAPITAL ONE FINANCIAL CORPORATION  0.249408   0         0             0
7709381       SYNCHRONY FINANCIAL                0.305126   1         1             0
7711411       Kikoff Inc.                        0.273523   0         0             0
7722317       JPMORGAN CHASE & CO.               0.195359   0         1             0
7731702       Paypal Holdings, Inc               0.298161   0         0             0
7740485       CITIBANK, N.A.                     0.485043   1         0             0
... 234 more rows
```
<!-- END output:02_issuer_lookup.sql -->

## 03: How often does the issuer lookup agree with the record, and how much better than chance?

Reproduces the issuer lookup's confusion counts, agreement and Cohen's kappa on DEV and held-out (`dev.issuer_rate` and `test.results.issuer_rate` in `results/metrics.json`; the README table's row 'Issuer relief rate').

Cohen's kappa is agreement corrected for chance: `(agreement − chance) / (1 − chance)`, where `chance = p_record × p_system + (1 − p_record) × (1 − p_system)` is the agreement two raters would reach by luck if they paid as often as these two do, independently. A system that always refuses agrees with the record most of the time and still has kappa 0.

- `WITH cells AS (...), rates AS (...), chance AS (...)`: a chain of named subqueries (common table expressions, CTEs); each can read the ones before it, which keeps every step small.
- `SUM(CASE WHEN label = 1 AND decision = 1 THEN 1 ELSE 0 END) AS both_pay`: counts the rows that meet the condition (conditional aggregation). The four sums are the four cells of the 2×2 table.
- `FROM issuer_lookup GROUP BY split`: one row per split, DEV and held-out.
- `rates`: agreement is the diagonal over n; `p_record_pays` and `p_system_pays` are the two margins over n.
- `cells.*`: all columns of `cells`, carried along.
- `CASE WHEN chance_agreement < 1 THEN ... END`: kappa is undefined (NULL) when the record and the system give the same single answer on every case (chance agreement is then 1); a `CASE` without `ELSE` returns NULL.
- The arithmetic is in the same order as `src/judge/metrics.py`, so the floats agree to the last digit.

<!-- BEGIN output:03_issuer_lookup_kappa.sql -->
```text
split  n    both_pay  pays_company_did_not  refuses_company_paid  both_refuse  system_pays  agreement  kappa
-----  ---  --------  --------------------  --------------------  -----------  -----------  ---------  --------
dev    200  43        49                    17                    91           92           0.67       0.318182
test   250  47        44                    28                    131          91           0.712      0.35368
```
<!-- END output:03_issuer_lookup_kappa.sql -->

## 04: For every system, how often does it pay, and in which direction does it err?

Reproduces four columns of the README table for every system at once: 'Matches the record', 'Pays (of these, the company did not pay)', 'Refuses where the company paid' and the held-out kappa point estimate (`test.results.<system>` in `results/metrics.json`).

- `FROM decisions AS d JOIN complaints AS c ON c.complaint_id = d.complaint_id`: each decision next to the company's recorded outcome. An inner `JOIN` is right here: every decision has its complaint.
- `WHERE c.split = 'test'`: held-out only (the table holds only held-out decisions; the filter states it).
- `GROUP BY d.system`: one bucket per system, so one pass gives every system's 2×2 table.
- `SUM(d.decision)`: how many the system pays; `SUM(CASE WHEN d.decision = 1 AND c.label = 0 ...)`: of those, how many the company did not pay; `SUM(CASE WHEN d.decision = 0 AND c.label = 1 ...)`: refusals where the company paid; `SUM(CASE WHEN d.decision = c.label ...)`: matches.
- `chance`: the chance agreement, as in 03; the final `SELECT` computes kappa from it as in 03.
- `JOIN systems AS s ON s.system = c.system ... ORDER BY s.position`: the README's names and row order.

<!-- BEGIN output:04_all_systems_confusion.sql -->
```text
system                                     n    agreement  system_pays  pays_company_did_not  refuses_company_paid  company_pays  kappa
-----------------------------------------  ---  ---------  -----------  --------------------  --------------------  ------------  --------
Always no relief                           250  0.7        0            0                     75                    75            0
Keyword rule (added later)                 250  0.688      33           18                    60                    75            0.115646
Company rule (planned)                     250  0.732      14           3                     64                    75            0.168734
Issuer relief rate (added later)           250  0.712      91           44                    28                    75            0.35368
TF-IDF, issuer names masked (added later)  250  0.684      64           34                    45                    75            0.214712
TF-IDF + logistic regression               250  0.724      66           30                    39                    75            0.319527
LLM judge v1 (plain rubric)                250  0.468      176          117                   16                    75            0.085282
LLM judge v2 (predicts the record)         250  0.712      39           18                    54                    75            0.205298
Guarded pipeline (v2 facts -> fact model)  250  0.736      19           5                     61                    75            0.200969
```
<!-- END output:04_all_systems_confusion.sql -->

## 05: Within one issuer, how many paid × unpaid pairs are there?

The README says that comparing only complaints about the same issuer leaves few pairs from few issuers. This reproduces both counts (`test.ranking.within_issuer_pairs` and `posthoc.within_issuer_by_issuer.issuers_with_pairs`), two ways. Result 1, the pairs of each issuer, was not reported before (added later; descriptive): two issuers carry more than half of the pairs.

(a) Product of grouped counts:

- `SUM(label) AS paid, SUM(1 - label) AS unpaid`: per issuer, the paid and the unpaid held-out complaints.
- `SUM(label) * SUM(1 - label) AS pairs`: every paid one can be paired with every unpaid one.
- `HAVING SUM(label) > 0 AND SUM(1 - label) > 0`: keep only issuers with both kinds. `HAVING` filters buckets after `GROUP BY`; `WHERE` filters rows before it.

(b) Self-join:

- `FROM complaints AS p JOIN complaints AS u ON u.issuer = p.issuer AND u.split = 'test' AND u.label = 0 WHERE p.split = 'test' AND p.label = 1`: the same table twice under two names, `p` for the paid side and `u` for the unpaid side; each output row is one pair from the same issuer.
- `COUNT(*)`: the number of pairs; `COUNT(DISTINCT p.issuer)`: the number of issuers that have one.

The last query puts (a) and (b) side by side with `CROSS JOIN` (each side is one row) and says whether they agree. The product is cheaper; the self-join is what 06 needs, because it gives the pairs themselves.

<!-- BEGIN output:05_same_issuer_pairs.sql -->
```text
issuer                                 paid  unpaid  pairs
-------------------------------------  ----  ------  -----
CITIBANK, N.A.                         23    15      345
JPMORGAN CHASE & CO.                   7     35      245
CAPITAL ONE FINANCIAL CORPORATION      5     25      125
SYNCHRONY FINANCIAL                    8     15      120
AMERICAN EXPRESS COMPANY               3     15      45
BANK OF AMERICA, NATIONAL ASSOCIATION  11    3       33
WELLS FARGO & COMPANY                  4     8       32
GOLDMAN SACHS BANK USA                 4     6       24
NAVY FEDERAL CREDIT UNION              2     8       16
DISCOVER BANK                          1     12      12
U.S. BANCORP                           2     5       10
BARCLAYS BANK DELAWARE                 1     6       6
Chime Financial Inc                    2     1       2

issuers_by_product  issuers_by_self_join  pairs_by_product  pairs_by_self_join  check_result
------------------  --------------------  ----------------  ------------------  ------------
13                  13                    1015              1015                agree
```
<!-- END output:05_same_issuer_pairs.sql -->

## 06: Within one issuer, how often does a system score a paid complaint above an unpaid one?

Reproduces the within-issuer AUC point estimate of every system with a score (`test.ranking.systems.<system>.within_issuer_auc` in `results/metrics.json`). The interval around it comes from resampling (bootstrap) and is not reproduced in SQL.

- `scored`: each held-out decision of a system with a ranking score (`JOIN systems ... WHERE s.scored = 1`) with its issuer, the record and the score; `COALESCE(d.score, 0.5)` gives a missing score the value 0.5, as the frozen ranking does (on the held-out split no scored system has a missing score, so this rule is not exercised).
- `pairs`: the self-join of 05, once per system (`ON u.system = p.system AND u.issuer = p.issuer AND u.label = 0`, `WHERE p.label = 1`).
- `CASE WHEN p.score > u.score THEN 1.0 WHEN p.score = u.score THEN 0.5 ELSE 0.0 END AS win`: a pair counts 1 if the paid complaint is scored higher, a half on a tie.
- `SUM(win) / COUNT(*)`: the share of pairs won, pooled over all issuers. `1.0`, `0.5` and `0.0` are not integer literals (REAL in SQLite, NUMERIC in Postgres), so `SUM(win)` is not of integer type and the division is not truncated.
- The issuer lookup gives every complaint of one issuer the same score, so within an issuer every pair is a tie and its AUC is exactly one half: the lookup cannot order complaints inside one company.

<!-- BEGIN output:06_within_issuer_auc.sql -->
```text
system                                     pairs  wins   within_issuer_auc
-----------------------------------------  -----  -----  -----------------
Issuer relief rate (added later)           1015   507.5  0.5
TF-IDF, issuer names masked (added later)  1015   612    0.602956
TF-IDF + logistic regression               1015   586    0.57734
LLM judge v1 (plain rubric)                1015   600    0.591133
LLM judge v2 (predicts the record)         1015   617    0.607882
Guarded pipeline (v2 facts -> fact model)  1015   611    0.60197
```
<!-- END output:06_within_issuer_auc.sql -->

## 07: Which issuers carry the volume, and does the lookup pay or refuse them?

Reproduces the issuer facts of the full report (`issuer` in `results/metrics.json`): the number of issuers in TRAIN, the highest and lowest smoothed rate among issuers with at least 100 TRAIN complaints (the report calls it the smoothed relief rate, the issuer lookup's score), and how many issuers with at least 20 TRAIN complaints paid more than half of them (on the raw rate, as the planned company rule does). Result 1, each issuer's share of TRAIN and the running share, was not reported before (added later; descriptive); the README quotes one figure from it, the share the five largest issuers hold. A window function computes a value for each row from other rows without collapsing them as `GROUP BY` does.

- `ROW_NUMBER() OVER (ORDER BY n_train DESC, lowest_complaint_id)`: numbers the issuers from the largest down; issuers with equal counts are ordered by their lowest TRAIN complaint ID (02).
- `SUM(n_train) OVER ()`: the total over all rows (an empty window is the whole result), repeated on every row, so each issuer's share is `n_train / total`.
- `SUM(n_train) OVER (ORDER BY n_train DESC, lowest_complaint_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`: a running total, from the first row to this one; divided by the total it gives the running share (the last row reaches 1).
- `ROWS BETWEEN ...` is written out because the default frame (`RANGE`) treats rows with equal sort keys as one step. Here `lowest_complaint_id` already makes every key unique, so the two give the same result; without that second key, `RANGE` would give every issuer with the same count the running total of the whole group (on this data, 119 of the 164 rows would change), and a test checks every row.
- Result 2: `RANK() OVER (ORDER BY smoothed_rate DESC)` among issuers with `WHERE n_train >= 100`; the window runs after `WHERE`, so the rank is among those issuers only. `RANK` gives tied rows the same rank; `ROW_NUMBER` would not.
- Result 3: plain aggregates; `SUM(CASE WHEN n_train >= 20 AND ... > 0.5 THEN 1 ELSE 0 END)` counts the issuers the planned company rule would pay.

<!-- BEGIN output:07_issuers_by_volume.sql -->
```text
volume_rank  issuer                                 n_train  share_of_train  running_share  smoothed_rate  lookup_says
-----------  -------------------------------------  -------  --------------  -------------  -------------  -----------
1            CITIBANK, N.A.                         1126     0.14075         0.14075        0.485043       pay
2            JPMORGAN CHASE & CO.                   850      0.10625         0.247          0.195359       refuse
3            CAPITAL ONE FINANCIAL CORPORATION      820      0.1025          0.3495         0.249408       refuse
4            SYNCHRONY FINANCIAL                    616      0.077           0.4265         0.305126       pay
5            BANK OF AMERICA, NATIONAL ASSOCIATION  581      0.072625        0.499125       0.631148       pay
6            GOLDMAN SACHS BANK USA                 553      0.069125        0.56825        0.458275       pay
7            AMERICAN EXPRESS COMPANY               466      0.05825         0.6265         0.296237       refuse
8            DISCOVER BANK                          380      0.0475          0.674          0.130792       refuse
9            WELLS FARGO & COMPANY                  355      0.044375        0.718375       0.202764       refuse
10           BARCLAYS BANK DELAWARE                 298      0.03725         0.755625       0.155873       refuse
11           Bread Financial Holdings, Inc.         210      0.02625         0.781875       0.418222       pay
12           Chime Financial Inc                    168      0.021           0.802875       0.0899368      refuse
13           U.S. BANCORP                           161      0.020125        0.823          0.20473        refuse
14           TD BANK US HOLDING COMPANY             135      0.016875        0.839875       0.365578       pay
15           NAVY FEDERAL CREDIT UNION              120      0.015           0.854875       0.238529       refuse
16           Block, Inc.                            104      0.013           0.867875       0.0790241      refuse
... 148 more rows

rate_rank  issuer                                 n_train  smoothed_rate
---------  -------------------------------------  -------  -------------
1          BANK OF AMERICA, NATIONAL ASSOCIATION  581      0.631148
2          CITIBANK, N.A.                         1126     0.485043
3          GOLDMAN SACHS BANK USA                 553      0.458275
4          Bread Financial Holdings, Inc.         210      0.418222
5          TD BANK US HOLDING COMPANY             135      0.365578
6          SYNCHRONY FINANCIAL                    616      0.305126
7          AMERICAN EXPRESS COMPANY               466      0.296237
8          CAPITAL ONE FINANCIAL CORPORATION      820      0.249408
9          NAVY FEDERAL CREDIT UNION              120      0.238529
10         U.S. BANCORP                           161      0.20473
11         WELLS FARGO & COMPANY                  355      0.202764
12         JPMORGAN CHASE & CO.                   850      0.195359
13         BARCLAYS BANK DELAWARE                 298      0.155873
14         DISCOVER BANK                          380      0.130792
15         Chime Financial Inc                    168      0.0899368
16         Block, Inc.                            104      0.0790241

issuers_in_train  train_complaints  issuers_paying_over_half
----------------  ----------------  ------------------------
164               8000              1
```
<!-- END output:07_issuers_by_volume.sql -->

## 08: When were the complaints received, and did the payout rate move over time on DEV?

The split is random, not by date, so the full report also scores the earlier and the later half of DEV apart. This reproduces the date ranges per split and the two halves of DEV (`date_halves` in `results/metrics.json`) and the date window of the records (`posthoc.data_window`).

- `ROW_NUMBER() OVER (PARTITION BY split ORDER BY date_received) AS pos`: numbers the complaints by date within each split; `PARTITION BY` restarts the numbering for each split.
- `COUNT(*) OVER (PARTITION BY split) AS n`: the size of the row's split, on every row.
- `WHERE split IN ('train', 'dev', 'test')`: the 12 complaints used only for smoke tests are left out, as in the report.
- `MAX(CASE WHEN pos = n / 2 + 1 THEN date_received END)`: picks the middle date (the same position Python's `dates[n // 2]` takes); the `CASE` is NULL on every other row, and `MAX` ignores NULLs. For an even n there are two middle dates and this takes the later one, as the Python does; Postgres's `PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY date_received)` takes the earlier one and would print 2025-01-05 for DEV instead of 2025-01-08. Every n here is even.
- Result 2: `ROW_NUMBER() OVER (ORDER BY date_received, CAST(complaint_id AS INTEGER))` numbers DEV by date, ties broken by complaint ID as `date_halves` in `src/judge/analysis.py` does; `CASE WHEN pos <= n / 2 THEN 'earlier' ELSE 'later' END` cuts it in two; the outer `GROUP BY half` gives each half's size, dates and relief rate.
- Result 3: the first and last date over TRAIN, DEV and held-out.

<!-- BEGIN output:08_dates.sql -->
```text
split  complaints  first_date  median_date  last_date
-----  ----------  ----------  -----------  ----------
train  8000        2023-08-25  2025-01-06   2025-12-31
dev    200         2023-08-30  2025-01-08   2025-12-31
test   250         2023-09-04  2024-12-16   2025-12-29

half     complaints  first_date  last_date   relief_base_rate
-------  ----------  ----------  ----------  ----------------
earlier  100         2023-08-30  2025-01-05  0.28
later    100         2025-01-08  2025-12-31  0.32

first_date  last_date   complaints
----------  ----------  ----------
2023-08-25  2025-12-31  8450
```
<!-- END output:08_dates.sql -->

## What is not reproduced in SQL

- **The intervals.** Every interval in the study is a bootstrap (complaints or issuers drawn with replacement by NumPy's random generator with a fixed seed) or a Wilson interval. A Wilson interval is a formula and could be written in SQL; the bootstrap intervals could only be matched by importing NumPy's draws, which would not be SQL doing the work.
- **The systems themselves.** The decisions of every system except the issuer lookup are loaded, not computed: they come from a fitted model, a model answer or a rule in the Python code. Only the issuer lookup is rebuilt from scratch in SQL (02), and its decisions match the loaded ones complaint by complaint. The company rule could be rebuilt the same way (the raw TRAIN rate, the overall rate for issuers with fewer than 20 TRAIN complaints, threshold one half) but was not.
