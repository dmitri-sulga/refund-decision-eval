"""The SQL in sql/ (added after the held-out run) against figures computed in Python from the same data.

Every column of every result the queries print is checked, row by row and in order, against the same figure
computed in Python (by the frozen code where it computes one: baselines.issuer_rate, Systems.predict,
metrics.point, metrics.within_group_auc, analysis.base_rates) and, where the study reports it, against
results/metrics.json; the descriptive breakdown printed for the first time (per-issuer rates, shares and pairs) is
checked against plain Python counts and against the leaves posthoc.py adds for it. Counts must match to the last
integer; rates, kappa and AUC within 1e-9 (1e-12 where the arithmetic is the same), and round to the six decimals
metrics.json keeps. None of these compares the SQL with its own output; only test_docs_show_what_the_queries_print
does, and it is a snapshot of the docs. If one of these fails, the query is wrong until shown otherwise: fix the
query, not the expected value.
"""
import json
from collections import Counter

import numpy as np
import pytest

from judge import config as C
from judge import metrics as M
from judge import sql as Q
from judge.analysis import SYSTEMS, base_rates
from judge.baselines import issuer_rate
from judge.frontpage import SHORT

METRICS = json.loads((C.RESULTS / "metrics.json").read_text())
TOL = 1e-9


@pytest.fixture(scope="module")
def world():
    from judge.data import load_records, load_splits

    S = Q.systems(load_records(), load_splits())
    con = Q.connect(S=S)
    return {"S": S, "con": con, "res": Q.run_all(con)}


def rows(world, name: str, k: int = 0) -> list[dict]:
    cols, data = world["res"][name][k]
    return [dict(zip(cols, r)) for r in data]


def same(x: float, exact: float, reported: float | None = None) -> bool:
    """Within TOL of the Python value, and rounding to the reported value."""
    ok = abs(x - exact) <= TOL
    return ok and (reported is None or round(x, 6) == reported)


def test_every_file_asks_a_question_and_runs():
    names = [p.name for p in Q.files()]
    assert names == sorted(names) and len(names) >= 8
    for p in Q.files():
        assert Q.question(p).endswith("?"), p.name
        assert Q.statements(p.read_text()), p.name


def test_base_rate_per_split(world):
    S = world["S"]
    exact = base_rates(S.records, S.splits)
    got = {r["split"]: r for r in rows(world, "01_base_rate_per_split.sql")}
    assert set(got) == set(S.splits)
    for split, r in got.items():
        y = [S.records[i]["label"] for i in S.splits[split]]
        assert r["complaints"] == len(y) == METRICS["base_rates"][split]["n"]
        assert r["paid"] == sum(y)
        assert same(r["relief_base_rate"], exact[split]["base_rate"], METRICS["base_rates"][split]["base_rate"])


def volume_order(S) -> list[tuple[str, int]]:
    """Issuers by TRAIN count, largest first; ties by the issuer's lowest TRAIN complaint ID (a number, so that the
    order does not depend on how a database compares text)."""
    n_by = Counter(S.records[i]["company"] for i in S.splits["train"])
    first = {}
    for i in S.splits["train"]:
        co = S.records[i]["company"]
        first[co] = min(first.get(co, int(i)), int(i))
    return sorted(n_by.items(), key=lambda t: (-t[1], first[t[0]]))


def test_issuer_train_rates_are_the_frozen_smoothed_rates(world):
    S = world["S"]
    _, base, rates = issuer_rate(S.splits["train"], S.records)
    n_by = Counter(S.records[i]["company"] for i in S.splits["train"])
    got = rows(world, "02_issuer_lookup.sql", 0)
    assert {r["issuer"] for r in got} == set(rates) and len(got) == len(rates)
    for r in got:
        assert r["n_train"] == n_by[r["issuer"]]
        assert abs(r["smoothed_rate"] - rates[r["issuer"]]) <= 1e-12
        assert r["lookup_says"] == ("pay" if rates[r["issuer"]] > base else "refuse")


def test_every_column_of_the_issuer_tables(world):
    """02 Result 1 and 07 Results 1 and 2, every column and the row order, against plain Python counts and the
    frozen smoothed rates (the running share is a window with a ROWS frame; a RANGE frame or a missing tie-breaker
    changes it on rows with equal TRAIN counts)."""
    S = world["S"]
    _, base, rates = issuer_rate(S.splits["train"], S.records)
    train = S.splits["train"]
    n_by = Counter(S.records[i]["company"] for i in train)
    k_by = Counter(S.records[i]["company"] for i in train if S.records[i]["label"] == 1)
    order = volume_order(S)
    lookup = rows(world, "02_issuer_lookup.sql", 0)
    assert [(r["issuer"], r["n_train"]) for r in lookup] == order
    for r in lookup:
        assert r["paid_train"] == k_by[r["issuer"]]
        assert abs(r["raw_rate"] - k_by[r["issuer"]] / n_by[r["issuer"]]) <= 1e-12
    total, run = sum(n_by.values()), 0
    vol = rows(world, "07_issuers_by_volume.sql", 0)
    for rank, (r, (co, n)) in enumerate(zip(vol, order, strict=True), 1):
        run += n
        assert (r["volume_rank"], r["issuer"], r["n_train"]) == (rank, co, n)
        assert abs(r["share_of_train"] - n / total) <= 1e-12 and abs(r["running_share"] - run / total) <= 1e-12
        assert abs(r["smoothed_rate"] - rates[co]) <= 1e-12
        assert r["lookup_says"] == ("pay" if rates[co] > base else "refuse")
    iv = METRICS["posthoc"]["issuer_volume"]                    # the descriptive share the README quotes
    assert round(vol[iv["k"] - 1]["running_share"], 6) == iv["share_of_k_largest"] and len(vol) == iv["issuers"]
    big = sorted((co for co, n in n_by.items() if n >= 100), key=lambda co: -rates[co])
    by_rate = rows(world, "07_issuers_by_volume.sql", 1)
    assert [r["issuer"] for r in by_rate] == big
    for r in by_rate:
        assert r["n_train"] == n_by[r["issuer"]] and abs(r["smoothed_rate"] - rates[r["issuer"]]) <= 1e-12
        assert r["rate_rank"] == 1 + sum(rates[co] > rates[r["issuer"]] for co in big)   # RANK: ties share a rank


def test_issuer_lookup_decides_every_held_out_complaint_as_the_frozen_code(world):
    S = world["S"]
    ids = S.splits["test"]
    pr = S.predict("issuer_rate", "test")
    want = {i: (int(d), float(p)) for i, d, p in zip(ids, pr["pred"], pr["p"])}
    got = rows(world, "02_issuer_lookup.sql", 1)
    assert [r["complaint_id"] for r in got] == sorted(ids, key=int)
    for r in got:
        assert r["decision"] == want[r["complaint_id"]][0], r["complaint_id"]
        assert abs(r["score"] - want[r["complaint_id"]][1]) <= 1e-12
        assert r["company_paid"] == S.records[r["complaint_id"]]["label"]
        assert r["issuer"] == S.records[r["complaint_id"]]["company"]
    seen = {S.records[i]["company"] for i in S.splits["train"]}
    unseen = [r for r in got if r["issuer_unseen_in_train"]]
    assert len(unseen) == sum(S.records[i]["company"] not in seen for i in ids) > 0   # the rule is exercised
    assert all(r["decision"] == 0 for r in unseen)


CELLS = {"both_pay": "relief_relief", "pays_company_did_not": "none_relief",
         "refuses_company_paid": "relief_none", "both_refuse": "none_none"}


def test_issuer_lookup_confusion_agreement_and_kappa(world):
    S = world["S"]
    got = {r["split"]: r for r in rows(world, "03_issuer_lookup_kappa.sql")}
    assert [r["split"] for r in rows(world, "03_issuer_lookup_kappa.sql")] == ["dev", "test"]   # no TRAIN row
    for split, rep in (("dev", METRICS["dev"]["issuer_rate"]), ("test", METRICS["test"]["results"]["issuer_rate"])):
        r = got[split]
        assert {k: r[k] for k in CELLS} == {k: rep["confusion"][v] for k, v in CELLS.items()}, split
        assert r["n"] == rep["n"] and r["system_pays"] == rep["confusion"]["relief_relief"] + rep["confusion"]["none_relief"]
        exact = M.point(S.labels(split), S.predict("issuer_rate", split)["pred"])
        assert same(r["agreement"], exact["accuracy"], rep["accuracy"]["value"]), split
        assert same(r["kappa"], exact["kappa"], rep["kappa"]["value"]), split


def test_every_system_on_held_out_in_one_query(world):
    S = world["S"]
    y = S.labels("test")
    name_of = {v: k for k, v in SHORT.items()}
    got = rows(world, "04_all_systems_confusion.sql")
    assert [name_of[r["system"]] for r in got] == list(SYSTEMS)
    for r in got:
        name = name_of[r["system"]]
        rep = METRICS["test"]["results"][name]
        c = rep["confusion"]
        assert r["n"] == rep["n"] and r["company_pays"] == c["relief_relief"] + c["relief_none"]
        assert (r["system_pays"], r["pays_company_did_not"], r["refuses_company_paid"]) == \
            (c["relief_relief"] + c["none_relief"], c["none_relief"], c["relief_none"]), name
        exact = M.point(y, S.predict(name, "test")["pred"])
        assert same(r["agreement"], exact["accuracy"], rep["accuracy"]["value"]), name
        assert same(r["kappa"], exact["kappa"], rep["kappa"]["value"]), name


def test_same_issuer_pairs_two_ways(world):
    S = world["S"]
    per = rows(world, "05_same_issuer_pairs.sql", 0)
    tot = rows(world, "05_same_issuer_pairs.sql", 1)[0]
    y = S.labels("test")
    co = np.array([S.records[i]["company"] for i in S.splits["test"]])
    want = {g: int((y[co == g] == 1).sum() * (y[co == g] == 0).sum()) for g in np.unique(co)}
    assert {r["issuer"]: r["pairs"] for r in per} == {g: n for g, n in want.items() if n}
    assert [r["pairs"] for r in per] == sorted((r["pairs"] for r in per), reverse=True)
    for r in per:
        assert (r["paid"], r["unpaid"]) == (int((y[co == r["issuer"]] == 1).sum()), int((y[co == r["issuer"]] == 0).sum()))
    assert {r["issuer"]: r["pairs"] for r in per} == METRICS["posthoc"]["within_issuer_by_issuer"]["pairs_by_issuer"]
    assert tot["issuers_by_product"] == tot["issuers_by_self_join"] == len(per) \
        == METRICS["posthoc"]["within_issuer_by_issuer"]["issuers_with_pairs"]
    assert tot["pairs_by_product"] == tot["pairs_by_self_join"] == sum(r["pairs"] for r in per) \
        == METRICS["test"]["ranking"]["within_issuer_pairs"] == METRICS["posthoc"]["within_issuer_by_issuer"]["pairs"]
    assert tot["check_result"] == "agree"


def test_within_issuer_auc_of_every_scored_system(world):
    S = world["S"]
    y = S.labels("test")
    co = np.array([S.records[i]["company"] for i in S.splits["test"]])
    name_of = {v: k for k, v in SHORT.items()}
    got = {name_of[r["system"]]: r for r in rows(world, "06_within_issuer_auc.sql")}
    assert set(got) == set(METRICS["test"]["ranking"]["systems"])
    for name, r in got.items():
        pr = S.predict(name, "test")
        sc = np.array(pr.get("p_raw", pr["p"]), dtype=float)
        sc = np.where(np.isnan(sc), 0.5, sc)
        exact = M.within_group_auc(y, sc, co)
        assert r["pairs"] == METRICS["test"]["ranking"]["within_issuer_pairs"]
        d = [sc[(co == g) & (y == 1)][:, None] - sc[(co == g) & (y == 0)][None, :] for g in np.unique(co)]
        assert r["wins"] == sum(float((x > 0).sum()) + 0.5 * float((x == 0).sum()) for x in d), name
        assert same(r["within_issuer_auc"], exact, METRICS["test"]["ranking"]["systems"][name]["within_issuer_auc"]["value"])


def test_issuers_by_volume_window(world):
    S = world["S"]
    facts = METRICS["issuer"]
    vol = rows(world, "07_issuers_by_volume.sql", 0)
    assert len(vol) == facts["issuers_in_train"]
    assert [r["volume_rank"] for r in vol] == list(range(1, len(vol) + 1))
    assert all(a["n_train"] >= b["n_train"] for a, b in zip(vol, vol[1:]))
    assert abs(vol[-1]["running_share"] - 1) <= 1e-12 and abs(sum(r["share_of_train"] for r in vol) - 1) <= 1e-12
    by_rate = rows(world, "07_issuers_by_volume.sql", 1)
    for r, (co, rate, n) in ((by_rate[0], facts["highest"]), (by_rate[-1], facts["lowest"])):
        assert (r["issuer"], round(r["smoothed_rate"], 6), r["n_train"]) == (co, rate, n)
    counts = rows(world, "07_issuers_by_volume.sql", 2)[0]
    assert counts["issuers_in_train"] == facts["issuers_in_train"]
    assert counts["train_complaints"] == len(S.splits["train"])
    assert counts["issuers_paying_over_half"] == facts["issuers_above_half"]


def test_dates_window(world):
    S = world["S"]
    dh = METRICS["date_halves"]
    per = {r["split"]: r for r in rows(world, "08_dates.sql", 0)}
    assert list(per) == ["train", "dev", "test"] and set(per) == set(dh["ranges"])   # no smoke-test row
    for split, rng in dh["ranges"].items():
        assert (per[split]["first_date"], per[split]["median_date"], per[split]["last_date"]) == \
            (rng["from"], rng["median"], rng["to"]), split
        assert per[split]["complaints"] == len(S.splits[split])
    dev = sorted(S.splits["dev"], key=lambda i: (S.records[i]["date_received"], int(i)))
    for r in rows(world, "08_dates.sql", 1):
        ids = dev[: len(dev) // 2] if r["half"] == "earlier" else dev[len(dev) // 2:]
        rep = dh[r["half"]]
        assert (r["complaints"], r["first_date"], r["last_date"]) == (rep["n"], rep["from"], rep["to"])
        assert same(r["relief_base_rate"], float(np.mean([S.records[i]["label"] for i in ids])), rep["base_rate"])
    window = rows(world, "08_dates.sql", 2)[0]
    dw = METRICS["posthoc"]["data_window"]
    assert (window["first_date"], window["last_date"]) == (dw["first"], dw["last"])
    assert window["complaints"] == sum(dw["splits"][k] for k in ("train", "dev", "test"))


def test_docs_show_what_the_queries_print(world):
    """A snapshot, not a check of the figures: the expected outputs in docs/sql.md are what the queries print."""
    text = Q.DOC.read_text()
    blocks = {mt.group(2): mt.group(3) for mt in Q.DOC_BLOCK_RE.finditer(text)}
    assert set(blocks) == {p.name for p in Q.files()}
    for name, body in blocks.items():
        assert body == "```text\n" + Q.file_output(world["res"][name]) + "\n```\n", \
            f"docs/sql.md is stale for {name}: run python -m judge.sql --docs"


def test_the_database_is_never_written_into_the_repository(tmp_path):
    target = C.ROOT / "results" / "tmp-sql-test.db"
    with pytest.raises(SystemExit):
        Q.connect(target)
    assert not target.exists()
    kept = tmp_path / "kept.db"                       # outside the repository, but an existing file is not overwritten
    kept.write_bytes(b"x")
    with pytest.raises(SystemExit):
        Q.connect(kept)
    assert kept.read_bytes() == b"x"
