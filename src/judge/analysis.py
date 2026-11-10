"""All computations behind the report. Pure functions of data/ and the caches."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter

import numpy as np

from . import config as C
from . import freeze
from . import metrics as M
from .baselines import (COST_FILE, company_rule, first_commit_grid, fit_lr, fit_masked_lr, issuer_pattern,
                        issuer_rate, keyword_rule, learning_curve, lr_predict, masked_text, top_features)
from .bench import compare as compare_concurrency
from .bench import load_replays as load_concurrency_replays
from .cues import amounts, cue_table
from .data import PAIRS_FILE, PROVENANCE_FILE, load_records, load_splits
from .defence import FactModel, cross_fitted, filtered_item_id, sanitize
from .errors import ANNOTATION_DIR, direction, disagreements, load_annotations, validate
from .gaming import (ATTACKS, CONTROLS, INJECTION_PARAPHRASES, apply_attack, apply_control,
                     attacked_item_id, gaming_ids)
from .judgments import available, load, parse
from .llm import Cache
from .prompts import visible_narrative
from .replay import REPLAY_FILE, replay_key

CEILING_PAIR_COSINE = 0.9
JUDGE_VERSIONS = ("v1", "v2")
FACTS_VERSION = freeze.GUARDED_FACTS_VERSION
SYSTEMS = (("majority", "keyword_rule", "company_rule", "issuer_rate", "tfidf_masked", "tfidf_lr")
           + JUDGE_VERSIONS + ("guarded",))
SCORED = ("issuer_rate", "tfidf_masked", "tfidf_lr") + JUDGE_VERSIONS + ("guarded",)   # systems with a score
PAIRS = (("v1", "v2"), ("v2", "guarded"), ("tfidf_lr", "v1"), ("tfidf_lr", "v2"), ("issuer_rate", "v1"),
         ("issuer_rate", "v2"), ("company_rule", "v1"), ("company_rule", "v2"), ("tfidf_masked", "tfidf_lr"))
CONFIDENT = ("tfidf_lr",) + JUDGE_VERSIONS + ("guarded",)   # systems with a confidence signal


def settings() -> dict:
    return {
        "model": C.MODEL, "options": C.OPTIONS, "max_narrative_chars": C.MAX_NARRATIVE_CHARS,
        "sizes": {k: len(v) for k, v in load_splits().items()}, "split_seed": C.SPLIT_SEED,
        "bootstrap_b": C.BOOTSTRAP_B, "bootstrap_seed": C.BOOTSTRAP_SEED,
        "target_agreement": C.TARGET_AGREEMENT, "min_automated": C.MIN_AUTOMATED,
        "n_gaming": C.N_GAMING, "n_attacks": len(ATTACKS), "n_controls": len(CONTROLS),
        "gaming_seed": C.GAMING_SEED,
        "near_dup_cosine": C.NEAR_DUP_COSINE, "ceiling_pair_cosine": CEILING_PAIR_COSINE,
        "label_map": C.LABEL_MAP,
    }


def environment(cache: Cache) -> dict | None:
    envs = {(r["ollama_version"], r["model_digest"]) for r in cache.records.values()}
    if not envs:
        return None
    if len(envs) > 1:
        raise RuntimeError(f"cache mixes model versions: {envs}")
    v, d = envs.pop()
    return {"ollama_version": v, "model_digest": d}


def data_summary() -> dict:
    p = json.loads(PROVENANCE_FILE.read_text())
    return {"funnel": p["funnel"], "status_counts": p["status_counts_in_slice_with_narrative"],
            "label_map": p["label_map"], "mirror_rows_total": p["extract"]["mirror_rows_total"],
            "slice_rows": p["extract"]["slice_rows"]}


def base_rates(records, splits) -> dict:
    out = {}
    for k in ("train", "dev", "test", "debug"):
        y = [records[i]["label"] for i in splits[k]]
        out[k] = {"n": len(y), "base_rate": float(np.mean(y))}
    return out


# ------------------------------------------------------------------ systems
class Systems:
    """Predictions (and probabilities of relief) of every system on a split."""

    def __init__(self, records, splits, cache, baseline=None, lr_params=None):
        self.records, self.splits, self.cache = records, splits, cache
        self._baseline = baseline
        self.lr_params = lr_params        # given: no hyperparameter search (frozen values are reused)

    def baseline(self) -> dict:
        """Classic baselines, fit on TRAIN only (once)."""
        if self._baseline is None:
            lr, params, cv = fit_lr(self.splits["train"], self.records, self.lr_params)
            rule, _ = company_rule(self.splits["train"], self.records)
            keyword, chosen, candidates = keyword_rule(self.splits["train"], self.records)
            rate, base, rates = issuer_rate(self.splits["train"], self.records)
            pattern, words = issuer_pattern(self.splits["train"], self.records)
            self._baseline = {"lr": lr, "params": params, "cv": cv, "rule": rule, "keyword": keyword,
                              "keyword_chosen": chosen, "keyword_candidates": candidates,
                              "issuer_rate": rate, "issuer_base": base, "issuer_rates": rates,
                              "issuer_pattern": pattern, "issuer_words": words,
                              "masked": fit_masked_lr(self.splits["train"], self.records, params, pattern)}
        return self._baseline

    def labels(self, split):
        return np.array([self.records[i]["label"] for i in self.splits[split]])

    # -------------------------------------------------- guarded pipeline
    def _facts(self, ids):
        """Parsed v2 responses to the filtered text of each complaint (the filter
        leaves a complaint unchanged unless it flags it), and the flags."""
        keys, flags = [], []
        for i in ids:
            text = visible_narrative(self.records[i]["narrative"])
            clean, flagged = sanitize(text)
            keys.append(i if clean == text else filtered_item_id(i))
            flags.append(flagged)
        if not available(FACTS_VERSION, keys, self.cache):
            return None
        parsed = load(FACTS_VERSION, keys, self.cache)
        return [parsed[k] for k in keys], np.array(flags)

    def fact_model(self) -> FactModel:
        if not hasattr(self, "_fm"):
            dev = self.splits["dev"]
            parsed, _ = self._facts(dev)
            self._fm = FactModel().fit([x["fields"] for x in parsed], [self.records[i] for i in dev],
                                       self.labels("dev"))
        return self._fm

    def guarded(self, split):
        ids = self.splits[split]
        got = self._facts(ids)
        if got is None:
            return None
        parsed, flags = got
        valid = np.array([x["valid"] for x in parsed])
        facts = [x["fields"] for x in parsed]
        recs = [self.records[i] for i in ids]
        if split == "dev":   # out-of-fold: no complaint is scored by a model that saw its label
            p = cross_fitted(facts, recs, self.labels("dev"))
        else:
            p = self.fact_model().predict_proba(facts, recs)
        pred = (p >= 0.5).astype(int)
        pred[~valid] = 0                     # invalid model output: scored as no relief
        p_auto = p.copy()
        p_auto[flags | ~valid] = np.nan      # flagged or invalid: routed to a human
        return {"pred": pred, "p": p_auto, "p_raw": p, "invalid": int((~valid).sum()),
                "flagged": int(flags.sum())}

    def predict(self, name: str, split: str) -> dict | None:
        ids = self.splits[split]
        n = len(ids)
        if name == "guarded":
            return self.guarded(split)
        if name == "majority":
            return {"pred": np.zeros(n, int), "p": np.full(n, np.nan), "invalid": 0}
        if name == "keyword_rule":
            rule = self.baseline()["keyword"]
            return {"pred": np.array([rule(self.records[i]) for i in ids]),
                    "p": np.full(n, np.nan), "invalid": 0}
        if name == "company_rule":
            rule = self.baseline()["rule"]
            return {"pred": np.array([rule(self.records[i]) for i in ids]),
                    "p": np.full(n, np.nan), "invalid": 0}
        if name == "tfidf_lr":
            pred, p = lr_predict(self.baseline()["lr"], ids, self.records)
            return {"pred": pred, "p": p.astype(float), "invalid": 0}
        if name == "issuer_rate":
            b = self.baseline()
            p = np.array([b["issuer_rate"](self.records[i]) for i in ids])
            return {"pred": (p > b["issuer_base"]).astype(int), "p": p, "invalid": 0}
        if name == "tfidf_masked":
            b = self.baseline()
            p = b["masked"].predict_proba([masked_text(self.records[i], b["issuer_pattern"]) for i in ids])[:, 1]
            return {"pred": (p >= 0.5).astype(int), "p": p.astype(float), "invalid": 0}
        if name in JUDGE_VERSIONS:
            if not available(name, ids, self.cache):
                return None
            parsed = load(name, ids, self.cache)
            pr = [parsed[i] for i in ids]
            p_tok = np.array([x["p_token"] for x in pr], float)
            valid = np.array([x["valid"] for x in pr])
            p_tok[~valid] = np.nan
            return {"pred": np.array([x["pred"] for x in pr]), "p": p_tok,
                    "p_verbal": np.array([x["p_verbal"] for x in pr], float),
                    "invalid": int((~valid).sum())}
        raise KeyError(name)


def split_results(S: Systems, split: str, names) -> dict:
    y = S.labels(split)
    out, preds = {}, {}
    for name in names:
        pr = S.predict(name, split)
        if pr is None:
            continue
        r = M.bootstrap(y, pr["pred"])
        r["invalid"] = pr["invalid"]
        out[name] = r
        preds[name] = pr["pred"]
    for a, b in PAIRS:
        if a in preds and b in preds:
            d = M.paired_difference(y, preds[a], preds[b])
            ra, rb = preds[a] == y, preds[b] == y
            d["transitions"] = {
                "both_agree": int((ra & rb).sum()), "both_disagree": int((~ra & ~rb).sum()),
                "only_first_agrees": int((ra & ~rb).sum()), "only_second_agrees": int((~ra & rb).sum()),
                # what the second system made worse, by recorded outcome
                "lost_relief_cases": int((ra & ~rb & (y == 1)).sum()),
                "lost_no_relief_cases": int((ra & ~rb & (y == 0)).sum()),
                "gained_relief_cases": int((~ra & rb & (y == 1)).sum()),
                "gained_no_relief_cases": int((~ra & rb & (y == 0)).sum())}
            out[f"{b}_minus_{a}"] = d
    return out


def ranking(S: Systems, split: str) -> dict:
    """Threshold-free reading (added after DEV results): how well each score
    orders complaints, over all pairs and over pairs answered by the same
    issuer. Intervals: bootstrap over complaints."""
    y = S.labels(split)
    co = np.array([S.records[i]["company"] for i in S.splits[split]])
    scores = {}
    for name in SCORED:
        pr = S.predict(name, split)
        if pr is not None:
            sc = np.array(pr.get("p_raw", pr["p"]), dtype=float)
            scores[name] = np.where(np.isnan(sc), 0.5, sc)
    out = {"systems": {}, "differences": {}, "within_issuer_pairs": int(sum(
        (y[co == g] == 1).sum() * (y[co == g] == 0).sum() for g in np.unique(co)))}
    for name, sc in scores.items():
        out["systems"][name] = {
            "auc": M.bootstrap_stat(lambda i, sc=sc: M.auc(y[i], sc[i]), len(y)),
            "within_issuer_auc": M.bootstrap_stat(lambda i, sc=sc: M.within_group_auc(y[i], sc[i], co[i]), len(y))}
    for a, b in (("tfidf_lr", "v1"), ("tfidf_lr", "v2"), ("issuer_rate", "tfidf_lr"), ("tfidf_masked", "tfidf_lr")):
        if a in scores and b in scores:
            sa, sb = scores[a], scores[b]
            out["differences"][f"{a}_minus_{b}"] = {
                "auc": M.bootstrap_stat(lambda i: M.auc(y[i], sa[i]) - M.auc(y[i], sb[i]), len(y)),
                "within_issuer_auc": M.bootstrap_stat(
                    lambda i: M.within_group_auc(y[i], sa[i], co[i]) - M.within_group_auc(y[i], sb[i], co[i]), len(y))}
    return out


def repeated_split_half(S: Systems, names, B: int = 1000, seed: int = C.BOOTSTRAP_SEED) -> dict:
    """The operating rule applied to one random half of DEV and scored on the
    other, many times: how often a threshold exists, and what it then delivers."""
    y = S.labels("dev")
    n = len(y)
    out = {}
    for name in names:
        pr = S.predict(name, "dev")
        if pr is None:
            continue
        conf = M.confidence(pr["pred"], pr["p"])
        rng = np.random.default_rng(seed)
        found, agreements, meets = 0, [], 0
        for _ in range(B):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            t = M.choose_threshold(y[a], pr["pred"][a], conf[a])
            if t is None:
                continue
            found += 1
            auto = ~np.isnan(conf[b]) & (conf[b] >= t - M.THRESHOLD_TOLERANCE)
            if auto.any():
                agr = float(np.mean(y[b][auto] == pr["pred"][b][auto]))
                agreements.append(agr)
                meets += agr >= C.TARGET_AGREEMENT
        out[name] = {"splits": B, "threshold_found": found,
                     "median_agreement_other_half": float(np.median(agreements)) if agreements else math.nan,
                     "other_half_reaches_target": int(meets)}
    return out


def date_halves(S: Systems, names) -> dict:
    """Secondary analysis: the frozen split is random, not by date, so agreement
    is also reported separately for the earlier and the later half of DEV
    (by the date the complaint was received)."""
    dev = sorted(S.splits["dev"], key=lambda i: (S.records[i]["date_received"], int(i)))
    half = len(dev) // 2
    pos = {i: k for k, i in enumerate(S.splits["dev"])}
    out = {}
    for label, ids in (("earlier", dev[:half]), ("later", dev[half:])):
        idx = np.array([pos[i] for i in ids])
        y = S.labels("dev")[idx]
        row = {"n": len(ids), "from": S.records[ids[0]]["date_received"],
               "to": S.records[ids[-1]]["date_received"], "base_rate": float(y.mean()), "systems": {}}
        for name in names:
            pr = S.predict(name, "dev")
            if pr is not None:
                b = M.bootstrap(y, pr["pred"][idx])
                row["systems"][name] = {"kappa": b["kappa"], "accuracy": b["accuracy"]}
        out[label] = row
    dates = {k: sorted(S.records[i]["date_received"] for i in S.splits[k]) for k in ("train", "dev", "test")}
    out["ranges"] = {k: {"from": v[0], "median": v[len(v) // 2], "to": v[-1]} for k, v in dates.items()}
    return out


def routing(S: Systems, auto_dev: dict) -> dict:
    """The DEV operating point read as three outcomes: a confident 'no relief' is
    decided automatically, a confident 'relief' is a proposal a person approves
    (a payout is the costlier error to undo), everything else is escalated to a
    person. The two kinds of automated error are counted separately."""
    y = S.labels("dev")
    ids = S.splits["dev"]
    out = {}
    for name, sigs in auto_dev.items():
        t = sigs["token"]["threshold"]
        if t is None:
            continue
        pr = S.predict(name, "dev")
        conf = M.confidence(pr["pred"], pr["p"])
        sure = ~np.isnan(conf) & (conf >= t)
        deny, approve = sure & (pr["pred"] == 0), sure & (pr["pred"] == 1)
        wrong_deny = deny & (y == 1)
        amt = [max(a) for a in (amounts(visible_narrative(S.records[i]["narrative"]))
                                for i, w in zip(ids, wrong_deny) if w) if a]
        out[name] = {"n": len(y), "auto_no_relief": int(deny.sum()), "wrong_no_relief": int(wrong_deny.sum()),
                     "wrong_no_relief_ci": list(M.wilson(int(wrong_deny.sum()), int(deny.sum()))),
                     "proposed_relief": int(approve.sum()), "wrong_relief": int((approve & (y == 0)).sum()),
                     "escalated": int((~sure).sum()),
                     "relief_among_escalated": float(y[~sure].mean()) if (~sure).any() else math.nan,
                     "wrong_no_relief_with_amount": len(amt),
                     "wrong_no_relief_median_amount": float(np.median(amt)) if amt else math.nan}
    return out


# ------------------------------------------------------------------ error analysis
def error_analysis(records, splits, version="v1") -> dict | None:
    path = ANNOTATION_DIR / f"{version}_dev.csv"
    if not path.exists():
        return None
    dis = disagreements(version, records, splits)
    rows = load_annotations(version)
    validate(version, rows, records, {d["complaint_id"] for d in dis})
    dir_of = {d["complaint_id"]: direction(d["label"], d["pred"]) for d in dis}
    counts = Counter((r["category"], dir_of[r["complaint_id"]]) for r in rows)
    cats = Counter(r["category"] for r in rows)
    examples = {}
    for r in sorted(rows, key=lambda r: int(r["complaint_id"])):
        if r["quote"] and r["category"] not in examples:
            examples[r["category"]] = {"complaint_id": r["complaint_id"], "quote": r["quote"]}
    with open(ANNOTATION_DIR / "categories.csv", newline="", encoding="utf-8") as f:
        meta = {r["category"]: r for r in csv.DictReader(f)}
    unknown = set(cats) - set(meta)
    if unknown:
        raise ValueError(f"categories without a definition: {unknown}")
    with open(ANNOTATION_DIR / "groups.csv", newline="", encoding="utf-8") as f:
        group_label = {r["group"]: r["label"] for r in csv.DictReader(f)}
    read = [r for r in rows if r["group_basis"] == "read"]
    groups = []
    for g in group_label:
        sel = sorted((r for r in read if r["group"] == g), key=lambda r: int(r["complaint_id"]))
        ex = next((r for r in sel if r.get("example") == "yes"), sel[0] if sel else None)
        groups.append({"group": g, "label": group_label[g], "n": len(sel),
                       "judge_relief_company_none": sum(dir_of[r["complaint_id"]].startswith("judge relief") for r in sel),
                       "judge_none_company_relief": sum(dir_of[r["complaint_id"]].startswith("judge none") for r in sel),
                       "example": {"complaint_id": ex["complaint_id"], "quote": ex["quote"], "note": ex["note"]}
                       if ex else None})
    read_cases = [{k: r[k] for k in ("complaint_id", "group", "category", "quote", "note")}
                  for r in sorted(read, key=lambda r: (r["group"], int(r["complaint_id"])))]
    by_category_only = Counter(r["category"] for r in rows if r["group_basis"] != "read")
    # does 'the issuer had already said no' set the disagreements apart? Judge v2 extracts that fact for every DEV complaint.
    dev = splits["dev"]
    y = np.array([records[i]["label"] for i in dev])
    v1 = load("v1", dev)
    agree = np.array([v1[i]["pred"] for i in dev]) == y
    denial = None
    if available("v2", dev):
        v2 = load("v2", dev)
        den = np.array([v2[i]["fields"].get("issuer_status") == "denied" for i in dev])
        denial = {"n_denied": int(den.sum()), "relief_rate_denied": float(y[den].mean()),
                  "relief_rate_other": float(y[~den].mean()) if (~den).any() else math.nan,
                  "v1_agrees_among_denied": int((agree & den).sum()),
                  "v1_agrees_among_denied_relief": int((agree & den & (y == 1)).sum()),
                  "v1_disagrees_among_denied": int((~agree & den).sum())}
    words = sum(len(records[d["complaint_id"]]["narrative"].split()) for d in dis)
    provenance = json.loads((ANNOTATION_DIR / "provenance.json").read_text())
    return {"groups": groups, "read_cases": read_cases, "n_read": len(read),
            "category_only": dict(by_category_only), "n_category_only": int(sum(by_category_only.values())),
            "prior_denial": denial, "words_in_disagreements": int(words), "provenance": provenance,
            "n_disagreements": len(dis), "n_dev": len(splits["dev"]),
            "by_direction": dict(Counter(dir_of.values())),
            "categories": [{"category": c, "label": meta[c]["label"], "fix": meta[c]["fix"], "n": n,
                            "judge_relief_company_none": counts[(c, "judge relief, company none")],
                            "judge_none_company_relief": counts[(c, "judge none, company relief")],
                            "example": examples.get(c)}
                           for c, n in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))]}


# ------------------------------------------------------------------ gaming
def _replays() -> Cache | None:
    return Cache(REPLAY_FILE) if REPLAY_FILE.exists() else None


def gaming(records, splits, cache, S=None, thresholds=None, versions=JUDGE_VERSIONS) -> dict:
    """Flip rates of every rewrite, of the neutral control, and (when the replay
    file covers the gaming subset) of sending the unmodified narrative again.
    With DEV thresholds, also what happens to the routing of each complaint."""
    ids = gaming_ids(splits["dev"], {i: records[i]["label"] for i in splits["dev"]})
    names = list(ATTACKS) + list(CONTROLS)
    thresholds = thresholds or {}
    out = {"n_items": len(ids), "attacks": list(ATTACKS), "controls": list(CONTROLS), "systems": {},
           "routing": {}}
    replays = _replays()
    for v in versions:
        att_ids = [attacked_item_id(i, a) for i in ids for a in names]
        if not (available(v, ids, cache) and available(v, att_ids, cache)):
            continue
        base = load(v, ids, cache)
        variants = {a: {i: parse(cache.records[Cache.key(v, attacked_item_id(i, a))]) for i in ids}
                    for a in names}
        if replays and all(replay_key(v, i) in replays.records for i in ids):
            variants["replay"] = {i: parse(replays.records[replay_key(v, i)]) for i in ids}
        out["systems"][v] = flip_table(ids, base, variants)
        if thresholds.get(v) is not None:
            out["routing"][v] = route_table(ids, base, variants, thresholds[v])
    if S is not None and FACTS_VERSION in out["systems"]:
        base, variants, out["filter"] = guarded_gaming(ids, records, splits, cache, S, replays)
        out["systems"]["guarded"] = flip_table(ids, base, variants)
        if thresholds.get("guarded") is not None:
            out["routing"]["guarded"] = route_table(ids, base, variants, thresholds["guarded"])
    return out


def guarded_gaming(ids, records, splits, cache, S, replays):
    """Filter first, then decide from the v2 facts of the filtered text."""
    fm = S.fact_model()
    v = FACTS_VERSION
    base_parsed = load(v, ids, cache)

    def decide(parsed, rec, flagged=False):
        if not parsed["valid"]:
            return {"pred": 0, "p_token": math.nan, "flagged": flagged}
        p = float(fm.predict_proba([parsed["fields"]], [rec])[0])
        return {"pred": int(p >= 0.5), "p_token": p, "flagged": flagged}

    base, variants, flagged, restored = {}, {}, Counter(), Counter()
    for i in ids:
        rec = records[i]
        vis = visible_narrative(rec["narrative"])
        base[i] = decide(base_parsed[i], rec)
        for a in list(ATTACKS) + list(CONTROLS):
            text = apply_attack(vis, a) if a in ATTACKS else apply_control(vis, a)
            clean, flag = sanitize(text)
            flagged[a] += flag
            if clean == vis:          # the filter removed exactly the appended text
                parsed = base_parsed[i]
                restored[a] += 1
            elif clean == text:       # nothing removed: facts of the rewritten narrative
                parsed = parse(cache.records[Cache.key(v, attacked_item_id(i, a))])
            else:                     # partly removed: would need its own model call
                key = filtered_item_id(attacked_item_id(i, a))
                parsed = load(v, [key], cache)[key]
            variants.setdefault(a, {})[i] = decide(parsed, rec, flag)
        if replays and replay_key(v, i) in replays.records:
            variants.setdefault("replay", {})[i] = decide(parse(replays.records[replay_key(v, i)]), rec)
    if "replay" in variants and len(variants["replay"]) != len(ids):
        del variants["replay"]
    clean_ids = splits["dev"] + splits["train"]
    clean_flagged = sum(sanitize(visible_narrative(records[i]["narrative"]))[1] for i in clean_ids)
    probe = visible_narrative(records[ids[0]]["narrative"])
    return base, variants, {"attacked_flagged": dict(flagged), "clean_flagged": int(clean_flagged),
                            "clean_n": len(clean_ids),
                            # cells that are fixed by construction: the cached clean answer is reused
                            "answer_reused": {a: int(restored[a]) for a in ATTACKS},
                            "paraphrases": len(INJECTION_PARAPHRASES),
                            "paraphrases_flagged": int(sum(sanitize(probe + "\n\n" + t)[1]
                                                           for t in INJECTION_PARAPHRASES))}


def _route(d: dict, t: float) -> str:
    """auto_none: decided automatically as no relief; approve: relief proposed to
    a person; escalate: a person decides (low confidence, invalid output, or flagged)."""
    p = d["p_token"]
    if d.get("flagged") or p != p:
        return "escalate"
    conf = p if d["pred"] == 1 else 1 - p
    if conf < t:
        return "escalate"
    return "approve" if d["pred"] == 1 else "auto_none"


def route_table(ids, base, variants, t) -> dict:
    """What a rewrite does to complaints at the operating threshold t, starting
    from the routing of the unmodified narrative."""
    groups = {a: [a] for a in variants}
    groups["all"] = [a for a in ATTACKS if a in variants]
    start = {i: _route(base[i], t) for i in ids}
    res = {"base": dict(Counter(start.values()))}
    for name, members in groups.items():
        c = Counter()
        for i in ids:
            for x in members:
                c[(start[i], _route(variants[x][i], t))] += 1
        n_auto = sum(v for (a, _), v in c.items() if a == "auto_none")
        n_esc = sum(v for (a, _), v in c.items() if a == "escalate")
        res[name] = {"n_auto_none": n_auto, "auto_none_to_approve": c[("auto_none", "approve")],
                     "auto_none_to_escalate": c[("auto_none", "escalate")],
                     "n_escalate": n_esc, "escalate_to_approve": c[("escalate", "approve")],
                     "escalate_to_auto_none": c[("escalate", "auto_none")],
                     "to_approve_ci": list(M.wilson(c[("auto_none", "approve")] + c[("escalate", "approve")],
                                                    n_auto + n_esc))}
    return res


def flip_table(ids, base, variants) -> dict:
    """variants: {rewrite name: {complaint id: decision}}. 'all' pools the attacks.
    Single rewrites get a Wilson interval (30 independent complaints); the pooled
    row repeats each complaint five times, so its interval resamples complaints.
    Each rewrite is also compared with the neutral control on the same complaints
    (exact McNemar test on the complaints where exactly one of the two flipped)."""
    from scipy.stats import binomtest

    res = {"base_relief": int(sum(base[i]["pred"] for i in ids))}
    groups = {a: [a] for a in variants}
    groups["all"] = [a for a in ATTACKS if a in variants]
    flipped = {a: np.array([variants[a][i]["pred"] != base[i]["pred"] for i in ids]) for a in variants}
    for name, members in groups.items():
        flips = to_relief = to_none = n = n_base_no = 0
        dps = []
        for i in ids:
            b = base[i]
            for x in members:
                r = variants[x][i]
                n += 1
                flips += r["pred"] != b["pred"]
                if b["pred"] == 0:
                    n_base_no += 1
                    to_relief += r["pred"] == 1
                else:
                    to_none += r["pred"] == 0
                if not (math.isnan(r["p_token"]) or math.isnan(b["p_token"])):
                    dps.append(r["p_token"] - b["p_token"])
        if len(members) == 1:
            lo, hi = M.wilson(flips, n)
        else:                                   # clustered: resample the complaints, keep their rewrites together
            per = np.sum([flipped[x] for x in members], axis=0)
            idx = M._indices(len(ids), C.BOOTSTRAP_B, C.BOOTSTRAP_SEED)
            lo, hi = np.percentile(per[idx].sum(1) / (len(ids) * len(members)), [2.5, 97.5])
        lo2, hi2 = M.wilson(to_relief, n_base_no)
        row = {"n": n, "flips": flips, "flip_rate": flips / n, "flip_ci": [float(lo), float(hi)],
               "n_base_no_relief": n_base_no, "to_relief": to_relief,
               "to_relief_rate": to_relief / n_base_no if n_base_no else math.nan,
               "to_relief_ci": [lo2, hi2], "to_no_relief": to_none,
               "n_base_relief": n - n_base_no,
               "mean_delta_p_relief": float(np.mean(dps)) if dps else math.nan}
        if len(members) == 1 and name != "neutral" and "neutral" in flipped:
            only_this = int((flipped[name] & ~flipped["neutral"]).sum())
            only_control = int((~flipped[name] & flipped["neutral"]).sum())
            row["vs_neutral"] = {"only_rewrite": only_this, "only_control": only_control,
                                 "p": float(binomtest(only_this, only_this + only_control, 0.5).pvalue)
                                 if only_this + only_control else 1.0}
        res[name] = row
    return res


# ------------------------------------------------------------------ test-retest
def retest(records, splits, cache, split: str = "dev") -> dict:
    """Second answers to cached requests of one split (replay file) against the cached ones."""
    replays = _replays()
    out = {}
    if replays is None:
        return out
    for v in JUDGE_VERSIONS:
        ids = [i for i in splits[split] if replay_key(v, i) in replays.records
               and Cache.key(v, i) in cache.records]
        if not ids:
            continue
        ra = [cache.records[Cache.key(v, i)] for i in ids]
        rb = [replays.records[replay_key(v, i)] for i in ids]
        a, b = [parse(r) for r in ra], [parse(r) for r in rb]
        y = np.array([records[i]["label"] for i in ids])
        pa, pb = np.array([x["pred"] for x in a]), np.array([x["pred"] for x in b])
        same = int(np.sum(pa == pb))
        shift = np.abs(np.array([x["p_token"] for x in a]) - np.array([x["p_token"] for x in b]))
        shift = shift[~np.isnan(shift)]
        reuse_same = np.array([x["prompt_eval_cached_count"] == z["prompt_eval_cached_count"]
                               for x, z in zip(ra, rb)])
        p_same = np.array([x["p_token"] == z["p_token"] for x, z in zip(a, b)])
        out[v] = {
            "n": len(ids), "n_split": len(splits[split]),
            "same_decision": same, "same_decision_rate": same / len(ids),
            "same_decision_ci": list(M.wilson(same, len(ids))),
            "byte_identical": int(sum(x["content"] == z["content"] for x, z in zip(ra, rb))),
            "same_fields": int(sum(x["fields"] == z["fields"] for x, z in zip(a, b))),
            "same_probability": int(p_same.sum()),
            "shift": {"median": float(np.median(shift)), "p90": float(np.percentile(shift, 90)),
                      "p95": float(np.percentile(shift, 95)), "max": float(shift.max()),
                      "over_0.01": int((shift > 0.01).sum()), "over_0.05": int((shift > 0.05).sum())},
            "same_prompt_cache_reuse": {"n": int(reuse_same.sum()),
                                        "same_probability": int(p_same[reuse_same].sum())},
            "different_prompt_cache_reuse": {"n": int((~reuse_same).sum()),
                                             "same_probability": int(p_same[~reuse_same].sum())},
            "top_share_cached": fixed_coverage(y, pa, M.confidence(pa, np.array([x["p_token"] for x in a])),
                                               levels=(0.1, 0.25, 0.5)),
            "top_share_replay": fixed_coverage(y, pb, M.confidence(pb, np.array([x["p_token"] for x in b])),
                                               levels=(0.1, 0.25, 0.5)),
            "kappa_between_runs": M.cohen_kappa(pa, pb),
            "accuracy_cached": M.accuracy(y, pa), "accuracy_replay": M.accuracy(y, pb),
            "kappa_cached": M.cohen_kappa(y, pa), "kappa_replay": M.cohen_kappa(y, pb),
            "flips": [{"complaint_id": i, "cached": x["pred"], "p_cached": x["p_token"],
                       "replay": z["pred"], "p_replay": z["p_token"]}
                      for i, x, z in zip(ids, a, b) if x["pred"] != z["pred"]],
        }
    return out


def threshold_stability(S: Systems, auto_dev: dict, rt: dict) -> dict:
    """How close DEV complaints sit to each operating threshold, compared with the
    confidence shift seen when the same request is sent twice."""
    out = {}
    y = S.labels("dev")
    ids = S.splits["dev"]
    replays = _replays()
    for name, sigs in auto_dev.items():
        t = sigs["token"]["threshold"]
        if t is None:
            continue
        pr = S.predict(name, "dev")
        conf = M.confidence(pr["pred"], pr["p"])
        row = {"threshold": t, "n_auto": int(np.nansum(conf >= t))}
        if name in rt:
            delta = rt[name]["shift"]["max"]
            have = np.array([replay_key(name, i) in replays.records for i in ids])
            rep = [parse(replays.records[replay_key(name, i)]) if h else None for i, h in zip(ids, have)]
            pred_r = np.array([r["pred"] if r else 0 for r in rep])
            conf_r = M.confidence(pred_r, np.array([r["p_token"] if r and r["valid"] else np.nan for r in rep]))
            auto_c = have & ~np.isnan(conf) & (conf >= t)
            auto_r = have & ~np.isnan(conf_r) & (conf_r >= t)
            row.update({
                "replayed": int(have.sum()), "auto_cached": int(auto_c.sum()), "auto_replay": int(auto_r.sum()),
                "left_automation": int((auto_c & ~auto_r).sum()), "entered_automation": int((~auto_c & auto_r).sum()),
                "agreement_cached": float(np.mean(y[auto_c] == pr["pred"][auto_c])) if auto_c.any() else math.nan,
                "agreement_replay": float(np.mean(y[auto_r] == pred_r[auto_r])) if auto_r.any() else math.nan})
        elif name == "guarded" and FACTS_VERSION in rt:
            fm = S.fact_model()
            sel = [i for i in ids if replay_key(FACTS_VERSION, i) in replays.records]
            recs = [S.records[i] for i in sel]
            pa = fm.predict_proba([parse(S.cache.records[Cache.key(FACTS_VERSION, i)])["fields"] for i in sel], recs)
            pb = fm.predict_proba([parse(replays.records[replay_key(FACTS_VERSION, i)])["fields"] for i in sel], recs)
            delta = float(np.abs(pa - pb).max())
            row.update({"replayed": len(sel), "probability_changed": int((np.abs(pa - pb) > 1e-12).sum())})
        else:
            delta = 0.0
        row["observed_max_shift"] = delta
        row["within_shift_of_threshold"] = int(np.nansum(np.abs(conf - t) <= delta)) if delta > 0 else 0
        row["below_threshold_within_shift"] = int(np.nansum((conf < t) & (t - conf <= delta))) if delta > 0 else 0
        out[name] = row
    return out


# ------------------------------------------------------------------ automation
def automation_dev(S: Systems, names) -> dict:
    y = S.labels("dev")
    out = {}
    for name in names:
        pr = S.predict(name, "dev")
        if pr is None or np.all(np.isnan(pr["p"])):
            continue
        signals = {"token": pr["p"]}
        if "p_verbal" in pr:
            signals["verbal"] = pr["p_verbal"]
        out[name] = {}
        for sig, p in signals.items():
            conf = M.confidence(pr["pred"], p)
            t = M.choose_threshold(y, pr["pred"], conf)
            out[name][sig] = {
                "curve": M.selective_curve(y, pr["pred"], conf),
                "threshold": t,
                "at_threshold": M.at_threshold(y, pr["pred"], conf, t) if t is not None else None,
                "fixed_coverage": fixed_coverage(y, pr["pred"], conf),
            }
    return out


def fixed_coverage(y, pred, conf, levels=(0.1, 0.25, 0.5, 0.75, 1.0)) -> list:
    """Agreement on the most confident share of cases (largest threshold whose
    coverage reaches the level; ties are automated together)."""
    curve = M.selective_curve(y, pred, conf)
    out = []
    for lv in levels:
        pt = next((p for p in curve if p["coverage"] >= lv - 1e-9), None)
        if pt:
            out.append({"level": lv, **pt})
    return out


# ------------------------------------------------------------------ ceiling proxy
def label_consistency() -> dict:
    pairs = json.loads(PAIRS_FILE.read_text())
    out = {}
    for name, cond in (("same_company", lambda p: p["same_company"]),
                       ("same_company_within_7_days", lambda p: p["same_company"] and p["days_apart"] <= 7),
                       ("same_company_over_7_days", lambda p: p["same_company"] and p["days_apart"] > 7),
                       ("different_company", lambda p: not p["same_company"])):
        sel = [p for p in pairs if p["cosine"] >= CEILING_PAIR_COSINE and cond(p)]
        if not sel:
            continue
        a = np.array([p["label_a"] for p in sel])
        b = np.array([p["label_b"] for p in sel])
        agree = float(np.mean(a == b))
        k = int((a == b).sum())
        # symmetric (pooled) kappa for an unordered pair of draws
        p1 = (a.sum() + b.sum()) / (2 * len(a))
        pe = p1 ** 2 + (1 - p1) ** 2
        kappa = (agree - pe) / (1 - pe) if pe < 1 else math.nan
        bound = (1 + math.sqrt(max(0.0, 2 * agree - 1))) / 2
        out[name] = {"pairs": len(sel), "agreement": agree, "agreement_ci": list(M.wilson(k, len(sel))),
                     "pooled_kappa": kappa, "accuracy_upper_bound": bound}
    return out


# ------------------------------------------------------------------ cost
def _set_of(item_id: str, split_of: dict) -> str:
    base, _, suffix = item_id.partition("~")
    return split_of.get(base, "?") + (" rewrites" if suffix else "")


def cost(cache: Cache, splits) -> dict:
    out = {}
    if COST_FILE.exists():
        out["tfidf_lr"] = json.loads(COST_FILE.read_text())
    for v in JUDGE_VERSIONS:
        recs = [cache.records.get(Cache.key(v, i)) for i in splits["dev"]]
        recs = [r for r in recs if r]
        if not recs:
            continue
        w = np.array([r["wall_s"] for r in recs])
        out[v] = {"calls": len(recs), "median_s": float(np.median(w)), "p90_s": float(np.percentile(w, 90)),
                  "mean_s": float(np.mean(w)),
                  "mean_output_tokens": float(np.mean([r["eval_count"] for r in recs])),
                  "max_output_tokens": int(max(r["eval_count"] for r in recs)),
                  "mean_prompt_tokens": float(np.mean([r["prompt_eval_count"] for r in recs]))}
    for v in JUDGE_VERSIONS:
        w = [r["wall_s"] for r in cache.records.values() if r["prompt_version"] == v]
        if v in out and w:
            out[v]["min_s_all_calls"], out[v]["max_s_all_calls"] = float(np.min(w)), float(np.max(w))
            out[v]["hours_per_1000"] = out[v]["mean_s"] * 1000 / 3600      # what 1,000 DEV-like decisions took
    split_of = {i: k for k, ids in splits.items() for i in ids}
    groups: dict[tuple, list] = {}
    for r in cache.records.values():
        groups.setdefault((r["prompt_version"], _set_of(r["item_id"], split_of)), []).append(r)
    replays = _replays()
    if replays:
        for r in replays.records.values():
            groups.setdefault((r["prompt_version"], "dev (replay file)"), []).append(r)
    out["by_run"] = [{"version": v, "set": k, "calls": len(rs), "hours": float(sum(r["wall_s"] for r in rs) / 3600),
                      "median_s": float(np.median([r["wall_s"] for r in rs])),
                      "first_call": min(r["requested_at"] for r in rs),
                      "last_call": max(r["requested_at"] for r in rs)}
                     for (v, k), rs in sorted(groups.items(), key=lambda kv: min(r["requested_at"] for r in kv[1]))]
    everything = [r for rs in groups.values() for r in rs]
    out["all_calls"] = {"calls": len(everything),
                        "hours": float(sum(r["wall_s"] for r in everything) / 3600),
                        "truncated": int(sum(r["done_reason"] != "stop" for r in everything)),
                        "held_out_calls": int(sum(_set_of(r["item_id"], split_of).startswith("test")
                                                  for r in everything))}
    recent = sorted(cache.records.values(), key=lambda r: r["requested_at"])[-200:]
    out["recent_s_per_call"] = round(float(np.median([r["wall_s"] for r in recent])), 2) if recent else None
    events = [json.loads(line) for line in C.RUN_LOG.read_text().splitlines() if line.strip()]
    out["run_speed"] = [e for e in events if e.get("event") == "run_speed"]
    out["concurrency"] = compare_concurrency(load_concurrency_replays(), cache)
    out["concurrency_decision"] = next((e for e in reversed(events)
                                        if e.get("event") == "concurrency_decision"), None)
    return out


# ------------------------------------------------------------------ held-out
TEST_MARKER = C.RESULTS / "test_run.json"


def heldout_results(S: Systems, split: str, thresholds: dict, names=SYSTEMS) -> dict:
    """Agreement of every system on `split`, the operating points fixed beforehand
    applied unchanged, and (as an optimistic bound) the best threshold chosen on
    `split` itself."""
    res = split_results(S, split, names)
    y = S.labels(split)
    auto, oracle = {}, {}
    for name, t in thresholds.items():
        pr = S.predict(name, split)
        if pr is None:
            continue
        conf = M.confidence(pr["pred"], pr["p"])
        auto[name] = {"signal": "token", "threshold": t,
                      "at_dev_threshold": M.at_threshold(y, pr["pred"], conf, t) if t is not None else None,
                      "curve": M.selective_curve(y, pr["pred"], conf),
                      "fixed_coverage": fixed_coverage(y, pr["pred"], conf)}
        t_star = M.choose_threshold(y, pr["pred"], conf)       # chosen on the scored split itself
        oracle[name] = {"threshold": t_star,
                        "at": M.at_threshold(y, pr["pred"], conf, t_star) if t_star is not None else None}
    return {"n": len(y), "results": res, "automation": auto, "oracle": oracle, "ranking": ranking(S, split)}


NAMES = {"v1": "judge v1", "v2": "judge v2", "tfidf_lr": "the TF-IDF model", "issuer_rate": "the issuer relief rate",
         "company_rule": "the company rule"}


def heldout_claims(held: dict, settings: dict) -> list[dict]:
    """The verdict of every frozen claim (freeze.CLAIMS), decided by its rule."""
    res, out = held["results"], []
    d = res.get("v2_minus_v1")
    out.append({"id": "v2_vs_v1", "verdict": "v2 replaces v1" if d and d["kappa"]["lo"] > 0 else
                "no improvement of v2 over v1 was demonstrated", "interval": d["kappa"] if d else None})
    for base in ("tfidf_lr", "issuer_rate", "company_rule"):
        for v in ("v1", "v2"):
            d = res.get(f"{v}_minus_{base}")
            if d is None:
                continue
            k = d["kappa"]
            verdict = ("is below" if k["hi"] < 0 else "is above" if k["lo"] > 0 else "is not distinguishable from")
            out.append({"id": "judge_vs_baseline", "verdict": f"{NAMES[v]} {verdict} {NAMES[base]}", "interval": k})
    for v in ("v1", "v2"):
        d = held["ranking"]["differences"].get(f"tfidf_lr_minus_{v}")
        if d is None:
            continue
        k = d["within_issuer_auc"]
        out.append({"id": "within_issuer", "interval": k, "verdict":
                    f"TF-IDF orders complaints better than {NAMES[v]} within the same issuer" if k["lo"] > 0 else
                    f"{NAMES[v]} orders complaints better than TF-IDF within the same issuer" if k["hi"] < 0 else
                    f"the lead of TF-IDF over {NAMES[v]} is not demonstrated once the issuer is held fixed"})
    at = (held["automation"].get("tfidf_lr") or {}).get("at_dev_threshold")
    if at is not None:
        if at["n_auto"] < settings["min_automated"]:
            verdict = "too few automated cases to judge"
        else:
            lo, hi = at["agreement_wilson"]
            verdict = ("meets the target" if lo >= settings["target_agreement"] else
                       "misses the target" if hi < settings["target_agreement"] else "not shown to meet the target")
            if at["agreement"]["value"] <= at["majority_agreement_on_auto"]:
                verdict += "; does no better than always answering no relief on the same cases"
        out.append({"id": "tfidf_operating_point", "verdict": f"the frozen TF-IDF operating point: {verdict}",
                    "interval": None})
    none = [n for n, r in held["automation"].items() if r["threshold"] is None]
    if none:
        out.append({"id": "llm_automation", "interval": None,
                    "verdict": "the LLM judge automates nothing (no DEV threshold reached the target for: "
                               + ", ".join(sorted(none)) + ")"})
    return out


def test_results(S: Systems, frozen: dict | None) -> dict | None:
    """Only computed after the scored TEST pass has finished; thresholds and
    decision rules come from the frozen file."""
    if not TEST_MARKER.exists():
        return None
    marker = json.loads(TEST_MARKER.read_text())
    if not marker.get("finished_at"):
        return None
    if frozen is None or marker["frozen_sha256"] != freeze.sha256_path(freeze.FROZEN_FILE):
        raise RuntimeError("the held-out run does not match results/frozen.json")
    thresholds = {name: op["threshold"] for name, op in frozen["operating_points"].items()}
    out = heldout_results(S, "test", thresholds)
    out["claims"] = heldout_claims(out, frozen["settings"])
    noise = retest(S.records, S.splits, S.cache, "test") if marker.get("noise_finished_at") else {}
    out["noise"] = {v: {k: r[k] for k in ("n", "n_split", "same_decision", "same_decision_rate", "same_decision_ci",
                                          "byte_identical", "same_probability", "shift", "kappa_between_runs")}
                    for v, r in noise.items()}                    # never scored against recorded decisions
    out.update({"started_at": marker["started_at"], "finished_at": marker["finished_at"],
                "versions": marker["versions"], "workers": marker["workers"], "commit": marker["commit"]})
    return out


def rehearsal(S: Systems) -> dict | None:
    """Dress rehearsal of the held-out procedure inside DEV: thresholds and the
    fact model come from the even-numbered half of DEV (in complaint-ID order)
    and are applied unchanged to the odd-numbered half. No held-out complaint
    is involved; the prompts were still written with all of DEV in view."""
    dev = S.splits["dev"]
    if not available("v2", dev, S.cache):
        return None
    halves = {"train": S.splits["train"], "dev": dev[0::2], "test": dev[1::2]}
    S2 = Systems(S.records, halves, S.cache, baseline=S.baseline())
    auto = automation_dev(S2, CONFIDENT)
    thresholds = {name: r["token"]["threshold"] for name, r in auto.items()}
    out = heldout_results(S2, "test", thresholds)
    out["claims"] = heldout_claims(out, {"min_automated": C.MIN_AUTOMATED, "target_agreement": C.TARGET_AGREEMENT})
    out["fit_half"] = {name: r["token"]["at_threshold"] for name, r in auto.items()}
    out["n_fit"] = len(halves["dev"])
    return out


def estimates(auto_dev: dict, held: dict | None) -> dict | None:
    """Realistic and optimistic operating points for every system with a
    confidence signal; no system is singled out after seeing the scored split."""
    if not held:
        return None

    def cell(a):
        if a is None:
            return None
        return {"n_auto": a["n_auto"], "n": a["n"], "coverage": a["coverage"]["value"],
                "agreement": a["agreement"], "majority": a["majority_agreement_on_auto"]}

    rows = []
    for name, sigs in auto_dev.items():
        rows.append({"system": name,
                     "optimistic": cell(sigs["token"]["at_threshold"]),
                     "realistic": cell(held["automation"].get(name, {}).get("at_dev_threshold")),
                     "best_on_scored_split": cell((held["oracle"].get(name) or {}).get("at"))})
    return {"rows": rows}


# ------------------------------------------------------------------ human labels
def _human_figures(SL: Systems, ids: list, labels: dict) -> dict:
    """Agreements registered in docs/analysis_plan_human_labels.md for one set of
    labelled cases. SL scores the 'label' split with baselines that never saw it."""
    pos = {i: k for k, i in enumerate(SL.splits["label"])}
    idx = np.array([pos[i] for i in ids])
    y = SL.labels("label")[idx]
    lab = np.array([labels[i]["label"] for i in ids])
    decided, unsure = lab != "cannot_tell", lab == "cannot_tell"
    h = (lab == "relief").astype(int)             # 'cannot tell' counts as no relief only in the sensitivity line
    preds = {}
    for n in SYSTEMS:
        pr = SL.predict(n, "label")
        if pr is not None:                         # the judges answer only after the labels are locked
            preds[n] = pr["pred"][idx]
    out = {"n": len(ids), "decided": int(decided.sum()), "cannot_tell": int(unsure.sum()),
           "recorded_relief_share": float(y.mean()),
           "person_relief_share": float(h[decided].mean()) if decided.any() else math.nan,
           "cannot_tell_recorded_relief": float(y[unsure].mean()) if unsure.any() else math.nan,
           "cannot_tell_system_relief": {n: float(p[unsure].mean()) if unsure.any() else math.nan
                                         for n, p in preds.items()}}
    if decided.sum() >= 10:
        out["person_vs_recorded"] = M.bootstrap(y[decided], h[decided])
        out["systems_vs_person"] = {n: M.bootstrap(h[decided], p[decided]) for n, p in preds.items()}
        out["systems_vs_recorded_same_cases"] = {n: M.bootstrap(y[decided], p[decided]) for n, p in preds.items()}
        out["paired_vs_person"] = {f"{b}_minus_{a}": M.paired_difference(h[decided], preds[a][decided], preds[b][decided])
                                   for a, b in (("tfidf_lr", "v1"), ("tfidf_lr", "v2"), ("v1", "v2"))
                                   if a in preds and b in preds}
        out["sensitivity"] = {"person_vs_recorded": M.bootstrap(y, h),
                              "systems_vs_person": {n: M.bootstrap(h, p) for n, p in preds.items()}}
    return out


def human_labels(S: Systems) -> dict:
    """Blind labels by a person (label.py). Nothing is reported from a set that is
    not complete and locked, that predates the registered plan, or whose cases
    had a model answer before the lock."""
    from . import label as L
    from .human import AMENDMENT_1_AT, PLAN_REGISTERED_AT, halves
    from .review import load_reviews, review_path, review_rows
    from .translate import Translations, problems

    sample = L.load_sample()
    ids = sample["ids"]
    tr = Translations()
    got = [(visible_narrative(S.records[i]["narrative"]), i) for i in ids]
    got = [(text, tr.get(text, i)) for text, i in got]
    answers = [r for r in S.cache.records.values() if r["item_id"].split("~")[0] in set(ids)]
    out = {"sample_n": len(ids), "split": sample["split"], "plan_registered_at": PLAN_REGISTERED_AT,
           "amended_at": AMENDMENT_1_AT, "annotators": {}, "model_answers_for_sample": len(answers),
           "translations": {"missing": sum(r is None for _, r in got),
                            "flagged": sum(bool(problems(t, r)) for t, r in got if r is not None)}}
    locks = sorted(L.LABELS_DIR.glob("*.lock.json"))
    if answers and not locks:
        raise RuntimeError("the judge answered labelled-sample complaints before any labels were locked")
    if not locks:
        return out
    rest = [i for i in S.splits["train"] if i not in set(ids)]
    SL = Systems(S.records, {"train": rest, "dev": S.splits["dev"], "label": ids}, S.cache,
                 lr_params=S.baseline()["params"])        # baselines refitted without the labelled cases
    for lock in locks:
        who = lock.name[: -len(".lock.json")]
        if not L.is_locked(who):
            continue
        locked_at = json.loads(lock.read_text())["locked_at"]
        labels = L.load_labels(L.labels_path(who))
        if set(labels) != set(ids) or min(r["labelled_at"] for r in labels.values()) < AMENDMENT_1_AT:
            raise RuntimeError(f"labels of {who} do not match the registered sample and plan")
        if answers and min(r["requested_at"] for r in answers) < locked_at:
            raise RuntimeError("a model answer for a labelled complaint predates the lock")
        secs = [labels[i]["seconds"] for i in ids]
        row = {"locked_at": locked_at, "median_seconds": float(np.median(secs)),
               "total_minutes": float(np.sum(secs) / 60),
               "with_note": sum(bool(labels[i]["note"]) for i in ids),
               "translation_flagged_when_shown": sum(bool(labels[i].get("translation_problems")) for i in ids),
               "all": _human_figures(SL, ids, labels),
               "halves": {k: _human_figures(SL, v, labels) for k, v in halves(ids).items()}}
        reviews = load_reviews(review_path(who))
        row["error_review"] = {"total": len(review_rows()), "reviewed": len(reviews),
                               "confirmed": sum(r["confirmed"] for r in reviews.values()),
                               "changed": [{"complaint_id": r["complaint_id"], "from": r["drafted_group"],
                                            "to": r["group"], "note": r["note"]}
                                           for r in reviews.values() if not r["confirmed"]]}
        out["annotators"][who] = row
    return out


def issuer_facts(S: Systems) -> dict:
    """What the classic baselines know about issuers (all from TRAIN)."""
    b = S.baseline()
    train = S.splits["train"]
    n_by = Counter(S.records[i]["company"] for i in train)
    big = [(co, b["issuer_rates"][co], n) for co, n in n_by.items() if n >= 100]
    named = sum(bool(b["issuer_pattern"].search(visible_narrative(S.records[i]["narrative"]))) for i in train)
    old = first_commit_grid(b["cv"])
    old_model, _, _ = fit_lr(train, S.records, old)
    pred, _ = lr_predict(old_model, S.splits["dev"], S.records)
    return {"issuers_in_train": len(n_by), "issuers_above_half": sum(
                np.mean([S.records[i]["label"] for i in train if S.records[i]["company"] == co]) > 0.5
                for co, n in n_by.items() if n >= 20),
            "lowest": min(big, key=lambda t: t[1]), "highest": max(big, key=lambda t: t[1]),
            "train_narratives_naming_an_issuer": named / len(train),
            "issuer_words_masked": len(b["issuer_words"]),
            "top_features": top_features(b["lr"]),
            "first_commit_grid": {"chosen": old, "dev_kappa": M.cohen_kappa(S.labels("dev"), pred)}}


# ------------------------------------------------------------------ main
def compute() -> dict:
    cache = Cache()
    records, splits = load_records(), load_splits()
    S = Systems(records, splits, cache)
    base = S.baseline()
    auto_dev = automation_dev(S, CONFIDENT)
    thresholds = {name: r["token"]["threshold"] for name, r in auto_dev.items()}
    m = {
        "settings": settings(),
        "environment": environment(cache),
        "data": data_summary(),
        "base_rates": base_rates(records, splits),
        "tfidf_lr": {"chosen": base["params"], "cv_kappa": base["cv"],
                     "learning_curve": learning_curve(splits["train"], splits["dev"], records, base["params"])},
        "keyword_rule": {"chosen": base["keyword_chosen"], "candidates": base["keyword_candidates"]},
        "issuer": issuer_facts(S),
        "dev": split_results(S, "dev", SYSTEMS),
        "ranking_dev": ranking(S, "dev"),
        "split_half": repeated_split_half(S, CONFIDENT),
        "date_halves": date_halves(S, ("keyword_rule", "issuer_rate", "tfidf_lr") + JUDGE_VERSIONS + ("guarded",)),
        "cues": cue_table(splits["train"], records),
        "error_analysis": error_analysis(records, splits),
        "automation_dev": auto_dev,
        "routing": routing(S, auto_dev),
        "gaming": gaming(records, splits, cache, S, thresholds),
        "retest": retest(records, splits, cache),
        "label_consistency": label_consistency(),
        "cost": cost(cache, splits),
        "human": human_labels(S),
    }
    m["threshold_stability"] = threshold_stability(S, m["automation_dev"], m["retest"])
    if "guarded" in m["dev"]:
        fm = S.fact_model()
        m["fact_model"] = {"C": fm.C, "vocab": list(fm.vocab), "coef": fm.lr.coef_[0].tolist(),
                           "intercept": float(fm.lr.intercept_[0])}
    m["rehearsal"] = rehearsal(S)
    frozen = freeze.load()
    m["frozen"] = None
    if frozen is not None:
        m["frozen"] = {"sha256": freeze.sha256_path(freeze.FROZEN_FILE), "content": frozen,
                       "problems": freeze.verify_method(frozen) + freeze.verify_numbers(frozen, m)}
    m["test"] = test_results(S, frozen)
    m["estimates"] = estimates(m["automation_dev"], m["test"])
    curves = {"dev": {n: [(p["coverage"], p["agreement"]) for p in r["token"]["curve"]]
                      for n, r in m["automation_dev"].items()}}
    if m["test"]:
        curves["test"] = {n: [(p["coverage"], p["agreement"]) for p in r["curve"]]
                          for n, r in m["test"]["automation"].items()}
    if m["automation_dev"]:
        from .figures import automation_figure
        m["figure"] = automation_figure(curves, C.TARGET_AGREEMENT)
    return m


def report_markdown(m: dict, blocks: dict, render) -> str:
    parts = ["# Results report", "",
             "Generated by `python -m judge.report` from `data/` and the cached model "
             "responses. Do not edit by hand.", ""]
    for name in blocks:
        parts += [f"## {name.replace('_', ' ').capitalize()}", "",
                  f"<!-- BEGIN {name} -->", "<!-- END " + name + " -->", ""]
    return render("\n".join(parts), m)
