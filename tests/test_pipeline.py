import json
import math

import pytest

from judge import config as C
from judge import llm
from judge.data import label_of, normalize_narrative, stratified_split
from judge.gaming import ATTACKS, apply_attack


def test_label_mapping_is_explicit():
    assert label_of("Closed with monetary relief") == 1
    assert label_of("Closed with explanation") == 0
    assert label_of("Closed with non-monetary relief") == 0
    for dropped in ("In progress", "Untimely response", "Closed", "Closed with relief", ""):
        assert label_of(dropped) is None
    assert set(C.LABEL_MAP.values()) == {0, 1}


def test_normalize_only_touches_whitespace():
    raw = "  I was charged {$45.00}\r\n\r\n\r\ntwice   by XXXX.  "
    out = normalize_narrative(raw)
    assert out == "I was charged {$45.00}\n\ntwice by XXXX."
    assert out.split() == raw.split()


def test_stratified_split_sizes_disjoint_and_deterministic():
    ids = [str(i) for i in range(1000)]
    labels = [1 if i % 10 < 3 else 0 for i in range(1000)]   # base rate 0.3
    sizes = {"a": 100, "b": 200, "rest": None}
    s1 = stratified_split(ids, labels, sizes, seed=11)
    s2 = stratified_split(ids, labels, sizes, seed=11)
    assert s1 == s2
    assert [len(s1[k]) for k in ("a", "b", "rest")] == [100, 200, 700]
    assert not set(s1["a"]) & set(s1["b"])
    lab = dict(zip(ids, labels))
    assert sum(lab[i] for i in s1["a"]) == 30
    assert sum(lab[i] for i in s1["b"]) == 60


def test_attacks_only_append():
    base = "My card was charged twice."
    for name in ATTACKS:
        out = apply_attack(base, name)
        assert out.startswith(base + "\n\n")
        assert len(out) > len(base) + 10


def _lp(tokens, top=None):
    """Build a raw Ollama-style logprobs list."""
    out = []
    for i, (t, lp) in enumerate(tokens):
        alts = (top or {}).get(i, [(t, lp)])
        out.append({"token": t, "logprob": lp, "bytes": list(t.encode()),
                    "top_logprobs": [{"token": a, "logprob": b, "bytes": []} for a, b in alts]})
    return out


def test_token_probability_reads_the_decision_position():
    toks = [("{", 0.0), ('"monetary_relief', 0.0), ('":', 0.0), (" true", -0.2), ("}", 0.0)]
    raw = _lp(toks, top={3: [(" true", -0.2), (" false", -1.8), ("!", -9999)]})
    content = "".join(t for t, _ in toks)
    p = llm.token_probability(content, llm.compact_logprobs(raw), "monetary_relief")
    assert p == pytest.approx(1 / (1 + math.exp(-1.8 + 0.2)))


def test_token_probability_when_alternative_missing_uses_floor():
    toks = [('{"monetary_relief":', 0.0), (" false", -0.01), ("}", 0.0)]
    raw = _lp(toks, top={1: [(" false", -0.01), (" ", -5.0)]})
    content = "".join(t for t, _ in toks)
    p = llm.token_probability(content, llm.compact_logprobs(raw), "monetary_relief")
    assert 0 < p < 0.01   # 'true' was not even in the top-k


def test_token_probability_returns_none_without_logprobs():
    assert llm.token_probability('{"monetary_relief": true}', None, "monetary_relief") is None


def test_cache_refuses_silent_prompt_change(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = llm.Cache(path)
    cache.put({"key": "v1|1", "request_sha256": "aaa", "content": "{}"})
    again = llm.Cache(path)
    assert again.get("v1|1", "aaa")["content"] == "{}"
    with pytest.raises(RuntimeError, match="stale"):
        again.get("v1|1", "bbb")
    assert again.get("v1|2", "aaa") is None


def test_request_hash_covers_messages_and_schema():
    b1 = llm.request_body([{"role": "user", "content": "a"}], {"type": "object"})
    b2 = llm.request_body([{"role": "user", "content": "b"}], {"type": "object"})
    b3 = llm.request_body([{"role": "user", "content": "a"}], {"type": "string"})
    assert len({llm.request_hash(b) for b in (b1, b2, b3)}) == 3
    assert json.loads(json.dumps(b1))["think"] is False
    assert b1["options"]["temperature"] == 0
