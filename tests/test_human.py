from judge.human import halves
from judge.label import load_sample


def test_calibration_and_evaluation_halves_follow_the_registered_rule():
    ids = load_sample()["ids"]
    from judge.data import load_splits
    from judge.human import sample_from
    assert ids == sample_from(load_splits()["train"])
    h = halves(ids)
    assert len(h["calibration"]) == len(h["evaluation"]) == 50
    assert set(h["calibration"]) | set(h["evaluation"]) == set(ids)
    assert not set(h["calibration"]) & set(h["evaluation"])
    assert h == halves(list(reversed(ids)))              # independent of presentation order
    import hashlib
    top = max(hashlib.sha256(i.encode()).hexdigest() for i in h["calibration"])
    assert all(hashlib.sha256(i.encode()).hexdigest() > top for i in h["evaluation"])


def test_sample_rule_uses_only_a_hash_of_the_id():
    import hashlib

    from judge.human import SAMPLE_SALT, sample_from

    ids = [str(i) for i in range(1000, 1300)]
    got = sample_from(ids, 10)
    key = lambda i: hashlib.sha256((SAMPLE_SALT + i).encode()).hexdigest()  # noqa: E731
    assert got == sorted(ids, key=key)[:10] and got == sample_from(list(reversed(ids)), 10)
