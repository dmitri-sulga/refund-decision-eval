from judge.human import halves
from judge.label import load_sample


def test_calibration_and_evaluation_halves_follow_the_registered_rule():
    ids = load_sample()["ids"]
    h = halves(ids)
    assert len(h["calibration"]) == len(h["evaluation"]) == 50
    assert set(h["calibration"]) | set(h["evaluation"]) == set(ids)
    assert not set(h["calibration"]) & set(h["evaluation"])
    assert h == halves(list(reversed(ids)))              # independent of presentation order
    import hashlib
    top = max(hashlib.sha256(i.encode()).hexdigest() for i in h["calibration"])
    assert all(hashlib.sha256(i.encode()).hexdigest() > top for i in h["evaluation"])
