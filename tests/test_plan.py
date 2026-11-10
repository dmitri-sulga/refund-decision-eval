import shutil
import subprocess

import pytest

from judge import config as C

FIRST_COMMIT = "37b32d1"


def test_analysis_plan_is_the_one_in_the_first_commit():
    if not shutil.which("git"):
        pytest.skip("git not available")
    got = subprocess.run(["git", "-C", str(C.ROOT), "show", f"{FIRST_COMMIT}:README.md"],
                         capture_output=True, text=True)
    if got.returncode != 0:
        pytest.skip("first commit not available (shallow or exported copy)")
    old = got.stdout
    plan = old[old.index("## Pre-registered analysis plan"): old.index("<!-- BEGIN settings -->")].rstrip()
    doc = (C.ROOT / "docs" / "analysis_plan.md").read_text()
    assert doc.rstrip().endswith(plan)
