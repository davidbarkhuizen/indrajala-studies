"""
scripts/measurement/window_study_timing.py: the end-to-end cases run and their breakdowns account for
the step; the learn and forward cases are window_recurrent_timing.py's. Nothing is timed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "measurement"))
import window_study_timing as study_timing  # scripts/ isn't a package


def test_the_default_case_is_the_long_test_at_1024_columns() -> None:
    assert [study_timing.timing.parse_case(case) for case in study_timing.DEFAULT_CASES] == [
        ("long-test", "family", {"p": 7, "f": 32, "n": 1024, "c": 1023}, "U1")
    ]


@pytest.mark.parametrize(
    "case", ["long-test-family-p5-f16-U1-n8-c5", "study-family-p5-f16-e2", "learn-family-p5-f16-B4-W3"]
)
def test_each_kind_runs_and_its_breakdown_accounts_for_the_step(case: str) -> None:
    step, inference = study_timing.case_step(case)
    step()
    assert inference == (not case.startswith("learn"))
    lines = study_timing.timing.breakdown(case, study_timing.case_step)
    shares = [float(line.rsplit(", ", 1)[1].rstrip("%")) for line in lines[1:]]
    assert all(share >= 0 for share in shares) and sum(shares) == pytest.approx(100, abs=0.5)
