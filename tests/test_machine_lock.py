"""The machine lock (ml's indrajala_ml/measurement/machine_lock.py) taken by the study command lines here."""

import sys
from pathlib import Path

import pytest
from indrajala_ml.measurement import machine_lock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "studies"))
import addition_study  # scripts/ isn't a package


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path))
    return tmp_path


def test_the_addition_run_is_refused_while_the_machine_is_taken(root: Path) -> None:
    argv = ["run", str(REPO / "scripts/studies/configs/addition_sweep.json")]
    with machine_lock.hold("run --bench cmd", "run"), pytest.raises(SystemExit, match="not started: "):
        addition_study.main(argv)
