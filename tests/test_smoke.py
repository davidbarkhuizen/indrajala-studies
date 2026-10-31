from pathlib import Path

import indrajala_math_rust
from indrajala_ml.studies import common

import indrajala_studies

REPO = Path(__file__).resolve().parents[1]


def test_ml_comes_from_the_submodule_and_the_crate_imports():
    assert Path(common.__file__).resolve().is_relative_to(REPO / "ml")
    assert Path(indrajala_studies.__file__).resolve().is_relative_to(REPO)
    assert callable(indrajala_math_rust.Array)
