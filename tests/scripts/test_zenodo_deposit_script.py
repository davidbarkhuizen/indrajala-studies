import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "paper" / "zenodo_deposit.py"
spec = importlib.util.spec_from_file_location("zenodo_deposit", SCRIPT)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules["zenodo_deposit"] = script
spec.loader.exec_module(script)


def test_a_placeholder_left_unfilled_is_refused(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text('{"description": "<p>A preprint (doi:[PAPER DOI]).</p>", "notes": "at tag [RELEASE TAG]"}')
    with pytest.raises(SystemExit, match=r"fill \[PAPER DOI\], \[RELEASE TAG\] first"):
        script.filled(path)
    path.write_text('{"description": "<p>A preprint (doi:10.5281/zenodo.1). <i>[sic]</i></p>"}')
    assert script.filled(path)["description"].startswith("<p>A preprint")
