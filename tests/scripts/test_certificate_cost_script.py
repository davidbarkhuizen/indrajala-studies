"""scripts/certnote/certificate_cost.py (the certificate note's stage 4): the cost reading from recertify's results."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "certnote"))
import certificate_cost as ccost  # scripts/ isn't a package


def result(job: str, task: str, k: int, reachable: int, windows: int, seconds: float) -> dict[str, Any]:
    entries = reachable * windows
    return {
        "job": {"id": job, "task": task, "columns": k},
        "certified": True,
        "reachable": reachable,
        "certify_seconds": 2 * seconds,
        "seconds": seconds,
        "check": {
            "valid": True,
            "entries": entries,
            "windows": windows,
            "bytes": 20 * entries,
            "structure_seconds": 0.1,
            "fidelity_seconds": seconds,
        },
    }


def test_families_are_read_with_their_sizes_costs_and_chow_words(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    rows = [
        result("a", "modular-sum-4", 1, 6, 9, 0.5),
        result("b", "modular-sum-4", 1, 9, 9, 1.0),
        result("c", "random-q8-1", 1, 17, 5, 0.2),
        result("d", "random-q8-1", 1, 17, 5, 0.2),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = {t["family"]: t for t in ccost.readings(ccost.rows([path]))}
    assert set(table) == {"modular-sum-4", "random-q8"}
    sums = table["modular-sum-4"]
    assert sums["jobs"] == 2 and sums["reachable"] == [6, 9] and sums["entries"] == [54, 81]
    assert sums["certify_seconds"] == [0.5, 1.0]  # the emitted certificate's own search, not every encoding
    assert sums["bytes_per_entry"] == 20.0 and sums["target_states"] == 4
    words = sums["chow_words"]
    assert words is not None and words[0] < words[1] < words[2]


def test_a_rerun_replaces_the_earlier_line(tmp_path: Path):
    first, second = tmp_path / "1.jsonl", tmp_path / "2.jsonl"
    first.write_text(json.dumps(result("a", "modular-sum-4", 1, 6, 9, 5.0)) + "\n")
    second.write_text(json.dumps(result("a", "modular-sum-4", 1, 6, 9, 1.0)) + "\n")
    (reading,) = ccost.readings(ccost.rows([first, second]))
    assert reading["jobs"] == 1 and reading["certify_seconds"] == [1.0, 1.0]
