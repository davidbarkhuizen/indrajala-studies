"""scripts/paper/certificate_note_tables.py (the certificate note's D11): numbers and tables from the result files, checked against a manuscript."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "paper"))
import certificate_note_tables as cnt  # scripts/ isn't a package


def result(job: str, sha: str, task: str, certified: bool = True) -> dict[str, Any]:
    return {
        "job": {"id": job, "sha256": sha, "task": task, "columns": 1, "network": f"m/{job}/step-1.network.json"},
        "certified": certified,
        "reproduced": True,
        "encodings_reproducing": 1,
        "reachable": 6,
        "certify_seconds": 0.2,
        "seconds": 0.1,
        "check": {
            "valid": certified,
            "agrees": True,
            "entries": 54,
            "windows": 9,
            "bytes": 1000,
            "structure_seconds": 0.01,
            "fidelity_seconds": 0.05,
            "differences": {"snaps_moved": 0, "leads_not_bitwise": 0},
        },
    }


def fault(table: str, rank: str, miss: float) -> dict[str, Any]:
    return {
        "table": table,
        "task": "modular-sum-4",
        "rank": rank,
        "columns": 1,
        "caught": True,
        "counterexample_windows": 2,
        "catalogue_miss": miss,
    }


def bound(table: str, labels_sound: int = 54) -> dict[str, Any]:
    return {
        "table": table,
        "network": f"m/{'curriculum-linear' if table == 't' else 'other'}/step-1.network.json",
        "task": "modular-sum-4",
        "entries": 54,
        "labels_sound": labels_sound,
        "labels_undecided": 0,
        "labels_contradicted": 0,
        "snaps_sound": 50,
        "snaps_covered": 4,
        "snaps_uncovered": 0,
        "outside": 0,
        "ulps": 4,
    }


def affine(table: str) -> dict[str, Any]:
    return {**bound(table), "snaps_sound": 54, "snaps_covered": 0, "cap": 1024, "condensed": 3, "seconds": 7200.0}


@pytest.fixture
def data(tmp_path: Path) -> Path:
    for directory in ("recertify", "faults", "bound", "affine"):
        (tmp_path / directory).mkdir()
    (tmp_path / "inventory.csv").write_text("file,sha256\na.json,aa\nb.json,aa\nc.json,bb\n")
    (tmp_path / "recertify" / "results.jsonl").write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                result("curriculum-linear-1.0", "aa", "modular-sum-4"),
                result("b-1.05", "bb", "modular-sum-4", False),
            ]
        )
    )
    (tmp_path / "faults" / "faults.jsonl").write_text(
        "".join(
            json.dumps(f) + "\n"
            for f in [fault("t", "rarest", 0.7), fault("t", "median", 1e-30), fault("t", "commonest", 0.0)]
        )
    )
    (tmp_path / "bound" / "bound.jsonl").write_text(
        json.dumps(bound("t"))
        + "\n"
        + json.dumps(bound("v", 50))
        + "\n"
        + json.dumps({"table": "u", "unsupported": "branches"})
        + "\n"
    )
    (tmp_path / "affine" / "affine.jsonl").write_text(json.dumps(affine("v")) + "\n")
    assert cnt.main(["--data", str(tmp_path), "manifest"]) == 0
    return tmp_path


def test_the_numbers_count_what_the_files_hold(data: Path):
    made = cnt.numbers(data)
    assert made["inventory-files"] == "3" and made["inventory-networks"] == "2"
    assert made["recert-jobs"] == "2" and made["recert-certified"] == "1" and made["recert-uncertified-invalid"] == "1"
    assert (
        made["faults-caught"] == "3" and made["faults-likely-missed"] == "1" and made["fault-worst-miss"] == "0.70000"
    )
    assert made["bound-tables"] == "2" and made["bound-unsupported"] == "1" and made["bound-full"] == "1"
    assert made["affine-tables"] == "1" and made["affine-full"] == "1" and made["affine-capped-tables"] == "1"
    assert made["affine-mismatch"] == "0" and made["bounds-decided"] == "2" and made["affine-longest-hours"] == "2"
    assert made["linear-jobs"] == "1" and made["linear-certified"] == "1" and made["linear-bound-full"] == "1"


def test_a_table_the_interval_left_undecided_without_an_affine_run_is_a_mismatch(data: Path):
    (data / "affine" / "affine.jsonl").write_text("")
    assert cnt.affine_numbers(data)["affine-mismatch"] == "1"


def test_the_bound_table_reads_both_bounds(data: Path):
    (row,) = cnt.bound_table(data)
    assert row.startswith("sum mod 4 & 2 & 1 & ") and row.endswith(" & 1 & 1 & 0 \\\\")


def test_the_affine_results_are_in_the_manifest(data: Path):
    (data / "affine" / "affine.jsonl").write_text("")
    assert any("affine/affine.jsonl" in p for p in cnt.manifest_problems(data))


def test_a_table_rows_miss_chances_as_the_note_writes_them(data: Path):
    (row,) = cnt.faults_table(data)
    assert row == "sum mod 4 & 3 & 3 & 2 & 0.70 & $10^{-30}$ & $0$ \\\\"


def test_check_passes_a_filled_manuscript_and_fails_a_changed_number(data: Path, tmp_path: Path):
    tex = tmp_path / "note.tex"
    tex.write_text(
        "\\pn{faults}{0} faults.\n\\begin{table}\\label{tab:faults}\n\\midrule\nold\n\\bottomrule\n\\end{table}\n"
    )
    assert cnt.main(["--data", str(data), "check", str(tex)]) == 1
    tex.write_text(tex.read_text().replace("old", cnt.faults_table(data)[0]))
    assert cnt.main(["--data", str(data), "fill", str(tex)]) == 0
    assert cnt.main(["--data", str(data), "check", str(tex)]) == 0


def test_a_changed_data_file_fails_the_manifest(data: Path):
    (data / "bound" / "bound.jsonl").write_text("")
    assert cnt.main(["--data", str(data), "numbers"]) == 1
