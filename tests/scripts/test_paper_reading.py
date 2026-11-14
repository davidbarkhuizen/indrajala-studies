"""scripts/paper/paper_reading.py: each network found where its backdrop logs, read with its backdrop's task, in fixed shares."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "paper"))
import paper_reading as script  # scripts/ isn't a package

from indrajala_studies.batch import BACKDROPS


def test_networks_are_found_where_each_backdrop_logs(tmp_path: Path) -> None:
    division = tmp_path / "attempts" / "d11-quotient-301"
    machine = tmp_path / "attempts" / "s4-302" / "seed302"
    for directory in (division, machine):
        directory.mkdir(parents=True)
        (directory / "certified-step-7.network.json").write_text("{}")
    (tmp_path / "attempts" / "s4-302" / "certified-step-9.network.json").write_text("{}")  # not where s4 logs
    found = list(script.networks(tmp_path))
    assert [(n.attempt, n.backdrop, n.seed, n.name) for n in found] == [
        ("d11-quotient-301", "d11-quotient", 301, "d11-quotient-301-certified-step-7"),
        ("s4-302", "s4", 302, "s4-302-certified-step-7"),
    ]


def test_task_args_follow_the_backdrop() -> None:
    assert script.task_args(BACKDROPS["s3"]) == ["--task", "word-problem-s3"]
    assert script.task_args(BACKDROPS["d11-quotient"]) == ["--task", "modular-division-11", "--arm", "quotient"]
    assert script.task_args(BACKDROPS["d5-family-wide"]) == [
        "--task",
        "modular-division-5",
        "--arm",
        "family",
        "--vocabulary",
        "sum",
        "difference",
        "product",
    ]


def test_shares_cover_every_network_once() -> None:
    items = [script.Network(f"s3-{i}", "s3", i, Path(f"{i}")) for i in range(7)]
    shares = [script.share(items, f"{i}/3") for i in range(3)]
    assert sorted(n.seed for part in shares for n in part) == list(range(7))


def test_a_cells_prefix_is_read_and_a_continuous_cell_has_no_machine() -> None:
    assert script.split("approximation.d5-family-wide-301") == ("d5-family-wide", 301)
    assert script.group("approximation.d5-family-wide-301") == "approximation.d5-family-wide"
    assert not script.coded("approximation-continuous.d5-family-wide-300")
    assert script.coded("approximation-linear.d5-family-wide-300") and script.coded("s3-300")


def test_the_probe_is_this_repositorys_and_a_machines_vocabulary_is_read_with_its_networks() -> None:
    """The probe moved here from ml (2026-10-31); the told plan's machine arms are read on their vocabulary."""
    assert script.PROBE.exists()
    assert script.task_args(BACKDROPS["told-matched-54"]) == [
        "--task",
        "random-planted-q16-b4-54",
        "--arm",
        "matched",
        "--vocabulary",
        "planted",
        "matched",
    ]
    assert script.task_args(BACKDROPS["t10-p2-0"]) == ["--task", "random-planted-q16-b4-0", "--arm", "planted"]


def test_told_reads_a_batchs_runs_into_the_five_tests_and_the_cost_measures(tmp_path: Path) -> None:
    """Stage 1's reading wired to a batch's links: here every run spent its budget, so nothing succeeds."""
    import json

    def spent(directory: Path) -> None:
        directory.mkdir(parents=True)
        rows = [{"step": 1, "end": "spent", "state_share": 0.5, "epochs": 401, "seconds": 1.0}]
        (directory / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    for arm in ("quotient", "sum", "matched"):
        for seed in (800, 801):
            spent(tmp_path / "runs" / f"told-d5-{arm}" / f"seed{seed}")
    for arm in ("alone", "planted", "matched"):
        for machine in (54, 55):
            spent(tmp_path / "runs" / f"told-{arm}-{machine}" / "seed840")
    (tmp_path / "certify").mkdir()
    (tmp_path / "long-tests").mkdir()
    lines = script.told(tmp_path)
    assert lines[0] == "division sum: 0 of 2 seeds succeeded"
    assert "planted alone: 0 of 2 machines, 0 runs succeeded" in lines
    assert sum("Holm 1" in line for line in lines) == 5
    assert "reading, division at p 5: undecided" in lines and "reading, planted machines: undecided" in lines
    assert sum("2 runs (1 + 1), 0 certified as run" in line for line in lines) == 6  # each arm's cost
    assert sum("0 ties" not in line and "ties" in line for line in lines) == 5  # each test's epochs, all tied
