"""indrajala_studies.formation: a run's formation on the step over depth plus k, and a stopping rule's effect."""

import json
from pathlib import Path

import pytest

from indrajala_studies import formation as fm
from indrajala_studies import window_tasks as wt


def run(name: str, shares: list[float], certified_at: int | None = None, depth_k: int = 4, epochs: int = 10) -> fm.Run:
    ends = tuple(
        fm.StepEnd(step, step * epochs, share, step == certified_at) for step, share in enumerate(shares, start=1)
    )
    return fm.Run(name, ends, depth_k)


def test_depth_plus_k_is_the_coverage_rules_measure() -> None:
    assert fm.depth_plus_k(wt.modular_machine(13)) == 4 and fm.depth_plus_k(wt.modular_machine(5)) == 3
    assert fm.depth_plus_k(wt.task_named("word-problem-s4").machine) == 5


def test_step_ends_count_epochs_and_keep_the_last_run() -> None:
    rows = [
        {"step": 1, "end": "passed", "epochs": 5, "state_share": 0.5, "certified": False},
        {"step": 1, "end": "passed", "epochs": 3, "state_share": 0.4, "certified": False},  # a re-run starts again
        {"step": 2, "epoch": 1, "state_share": 0.9},  # a diagnosis, not an end
        {"step": 2, "end": "passed", "epochs": 4, "state_share": 1.0, "certified": True},
    ]
    assert fm.step_ends(rows) == (fm.StepEnd(1, 3, 0.4, False), fm.StepEnd(2, 7, 1.0, True))


def test_a_cut_line_is_skipped(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"step": 1, "end": "spent", "epochs": 2, "state_share": 0.3}) + '\n{"step": 2, "en')
    assert fm.read_run(log, 4).ends == (fm.StepEnd(1, 2, 0.3, False),)


def test_a_rule_stops_at_its_point_before_a_certificate_and_counts_certificates_per_epoch() -> None:
    late = run("late", [0.2, 0.5, 0.8, 0.95, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], certified_at=11)
    never = run("never", [0.2] * 20)
    early = run("early", [0.5, 1.0, 1.0], certified_at=3)
    rule = fm.Rule(2.5, 0.99)  # step 10 at depth plus k 4
    assert fm.stop(never, rule) == never.ends[9] and fm.stop(early, rule) is None
    assert fm.stop(late, rule) is None  # formed (1.0) by step 10
    assert fm.stop(late, fm.Rule(1.0, 0.99)) == late.ends[3]  # at step 4 its share is 0.95
    assert fm.formed_step(late) == 6 and fm.formed_step(never) is None
    e = fm.effect([late, never, early], fm.Rule(1.0, 0.99))
    assert (e.stopped, e.lost, e.certified) == (3 - 1, 1, 2)  # early formed at step 2, before step 4
    spent, saved = 110 + 200 + 30, (110 - 40) + (200 - 40)
    assert e.saved == pytest.approx(saved / spent)
    assert e.gain == pytest.approx((1 / (spent - saved)) / (2 / spent))


def test_unformed_later_counts_the_runs_not_formed_by_the_point() -> None:
    runs = [
        run("a", [0.5] * 11 + [1.0], certified_at=12),
        run("b", [0.5] * 12),
        run("c", [1.0] * 12),
        run("d", [0.5] * 5),
    ]
    assert fm.unformed_later(runs, 2.5) == (1, 2)  # c formed before step 10, d ended before it
