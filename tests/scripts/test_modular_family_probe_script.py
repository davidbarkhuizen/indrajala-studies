"""scripts/studies/modular_family_probe.py's run command: which runs a pool trains, and where."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "studies"))
import modular_family_probe as script  # scripts/ isn't a package

TINY = ["--factors", "1", "--budget-epochs", "1", "--workers", "2"]


def test_runs_trains_the_chosen_task_and_seed_pairs_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "locks"))  # the study's machine lock, not the machine's
    out = tmp_path / "out"
    assert script.main(["run", "--runs", "word-problem-s3:1", "move-to-front-3:0", *TINY, "--out", str(out)]) == 0
    trained = sorted(str(log.parent.relative_to(out)) for log in out.glob("*/*/log.jsonl"))
    assert trained == ["move-to-front-3/seed0", "word-problem-s3/seed1"]


def test_runs_names_its_tasks_so_task_is_refused(tmp_path: Path):
    with pytest.raises(SystemExit):
        script.main(["run", "--runs", "word-problem-s3:1", "--task", "word-problem-s3", "--out", str(tmp_path)])


def test_run_needs_an_epoch_budget_and_has_no_wall_clock_one_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "locks"))
    # epochs are the binding budget (the owner, 2026-10-23); --budget-minutes is a runaway guard alone
    with pytest.raises(SystemExit):
        script.main(["run", "--runs", "word-problem-s3:1", "--factors", "1", "--out", str(tmp_path)])
    out = tmp_path / "out"
    assert script.main(["run", "--runs", "word-problem-s3:1", *TINY, "--out", str(out)]) == 0
    saved = json.loads((out / "seed1" / "checkpoint.json").read_text())
    assert "budget_seconds" not in saved["run"] and saved["run"]["budget_epochs"] == 1  # a guard, not identity


def test_answers_proof_checks_the_answers_only_stops_network_by_chows_test_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "locks"))
    out = tmp_path / "out"
    assert script.main(["run", "--runs", "word-problem-s3:1", *TINY, "--out", str(out)]) == 0
    log = out / "seed1" / "log.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    end = next(r for r in rows if "end" in r)
    assert (log.parent / f"step-{end['step']}.network.json").exists()  # the cell keeps every step's network
    # the step's answers made all right, uncertified: the stop a false discovery, which Chow's test set refutes
    end.update(certified=False, minimised=1, accuracy=dict.fromkeys(end["accuracy"], 1.0), long={"24": 1.0})
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    capsys.readouterr()
    args = ["answers-proof", str(log), "--task", "word-problem-s3", "--slack", "1", "10", "--limit", "1000"]
    assert script.main([*args, "--out", str(tmp_path / "proof")]) == 0
    printed = capsys.readouterr().out
    assert f"x1 bound 6: false-discovery, stop {end['step']}, certified None; refuted" in printed
    assert "x10 bound 60: false-discovery" in printed and "out of reach" in printed
    (saved,) = (tmp_path / "proof").glob("*.answers-proof.json")
    exact = json.loads(saved.read_text())["bounds"][0]
    assert exact["chow"]["equivalent"] is False and exact["chow"]["counterexample"]


def test_factor_shares_reads_a_trained_network_and_writes_every_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "locks"))
    out = tmp_path / "out"
    assert script.main(["run", "--runs", "word-problem-s3:1", *TINY, "--out", str(out)]) == 0
    network = out / "seed1" / "network.json"
    capsys.readouterr()
    args = ["factor-shares", str(network), "--task", "word-problem-s3", "--show", "0"]
    assert script.main([*args, "--out", str(tmp_path / "shares")]) == 0
    assert f"{network}: state " in capsys.readouterr().out
    (saved,) = (tmp_path / "shares").glob("*.factor-shares.json")
    read = json.loads(saved.read_text())
    assert len(read["factors"]) == 1 and set(read["factors"][0]["named"]) >= {"parity", "places {0}"}
    assert all(0.0 <= v <= 1.0 for v in read["factors"][0]["named"].values())
