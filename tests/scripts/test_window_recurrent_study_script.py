"""scripts/studies/window_recurrent_study.py's run, check and certify commands."""

import contextlib
import json
import sys
from pathlib import Path

import pytest
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "studies"))
import window_recurrent_study as script  # scripts/ isn't a package


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    path = tmp_path / "curriculum-k1-s4-m3-seed0" / "step-1.network.json"
    path.parent.mkdir()
    wr.build(wt.ADDITION, 1, 4, 3, seed=0).save(str(path))
    return path


def test_check_runs_the_long_tests_as_asked(saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "check.json"
    argv = ["check", str(saved), "--n", "16", "--properties", "B1", "M3", "--sampled", "30"]
    assert script.main([*argv, "--per-stratum", "2", "--doubling", "--json", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == ["n 16, B1", "n 16, M3"]
    result = json.loads(out.read_text())["lengths"]["16"]
    assert result["properties"]["B1"]["cases"] == 30
    assert sorted(map(int, result["properties"]["M3"]["strata"])) == [1, 2, 4, 8, 16]


def test_certify_writes_each_networks_certificate(saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "certificates"
    assert script.main(["certify", str(saved), "--out", str(out)]) == 0
    assert capsys.readouterr().out.startswith("curriculum-k1-s4-m3-seed0-step-1: ")
    certificate = json.loads((out / "curriculum-k1-s4-m3-seed0-step-1.json").read_text())
    assert certificate["network"] == str(saved) and len(certificate["sha256"]) == 64
    assert certificate["codes"] == [3] and certificate["evaluations"] == 27 * len(certificate["reachable"])


def test_certify_strict_records_its_snap_ratio(saved: Path, tmp_path: Path):
    out = tmp_path / "strict"
    assert script.main(["certify", str(saved), "--out", str(out), "--snap-ratio", "1"]) == 0
    certificate = json.loads((out / "curriculum-k1-s4-m3-seed0-step-1.json").read_text())
    assert certificate["snap_ratio"] == 1.0 and certificate["snap"]["short"] == 0


@pytest.fixture
def saved_division(tmp_path: Path) -> Path:
    path = tmp_path / "modular-division-5-curriculum-k1-s4-m4-seed0" / "step-1.network.json"
    path.parent.mkdir()
    wr.build(wt.ModularDivision(5), 1, 4, 4, seed=0).save(str(path))
    return path


def test_check_and_certify_a_division_network(saved_division: Path, tmp_path: Path):
    out = tmp_path / "check.json"
    argv = ["check", str(saved_division), "--task", "modular-division-5", "--n", "16", "--properties", "U1", "S2"]
    assert script.main([*argv, "--sampled", "30", "--per-stratum", "2", "--doubling", "--json", str(out)]) == 0
    result = json.loads(out.read_text())["lengths"]["16"]["properties"]
    assert result["U1"]["cases"] == 30 and sorted(map(int, result["S2"]["strata"])) == [1, 2, 4, 8, 16]
    certificates = tmp_path / "certificates"
    assert (
        script.main(["certify", str(saved_division), "--task", "modular-division-5", "--out", str(certificates)]) == 0
    )
    certificate = json.loads((certificates / "modular-division-5-curriculum-k1-s4-m4-seed0-step-1.json").read_text())
    assert certificate["task"] == "modular-division-5" and certificate["evaluations"] == 9 * len(
        certificate["reachable"]
    )


def test_run_arm_runs_that_arms_runs_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "config.json"
    matrix = [{"arm": arm, "k": [1], "s": [4], "m": [3]} for arm in wr.ARMS]
    config.write_text(json.dumps({"directory": str(tmp_path / "study"), "seeds": 2, "matrix": matrix}))
    ran: list[wr.Study] = []

    def run(study: wr.Study, log: object) -> None:
        ran.append(study)

    def report(study: wr.Study) -> str:
        return ""

    monkeypatch.setattr(script.machine_lock, "study", contextlib.nullcontext)
    monkeypatch.setattr(script.wr, "run", run)
    monkeypatch.setattr(script.wr, "report", report)
    assert script.main(["run", str(config), "--arm", "all-at-once", "--workers", "2"]) == 0
    (study,) = ran
    assert [spec.key for spec in study.runs] == ["all-at-once-k1-s4-m3-seed0", "all-at-once-k1-s4-m3-seed1"]
    assert study.workers == 2 and study.directory == tmp_path / "study"


def test_probe_prints_and_keeps_a_factored_studys_figures(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tmp_path / "study.json"
    config.write_text(
        json.dumps(
            {
                "directory": str(tmp_path / "study"),
                "task": "modular-division-5",
                "epoch_examples": 64,
                "screen": 20,
                "sampled": 30,
                "relation_sampled": 30,
                "per_stratum": 2,
                "steps": 3,
                "certify_steps": True,
                "matrix": [{"k": [1], "factors": [[4, 8], [4, 8]], "reads": "before_state"}],
            }
        )
    )
    out = tmp_path / "probe.json"
    assert script.main(["probe", str(config), "--epochs", "1", "--json", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == ["step 3 epoch 1", "whole check at 3", "certificate"]
    figures = json.loads(out.read_text())
    assert [row["step"] for row in figures["epochs"]] == [3] and not (tmp_path / "study").exists()


def test_diagnose_writes_each_steps_diagnosis_once_and_follow_ends_with_the_lock_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "ab-runs"))  # no holder: --follow makes one pass
    config = tmp_path / "study.json"
    config.write_text(
        json.dumps(
            {
                "directory": str(tmp_path / "study"),
                "task": "modular-division-5",
                "matrix": [{"k": [1], "factors": [[4, 8], [4, 8]], "reads": "before_state"}],
            }
        )
    )
    study = wr.load_config(config)
    spec = study.runs[0]
    run = study.models / spec.key
    run.mkdir(parents=True)
    wr.build(wt.task_named(spec.task), 1, 0, None, 0, recurrence=spec.recurrence).save(str(run / "step-1.network.json"))
    (run / "step-1.json").write_text(json.dumps({"step": 1, "stop": "plateau", "epochs": 2, "break": 0}))
    argv = ["diagnose", str(config), "--n", "3", "--cases", "20"]
    assert script.main([*argv, "--follow", "--interval", "0"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == [f"{spec.key} step-1"]
    assert "f1 A " in lines[0] and "| state " in lines[0]
    assert json.loads((run / "step-1.diagnosis.json").read_text())["lengths"] == [3]
    assert script.main(argv) == 0 and capsys.readouterr().out == ""
