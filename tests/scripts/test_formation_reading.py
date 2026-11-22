"""scripts/studies/formation_reading.py: a survey reads each run's machine from its path; a batch's from its backdrop."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "studies"))
import formation_reading as script  # scripts/ isn't a package


def write(directory: Path, shares: list[float], certified_at: int | None = None) -> None:
    directory.mkdir(parents=True)
    rows = [
        {"step": s, "end": "passed", "epochs": 10, "state_share": share, "certified": s == certified_at}
        for s, share in enumerate(shares, start=1)
    ]
    (directory / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_survey_reads_the_machine_from_the_path_and_leaves_other_methods_out(tmp_path: Path) -> None:
    write(tmp_path / "general-algebras" / "round1" / "word-problem-s4" / "seed1", [0.5, 1.0], certified_at=2)
    write(tmp_path / "paper" / "t2-t5" / "attempts" / "d13-family-300", [0.1] * 12)
    write(tmp_path / "paper" / "t2-t5" / "attempts" / "rq16-301" / "seed301", [0.1] * 13)
    write(tmp_path / "state-formation" / "option1" / "probe-d55" / "s3-seed1" / "seed1", [1.0], certified_at=1)
    found = {
        label: run.depth_k
        for log in sorted(tmp_path.rglob("log.jsonl"))
        if (made := script.survey_run(log, tmp_path))
        for label, run in [made]
    }
    assert found == {"s4": 5, "division": 4, "random": 5}
    lines = script.survey([tmp_path], [2.5], script.rules(["2.5:0.9"]))
    assert lines[0] == "3 curriculum runs, 1 certified"
    assert "division (1), not formed by x(d+k), certified later: 2.5: 0/1" in lines
    assert lines[-1].startswith("2.5(d+k), share < 0.9: stops 2, loses 0 of 1 certificates")


def test_a_batch_reads_each_attempt_on_its_backdrops_machine(tmp_path: Path) -> None:
    write(tmp_path / "attempts" / "d23-family-600", [0.5] * 12)
    write(tmp_path / "attempts" / "d19-family-601", [0.5, 1.0, 1.0], certified_at=3)
    lines = script.batch(tmp_path, script.rules(["2.5:0.9"]))
    assert lines[0] == "2 runs, 1 certified"
    assert lines[1] == "  2.5(d+k), share < 0.9: d23-family-600 at step 10, epoch 100 of 120, share 0.50"
