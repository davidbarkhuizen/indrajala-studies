"""The window networks' diagnosis (indrajala_studies/window_diagnostics.py), on stand-in networks."""

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from indrajala_studies import window_diagnostics as wd
from indrajala_studies import window_tasks as wt

DIVISION = wt.ModularDivision(5)


class _Backend:
    def matrix(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [list(row) for row in rows]


class MachineNetwork:
    """A stand-in at k 1 that runs a task's machine, each window's codes a function of the machine's state."""

    backend = _Backend()
    columns = 1

    def __init__(self, task: wt.Task, codes: Callable[[int], tuple[int, ...]], wrong: int | None = None) -> None:
        self.task, self.codes, self.wrong = task, codes, wrong

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> list[tuple[float, ...]]:
        return [tokens for tokens, _labels in rows]

    def classify_rows(self, prepared: Sequence[tuple[float, ...]]) -> list[tuple[int, ...]]:
        """The machine's answers, except a class it gets wrong (answered 0), if any."""
        answers = [self.task.machine.run([int(x) for x in tokens]) for tokens in prepared]
        return [tuple(0 if label == self.wrong else label for label in labels) for labels in answers]

    def state_codes(self, X: Sequence[Sequence[float]]) -> list[list[tuple[int, ...]]]:
        machine = self.task.machine
        result: list[list[tuple[int, ...]]] = []
        for row in X:
            state = machine.start
            codes = [self.codes(state)]
            for token in row:
                state = int(machine.transition[state, int(token)])
                codes.append(self.codes(state))
            result.append(codes)
        return result


def residues(state: int) -> tuple[int, int]:
    return divmod(state, 5)  # division's state A * 5 + B


def test_a_share_is_one_when_the_code_determines_x_and_zero_when_independent():
    assert wd.share([(c, c % 2) for c in range(8)]) == 1.0
    assert wd.share([(c, x) for c in range(4) for x in range(4)]) == 0.0
    assert wd.share([(c, 7) for c in range(4)]) == 1.0  # x doesn't vary


def test_factors_holding_the_residues_share_all_of_each_and_the_state():
    diagnosis = wd.diagnose(MachineNetwork(DIVISION, residues), DIVISION, (3, 6), 60)
    a, b = (factor["shares"] for factor in diagnosis["factors"])
    assert a["A"] == pytest.approx(1.0) and a["A=0"] == pytest.approx(1.0) and a["B"] < 0.5
    assert b["B"] == pytest.approx(1.0) and b["B=0"] == pytest.approx(1.0) and b["A"] < 0.5
    assert diagnosis["state"]["share"] == pytest.approx(1.0)
    assert all(right == total for right, total in diagnosis["classes"].values())


def test_factors_tracking_only_zero_share_the_zero_tests_not_the_residues():
    def zeros(state: int) -> tuple[int, int]:
        A, B = residues(state)
        return int(A == 0), int(B == 0)

    diagnosis = wd.diagnose(MachineNetwork(DIVISION, zeros, wrong=3), DIVISION, (3, 6), 60)
    a, b = (factor["shares"] for factor in diagnosis["factors"])
    assert a["A=0"] == pytest.approx(1.0) and a["A"] < 0.5
    assert b["B=0"] == pytest.approx(1.0) and b["B"] < 0.5
    right, total = diagnosis["classes"]["3"]
    assert right == 0 and total > 0  # the class it answers wrong
    assert diagnosis["factors"][0]["codes_used"] == 2


def test_addition_is_read_by_its_carry_and_a_machine_without_coordinates_by_its_state():
    diagnosis = wd.diagnose(MachineNetwork(wt.ADDITION, lambda s: (s,)), wt.ADDITION, (3,), 30)
    assert set(diagnosis["factors"][0]["shares"]) == {"carry", "carry=0", "answer"}
    assert diagnosis["factors"][0]["shares"]["carry"] == pytest.approx(1.0)
    bare = dataclasses.replace(wt.ADDITION.machine, coordinates=None)
    assert {name: of(1) for name, of in wd.coordinates(bare).items()} == {"state": 1}


def test_each_new_step_is_diagnosed_once_beside_its_network(tmp_path: Path):
    network = MachineNetwork(DIVISION, residues)

    def save(step: int) -> None:
        (tmp_path / f"step-{step}.json").write_text(
            json.dumps({"step": step, "stop": "plateau", "epochs": 3, "break": 0, "certificate": {"certified": False}})
        )
        (tmp_path / f"step-{step}.network.json").write_text("{}")

    def load(path: str) -> Any:
        return network

    save(1)
    save(2)
    lines = wd.diagnose_new(tmp_path, DIVISION, load, (3,), 20)
    assert [text.split(":")[0] for text in lines] == [f"{tmp_path.name} step-1", f"{tmp_path.name} step-2"]
    assert "f0 A 1.00" in lines[0] and "certified False" in lines[0]
    written = json.loads((tmp_path / "step-1.diagnosis.json").read_text())
    assert written["step"]["stop"] == "plateau" and written["task"] == DIVISION.name
    assert wd.diagnose_new(tmp_path, DIVISION, load, (3,), 20) == []
    save(10)
    assert [text.split(":")[0] for text in wd.diagnose_new(tmp_path, DIVISION, load, (3,), 20)] == [
        f"{tmp_path.name} step-10"
    ]


def test_congruence_shares_read_a_factor_holding_a_place_and_one_holding_nothing():
    machine = wt.task_named("word-problem-s3").machine
    named = wt.named_labels(machine.coordinates)
    place = named["places {1}"]
    windows = [wd.Window(0, 0, s, (int(place[s]), 0)) for s in range(machine.states) for _ in range(5)]
    read = wd.congruence_shares(windows, machine)
    held, idle = read["factors"]
    assert held["named"]["places {1}"] == 1.0 and held["named"]["parity"] == 0.0  # the sign independent of a place
    assert all(v == 0.0 for v in idle["named"].values()) and idle["state"] == 0.0
    assert set(held["named"]) == {"first 2", "places {0}", "places {1}", "places {2}", "parity"}
    assert 0.0 < read["state"] < 1.0 and read["state"] == held["state"]
