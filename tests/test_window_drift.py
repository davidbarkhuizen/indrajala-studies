"""The drift profile (indrajala_studies/window_drift.py), on stand-in networks at k 1."""

import itertools
from collections.abc import Sequence

import numpy as np
import pytest
from indrajala_ml.studies import window_tasks as wt

from indrajala_studies import window_drift as wdr

ADDITION = wt.ADDITION


class _Backend:
    @staticmethod
    def matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
        return np.array(rows, dtype=float)

    @staticmethod
    def argmax_token_rows(matrix: np.ndarray, classes: int) -> list[tuple[int, ...]]:
        return [tuple(int(i) for i in row.reshape(-1, classes).argmax(axis=1)) for row in matrix]


class CarryNetwork:
    """
    A stand-in that runs the carry machine: its state the carry's one-hot vector plus drift times
    the columns read, its digit right except from column wrong_from on, where it answers 0 for a 1.
    """

    backend = _Backend()
    columns = 1
    class_count = 3

    def __init__(self, drift: float = 0.0, wrong_from: int | None = None) -> None:
        self.drift, self.wrong_from = drift, wrong_from

    def window_states(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        machine = ADDITION.machine
        rows, width = X.shape
        carry = np.zeros(rows, dtype=int)
        states = [np.eye(3)[carry]]
        outputs = np.zeros((rows, width * 3))
        for t in range(width):
            tokens = X[:, t].astype(int)
            labels = machine.output[carry, tokens]
            if self.wrong_from is not None and t >= self.wrong_from:
                labels = np.where(labels == 1, 0, labels)
            outputs[np.arange(rows), t * 3 + labels] = 1.0
            carry = machine.transition[carry, tokens]
            states.append(np.eye(3)[carry] + self.drift * (t + 1))
        return outputs, states, states


def test_the_carry_machine_itself_neither_errs_nor_drifts():
    result = wdr.profile(CarryNetwork(), ADDITION, 32, cases=40, bins=4)
    assert result["cases_wrong"] == 0 and result["wrong_by_bin"] == [0, 0, 0, 0] and result["contexts"] == []
    assert result["separation"] == pytest.approx(np.sqrt(2))
    assert all(d["shift"] == 0 and d["spread"] == 0 for drift in result["drift"] for d in drift.values())


def test_wrong_digits_fall_in_their_bins_and_contexts():
    result = wdr.profile(CarryNetwork(wrong_from=16), ADDITION, 32, cases=40, bins=4)
    assert result["wrong_by_bin"][:2] == [0, 0] and min(result["wrong_by_bin"][2:]) > 0
    assert result["cases_wrong"] == 40
    occurring = {(c["entering"], c["leaving"]): c for c in result["contexts"]}
    assert all(0 < c["wrong"] <= c["occurrences"] for c in occurring.values())
    assert sum(c["wrong"] for c in occurring.values()) == sum(result["wrong_by_bin"]) + result["wrong_past_n"]


def test_a_state_moving_along_the_input_shifts_more_with_each_bin():
    result = wdr.profile(CarryNetwork(drift=0.01), ADDITION, 32, cases=40, bins=4)
    shifts = [[d["shift"] for d in drift.values()] for drift in result["drift"]]
    assert shifts[0] == [0, 0, 0]
    assert all(min(later) > max(earlier) for earlier, later in itertools.pairwise(shifts))
    assert any(line.startswith("   bin 3:") for line in wdr.lines("drifting", result))
