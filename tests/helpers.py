"""
The test doubles these tests share with ml's, copied (the studies-repo workplan, D5) from ml at
ba5fa14: CarryTable from tests/studies/helpers.py, CarryMachine and TaskMachine from
tests/studies/test_window_recurrent_certificate.py.
"""

import math

import numpy as np
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_tasks as wt

ADDITION = wt.ADDITION
ALL_TWOS = 26  # the column token of digits (2, 2, 2): column sum 6
NEAR = 1.01  # a distance ratio inside SNAP_RATIO


class CarryTable:
    """The carry machine at k 1 as a WindowMachine, with flip (state, window) handing on the next carry instead."""

    def __init__(self, flip: tuple[int, int] | None = None) -> None:
        self.flip = flip

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return [3]

    def start(self) -> tuple[tuple[int, ...], list[np.ndarray]]:
        return (0,), [np.array([0.0, 1.0, 1.0])]

    def step(self, state: tuple[int, ...], windows: np.ndarray) -> wc.Steps:
        (carry,) = state
        labels, after = ADDITION.machine.windows(windows)
        digits, nexts = labels[carry], after[carry].copy()
        if self.flip is not None and self.flip[0] == carry:
            nexts[windows[:, 0] == self.flip[1]] = (nexts[windows[:, 0] == self.flip[1]] + 1) % 3
        distances = np.ones((len(windows), 3))
        distances[np.arange(len(windows)), nexts] = 0.0
        return wc.Steps(digits, np.full(digits.shape, 5.0), nexts[:, np.newaxis], [distances])


class CarryMachine:
    """
    A stand-in transducer: code q holds the carry q % 3, copies codes a carry (q = carry + 3 *
    copy, the copy the window's last token's parity), each window's digits the sum's and its next
    code its carry out's, every other code twice as far as the one it snaps to. Its faults:
    "drops" hands a carry out of 2 on as 1; "inside" hands on no carry (it carries only inside a
    window); "snap-tie" puts the code of the next carry, "copy-tie" the carry's other copy, at
    NEAR after an all-twos window; "digit-tie" gives an all-twos window's digits a lead of 1e-9.
    """

    def __init__(self, k: int, copies: int = 1, fault: str | None = None) -> None:
        self.columns = k
        self.copies = copies
        self.fault = fault
        self.count = 3 * copies

    @property
    def codes(self) -> list[int]:
        return [self.count]

    def start(self) -> tuple[wc.State, list[np.ndarray]]:
        return (0,), [self._distances(np.zeros(1, dtype=np.int64))[0]]

    def _distances(self, codes: np.ndarray) -> np.ndarray:
        distances = np.full((len(codes), self.count), 2.0)
        distances[np.arange(len(codes)), codes] = 1.0
        return distances

    def step(self, state: wc.State, windows: np.ndarray) -> wc.Steps:
        (code,) = state
        want_digits, want_carry = wt.ADDITION.machine.windows(windows)
        digits, carry = want_digits[code % 3], want_carry[code % 3].copy()
        if self.fault == "drops":
            carry[carry == 2] = 1
        if self.fault == "inside":
            carry[:] = 0
        codes = carry + 3 * (windows[:, -1] % self.copies)
        distances = self._distances(codes)
        twos = (windows == ALL_TWOS).all(axis=1)
        if self.fault == "snap-tie":
            distances[twos, (carry[twos] + 1) % 3 + 3 * (windows[twos, -1] % self.copies)] = NEAR
        if self.fault == "copy-tie":
            distances[twos, carry[twos] + 3 * ((windows[twos, -1] + 1) % self.copies)] = NEAR
        leads = np.full(digits.shape, 5.0)
        if self.fault == "digit-tie":
            leads[twos] = 1e-9
        return wc.Steps(digits, leads, codes[:, np.newaxis], [distances])


class TaskMachine:
    """
    A stand-in transducer for any task: code q holds the target state q % S (S the target's
    states), copies of each (q = state + S * copy, the copy the window's last token's parity), each
    window's labels and next code the target's. Its faults: "inside" hands on the start state (it
    keeps nothing across windows); "forgets-b", for modular division, hands on B as 0.
    """

    def __init__(self, task: wt.Task, k: int, copies: int = 1, fault: str | None = None) -> None:
        self.task, self.columns, self.copies, self.fault = task, k, copies, fault
        self.states = task.machine.states
        self.count = self.states * copies

    @property
    def codes(self) -> list[int]:
        return [self.count]

    def start(self) -> tuple[wc.State, list[np.ndarray]]:
        return (0,), [self._distances(np.zeros(1, dtype=np.int64))[0]]

    def _distances(self, codes: np.ndarray) -> np.ndarray:
        distances = np.full((len(codes), self.count), 2.0)
        distances[np.arange(len(codes)), codes] = 1.0
        return distances

    def step(self, state: wc.State, windows: np.ndarray) -> wc.Steps:
        (code,) = state
        labels, after = self.task.machine.windows(windows)
        target = code % self.states
        following = after[target].copy()
        if self.fault == "inside":
            following[:] = self.task.machine.start
        if self.fault == "forgets-b":
            p = math.isqrt(self.states)
            following = following // p * p
        codes = following + self.states * (windows[:, -1] % self.copies)
        return wc.Steps(
            labels[target], np.full(labels[target].shape, 5.0), codes[:, np.newaxis], [self._distances(codes)]
        )
