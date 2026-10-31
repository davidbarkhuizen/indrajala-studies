"""
The window-recurrent adder's certificate (indrajala_studies/window_recurrent_certificate.py)
against stand-in transducers that add, drop a carry, carry only inside a window or snap or choose a
digit on a near tie, and against real networks, whose machine is checked against their own
forward pass.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import BEFORE_STATE, AttentionPooling, Codes, Factor, WindowRecurrence

from indrajala_studies import addition_data as ad
from indrajala_studies import modular_data as md
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt

ALL_TWOS = 26  # the column token of digits (2, 2, 2): column sum 6
NEAR = 1.01  # a distance ratio inside SNAP_RATIO


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


class MachineNetwork:
    """A WindowMachine as the network WindowModel reads: each example's windows run in turn from the start."""

    def __init__(self, machine: wc.WindowMachine) -> None:
        self.machine = machine

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> list[tuple[float, ...]]:
        return [tokens for tokens, _labels in rows]

    def classify_rows(self, prepared: Sequence[tuple[float, ...]]) -> list[tuple[int, ...]]:
        k = self.machine.columns
        predictions: list[tuple[int, ...]] = []
        for tokens in prepared:
            state, digits = self.machine.start()[0], list[int]()
            for w in range(0, len(tokens), k):
                steps = self.machine.step(state, np.array([tokens[w : w + k]], dtype=np.int64))
                digits += [int(d) for d in steps.digits[0]]
                state = tuple(int(c) for c in steps.states[0])
            predictions.append(tuple(digits))
        return predictions


def fails_as_said(counterexample: wc.Counterexample, machine: CarryMachine) -> bool:
    """The counterexample's windows run through the machine give what it says they give."""
    state, digits = machine.start()[0], list[int]()
    for window, before in zip(counterexample.windows, counterexample.codes, strict=True):
        assert state == before
        steps = machine.step(state, np.array([window], dtype=np.int64))
        state, digits = tuple(int(c) for c in steps.states[0]), [int(d) for d in steps.digits[0]]
    if counterexample.kind == "outputs":
        return digits == counterexample.got != counterexample.expected
    return counterexample.got != counterexample.expected


# the stand-ins


@pytest.mark.parametrize("k", [1, 2, 4])
def test_an_adding_machine_is_certified_and_minimises_to_the_carry_automaton(k: int):
    certificate = wc.certify(CarryMachine(k))
    assert certificate.certified and certificate.counterexample is None
    assert certificate.reachable == [(0,), (1,), (2,)] and certificate.phi == {(0,): 0, (1,): 1, (2,): 2}
    assert certificate.minimised == [[(0,)], [(1,)], [(2,)]]
    assert certificate.snap.smallest == 2.0 and certificate.snap.short == 0
    assert certificate.evaluations == 3 * 27**k


def test_copies_of_a_carry_are_certified_and_merged_by_the_minimised_machine():
    certificate = wc.certify(CarryMachine(2, copies=2))
    assert certificate.certified and sorted(certificate.reachable) == [(q,) for q in range(6)]
    assert certificate.phi == {(q,): q % 3 for q in range(6)}
    assert certificate.minimised == [[(0,), (3,)], [(1,), (4,)], [(2,), (5,)]]


@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_machine_that_drops_a_carry_of_two_fails_on_the_first_all_twos_window(k: int):
    certificate = wc.certify(CarryMachine(k, fault="drops"))
    assert not certificate.certified and certificate.wrong_outputs == 0 and certificate.wrong_states > 0
    assert certificate.wrong_ties == 0
    assert certificate.reachable == [(0,), (1,)]
    counterexample = certificate.counterexample
    assert counterexample is not None and counterexample.kind == "state"
    # the smallest: the first window in order whose carry out is 2, from the start
    assert counterexample.windows == [[0] * (k - 1) + [ALL_TWOS]] and counterexample.codes == [(0,)]
    assert counterexample.expected == [2] and counterexample.got == [1]
    assert fails_as_said(counterexample, CarryMachine(k, fault="drops"))
    assert counterexample.after == [0] * k  # the carries 2 and 1 differ on a zero column, a zero window
    assert wt.ADDITION.case_json(counterexample.columns()) == {"triple": [2 * 3 ** (k - 1)] * 3, "n": 2 * k}


@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_machine_that_carries_only_inside_a_window_fails_on_the_first_carry_out(k: int):
    certificate = wc.certify(CarryMachine(k, fault="inside"))
    assert not certificate.certified and certificate.reachable == [(0,)]
    counterexample = certificate.counterexample
    assert counterexample is not None and counterexample.kind == "state"
    # token 5 is the digits (0, 1, 2), the first column sum of 3 in token order
    assert counterexample.windows == [[0] * (k - 1) + [5]]
    assert counterexample.expected == [1] and counterexample.got == [0]
    # as an addition at alignment 0, through the zero window, the network gets its sum wrong
    case = wt.ADDITION.case_json(counterexample.columns())
    tokens, labels = wr.example(wt.ADDITION, tuple(case["triple"]), case["n"] - k, k, 0)
    network = MachineNetwork(CarryMachine(k, fault="inside"))
    assert network.classify_rows([tokens]) != [labels]


def test_a_snap_near_tie_branches_and_fails_when_the_other_code_carries_otherwise():
    certificate = wc.certify(CarryMachine(2, fault="snap-tie"))
    assert not certificate.certified and certificate.snap.short > 0
    assert certificate.snap.smallest == pytest.approx(NEAR)
    counterexample = certificate.counterexample
    # the network's own snap is right; the near tie's other code isn't
    assert certificate.wrong_ties == certificate.wrong_states > 0
    assert counterexample is not None and counterexample.kind == "state"
    assert counterexample.windows == [[ALL_TWOS, ALL_TWOS]] and counterexample.expected == [2]
    assert counterexample.got == [0]


def test_a_near_tie_to_another_carry_splits_the_minimised_machine():
    # the tie's other code is another carry: a state the machine itself tells apart from the snap's
    certificate = wc.certify(CarryMachine(2, fault="snap-tie"))
    assert certificate.split_ties > 0
    assert wc.certify(CarryMachine(2, fault="snap-tie"), snap_ratio=1.0).split_ties == 0


def test_strict_certifies_a_machine_wrong_only_on_a_near_tie():
    certificate = wc.certify(CarryMachine(2, fault="snap-tie"), snap_ratio=1.0)
    assert certificate.certified and certificate.snap.short == 0 and certificate.wrong_ties == 0


def test_a_snap_near_tie_between_copies_of_a_carry_is_certified_and_counted():
    certificate = wc.certify(CarryMachine(2, copies=2, fault="copy-tie"))
    assert certificate.certified and certificate.snap.short > 0
    assert certificate.split_ties == 0  # the copies minimise to one state: the tie splits nothing
    assert certificate.snap.smallest == pytest.approx(NEAR) and certificate.snap.window is not None


def test_a_digits_near_tie_fails_the_certificate_though_every_digit_is_right():
    certificate = wc.certify(CarryMachine(1, fault="digit-tie"))
    assert certificate.correct and not certificate.certified
    assert certificate.lead.short == 3 and certificate.lead.window == [ALL_TWOS]
    assert certificate.to_json()["certified"] is False


def test_a_certified_machine_passes_the_long_tests():
    k = 4
    assert wc.certify(CarryMachine(k, copies=2)).certified
    model = wr.WindowModel(MachineNetwork(CarryMachine(k, copies=2)), k, wt.ADDITION)
    catalogue = ad.catalogue(sampled=20, per_stratum=2, doubling=True)
    long = [p for p in catalogue if p.id in ("B1", "M3")]
    results = model.evaluate(1024, long)
    assert all(r.holds for r in results)
    assert {r.id: r.reach() for r in results}["M3"] == 1024


def test_a_counterexample_is_json_with_its_case():
    value = wc.certify(CarryMachine(1, fault="drops")).to_json()
    assert value["certified"] is False and value["phi"] == [[[0], 0], [[1], 1]] and value["task"] == "addition"
    assert value["counterexample"]["case"] == {"triple": [2, 2, 2], "n": 2} and "case" not in value
    assert "fails" in wc.summary({**value, "seconds": 0.0})


# against another target: modular division (D17)


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


# the certificate's code is the same at every p; above 23 a prime only adds minutes (80 s at 41, k 2)
CERTIFIED_PRIMES = [p for p in md.PRIMES if p <= 23]


@pytest.mark.parametrize("p", CERTIFIED_PRIMES)
@pytest.mark.parametrize("k", [1, 2])
def test_a_dividing_machine_is_certified_against_the_modular_automaton(p: int, k: int):
    task = wt.ModularDivision(p)
    certificate = wc.certify(TaskMachine(task, k, copies=2), task)
    assert certificate.certified and certificate.counterexample is None and certificate.task == task.name
    assert sorted(certificate.reachable) == [(q,) for q in range(2 * p * p)]
    assert certificate.phi == {q: q[0] % (p * p) for q in certificate.reachable}
    assert len(certificate.minimised) == p * p
    assert certificate.evaluations == len(certificate.reachable) * 9**k


@pytest.mark.parametrize("q", [4, 5])
@pytest.mark.parametrize("k", [1, 2])
def test_a_summing_machine_is_certified_against_the_sum_automaton(q: int, k: int):
    task = wt.ModularSum(q)
    certificate = wc.certify(TaskMachine(task, k, copies=2), task)
    assert certificate.certified and certificate.counterexample is None and certificate.task == task.name
    assert certificate.phi == {c: c[0] % q for c in certificate.reachable} and len(certificate.minimised) == q


@pytest.mark.parametrize("fault", ["inside", "forgets-b"])
@pytest.mark.parametrize("p", CERTIFIED_PRIMES)
def test_a_faulty_dividing_machine_fails_with_a_case_it_gets_wrong(fault: str, p: int):
    task, k = wt.ModularDivision(p), 2
    machine = TaskMachine(task, k, fault=fault)
    certificate = wc.certify(machine, task)
    assert not certificate.certified and certificate.wrong_states > 0 and certificate.wrong_outputs == 0
    counterexample = certificate.counterexample
    assert counterexample is not None and counterexample.kind == "state"
    assert len(counterexample.after) == k
    case = certificate.to_json()["counterexample"]["case"]
    assert case == task.case_json(counterexample.columns()) and case["n"] == len(counterexample.columns())
    # the case at alignment 0, through the harness's encoding: the machine's answers aren't the task's
    tokens, labels = wr.example(task, tuple(case["pair"]), case["n"], k, 0)
    assert MachineNetwork(machine).classify_rows([tokens]) != [labels]


def test_a_certified_dividing_machine_passes_the_long_tests():
    task, k = wt.ModularDivision(7), 2
    assert wc.certify(TaskMachine(task, k, copies=2), task).certified
    model = wr.WindowModel(MachineNetwork(TaskMachine(task, k, copies=2)), k, task)
    catalogue = task.catalogue(sampled=20, relation_sampled=0, per_stratum=2, doubling=True)
    results = {r.id: r for r in model.evaluate(256, [p for p in catalogue if p.id in ("U1", "S2")])}
    assert all(r.holds for r in results.values()) and task.reach(results["S2"]) == 256


# real networks


@pytest.mark.parametrize("network_class", [WindowRecurrentArrayNetwork, WindowRecurrentRustArrayNetwork])
def test_a_networks_machine_is_its_forward_pass_window_by_window(network_class: Any):
    k = 2
    network: Any = network_class.randomized(
        (k,), wr.window_specs(wt.ADDITION), WindowRecurrence((Factor(4, Codes(4)),)), Adam(), seed=3
    )
    machine = wc.NetworkMachine(network)
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 27, size=(40, 3 * k))
    expected = network.classify_rows(network.prepare_dataset([(tuple(map(float, r)), (0,) * len(r)) for r in rows]))
    codes = network.state_codes(network.backend.matrix(rows.astype(float).tolist()))
    for row, want, want_states in zip(rows, expected, codes, strict=True):
        state, distances = machine.start()
        assert state == want_states[0] and int(np.argmin(distances[0])) == state[0]
        digits: list[int] = []
        for w, window in enumerate(row.reshape(-1, k)):
            steps = machine.step(state, window[np.newaxis, :])
            digits += [int(d) for d in steps.digits[0]]
            state = tuple(int(c) for c in steps.states[0])
            assert state == want_states[w + 1]
        assert tuple(digits) == tuple(want)


@pytest.mark.parametrize("network_class", [WindowRecurrentArrayNetwork, WindowRecurrentRustArrayNetwork])
def test_a_continuous_network_has_no_machine_to_certify(network_class: Any):
    network: Any = network_class.randomized(
        (2,), wr.window_specs(wt.ADDITION), WindowRecurrence((Factor(4, None),)), Adam(), seed=3
    )
    with pytest.raises(AssertionError, match="continuous"):
        wc.NetworkMachine(network)


def test_an_untrained_networks_counterexample_is_its_own_output():
    k = 2
    network = wr.build(wt.ADDITION, k, 4, 4, seed=1)
    certificate = wc.certify(wc.NetworkMachine(network))
    assert not certificate.certified and set(certificate.reachable) <= {(c,) for c in range(4)}
    counterexample = certificate.counterexample
    assert counterexample is not None
    tokens = [float(t) for window in counterexample.windows for t in window]
    (prediction,) = network.classify_rows(network.prepare_dataset([(tuple(tokens), (0,) * len(tokens))]))
    if counterexample.kind == "outputs":
        assert list(prediction[-k:]) == counterexample.got != counterexample.expected


# a factored state (the factored-state workplan, stage 6)


class FactoredDivision:
    """
    A stand-in transducer for modular division whose state is two factors of p codes each: factor g
    holds the linear form forms[g] of the target's (A, B), the pair an invertible change of
    coordinates, so the machine is division's. Its faults: "b-slips" moves factor 2's next code on
    by one after a window whose last token is 1; "b-tie" puts factor 2's next code plus one at NEAR
    after a window whose last token is 0.
    """

    def __init__(
        self, p: int, k: int, forms: tuple[tuple[int, int], tuple[int, int]], fault: str | None = None
    ) -> None:
        self.p, self.columns, self.fault = p, k, fault
        self.task = wt.ModularDivision(p)
        self.forward = {
            A * p + B: tuple((alpha * A + beta * B) % p for alpha, beta in forms) for A in range(p) for B in range(p)
        }
        self.inverse = {state: s for s, state in self.forward.items()}
        assert len(self.inverse) == p * p, "the forms must be an invertible change of coordinates"

    @property
    def codes(self) -> list[int]:
        return [self.p, self.p]

    def _distances(self, codes: np.ndarray) -> np.ndarray:
        distances = np.full((len(codes), self.p), 2.0)
        distances[np.arange(len(codes)), codes] = 1.0
        return distances

    def start(self) -> tuple[wc.State, list[np.ndarray]]:
        state = self.forward[self.task.machine.start]
        return state, [self._distances(np.array([c]))[0] for c in state]

    def step(self, state: wc.State, windows: np.ndarray) -> wc.Steps:
        labels, after = self.task.machine.windows(windows)
        target = self.inverse[state]
        states = np.array([self.forward[int(s)] for s in after[target]], dtype=np.int64).reshape(-1, 2)
        if self.fault == "b-slips":
            slips = windows[:, -1] == 1
            states[slips, 1] = (states[slips, 1] + 1) % self.p
        distances = [self._distances(states[:, 0]), self._distances(states[:, 1])]
        if self.fault == "b-tie":
            ties = np.flatnonzero(windows[:, -1] == 0)
            distances[1][ties, (states[ties, 1] + 1) % self.p] = NEAR
        return wc.Steps(labels[target], np.full(labels[target].shape, 5.0), states, distances)


@pytest.mark.parametrize(
    "forms,names",
    [(((1, 0), (0, 1)), ["A", "B"]), (((1, 1), (1, 4)), ["A + B", "A + 4B"])],
    ids=["A and B", "A + B and A - B"],
)
@pytest.mark.parametrize("k", [1, 2])
def test_a_factored_dividing_machine_is_certified_and_its_factors_named(forms: Any, names: list[str], k: int):
    p = 5
    certificate = wc.certify(FactoredDivision(p, k, forms), wt.ModularDivision(p))
    assert certificate.certified and certificate.counterexample is None
    assert certificate.codes == [p, p] and len(certificate.reachable) == p * p
    assert all(len(state) == 2 for state in certificate.reachable)
    assert len(certificate.minimised) == p * p
    assert [f.name for f in certificate.factors] == names
    assert all(f.function and len(f.partition) == p for f in certificate.factors)
    value = certificate.to_json()
    assert value["factors"][0]["name"] == names[0] and value["phi"][0] == [list(certificate.start), 0]


def test_a_factor_whose_next_code_slips_fails_with_its_state_in_the_counterexample():
    p, k = 5, 1
    machine = FactoredDivision(p, k, ((1, 0), (0, 1)), fault="b-slips")
    certificate = wc.certify(machine, wt.ModularDivision(p))
    assert not certificate.certified and certificate.wrong_states > 0
    counterexample = certificate.counterexample
    assert counterexample is not None and counterexample.kind == "state"
    # the slipped code is taken first for the state its window should reach (phi is fixed by the
    # first window to reach a state), so the failure shows where another window reaches it
    assert all(len(state) == 2 for state in counterexample.codes)
    assert counterexample.got != counterexample.expected
    # factor 1 still holds A; factor 2 no longer a function of the target's state
    assert certificate.factors[0].name == "A" and not certificate.factors[1].function


def test_a_near_tie_in_one_factor_branches_over_that_factors_codes():
    p, k = 5, 1
    task = wt.ModularDivision(p)
    certificate = wc.certify(FactoredDivision(p, k, ((1, 0), (0, 1)), fault="b-tie"), task)
    assert certificate.snap.short > 0 and certificate.snap.smallest == pytest.approx(NEAR)
    # the other code's state is a wrong one: wrong on the near tie, and from the state it branches to
    assert not certificate.certified and certificate.wrong_ties > 0
    strict = wc.certify(FactoredDivision(p, k, ((1, 0), (0, 1)), fault="b-tie"), task, snap_ratio=1.0)
    assert strict.certified and strict.snap.short == 0


def test_next_states_branch_over_every_combination_of_the_factors_near_ties():
    states = np.array([[0, 1], [2, 3]])
    near = [np.array([[False, True, False], [False] * 3]), np.array([[False] * 4, [False, False, False, False]])]
    near[1][0, 3] = True
    windows, rows = wc.next_states(states, near)
    assert windows.tolist() == [0, 1, 0, 0, 0]
    assert rows.tolist() == [[0, 1], [2, 3], [0, 3], [1, 1], [1, 3]]


def test_a_partition_is_named_by_its_linear_form_or_not_at_all():
    coordinates = wt.modular_machine(3).coordinates
    by_a = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert wc.name_partition(by_a, coordinates) == "A"
    assert wc.name_partition([[0, 1, 2, 3, 4, 5, 6, 7, 8]], coordinates) is None
    assert wc.name_partition(by_a, None) is None
    # the carry automaton's states are its one coordinate
    certificate = wc.certify(CarryMachine(1))
    assert [f.name for f in certificate.factors] == ["carry"]


@pytest.mark.parametrize("network_class", [WindowRecurrentArrayNetwork, WindowRecurrentRustArrayNetwork])
def test_a_factored_networks_machine_is_its_forward_pass_window_by_window(network_class: Any):
    k = 2
    recurrence = WindowRecurrence((Factor(2, Codes(3)), Factor(3, Codes(4))), pooling=AttentionPooling(BEFORE_STATE))
    network: Any = network_class.randomized((k,), wr.window_specs(wt.ADDITION), recurrence, Adam(), seed=3)
    machine = wc.NetworkMachine(network)
    assert machine.codes == [3, 4]
    rows = np.random.default_rng(0).integers(0, 27, size=(20, 3 * k))
    expected = network.classify_rows(network.prepare_dataset([(tuple(map(float, r)), (0,) * len(r)) for r in rows]))
    states = network.state_codes(network.backend.matrix(rows.astype(float).tolist()))
    for row, want, want_states in zip(rows, expected, states, strict=True):
        state, distances = machine.start()
        assert state == want_states[0] and [int(np.argmin(d)) for d in distances] == list(state)
        digits: list[int] = []
        for w, window in enumerate(row.reshape(-1, k)):
            steps = machine.step(state, window[np.newaxis, :])
            digits += [int(d) for d in steps.digits[0]]
            state = tuple(int(c) for c in steps.states[0])
            assert state == want_states[w + 1]
        assert tuple(digits) == tuple(want)
    certificate = wc.certify(machine)
    assert all(len(state) == 2 for state in certificate.reachable) and len(certificate.factors) == 2
