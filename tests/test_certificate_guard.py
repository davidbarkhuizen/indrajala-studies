"""The certificate guard (indrajala_studies/certificate_guard.py): its read, and the updates it undoes."""

import random

import numpy as np
import pytest

from indrajala_studies import certificate_guard as cg
from indrajala_studies import modular_family as mf
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_replay as wrp
from indrajala_studies import window_tasks as wt
from tests.helpers import CarryTable

ADDITION = wt.ADDITION


def test_the_read_finds_the_target_right_and_any_flipped_transition_wrong_as_the_machine_table_does():
    assert cg.right(CarryTable(), ADDITION)
    for carry in range(len(ADDITION.machine.transition)):
        for window in (0, 13, 26):
            flipped = CarryTable(flip=(carry, window))
            assert cg.right(flipped, ADDITION) == (wrp.machine_table(flipped, ADDITION).wrong == 0)


def test_a_machine_without_its_own_step_states_is_stepped_a_state_at_a_time():
    machine, windows = CarryTable(), wc.all_windows(1, ADDITION.machine.tokens)
    stacked = wc.step_states(machine, [(1,), (0,)], windows)
    one, zero = machine.step((1,), windows), machine.step((0,), windows)
    assert np.array_equal(stacked.digits, np.concatenate([one.digits, zero.digits]))
    assert np.array_equal(stacked.states, np.concatenate([one.states, zero.states]))
    assert np.array_equal(stacked.distances[0], np.concatenate([one.distances[0], zero.distances[0]]))


def test_a_networks_states_stepped_in_one_pass_are_its_steps_a_state_at_a_time_bit_for_bit():
    encoding = mf.Encoding(5, mf.FAMILY)
    machine = mf.QueryMachine(mf.network(encoding, [(4, 4), (4, 4)], 0, reads="output"), encoding, 0)
    columns = np.arange(9, dtype=np.int64)[:, np.newaxis]
    states = [(0, 0), (1, 2), (3, 1)]  # a code per factor, reached or not
    stacked = wc.step_states(machine, states, columns)
    for i, state in enumerate(states):
        alone = machine.step(state, columns)
        rows = slice(9 * i, 9 * (i + 1))
        assert np.array_equal(stacked.digits[rows], alone.digits)
        assert np.array_equal(stacked.leads[rows], alone.leads)
        assert np.array_equal(stacked.states[rows], alone.states)
        assert all(np.array_equal(s[rows], a) for s, a in zip(stacked.distances, alone.distances, strict=True))


def _family_network_and_batches(batches: int) -> tuple[object, list[list[mf.Example]]]:
    encoding = mf.Encoding(5, mf.FAMILY)
    network = mf.network(encoding, [(4, 4)], 0)
    rng = random.Random(0)
    return network, [[encoding.example(mf.columns(3, rng)) for _ in range(8)] for _ in range(batches)]


def _stub_reads(monkeypatch: pytest.MonkeyPatch, reads: list[bool]) -> list[bool]:
    left = list(reads)

    def stub(machine: object, task: object) -> bool:
        return left.pop(0)

    monkeypatch.setattr(cg, "right", stub)
    return left


def _parameters(network: object) -> np.ndarray:
    return wrp._parameters(network)  # pyright: ignore[reportPrivateUsage]


def test_the_guard_undoes_an_update_that_makes_a_right_machine_wrong(monkeypatch: pytest.MonkeyPatch):
    # right when armed, wrong after the first update (undone), right after the second (kept)
    network, batches = _family_network_and_batches(3)
    left = _stub_reads(monkeypatch, [True, False, True, True])
    guard = cg.Guard(lambda n: n, ADDITION)
    guard.arm(network)
    start = (_parameters(network), network.optimizer.state().t, network.rng.state)  # pyright: ignore
    guard.learn_batch(network, 0.003, batches[0])
    assert guard.rollbacks == 1
    assert np.array_equal(_parameters(network), start[0])
    assert network.optimizer.state().t == start[1] and network.rng.state == start[2]  # pyright: ignore
    guard.learn_batch(network, 0.003, batches[1])
    assert guard.rollbacks == 1 and network.optimizer.state().t == start[1] + 1  # pyright: ignore
    assert not np.array_equal(_parameters(network), start[0])
    guard.learn_batch(network, 0.003, batches[2])
    assert left == []  # a read after every update
    assert guard.updates == 3


def test_an_armed_guard_keeps_every_update_until_the_machine_is_right(monkeypatch: pytest.MonkeyPatch):
    # wrong when armed and after the first update (kept), right after the second, wrong after the third (undone)
    network, batches = _family_network_and_batches(3)
    _stub_reads(monkeypatch, [False, False, True, False])
    guard = cg.Guard(lambda n: n, ADDITION)
    guard.arm(network)
    for batch in batches:
        guard.learn_batch(network, 0.003, batch)
    assert guard.rollbacks == 1 and network.optimizer.state().t == 2  # pyright: ignore


def test_an_unarmed_guard_or_another_network_trains_unread(monkeypatch: pytest.MonkeyPatch):
    network, batches = _family_network_and_batches(2)
    other, _ = _family_network_and_batches(0)
    _stub_reads(monkeypatch, [True])  # the arm's read; another would fail on the empty list
    guard = cg.Guard(lambda n: n, ADDITION)
    guard.learn_batch(network, 0.003, batches[0])
    guard.arm(other)
    guard.learn_batch(network, 0.003, batches[1])
    assert guard.rollbacks == guard.updates == 0 and network.optimizer.state().t == 2  # pyright: ignore
    guard.disarm()
    assert not guard.armed
