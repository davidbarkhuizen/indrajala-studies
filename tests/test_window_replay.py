"""The code margins (indrajala_studies/window_replay.py), on a stand-in snapped network at k 1."""

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
from indrajala_ml.studies import window_tasks as wt

from indrajala_studies import window_replay as wrp
from tests.helpers import CarryTable

ADDITION = wt.ADDITION
OFFSET = 0.1  # every state's distance from its code, per coordinate


class _Backend:
    @staticmethod
    def matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
        return np.array(rows, dtype=float)


class SnappedCarryNetwork:
    """
    A stand-in that runs the carry machine: its state the carry's one-hot vector plus OFFSET in
    every coordinate, beside a codebook of the one-hot vectors and one code no state is near.
    """

    backend = _Backend()
    columns = 1

    def __init__(self, snapped: bool = True) -> None:
        states = len(ADDITION.machine.transition)
        E = np.vstack([np.eye(states), np.full(states, 5.0)]) if snapped else None
        self.states = states
        self.recurrence = SimpleNamespace(factors=[SimpleNamespace(state_size=states)])
        self.recurrence_layer = SimpleNamespace(
            factors=[SimpleNamespace(codes=None if E is None else SimpleNamespace(E=E))]
        )

    def window_states(self, X: np.ndarray) -> tuple[None, list[np.ndarray], list[np.ndarray]]:
        machine = ADDITION.machine
        rows, width = X.shape
        carry = np.zeros(rows, dtype=int)
        states = [np.eye(self.states)[carry] + OFFSET]
        for t in range(width):
            carry = machine.transition[carry, X[:, t].astype(int)]
            states.append(np.eye(self.states)[carry] + OFFSET)
        return None, states, states


def test_the_margins_read_the_codes_in_use_and_not_the_unused_one():
    margins = wrp.code_margins(SnappedCarryNetwork(), ADDITION, [4, 8], cases=40)
    assert margins is not None
    (factor,) = margins["factors"]
    used = factor["codes_used"]
    assert len(used) >= 2 and max(used) < SnappedCarryNetwork().states  # the far code is never used
    assert factor["closest_used"] == pytest.approx(np.sqrt(2))
    states = SnappedCarryNetwork().states
    assert factor["farthest_state"] == pytest.approx(OFFSET * np.sqrt(states))
    # the next nearest code: one coordinate at 1 + OFFSET, one at 1 - OFFSET, the rest at OFFSET
    q2 = (1 + OFFSET) ** 2 + (1 - OFFSET) ** 2 + (states - 2) * OFFSET**2
    assert margins["least_snap_ratio"] == pytest.approx(np.sqrt(q2 / (states * OFFSET**2)))
    assert "closest pair 1.414" in wrp.margins_text(margins)


def test_an_unsnapped_network_has_no_margins():
    assert wrp.code_margins(SnappedCarryNetwork(snapped=False), ADDITION, [4], cases=10) is None
    assert wrp.margins_text(None) == "no snapped factor"


def test_the_machine_table_of_the_target_has_no_wrong_pair_and_no_change_against_itself():
    table = wrp.machine_table(CarryTable(), ADDITION)
    assert table.wrong == 0
    assert len(table.rows) == len(ADDITION.machine.transition)
    assert wrp.table_changes(table, wrp.machine_table(CarryTable(), ADDITION)) == {
        "pairs_changed": 0,
        "states_gained": 0,
        "states_lost": 0,
    }


def test_one_flipped_transition_is_one_pair_changed_and_one_wrong():
    window = 26  # (2, 2, 2) from carry 0, handed on as the next carry
    flipped = wrp.machine_table(CarryTable(flip=(0, window)), ADDITION)
    assert wrp.table_changes(wrp.machine_table(CarryTable(), ADDITION), flipped)["pairs_changed"] == 1
    assert flipped.wrong == 1


def test_the_guard_undoes_an_update_that_makes_a_right_machine_wrong(monkeypatch: pytest.MonkeyPatch):
    # a real network at k 1; the machine read off is stubbed: right at the start, wrong after the
    # first update, right after every other, so the first batch is undone and the rest kept
    from indrajala_ml.studies import window_recurrent_study as wr

    network = wr.build(ADDITION, 1, 4, 4, seed=0, window="linear")
    reads = iter([0, 5] + [0] * 100)

    def stub_table(machine: object, task: object) -> wrp.Table:
        return wrp.Table((0,), {}, next(reads))

    monkeypatch.setattr(wrp, "machine_table", stub_table)
    settings = wr.Settings(epoch_examples=6 * 64, batch_size=64, screen=5)
    starts: list[tuple[np.ndarray, int, object]] = []
    original = network.learn_batch

    def learn_batch(rate: float, batch: object) -> None:
        # each batch's starting weights, optimizer step and generator state
        starts.append((wrp._parameters(network), network.optimizer.state().t, network.rng.state))
        original(rate, batch)

    network.learn_batch = learn_batch
    result = wrp.replay(network, ADDITION, [1, 2], settings, 1, "guard-test", guard=True)
    (epoch,) = result["epochs"]
    assert epoch["rollbacks"] == 1
    assert [row.get("rolled_back", 0.0) for row in epoch["batches"]] == [1.0] + [0.0] * 5
    # the first batch undone: the second starts exactly where the first did, the third after one step
    (w0, t0, g0), (w1, t1, g1), (w2, t2, _) = starts[:3]
    assert np.array_equal(w1, w0) and t1 == t0 and g1 == g0
    assert not np.array_equal(w2, w1) and t2 == t1 + 1
