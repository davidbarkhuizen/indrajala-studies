"""chow: Chow's test set (the W-method), an exact answers-only check under a state bound."""

import itertools

import numpy as np
import pytest

from indrajala_studies import chow
from indrajala_studies import machine_reading as mr
from indrajala_studies import window_tasks as wt

# a mod-3 counter: token 0 the identity, token 1 adds one; the output the count after the token
COUNTER = wt.Mealy(np.array([[0, 1], [1, 2], [2, 0]]), np.array([[0, 1], [1, 2], [2, 0]]))


def doubled(machine: wt.Mealy) -> wt.Mealy:
    """The machine with every state copied (state s and s + n equivalent): 2n states, n after minimising."""
    n = machine.states
    transition = np.concatenate([machine.transition + n * (np.arange(n) % 2)[:, None], machine.transition])
    return wt.Mealy(transition, np.concatenate([machine.output, machine.output]))


def same(a: wt.Mealy, b: wt.Mealy, length: int) -> bool:
    """Every word up to `length` tokens gives the same labels in both."""
    return all(a.run(w) == b.run(w) for n in range(length + 1) for w in itertools.product(range(a.tokens), repeat=n))


def test_minimised_merges_equivalent_states_and_keeps_the_behaviour() -> None:
    copy = doubled(COUNTER)
    assert copy.states == 6
    m = chow.minimised(copy)
    assert m.states == 3 and m.start == 0
    assert same(m, copy, 6)
    assert chow.minimised(wt.carry_machine()).states == 3


def test_minimised_drops_unreachable_states() -> None:
    machine = wt.Mealy(np.array([[0, 1], [1, 0], [2, 2]]), np.array([[0, 1], [1, 0], [2, 2]]))
    assert chow.minimised(machine).states == 2


def test_the_cover_and_the_characterisation_set() -> None:
    cover = chow.transition_cover(COUNTER)
    assert cover[0] == () and set(cover) >= {(0,), (1,), (1, 1), (1, 0), (1, 1, 0), (1, 1, 1)}
    w = chow.characterisation_set(COUNTER)
    for s, r in itertools.combinations(range(3), 2):  # every pair told apart by some word of W
        assert any(
            wt.Mealy(COUNTER.transition, COUNTER.output, start=s).run(word)
            != wt.Mealy(COUNTER.transition, COUNTER.output, start=r).run(word)
            for word in w
        )


def test_the_size_counts_the_words_before_repeats() -> None:
    for bound in (3, 4, 5):
        assert len(list(chow.test_set(COUNTER, bound))) <= chow.test_set_size(COUNTER, bound)
    with pytest.raises(AssertionError):
        chow.test_set_size(COUNTER, 2)


def test_an_equivalent_machine_passes_at_any_bound() -> None:
    target = wt.random_sized_machine(6, 0)
    for bound in (6, 7, 8):
        result = chow.check(doubled(target), chow.machine_query(target), bound)
        assert result.equivalent and result.states == 6 and result.counterexample is None and result.words > 0


def test_a_machine_larger_than_the_bound_fails_without_a_query() -> None:
    result = chow.check(wt.random_sized_machine(6, 0), chow.machine_query(COUNTER), 5)
    assert not result.equivalent and result.words == 0


def test_every_wrong_machine_within_the_bound_is_caught() -> None:
    """
    Chow's theorem, exhaustively on small machines: every single-transition or single-label change to a
    4-state target that makes it inequivalent is caught at the bound 4 (and the counterexample is real).
    """
    target = wt.random_sized_machine(4, 1)
    caught = 0
    for s, t in itertools.product(range(target.states), range(target.tokens)):
        for r in range(target.states):
            transition = target.transition.copy()
            transition[s, t] = r
            mutant = wt.Mealy(transition, target.output.copy())
            if same(mutant, target, 2 * target.states):
                continue
            result = chow.check(mutant, chow.machine_query(target), target.states)
            assert not result.equivalent and result.counterexample is not None
            assert mutant.run(result.counterexample) != target.run(result.counterexample)
            caught += 1
    assert caught > 0


def test_a_smaller_machine_right_on_short_words_is_caught_at_the_targets_bound() -> None:
    """
    A mod-6 counter labelled by whether its count is at least 3, against a 3-state machine that
    agrees with it until the count reaches 3: the check at the target's 6 states finds the word.
    """
    target = wt.Mealy(
        np.array([[s, (s + 1) % 6] for s in range(6)]),
        np.array([[int(s >= 3), int((s + 1) % 6 >= 3)] for s in range(6)]),
    )
    wrong = wt.Mealy(np.array([[0, 1], [1, 2], [2, 0]]), np.array([[0, 0], [0, 0], [0, 1]]))
    assert not chow.check(wrong, chow.machine_query(target), 6).equivalent


def test_the_limit_refuses_rather_than_claiming_equivalence() -> None:
    target = wt.random_sized_machine(6, 0)
    with pytest.raises(ValueError, match="exceed the limit"):
        chow.check(target, chow.machine_query(target), 12, limit=100)


def test_a_walk_reads_as_a_mealy_machine() -> None:
    walk = mr.Walk([(0,), (1,), (2,)], COUNTER.transition, COUNTER.output[:, :, None])
    machine = mr.mealy(walk)
    assert chow.check(machine, chow.machine_query(COUNTER), 3).equivalent
