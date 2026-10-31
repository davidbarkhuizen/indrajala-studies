"""machine_reading: a network's machine read against its target (the code-consolidation workplan's stage 2)."""

import numpy as np
import pytest

from indrajala_studies import machine_reading as mr
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_tasks as wt

# a mod-3 counter: token 0 the identity, token 1 adds one; the output the count after the token
COUNTER = wt.Mealy(np.array([[0, 1], [1, 2], [2, 0]]), np.array([[0, 1], [1, 2], [2, 0]]))


class Table:
    """A snapped machine given by tables, one factor: next[c][t] the code after t, out[c][t] its label."""

    def __init__(self, next: list[list[int]], out: list[list[int]]) -> None:
        self.next, self.out = next, out

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return [len(self.next)]

    def start(self) -> tuple[wc.State, list[wc.FloatArray]]:
        return (0,), [np.zeros(len(self.next))]

    def step(self, state: wc.State, windows: wc.IntArray) -> wc.Steps:
        tokens = [int(w[0]) for w in windows]
        c = state[0]
        return wc.Steps(
            np.array([[self.out[c][t]] for t in tokens], dtype=np.int64),
            np.ones((len(tokens), 1)),
            np.array([[self.next[c][t]] for t in tokens], dtype=np.int64),
            [np.zeros((len(tokens), len(self.next)))],
        )


def test_a_copy_of_the_target_is_exact_at_every_length_and_breaks_nothing():
    walk = mr.product(Table([[0, 1], [1, 2], [2, 0]], [[0, 1], [1, 2], [2, 0]]), COUNTER)
    assert len(walk.codes) == 3 and len(walk.pairs) == 3
    rows = mr.by_length(walk, [1, 5, 20], mr.word_length(COUNTER))
    assert all(r["last"] == r["exact"] == r["named"] == 1.0 for r in rows)
    result = mr.breaks(walk, mr.word_length(COUNTER))
    assert result["first_codes"] == 3 and result["right_pairs"] == 3
    assert result["first"] == {"broken": 0, "transitions": 6} and result["extra"] == {"broken": 0, "transitions": 0}


def test_an_extra_code_that_breaks_is_counted_apart_from_the_first_codes():
    # code 3 stands for count 0 after 2 + 1 (count 0's first code is 0), right for one token, but its
    # token 1 goes to code 2 (count 2) where count 1 is due
    walk = mr.product(Table([[0, 1], [1, 2], [2, 3], [3, 2]], [[0, 1], [1, 2], [2, 0], [0, 1]]), COUNTER)
    assert len(walk.codes) == 4
    d = mr.word_length(COUNTER)
    extra = walk.pairs.index((3, 0))
    assert mr.right_codes(walk, 1)[extra]  # its outputs right one token ahead
    assert not mr.right_codes(walk, d)[extra]  # but not d tokens ahead
    result = mr.breaks(walk, 1)
    assert result["first_right"] == {"broken": 0, "transitions": 6}  # the core, 0 1 2, exact
    assert result["extra_right"] == {"broken": 1, "transitions": 2}  # code 3's token 1
    rows = mr.by_length(walk, [1, 4, 5, 8], d)
    exact = [r["exact"] for r in rows]
    # the first wrong answer is the fifth token of 1 1 1 1 t: exact to length 4, then falling
    assert exact[:2] == [1.0, 1.0] and exact[-1] < exact[2] < 1.0


def test_the_counterexamples_are_the_shortest_words_to_the_pairs_wrong_for_their_states():
    # the extra code above: code 3 stands for count 0 after 1 1 1 and is wrong d tokens ahead; from it
    # codes 3 and 2 alternate while the count runs on, so each later pair is wrong too (R4)
    walk = mr.product(Table([[0, 1], [1, 2], [2, 3], [3, 2]], [[0, 1], [1, 2], [2, 0], [0, 1]]), COUNTER)
    d = mr.word_length(COUNTER)
    words = mr.counterexamples(walk, d, 10)
    assert words == [[1] * n for n in range(3, 8)]  # the five pairs past the exact core 0 1 2
    assert mr.counterexamples(walk, d, 1) == [[1, 1, 1]]  # at most limit, shortest first
    for word in words:  # each word reaches a pair whose code is wrong for its state
        pair = 0
        for token in word:
            pair = int(walk.next[pair][token])
        assert not mr.right_codes(walk, d)[pair]
    # a copy of the target has none
    exact = mr.product(Table([[0, 1], [1, 2], [2, 0]], [[0, 1], [1, 2], [2, 0]]), COUNTER)
    assert mr.counterexamples(exact, d, 10) == []
    assert mr.summary(walk, d) == {"codes": 4, "pairs": len(walk.pairs), "right": int(mr.right_codes(walk, d).sum())}


def test_a_factor_s_partition_is_named_only_over_half_the_states_or_more_and_in_two_blocks_or_more():
    coordinates = wt.Coordinates(("A",), np.array([[0], [1], [0], [1], [0], [1]]), 2)
    assert wc.name_partition([[0, 2, 4], [1, 3, 5]], coordinates) == "A"
    assert wc.name_partition([[0, 2], [1, 3]], coordinates) == "A"  # 4 of 6 states reached
    assert wc.name_partition([[0, 1, 2, 3, 4, 5]], coordinates) is None  # one block: a constant form fits
    assert wc.name_partition([[0], [1]], coordinates) is None  # 2 of 6 reached: too few to say


@pytest.mark.parametrize("name", ["word-problem-s4", "random-4-0", "modular-division-5"])
def test_d_is_the_target_s_distinguishing_length_plus_one(name: str):
    machine = wt.task_named(name).machine
    distinguishing = machine.distinguishing_length()
    assert distinguishing is not None and mr.word_length(machine) == distinguishing + 1


def test_a_code_that_behaves_as_another_at_every_depth_is_its_duplicate_and_pulled_to_it():
    # code 3 is code 0 again (count 0 after 1 1 1), the same outputs and next codes
    walk = mr.walk(Table([[0, 1], [1, 2], [2, 3], [3, 1]], [[0, 1], [1, 2], [2, 0], [0, 1]]), 2)
    assert walk.codes == [(0,), (1,), (2,), (3,)] and walk.outputs.shape == (4, 2, 1)
    for d in (1, 2, 5):
        assert mr.groups(walk, d) == [[0, 3], [1], [2]]
    assert mr.consolidation(walk, mr.groups(walk, 2)) == {(3,): (0,)}


def test_codes_are_grouped_only_as_far_as_d_tokens_tell_them_apart():
    # code 3 answers as code 0 for one token, but its token 1 goes to code 2, not code 1
    walk = mr.walk(Table([[0, 1], [1, 2], [2, 3], [3, 2]], [[0, 1], [1, 2], [2, 0], [0, 1]]), 2)
    assert mr.groups(walk, 1) == [[0, 3], [1], [2]]
    assert mr.groups(walk, 2) == [[0], [1], [2], [3]]
    assert mr.consolidation(walk, mr.groups(walk, 2)) == {}


def test_the_product_s_codes_are_the_walk_s():
    table = Table([[0, 1], [1, 2], [2, 3], [3, 2]], [[0, 1], [1, 2], [2, 0], [0, 1]])
    assert mr.product(table, COUNTER).codes == mr.walk(table, 2).codes


class Pairs:
    """A snapped machine of two factors given by a table over (code 0, code 1): next[(a, b)][t], out[(a, b)][t]."""

    def __init__(self, next: dict[wc.State, list[wc.State]], out: dict[wc.State, list[int]]) -> None:
        self.next, self.out = next, out

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return [2, 2]

    def start(self) -> tuple[wc.State, list[wc.FloatArray]]:
        return (0, 0), [np.zeros(2), np.zeros(2)]

    def step(self, state: wc.State, windows: wc.IntArray) -> wc.Steps:
        tokens = [int(w[0]) for w in windows]
        return wc.Steps(
            np.array([[self.out[state][t]] for t in tokens], dtype=np.int64),
            np.ones((len(tokens), 1)),
            np.array([self.next[state][t] for t in tokens], dtype=np.int64),
            [np.zeros((len(tokens), 2)), np.zeros((len(tokens), 2))],
        )


def test_a_copy_of_the_target_determines_its_finest_named_congruence_and_covers_it():
    target = wt.task_named("move-to-front-3").machine
    copy = Table(target.transition.tolist(), target.output.tolist())
    result = mr.approximations(mr.product(copy, target), target)
    (factor,) = result["factors"]
    assert result["states"] == result["meet"] == 6 and result["cover"]
    assert factor["function"] and factor["congruence"] and factor["autonomous"] == 1.0
    assert factor["approximation"] == [1] * 6
    assert factor["determines"] == ["first 2"]  # the whole order: every coarser named congruence implied


def _toggle(t: int, s: wc.State) -> wc.State:
    """Token 0 the identity, token 1 toggles the first bit, token 2 the second."""
    a, b = s
    return (a, b) if t == 0 else (1 - a, b) if t == 1 else (a, 1 - b)


def test_a_counter_mod_4_in_two_bits_is_a_cascade_not_a_parallel_cover():
    # the target: a counter mod 4 (token 0 holds, 1 and 2 add one), states (low, high); the low bit
    # alone a counter mod 2, the high bit flipping on the low bit's carry: the series form
    def count(t: int, s: wc.State) -> wc.State:
        value = (s[0] + 2 * s[1] + (t > 0)) % 4
        return (value % 2, value // 2)

    states: list[wc.State] = [(a, b) for b in range(2) for a in range(2)]
    transition = np.array([[states.index(count(t, s)) for t in range(3)] for s in states])
    target = wt.Mealy(transition, np.array([[states[int(r)][1] for r in row] for row in transition]))
    network = Pairs(
        {s: [count(t, s) for t in range(3)] for s in states},
        {s: [count(t, s)[1] for t in range(3)] for s in states},
    )
    result = mr.approximations(mr.product(network, target), target)
    low, high = result["factors"]
    assert low["congruence"] and low["approximation"] == [2, 2] and low["autonomous"] == 1.0
    assert not high["congruence"] and high["approximation"] == [4] and high["autonomous"] < 1.0
    assert not result["cover"] and result["meet"] == 2  # the parallel reading misses it
    assert result["cascade"] == [0]  # the low bit on top, the high bit reading it


def test_two_counters_mod_2_each_an_approximation_together_a_cover():
    # the target: two bits each a counter mod 2 (_toggle), the output the first bit; a network holding
    # each bit in its own factor
    states: list[wc.State] = [(a, b) for a in range(2) for b in range(2)]
    transition = np.array([[states.index(_toggle(t, s)) for t in range(3)] for s in states])
    target = wt.Mealy(transition, np.array([[states[int(r)][0] for r in row] for row in transition]))
    network = Pairs(
        {s: [_toggle(t, s) for t in range(3)] for s in states},
        {s: [_toggle(t, s)[0] for t in range(3)] for s in states},
    )
    result = mr.approximations(mr.product(network, target), target)
    assert result["cover"] and result["meet"] == 4
    for factor in result["factors"]:
        assert factor["codes"] == 2 and factor["function"] and factor["congruence"]
        assert factor["approximation"] == [2, 2] and factor["autonomous"] == 1.0
        assert factor["determines"] == []  # no coordinates, no names
    assert result["cascade"] == [0, 1]  # a parallel product: a cascade of either over the other


def test_divisions_sum_and_difference_held_apart_read_as_its_lines_and_cover_it():
    # a network holding division's state as (A + B, A - B) mod 5, a factor each: the two named lines
    p = 5
    target = wt.modular_machine(p)
    assert target.coordinates is not None
    values = target.coordinates.values

    def held(s: int) -> wc.State:
        A, B = (int(x) for x in values[s])
        return ((A + B) % p, (A - B) % p)

    state_of = {held(s): s for s in range(target.states)}
    network = Pairs(
        {code: [held(int(r)) for r in target.transition[s]] for code, s in state_of.items()},
        {code: target.output[s].tolist() for code, s in state_of.items()},
    )
    result = mr.approximations(mr.product(network, target), target)
    total, difference = result["factors"]
    assert result["cover"] and result["meet"] == p * p
    assert total["congruence"] and total["approximation"] == [p] * p and total["determines"] == ["A + B"]
    assert difference["congruence"] and difference["determines"] == ["A - B"]


def test_a_planted_machines_planted_approximation_is_named_when_given_and_t11s_covers_are_not_added():
    # the paper workplan's T4' (D27): a network holding a planted machine's block in one factor and the
    # state in the other reads "planted" for the first; without the task's names it determines nothing
    name = "random-planted-q16-b4-0"
    target = wt.task_named(name).machine
    blocks = wt.planted_blocks(16, 4)

    def held(s: int) -> wc.State:
        return (int(blocks[s]), s)

    network = Pairs(
        {held(s): [held(int(r)) for r in target.transition[s]] for s in range(target.states)},
        {held(s): target.output[s].tolist() for s in range(target.states)},
    )
    block, state = mr.approximations(mr.product(network, target), target, wt.planted_labels(name))["factors"]
    assert block["congruence"] and block["approximation"] == [4] * 4 and block["determines"] == ["planted"]
    assert state["approximation"] == [1] * 16
    unnamed = mr.approximations(mr.product(network, target), target)["factors"][0]
    assert unnamed["approximation"] == [4] * 4 and unnamed["determines"] == []
    assert wt.planted_labels("word-problem-s3") == {} and wt.planted_labels("modular-division-5") == {}
