"""
The window-recurrent study's tasks (indrajala_studies/window_tasks.py): each target machine's
labels are its task's answers, the machines are minimal, and a word of columns reads back as its case.
"""

import dataclasses
import itertools
import random

import numpy as np
import pytest

from indrajala_studies import addition_data as ad
from indrajala_studies import modular_data as md
from indrajala_studies import modular_family as mf
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_tasks as wt


def reachable(machine: wt.Mealy) -> list[int]:
    seen, queue = {machine.start}, [machine.start]
    for s in queue:
        for t in range(machine.tokens):
            r = int(machine.transition[s, t])
            if r not in seen:
                seen.add(r)
                queue.append(r)
    return queue


def test_the_carry_machines_labels_are_the_sums_digits_at_any_alignment():
    rng = random.Random(0)
    for _ in range(500):
        n, a, padding = rng.randrange(1, 12), rng.randrange(4), rng.randrange(1, 6)
        triple = ad.uniform_triple(n, rng)
        tokens = (0,) * a + wt.ADDITION.column_tokens(triple, n) + (0,) * padding
        assert wt.ADDITION.machine.run(tokens) == (0,) * a + ad.digits(sum(triple), n + padding)


@pytest.mark.parametrize("p", md.PRIMES)
def test_the_modular_machines_labels_are_the_running_quotients_unchanged_by_zero_columns(p: int):
    task = wt.ModularDivision(p)
    rng = random.Random(p)
    for _ in range(500):
        n, a, padding = rng.randrange(1, 12), rng.randrange(4), rng.randrange(6)
        pair = md.uniform_pair(n, rng)
        labels = task.machine.run((0,) * a + task.column_tokens(pair, n) + (0,) * padding)
        answers = md.answers(pair, n, p)
        assert labels == (md.undefined(p),) * a + answers + (answers[-1],) * padding


@pytest.mark.parametrize("q", [4, 5, 13])
def test_the_sum_machines_labels_are_the_running_sums(q: int):
    task = wt.task_named(f"modular-sum-{q}")
    assert isinstance(task, wt.ModularSum) and task.reach_id is None and task.machine.classes == q
    rng = random.Random(q)
    for _ in range(500):
        n, a = rng.randrange(1, 12), rng.randrange(4)
        pair = md.uniform_pair(n, rng)
        assert task.machine.run((0,) * a + task.column_tokens(pair, n)) == (0,) * a + md.sums(pair, n, q)


@pytest.mark.parametrize(
    ("task", "states"),
    [
        (wt.ADDITION, 3),
        (wt.ModularDivision(5), 25),
        (wt.ModularDivision(7), 49),
        (wt.ModularSum(4), 4),
        (wt.ModularSum(5), 5),
        (wt.ModularSum(13), 13),
    ],
)
def test_each_machine_is_minimal_every_state_reachable(task: wt.Task, states: int):
    machine = task.machine
    assert machine.states == states and len(reachable(machine)) == states
    # as the certificate's machine of one factor: a state a tuple of one code
    D = {(s,): machine.output[s].astype(np.int8) for s in range(states)}
    G = {(s,): machine.transition[s].astype(np.int16)[:, np.newaxis] for s in range(states)}
    assert len(wc.minimise([(s,) for s in range(states)], D, G)) == states


def labels_from(machine: wt.Mealy, state: int, word: list[int]) -> list[int]:
    labels: list[int] = []
    for token in word:
        labels.append(int(machine.output[state, token]))
        state = int(machine.transition[state, token])
    return labels


def test_two_states_are_told_apart_by_the_shortest_word():
    carry = wt.ADDITION.machine
    assert carry.distinguishing(0, 2) == [0]  # a zero column's digit is the carry
    machine = wt.ModularDivision(5).machine
    for s, r in itertools.combinations(range(machine.states), 2):
        word = machine.distinguishing(s, r)
        mine, theirs = labels_from(machine, s, word), labels_from(machine, r, word)
        assert mine[:-1] == theirs[:-1] and mine[-1] != theirs[-1]
        shorter = itertools.product(range(machine.tokens), repeat=len(word) - 1)
        assert all(labels_from(machine, s, list(w)) == labels_from(machine, r, list(w)) for w in shorter)


def test_a_word_reads_back_as_its_case():
    rng = random.Random(4)
    triple = ad.uniform_triple(7, rng)
    assert wt.ADDITION.case_json(wt.ADDITION.column_tokens(triple, 7)) == {"triple": list(triple), "n": 7}
    task = wt.ModularDivision(7)
    pair = md.uniform_pair(7, rng)
    assert task.case_json(task.column_tokens(pair, 7)) == {"pair": list(pair), "n": 7}


def test_tasks_are_named():
    assert wt.task_named("addition") is wt.ADDITION
    assert wt.task_named("modular-division-7") == wt.ModularDivision(7)
    assert wt.ModularDivision(5).name == "modular-division-5"
    assert (wt.ADDITION.machine.tokens, wt.ADDITION.machine.classes) == (27, 3)
    assert (wt.ModularDivision(7).machine.tokens, wt.ModularDivision(7).machine.classes) == (9, 8)
    with pytest.raises(AssertionError):
        wt.task_named("modular-division-4")


def readout(machine: wt.Mealy, *labels: np.ndarray) -> wt.Mealy:
    """The machine with its outputs replaced by readouts of the state each token leads to."""
    combined = np.zeros(machine.states, dtype=np.int64)
    for label in labels:
        combined = combined * (int(label.max()) + 1) + label
    return dataclasses.replace(machine, output=combined[machine.transition])


@pytest.mark.parametrize("p", [5, 7])
def test_division_readouts_approximations_and_the_cover(p: int):
    """
    The division paper's algebra (P2, "The target machine and its cover"): the sum and difference
    are congruences of division's machine (approximations), the quotient and the product aren't;
    the sum and difference together are injective on states (they cover it, odd p), the quotient
    with the sum isn't; every readout set trained on is 1-distinguishable, the sum alone never; each
    column permutes the states (a permutation automaton).
    """
    machine = wt.modular_machine(p)
    assert machine.coordinates is not None
    A, B = machine.coordinates.values.T
    total, difference, product = (A + B) % p, (A - B) % p, (A * B) % p
    quotient = np.array([md.quotient(int(a), int(b), p) for a, b in zip(A, B, strict=True)])
    assert machine.is_congruence(total) and machine.is_congruence(difference)
    assert not machine.is_congruence(quotient) and not machine.is_congruence(product)
    assert len(set(zip(total, difference, strict=True))) == p * p
    assert len(set(zip(quotient, total, strict=True))) < p * p
    assert machine.distinguishing_length() == 1
    for labels in [(quotient,), (quotient, total), (total, difference), (quotient, total, difference), (product,)]:
        assert readout(machine, *labels).distinguishing_length() == 1
    assert readout(machine, total).distinguishing_length() is None
    assert all(len(set(machine.transition[:, t])) == machine.states for t in range(machine.tokens))


def test_the_congruence_closure_is_the_finest_congruence_coarser_than_the_labels():
    counter = wt.Mealy(np.array([[0, 1], [1, 2], [2, 0]]), np.array([[0, 1], [1, 2], [2, 0]]))
    assert counter.congruence_closure([0, 1, 2]).tolist() == [0, 1, 2]
    # joining counts 0 and 1 joins their successors 1 and 2: one block
    assert counter.congruence_closure([0, 0, 1]).tolist() == [0, 0, 0]
    machine = wt.modular_machine(5)
    assert machine.coordinates is not None
    A, B = machine.coordinates.values.T
    total, quotient = (A + B) % 5, np.array([md.quotient(int(a), int(b), 5) for a, b in zip(A, B, strict=True)])
    closed = machine.congruence_closure(total)  # an approximation is its own closure
    assert len(set(zip(closed.tolist(), total.tolist(), strict=True))) == len(set(total.tolist()))
    coarser = machine.congruence_closure(quotient)
    assert machine.is_congruence(coarser)
    assert all(coarser[s] == coarser[r] for s in range(25) for r in range(25) if quotient[s] == quotient[r])


def test_order_labels_name_move_to_fronts_and_the_word_problems_congruences():
    mtf = wt.task_named("move-to-front-4").machine
    named = wt.order_labels(mtf.coordinates)
    congruent = [k for k, v in named.items() if mtf.is_congruence(v)]
    assert len(named) == 29 and len(congruent) == 17 and not any(k.startswith(("places", "parity")) for k in congruent)
    assert "pairings" not in congruent and all(f"ahead {{{i}}}" in congruent for i in range(4))
    assert sorted(np.bincount(named["order {1, 3}"]).tolist()) == [12, 12]  # a flip-flop: an age matrix's bit
    s3 = wt.task_named("word-problem-s3").machine
    named = wt.order_labels(s3.coordinates)
    assert [k for k, v in named.items() if s3.is_congruence(v)] == [
        "first 2",
        "places {0}",
        "places {1}",
        "places {2}",
        "parity",
    ]
    assert np.bincount(named["places {0}"]).tolist() == [2, 2, 2]  # S3 acting on 3 points
    assert np.bincount(named["parity"]).tolist() == [3, 3]  # A3's cosets
    assert wt.order_labels(wt.modular_machine(5).coordinates) == {}
    s4 = wt.task_named("word-problem-s4").machine
    named = wt.order_labels(s4.coordinates)
    assert s4.is_congruence(named["pairings"]) and np.bincount(named["pairings"]).tolist() == [4] * 6  # S4 / V4
    assert not any(s4.is_congruence(v) for k, v in named.items() if k.startswith("ahead"))


@pytest.mark.parametrize("name", wt.COVERED)
def test_t11s_given_pairs_are_named(name: str):
    coordinates = wt.task_named(name).machine.coordinates
    named = wt.order_labels(coordinates)
    pair = [wt.query_partitions(name)[query] for query in mf.COVER]
    assert all(any(np.array_equal(given, v) for v in named.values()) for given in pair)  # D25: T4's tool names both


@pytest.mark.parametrize("p", [5, 7])
def test_linear_labels_are_all_of_divisions_congruences(p: int):
    machine = wt.modular_machine(p)
    named = wt.linear_labels(machine.coordinates)
    assert len(named) == p + 1 and {"A", "B", "A + B", "A - B", "A + 2B"} <= set(named)
    assert all(machine.is_congruence(v) and sorted(np.bincount(v).tolist()) == [p] * p for v in named.values())
    assert wt.named_labels(machine.coordinates).keys() == named.keys()

    # the finest congruence joining two states is the line through their difference, so every one is a join of these
    def blocks(labels: np.ndarray) -> tuple[int, ...]:
        first: dict[int, int] = {}
        return tuple(first.setdefault(int(x), len(first)) for x in labels)

    lines = {blocks(v) for v in named.values()}
    for r in range(1, p * p):
        assert blocks(machine.congruence_closure([0 if s in (0, r) else s for s in range(p * p)])) in lines
    assert (
        wt.named_labels(wt.task_named("move-to-front-4").machine.coordinates).keys()
        == wt.order_labels(wt.task_named("move-to-front-4").machine.coordinates).keys()
    )


def test_principal_congruences_are_divisions_lines_s3s_cosets_and_none_for_a_random_machine():
    """
    Division's machine at p has p + 1 approximations of p blocks (its lines: A + kB and B), each a
    principal congruence; S3's are its three point stabilisers' cosets (3 blocks) and the sign (2);
    random-q16-0 has none (every pair of states joined closes to one block).
    """
    p = 5
    lines = wt.modular_machine(p).principal_congruences()
    assert len(lines) == p + 1 and all(max(labels) + 1 == p for labels in lines)
    assert all(wt.modular_machine(p).is_congruence(labels) for labels in lines)
    s3 = wt.task_named("word-problem-s3").machine.principal_congruences()
    assert sorted(max(labels) + 1 for labels in s3) == [2, 3, 3, 3]
    assert wt.task_named("random-q16-0").machine.principal_congruences() == []


def test_a_planted_random_machine_has_its_blocks_as_a_congruence_and_random_q_s_shape():
    """
    random-planted-q16-b4-0: random-q16's tokens (the identity first) and 4 classes, every state
    reachable, minimal, and its planted blocks (s mod 4) a congruence, where random-q16-0 is primitive.
    """
    planted = wt.task_named("random-planted-q16-b4-0").machine
    plain = wt.task_named("random-q16-0").machine
    assert (planted.states, planted.tokens, planted.classes) == (plain.states, plain.tokens, plain.classes)
    assert (planted.transition[:, 0] == np.arange(16)).all()
    assert len(planted.reachable()) == 16 and planted.distinguishing_length() is not None
    assert planted.is_congruence(wt.planted_blocks(16, 4))
    assert tuple(wt.planted_blocks(16, 4).tolist()) in planted.principal_congruences()  # its blocks numbered as found


def test_depth_token_ranks_and_synchronising_read_a_reset_machine_and_a_group():
    """
    A 3-state machine with a shift (rank 3) and a reset to 0 (rank 1): depth 2 (0, 1, 2 along the
    shift), synchronising (the reset is a reset word); S3's word problem permutes its states, so every
    token's rank is 6 and no word joins two states.
    """
    transition = np.array([[1, 0], [2, 0], [0, 0]], dtype=np.int64)
    machine = wt.Mealy(transition, transition)
    assert machine.depth() == 2 and machine.token_ranks() == [3, 1] and machine.synchronising()
    s3 = wt.task_named("word-problem-s3").machine
    assert s3.token_ranks() == [6] * s3.tokens and not s3.synchronising()


def test_synchronising_needs_every_pair_joined_not_one_token_of_low_rank():
    """
    Two cycles each held by a token that joins within a cycle only: ranks below the states, yet the
    cycles never meet, so no reset word (Cerny's pairs: (0, 2) is never joined).
    """
    transition = np.array([[1, 0], [0, 0], [3, 2], [2, 2]], dtype=np.int64)
    machine = wt.Mealy(transition, transition)
    assert machine.token_ranks() == [4, 2] and not machine.synchronising()
