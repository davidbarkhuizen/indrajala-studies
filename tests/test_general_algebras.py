"""The general algebras' machines and their probe target (indrajala-research's docs/workplans/general-algebras-workplan.md, D1-D3, D6, D7)."""

import json
import math
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from indrajala_studies import modular_family as mf
from indrajala_studies import window_tasks as wt

CLASSES = ("word-problem-s{n}", "move-to-front-{n}", "random-{n}-0")
SIZES = (3, 4, 5)
# the workplan's feasibility table: n -> codes a factor, the exclusion bound (the cap)
FEASIBILITY = {3: (8, 69), 4: (15, 248), 5: (32, 1143)}


def _tasks() -> list[tuple[int, str]]:
    return [(n, form.format(n=n)) for n in SIZES for form in CLASSES]


@pytest.mark.parametrize(("n", "name"), _tasks())
def test_every_machine_has_n_factorial_reachable_minimal_states_n_plus_one_tokens_and_n_classes(n: int, name: str):
    machine = wt.task_named(name).machine
    assert machine.states == math.factorial(n) and machine.tokens == n + 1 and machine.classes == n
    assert len(machine.reachable()) == machine.states
    assert machine.distinguishing_length() is not None  # minimal (D2)
    assert (machine.transition[:, 0] == np.arange(machine.states)).all()  # token 0 the identity


@pytest.mark.parametrize(("n", "name"), _tasks())
def test_the_cells_are_feasible_with_the_surplus_held(n: int, name: str):
    q = wt.task_named(name).machine.states
    codes, cap = FEASIBILITY[n]
    factors = mf.scaled_factors(q, Fraction(32, 11), 2)
    assert factors == [(16, codes), (16, codes)]
    assert codes * codes >= 8 * q  # joint codes at least 8 times the states (8.5 at 120, division p 11's)
    assert mf.exclusion_bound(q, factors) == cap


@pytest.mark.parametrize("n", SIZES)
def test_the_group_s_tokens_are_permutations_and_move_to_front_s_are_idempotent(n: int):
    group, band = wt.word_problem_machine(n), wt.move_to_front_machine(n)
    for t in range(group.tokens):
        assert sorted(group.transition[:, t]) == list(range(group.states))  # a bijection
    for t in range(1, band.tokens):
        after = band.transition[:, t]
        assert (band.transition[after, t] == after).all()  # t t = t: no group part
        assert len(set(after)) < band.states


def test_the_group_s_output_is_the_item_at_position_1_and_move_to_front_s_the_last_item():
    states = wt.orderings(3)
    group, band = wt.word_problem_machine(3), wt.move_to_front_machine(3)
    # from (0, 1, 2): swap positions 1 and 3, then rotate: (2, 1, 0) -> (1, 0, 2)
    assert [states[s] for s in group.path([2, 3])] == [(2, 1, 0), (1, 0, 2)]
    assert group.run([2, 3]) == (2, 1)
    # access item 2, then item 1: (2, 0, 1) -> (1, 2, 0); the last item each time
    assert [states[s] for s in band.path([3, 2])] == [(2, 0, 1), (1, 2, 0)]
    assert band.run([3, 2]) == (1, 0)


def test_a_random_machine_is_the_same_for_its_seed_and_another_for_another():
    assert (wt.random_machine(4, 0).transition == wt.task_named("random-4-0").machine.transition).all()
    assert not (wt.random_machine(4, 0).transition == wt.random_machine(4, 1).transition).all()


def test_the_machine_task_passes_a_right_predictor_and_fails_a_wrong_one_and_s1_covers_every_state():
    task = wt.task_named("move-to-front-3")
    assert isinstance(task, wt.MachineTask)
    properties = task.catalogue(50, 0, 3)
    right = task.evaluate(lambda words: [task.machine.run((*w, 0)) for w in words], 8, properties)
    assert all(r.holds for r in right) and [r.cases for r in right] == [50, 18]
    assert set(right[1].strata) == set(range(6))
    wrong = task.evaluate(lambda words: [(0,) * (len(w) + 1) for w in words], 8, properties)
    assert not any(r.holds for r in wrong) and wrong[0].failures
    assert task.result_json(wrong[0])["failures"][0] == {"word": list(wrong[0].failures[0])}


def test_a_machine_target_is_one_token_a_window_its_kinds_by_token():
    encoding = mf.target_named("word-problem-s3")
    assert isinstance(encoding, mf.MachineEncoding)
    assert (encoding.width, encoding.tokens, encoding.classes, encoding.target_states) == (1, 4, 3, 6)
    word = encoding.draw(7, random.Random(0))
    assert len(word) == 8 and set(word) <= set(range(4)) and word[-1] == 0  # the final identity
    tokens, labels = encoding.example(word)
    assert tokens == tuple(float(t) for t in word) and labels == encoding.task.machine.run(word)
    assert encoding.kinds(tokens, labels) == [f"token-{t}" for t in word]
    assert set(encoding.quantities) == {"position-1", "position-2", "position-3"}
    assert isinstance(mf.target_named("modular-division-5"), mf.Encoding)


def test_a_probe_run_trains_a_machine_and_certifies_each_step_against_it(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.target_named("move-to-front-3"), [(4, 4), (4, 4)], 0, 1e9, tmp_path, settings, cap=2, fork=mf.Fork())
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    ends = [r for r in rows if "end" in r]
    assert [r["step"] for r in ends] == [1, 2]
    assert set(rows[0]["accuracy"]) <= {f"token-{t}" for t in range(4)}
    assert set(rows[0]["factors"][0]["shares"]) == {"position-1", "position-2", "position-3"}
    assert all(isinstance(r["certified"], bool) and r["minimised"] >= 1 for r in ends)
    certificate = mf.certify_file(tmp_path / "network.json", mf.target_named("move-to-front-3"), "move-to-front-3")
    assert certificate["task"] == "move-to-front-3" and certificate["slot"] == 0


def test_a_planted_query_asks_the_block_of_the_state_after_each_token_and_only_the_token_is_certified():
    """
    T10's P2 (the paper workplan's D19): a window is the token, then the query token (the machine's
    tokens' next, 5), its answer the planted block (state mod b) of the state the token leads to.
    """
    name = "random-planted-q8-b2-0"
    encoding = mf.target_named(name, mf.PLANTED)
    assert isinstance(encoding, mf.MachineEncoding)
    assert (encoding.width, encoding.tokens, encoding.query_tokens, encoding.classes) == (2, 6, (5,), 4)
    word = encoding.draw(6, random.Random(1))
    assert set(word) <= set(range(5))  # tokens drawn from the machine's, never the query's
    tokens, labels = encoding.example(word)
    machine = encoding.task.machine
    assert tokens[0::2] == tuple(float(t) for t in word) and set(tokens[1::2]) == {5.0}
    assert labels[0::2] == machine.run(word) and labels[1::2] == tuple(s % 2 for s in machine.path(word))
    kinds = encoding.kinds(tokens, labels)
    assert kinds[0::2] == [f"token-{t}" for t in word] and set(kinds[1::2]) == {"planted"}
    assert encoding.certified("token-3") and not encoding.certified("planted")
    with pytest.raises(AssertionError, match="queries"):
        mf.target_named("random-q16-0", mf.PLANTED)  # primitive: nothing to ask


@pytest.mark.parametrize(
    ("name", "blocks", "monoids"),
    [("word-problem-s3", (2, 3), (2, 6)), ("word-problem-s4", (4, 6), (24, 6)), ("move-to-front-4", (6, 8), (10, 15))],
)
def test_a_cover_query_asks_t11s_registered_pair_of_approximations(
    name: str, blocks: tuple[int, int], monoids: tuple[int, int]
):
    """
    T11 (the paper workplan's D20): the coarsest pair of approximations that together determine the
    state, asked as two queries after each token: S3 the sign and a point's place, S4 a point's place
    and S3 (the Klein four-group's cosets), move-to-front-4 the order of items 0, 1 and 3 and the set
    of items more recent than item 2.
    """
    encoding = mf.target_named(name, mf.COVER)
    assert isinstance(encoding, mf.MachineEncoding) and encoding.width == 3
    machine = encoding.task.machine
    pair = [wt.query_partitions(name)[query] for query in mf.COVER]
    assert tuple(int(labels.max()) + 1 for labels in pair) == blocks
    assert tuple(machine.quotient_monoid(wt.canonical(labels)) for labels in pair) == monoids
    assert all(machine.is_congruence(labels) for labels in pair) and len(set(zip(*pair, strict=True))) == machine.states
    word = encoding.draw(5, random.Random(2))
    tokens, labels = encoding.example(word)
    path = machine.path(word)
    assert labels[0::3] == machine.run(word) and set(tokens[1::3]) == {float(encoding.query_tokens[0])}
    assert labels[1::3] == tuple(int(pair[0][s]) for s in path) and labels[2::3] == tuple(int(pair[1][s]) for s in path)
    assert encoding.kinds(tokens, labels)[1:3] == ["cover-a", "cover-b"] and not encoding.certified("cover-a")


def test_the_coarsest_cover_is_s3s_registered_pair_and_a_primitive_machine_has_none():
    machine = wt.task_named("word-problem-s3").machine
    assert wt.coarsest_cover(machine) == tuple(
        wt.canonical(wt.query_partitions("word-problem-s3")[q]) for q in mf.COVER
    )
    assert len(machine.congruences()) == 4  # the three points' places and the sign
    with pytest.raises(AssertionError, match="primitive"):
        wt.coarsest_cover(wt.task_named("random-q12-0").machine)


def test_a_planted_query_run_trains_certifies_and_long_tests_the_machine_on_slot_0(tmp_path: Path):
    name = "random-planted-q8-b2-0"
    encoding = mf.target_named(name, mf.PLANTED)
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(encoding, [(4, 4), (4, 4)], 0, 1e9, tmp_path, settings, cap=1)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert "planted" in rows[0]["accuracy"] and any(k.startswith("token-") for k in rows[0]["accuracy"])
    certificate = mf.certify_file(tmp_path / "network.json", encoding, name)
    assert certificate["task"] == name and certificate["slot"] == 0
    long = mf.long_test(tmp_path / "network.json", encoding, name, lengths=(8,), properties=("U1",))
    assert set(long["lengths"]["8"]["properties"]) == {"U1"}


# B's origins (the code-consolidation workplan's stage 2b): random-4's machine with bijections only,
# and at other sizes; each probe's machine and its cell's feasibility (codes a factor, the cap)
B_PROBES = {
    "random-perm-4-0": (24, 15, 248),
    "random-q8-1": (8, 9, 88),
    "random-q12-0": (12, 11, 132),
    "random-q16-0": (16, 12, 159),
}


@pytest.mark.parametrize("name", sorted(B_PROBES))
def test_b_s_probe_machines_are_reachable_minimal_with_random_4_s_tokens_and_classes(name: str):
    states, codes, cap = B_PROBES[name]
    machine = wt.task_named(name).machine
    assert machine.states == states and machine.tokens == 5 and machine.classes == 4
    assert len(machine.reachable()) == states and machine.distinguishing_length() is not None
    assert (machine.transition[:, 0] == np.arange(states)).all()  # token 0 the identity
    factors = mf.scaled_factors(states, Fraction(32, 11), 2)
    assert factors == [(16, codes), (16, codes)] and mf.exclusion_bound(states, factors) == cap


def test_random_perm_s_tokens_are_bijections_and_random_4_s_merge_states():
    perm, maps = wt.task_named("random-perm-4-0").machine, wt.task_named("random-4-0").machine
    assert all(len(set(perm.transition[:, t])) == 24 for t in range(5))
    assert any(len(set(maps.transition[:, t])) < 24 for t in range(1, 5))
