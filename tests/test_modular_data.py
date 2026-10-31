"""Modular division's and the sum's encoding, answers, training mixture and catalogues (indrajala_studies/modular_data.py)."""

import random
from collections.abc import Sequence

import pytest

from indrajala_studies import modular_data as md


def direct(pair: md.Pair, n: int, p: int) -> md.Answers:
    """The running quotients from the prefixes as whole numbers, most significant first."""
    a, b = pair
    return tuple(md.quotient(a // 3 ** (n - i), b // 3 ** (n - i), p) for i in range(1, n + 1))


def test_a_pair_reads_most_significant_first():
    # a = 21 = 210 in base 3, b = 5 = 012: columns (2, 0), (1, 1), (0, 2)
    assert md.column_tokens((21, 5), 3) == (3 * 2 + 0, 3 * 1 + 1, 3 * 0 + 2)


@pytest.mark.parametrize("p", md.PRIMES)
def test_the_answers_are_the_running_quotients_of_the_prefixes(p: int):
    rng = random.Random(p)
    for _ in range(500):
        n = rng.randrange(1, 12)
        pair = md.uniform_pair(n, rng)
        answers = md.answers(pair, n, p)
        assert answers == direct(pair, n, p)
        a, b = pair
        assert answers[-1] == (md.undefined(p) if b % p == 0 else a * pow(b, -1, p) % p)


def test_a_zero_prefix_leaves_the_answer_undefined_for_that_many_columns_then_defined():
    rng = random.Random(1)
    for length in range(1, 9):
        pair = md.zero_prefix_pair(8, rng, length)
        answers = md.answers(pair, 8, 5)
        assert answers[:length] == (md.undefined(5),) * length
        if length < 8:
            assert md.column_tokens(pair, 8)[length] % 3 != 0  # b's next digit isn't zero
            assert answers[length] != md.undefined(5)


def test_the_training_mixture_never_draws_a_held_out_pair_and_holds_out_a_tenth():
    rng = random.Random(2)
    pairs = [md.training_pair(i, 8, rng) for i in range(3000)]
    assert not any(md.held_out(pair, 8) for pair in pairs)
    assert sum(pair[0] == 0 for pair in pairs[2::3]) == 1000  # kind 2: a zero dividend
    share = sum(md.held_out(md.uniform_pair(8, rng), 8) for _ in range(20_000)) / 20_000
    assert 0.08 < share < 0.12


def test_the_catalogue_is_exhaustive_at_short_lengths_and_stratified_by_state_and_prefix():
    by_id = {p.id: p for p in md.catalogue(7, sampled=50, per_stratum=4)}
    assert len(md.property_cases(by_id["U1"], 3)) == 9**3
    assert len(md.property_cases(by_id["U1"], 8)) == 50
    held = md.property_cases(by_id["U2"], 3)
    assert held and all(md.held_out(c.pair, 3) for c in held)
    states = md.property_cases(by_id["S1"], 8)
    assert len(states) == 49 * 4 and {c.stratum for c in states} == set(range(49))
    assert all(c.pair[0] % 7 * 7 + c.pair[1] % 7 == c.stratum for c in states)
    # at one column only the residues 0-2 are reachable
    assert {c.stratum for c in md.property_cases(by_id["S1"], 1)} == {a * 7 + b for a in range(3) for b in range(3)}
    runs = md.property_cases(by_id["S2"], 8)
    assert sorted({c.stratum or 0 for c in runs}) == list(range(1, 9))
    doubling = {p.id: p for p in md.catalogue(7, per_stratum=2, doubling=True)}["S2"]
    assert sorted({c.stratum or 0 for c in md.property_cases(doubling, 64)}) == [1, 2, 4, 8, 16, 32, 64]


def test_the_same_property_gives_the_same_cases_and_another_p_others():
    five, seven = md.catalogue(5, sampled=20)[0], md.catalogue(7, sampled=20)[0]
    assert md.property_cases(five, 9) == md.property_cases(five, 9)
    assert md.property_cases(five, 9) != md.property_cases(seven, 9)


def test_evaluate_counts_each_strata_and_keeps_the_first_failures():
    prop = {p.id: p for p in md.catalogue(5, per_stratum=3)}["S2"]

    def wrong_on_long_runs(pairs: Sequence[md.Pair]) -> list[md.Answers]:
        return [md.answers(pair, 6, 5) if pair[1] >= 3**2 else (0,) * 6 for pair in pairs]

    (result,) = md.evaluate(wrong_on_long_runs, 6, [prop])
    assert result.cases == 18 and not result.holds
    assert result.reach() == 3 and result.strata[4] == (0, 3)
    assert len(result.failures) == 3 and all(c.pair[1] < 9 for c in result.failures)


@pytest.mark.parametrize("q", [2, 4, 5, 7, 11])
def test_the_sums_are_the_running_sums_of_the_prefixes(q: int):
    rng = random.Random(q)
    for _ in range(500):
        n = rng.randrange(1, 12)
        pair = md.uniform_pair(n, rng)
        a, b = pair
        sums = md.sums(pair, n, q)
        assert sums == tuple((a // 3 ** (n - i) + b // 3 ** (n - i)) % q for i in range(1, n + 1))


@pytest.mark.parametrize("q", [4, 5])
def test_the_sum_catalogue_checks_every_final_sum_against_the_running_sums(q: int):
    u1, u2, s1 = md.sum_catalogue(q, sampled=200, per_stratum=7)
    assert (u1.id, u2.id, s1.id) == ("U1", "U2", "S1") and all(p.operation == "sum" for p in (u1, u2, s1))
    cases = md.property_cases(s1, 4)
    assert {c.stratum for c in cases} == set(range(q)) and len(cases) == 7 * q
    assert all((c.pair[0] + c.pair[1]) % q == c.stratum for c in cases)

    def right(pairs: Sequence[md.Pair]) -> list[md.Answers]:
        return [md.sums(pair, 4, q) for pair in pairs]

    def wrong(pairs: Sequence[md.Pair]) -> list[md.Answers]:
        return [tuple((x + 1) % q for x in md.sums(pair, 4, q)) for pair in pairs]  # every answer off by one

    assert all(r.holds for r in md.evaluate(right, 4, (u1, u2, s1)))
    assert not any(r.holds for r in md.evaluate(wrong, 4, (u1, s1)))


def test_divisions_cases_are_drawn_as_before_the_sum():
    # the seed string keeps E7's and E7''s cases: "modular-division-p-<id>-n"
    prop = md.catalogue(5, sampled=50, per_stratum=3)[2]
    assert md.property_cases(prop, 9) == prop.generate(9, random.Random("modular-division-5-S1-9"))
