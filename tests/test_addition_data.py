"""The addition study's catalogue (indrajala_studies/addition_data.py), against stand-in adders."""

import itertools
from random import Random

import pytest

from indrajala_studies.addition_data import (
    BASE,
    CATALOGUE,
    COLUMN_SUMS,
    DEPENDENT,
    EXHAUSTIVE_LIMIT,
    RESETS,
    Case,
    PropertyResult,
    Triple,
    carries,
    catalogue,
    column_sums,
    digits,
    distance,
    doubling_distances,
    evaluate,
    from_digits,
    held_out,
    property_cases,
    training_triple,
    training_triples,
)
from tests.addition_adders import correct, drops_third, no_carry, reach

PROPERTIES = {prop.id: prop for prop in CATALOGUE}


def by_id(results: list[PropertyResult]) -> dict[str, PropertyResult]:
    return {result.id: result for result in results}


def failing(results: list[PropertyResult]) -> set[str]:
    return {result.id for result in results if not result.holds}


def test_digits_are_least_significant_first_and_round_trip():
    assert digits(5, 3) == (2, 1, 0)
    assert from_digits((2, 1, 0)) == 5
    assert all(from_digits(digits(x, 4)) == x for x in range(BASE**4))
    with pytest.raises(AssertionError):
        digits(27, 3)


def test_the_column_maps_are_three_resets_and_four_dependent():
    # the workplan's table: a reset's carry out is the same for every carry in
    for s in COLUMN_SUMS:
        outs = {(s + k) // BASE for k in range(BASE)}
        assert (len(outs) == 1) == (s in RESETS)
    assert sorted(RESETS + DEPENDENT) == list(COLUMN_SUMS)


def test_carries_and_distance_by_hand():
    # 22 + 22 + 22 in base 3 (digits 2, 2): column sums 6, 6; carries 0, 2, 2
    assert column_sums((8, 8, 8), 2) == (6, 6)
    assert carries((6, 6)) == (0, 2, 2)
    # column sums 3, 1, 2, 5: column 4's carry passes three dependent columns down to the reset
    assert distance((3, 1, 2, 5), 4) == 3
    assert distance((3, 1, 2, 5), 1) == 0
    assert distance((1, 2), 2) == 2  # down to column 0's carry in


def test_held_out_is_a_tenth_and_every_order_alike():
    n = 3
    triples: list[Triple] = [(a, b, c) for a, b, c in itertools.product(range(BASE**n), repeat=3)]
    fraction = sum(held_out(t, n) for t in triples) / len(triples)
    assert 0.09 < fraction < 0.11
    for t in triples[::97]:
        assert len({held_out((x, y, z), n) for x, y, z in itertools.permutations(t)}) == 1


def test_training_triples_are_the_three_kinds_in_turn_and_never_held_out():
    n = 4
    triples = training_triples(n, 3_000, Random(1))
    assert not any(held_out(t, n) for t in triples)
    assert all(0 in t for t in triples[2::3])  # the two-operand third
    assert training_triples(n, 30, Random(1)) == triples[:30]


def test_each_generators_cases_meet_its_statement():
    n = 4
    cases = {prop.id: property_cases(prop, n) for prop in CATALOGUE}
    assert all(0 in case.triple for case in cases["B2"])
    assert all(held_out(case.triple, n) for case in cases["B3"])
    assert all(max(case.triple) < BASE ** (n - 1) for case in cases["A2"])
    assert all(sorted(case.triple)[:2] == [0, 0] for case in cases["A3"])
    assert all(max(column_sums(case.triple, n)) <= 2 for case in cases["M1"])
    for case in cases["M2"]:
        assert case.column is not None and case.stratum is not None
        assert column_sums(case.triple, n)[case.column - 1] == RESETS[case.stratum]
    for case in cases["M3"]:
        assert case.column is not None
        assert distance(column_sums(case.triple, n), case.column) == case.stratum
    assert {case.stratum for case in cases["M3"]} == set(range(1, n + 1))
    for case in cases["M4"]:
        assert carries(column_sums(case.triple, n))[n] == case.stratum
    assert {column_sums(case.triple, n) for case in cases["M5"]} == set(itertools.product(COLUMN_SUMS, repeat=n))


def test_the_coverage_is_the_workplans():
    counts = {n: {prop.id: len(property_cases(prop, n)) for prop in CATALOGUE} for n in (4, 6)}
    assert counts[4]["B1"] == 531_441  # every triple
    assert counts[6]["B1"] == 100_000
    assert counts[4]["B2"] == 3 * 81**2
    assert counts[6]["A3"] == 3 * 729
    assert counts[6]["M3"] == 6 * 1_000
    assert counts[4]["M5"] == 7**4 * 4
    assert counts[6]["M5"] == 7**6


def test_a_sized_catalogue_samples_wherever_a_space_is_past_the_limit():
    # the window-recurrent adder's test lengths (W17): every property affordable at any n
    sized = catalogue(sampled=300, relation_sampled=40, per_stratum=7)
    for n in (12, 64):
        counts = {prop.id: len(property_cases(prop, n)) for prop in sized}
        assert counts == {
            "B1": 300,
            "B2": 300,
            "B3": 300,
            "A1": 40,
            "A2": 40,
            "A3": 300,
            "M1": 300,
            "M2": n * 3 * 16,
            "M3": n * 7,
            "M4": 3 * 7,
            "M5": 300,
        }
    m5 = next(prop for prop in sized if prop.id == "M5")
    assert 7**7 > EXHAUSTIVE_LIMIT and len(property_cases(m5, 7)) == 300
    # the default sizes are CATALOGUE's
    for mine, theirs in zip(catalogue(), CATALOGUE, strict=True):
        assert property_cases(mine, 4) == property_cases(theirs, 4)


def test_a_correct_adder_passes_a_sized_catalogue_at_a_long_length():
    results = evaluate(correct, 40, catalogue(sampled=200, relation_sampled=20, per_stratum=5), shrink=False)
    assert failing(results) == set()
    assert by_id(results)["M3"].reach() == 40


def test_doubling_takes_m3_at_powers_of_two_and_the_length():
    assert doubling_distances(1) == [1]
    assert doubling_distances(8) == [1, 2, 4, 8]
    assert doubling_distances(12) == [1, 2, 4, 8, 12]
    doubled = next(prop for prop in catalogue(per_stratum=5, doubling=True) if prop.id == "M3")
    cases = property_cases(doubled, 1024)
    assert len(cases) == 11 * 5 and {case.stratum for case in cases} == set(doubling_distances(1024))
    for case in cases:
        assert case.column is not None
        assert distance(column_sums(case.triple, 1024), case.column) == case.stratum
    results = evaluate(correct, 1024, [doubled], shrink=False)
    assert failing(results) == set() and by_id(results)["M3"].reach() == 1024


def test_training_draws_held_out_triples_only_when_asked():
    rng = Random(2)
    assert not any(held_out(training_triple(i, 2, rng), 2) for i in range(2000))
    assert any(held_out(training_triple(i, 2, rng, hold_out=False), 2) for i in range(2000))


def test_property_cases_are_the_same_on_every_call():
    prop = PROPERTIES["M3"]
    assert property_cases(prop, 4) == property_cases(prop, 4)


def test_a_correct_adder_has_every_property():
    results = evaluate(correct, 3)
    assert failing(results) == set()
    assert by_id(results)["M3"].reach() == 3


def test_a_carry_free_adder_has_the_algebra_and_fails_every_carry():
    # the algebraic properties alone don't make an adder: the oracle properties are needed
    results = evaluate(no_carry(3), 3)
    assert failing(results) == {"B1", "B2", "B3", "M2", "M3", "M4", "M5"}
    assert by_id(results)["M3"].reach() == 0


def test_a_short_reach_adder_fails_beyond_its_reach():
    for length in (1, 2):
        results = evaluate(reach(4, length), 4, [PROPERTIES[i] for i in ("M1", "M2", "M3")])
        assert failing(results) == {"M3"}
        assert by_id(results)["M3"].reach() == length


def test_dropping_an_operand_breaks_commutativity_and_identity():
    assert {"B1", "B2", "A1", "A3"} <= failing(evaluate(drops_third, 3))


def test_a_failure_shrinks_to_one_carrying_column():
    n = 3
    result = by_id(evaluate(no_carry(n), n, [PROPERTIES["B1"]]))["B1"]
    assert len(result.failures) == 3
    for case in result.failures:
        sums = column_sums(case.triple, n)
        assert sum(sums) == BASE and sum(s > 0 for s in sums) == 1  # the smallest carry


def test_a_shrunk_column_case_keeps_its_column():
    result = by_id(evaluate(reach(4, 1), 4, [PROPERTIES["M3"]]))["M3"]
    for case in result.failures:
        assert isinstance(case, Case) and case.column is not None
        assert not PROPERTIES["M3"].verify([case], reach(4, 1), 4)[0]


def test_a_limit_screens_a_spread_of_the_same_cases():
    results = evaluate(correct, 4, limit=500)
    assert all(result.cases <= 500 for result in results)
    assert by_id(results)["M3"].strata.keys() == {1, 2, 3, 4}


def test_digits_and_from_digits_are_the_per_digit_formulas_at_every_length():
    # the formulas they replaced (the window-program workplan's stage 1): quadratic in n
    rng = Random(0)
    for n in [*range(65), 127, 128, 129, 1024]:
        for x in {0, BASE**n - 1, rng.randrange(BASE**n), rng.randrange(BASE**n)}:
            ds = digits(x, n)
            assert ds == tuple((x // BASE**t) % BASE for t in range(n))
            assert from_digits(ds) == x
        # any integers, not only digits (a prediction's labels)
        free = [rng.randrange(-5, 9) for _ in range(n)]
        assert from_digits(free) == sum(d * BASE**t for t, d in enumerate(free))
