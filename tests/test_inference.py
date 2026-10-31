"""
indrajala_studies/inference.py against published values (Fisher's tea-tasting table, the
control's and D4's figures, Wilson's interval) and independent oracles: Mann-Whitney's exact p by
enumerating every split, Cochran-Armitage over two groups as Pearson's chi-square.
"""

import itertools
import math
from collections.abc import Sequence

import pytest

from indrajala_studies import inference as st


def test_wilson_interval():
    assert st.wilson(8, 10) == pytest.approx((0.490162, 0.943318), abs=1e-6)
    assert st.wilson(0, 10) == pytest.approx((0.0, 0.277533), abs=1e-6) and st.wilson(0, 10)[0] == 0.0
    assert st.wilson(20, 20)[1] == 1.0 and st.wilson(0, 0) == (0.0, 1.0)
    low, high = st.wilson(10, 20)
    assert low == pytest.approx(1 - high)


def test_fisher_exact_two_sided():
    assert st.fisher_exact(3, 1, 1, 3) == pytest.approx(0.485714, abs=1e-6)  # the lady tasting tea
    assert st.fisher_exact(10, 0, 8, 2) == pytest.approx(0.473684, abs=1e-6)  # the control at k 4
    assert st.fisher_exact(2, 3, 4, 1) == pytest.approx(0.523810, abs=1e-6)  # the control at k 1
    assert st.fisher_exact(16, 4, 8, 12) == pytest.approx(0.0225, abs=1e-4)  # D4: 0.8 against 0.4 at 20
    assert st.fisher_exact(15, 5, 10, 10) == pytest.approx(0.19, abs=0.005)  # D4: 0.75 against 0.5
    assert st.fisher_exact(5, 5, 5, 5) == 1.0 and st.fisher_exact(0, 0, 0, 0) == 1.0
    # the same test whichever way the table is read
    assert st.fisher_exact(4, 16, 12, 8) == st.fisher_exact(16, 4, 8, 12) == st.fisher_exact(16, 8, 4, 12)


def test_fisher_exact_greater_one_sided():
    assert st.fisher_exact_greater(3, 1, 1, 3) == pytest.approx(17 / 70)  # the lady tasting tea, one-sided
    assert st.fisher_exact_greater(1, 3, 3, 1) == pytest.approx(69 / 70)  # the other direction
    assert st.fisher_exact_greater(16, 4, 8, 12) == pytest.approx(0.0112, abs=1e-4)  # half D4's two-sided
    assert st.fisher_exact_greater(0, 0, 0, 0) == 1.0 and st.fisher_exact_greater(0, 5, 0, 5) == 1.0
    # a table and its mirror (arms swapped, columns swapped) are the same alternative
    assert st.fisher_exact_greater(13, 7, 1, 19) == st.fisher_exact_greater(19, 1, 7, 13)


def _enumerated_p(first: Sequence[float], second: Sequence[float]) -> float:
    """Mann-Whitney's two-sided exact p by every split of the pooled values."""
    pooled = [*first, *second]

    def u(chosen: tuple[int, ...]) -> float:
        rest = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        return sum(1.0 if pooled[i] > y else 0.5 if pooled[i] == y else 0.0 for i in chosen for y in rest)

    observed = u(tuple(range(len(first))))
    splits = [u(chosen) for chosen in itertools.combinations(range(len(pooled)), len(first))]
    below = sum(x <= observed for x in splits) / len(splits)
    above = sum(x >= observed for x in splits) / len(splits)
    return min(1.0, 2 * min(below, above))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ([0.39, 1.05, 0.26, 0.52, 0.71], [6.4, 4.8, 12.3, 0.6, 7.1, 5.5]),
        ([3, 9, 1, 7], [2, 8, 4, 6, 5]),
        ([10.0], [1.0, 2.0, 3.0]),
    ],
)
def test_mann_whitney_exact_against_enumeration(first: Sequence[float], second: Sequence[float]):
    result = st.mann_whitney(first, second)
    assert result.exact and result.p == pytest.approx(_enumerated_p(first, second), abs=1e-12)
    assert st.mann_whitney(second, first).u == len(first) * len(second) - result.u
    assert st.mann_whitney(second, first).p == pytest.approx(result.p)


def test_mann_whitney_with_ties_is_near_the_enumeration():
    first, second = [1, 2, 2, 3, 5, 5, 6], [2, 4, 5, 7, 8, 8, 9, 9]
    result = st.mann_whitney(first, second)
    assert not result.exact and result.u == 10.0  # 0 + 2 x 0.5 + 1 + 2 x 2.5 + 3
    assert result.p == pytest.approx(_enumerated_p(first, second), abs=0.02)
    assert st.mann_whitney([1, 1], [1, 1, 1]).p == 1.0


def test_cochran_armitage_over_two_groups_is_pearsons_chi_square():
    a, b, c, d = 16, 4, 8, 12  # successes and failures of two groups
    chi_square = (a + b + c + d) * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    trend = st.cochran_armitage([a, c], [a + b, c + d], [0, 1])
    assert trend.z**2 == pytest.approx(chi_square) and trend.z < 0
    assert trend.p == pytest.approx(math.erfc(math.sqrt(chi_square / 2)))


def test_cochran_armitage_trend():
    successes, trials, scores = [2, 3, 5, 8], [10, 10, 10, 10], [3, 4, 8, 16]
    trend = st.cochran_armitage(successes, trials, scores)
    assert trend.z > 0 and trend.p < 0.01
    # a linear change of the scores leaves the test alone; reversing them flips the sign
    assert st.cochran_armitage(successes, trials, [2 * t + 5 for t in scores]).z == pytest.approx(trend.z)
    assert st.cochran_armitage(successes, trials, [-t for t in scores]).z == pytest.approx(-trend.z)
    assert st.cochran_armitage([0, 0], [5, 5], [1, 2]) == st.Trend(0.0, 1.0)
    assert st.cochran_armitage([3, 3, 3], [10, 10, 10], [1, 2, 3]).z == 0.0


def test_holm_adjusts_step_down():
    assert st.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert st.holm([0.5, 0.9]) == [1.0, 1.0] and st.holm([]) == []


def _brute_rank_sum_greater(first: list[float], second: list[float]) -> float:
    pooled = [*first, *second]
    ranks = [sum(1.0 if w < v else 0.5 if w == v else 0.0 for w in pooled) + 0.5 for v in pooled]
    observed = sum(ranks[: len(first)])
    splits = list(itertools.combinations(range(len(pooled)), len(first)))
    return sum(sum(ranks[i] for i in split) >= observed - 1e-9 for split in splits) / len(splits)


@pytest.mark.parametrize(
    ("first", "second"),
    [([2, 1, 1, 0], [0, 0, 1]), ([1, 1, 0, 0, 0], [0, 0, 0, 0]), ([3.0, 5.0, 4.0], [1.0, 2.0, 6.0]), ([0, 0], [1, 2])],
)
def test_the_exact_rank_sum_test_is_the_share_of_splits_at_least_as_extreme(first: list[float], second: list[float]):
    assert st.rank_sum_greater(first, second) == pytest.approx(_brute_rank_sum_greater(first, second))


def test_the_rank_sum_bars_are_d19s():
    """T10's bars (the paper workplan's D19): with P0 at 0, 5 of 10 planted machines p 0.016, 4 of 10 0.043; one P0 success, 5 of 10 0.070."""
    assert st.rank_sum_greater([1] * 5 + [0] * 5, [0] * 10) == pytest.approx(252 / 15504)
    assert st.rank_sum_greater([2] * 4 + [0] * 6, [0] * 10) == pytest.approx(0.0433, abs=1e-4)
    assert st.rank_sum_greater([1] * 5 + [0] * 5, [1] + [0] * 9) == pytest.approx(0.0704, abs=1e-4)


@pytest.mark.parametrize("differences", [[1, 2, 3], [1, -1, 2, 0, 2], [-1, -2, 1], [0, 0], [2, 2, 1, -1, 1]])
def test_the_exact_signed_rank_test_is_the_share_of_sign_assignments_at_least_as_extreme(differences: list[float]):
    nonzero = [d for d in differences if d != 0]
    if not nonzero:
        assert st.signed_rank_greater(differences) == 1.0
        return
    sizes = [abs(d) for d in nonzero]
    ranks = [sum(1.0 if w < v else 0.5 if w == v else 0.0 for w in sizes) + 0.5 for v in sizes]
    observed = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    signs = list(itertools.product((0, 1), repeat=len(ranks)))
    brute = sum(sum(r for r, s in zip(ranks, sign, strict=True) if s) >= observed - 1e-9 for sign in signs) / len(signs)
    assert st.signed_rank_greater(differences) == pytest.approx(brute)
