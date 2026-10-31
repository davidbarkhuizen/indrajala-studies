"""
The statistics the window-recurrent claims are analysed by, fixed before any run
(indrajala-research/docs/research-notes/window-recurrent-report.md, D4): a rate's Wilson 95% interval, Fisher's exact test of
two arms, the Cochran-Armitage test of a trend over a cell's code counts, the Mann-Whitney test of
two samples (examples to pass), and Holm's correction over a family of tests. By hand (no scipy):
Fisher's test and Mann-Whitney's without ties are exact, in integers; Cochran-Armitage and
Mann-Whitney with ties take the normal approximation. The paper workplan's T10 (D19) adds exact
one-sided rank tests with ties (midranks), for small samples of tied counts: the rank sum of two
samples and the signed rank of paired ones, each a permutation test counted in integers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from statistics import NormalDist

Z95 = NormalDist().inv_cdf(0.975)


def _two_sided(z: float) -> float:
    """A standard normal statistic's two-sided p value."""
    return math.erfc(abs(z) / math.sqrt(2))


def wilson(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """A rate's Wilson score interval (95% by default); no trials, the whole of [0, 1]."""
    assert 0 <= successes <= trials
    if trials == 0:
        return 0.0, 1.0
    rate = successes / trials
    centre = rate + z * z / (2 * trials)
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
    scale = 1 + z * z / trials
    low = 0.0 if successes == 0 else (centre - spread) / scale  # exact at the ends, not 1e-17 off
    high = 1.0 if successes == trials else (centre + spread) / scale
    return low, high


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """
    The two-sided p value of Fisher's exact test of the table [[a, b], [c, d]] (rows: the arms;
    columns: working, not): the probability, under the margins fixed, of a table no more likely
    than this one, its terms compared exactly as integers.
    """
    assert min(a, b, c, d) >= 0
    first, second, working = a + b, c + d, a + c
    low, high = max(0, working - second), min(working, first)
    weights = {x: math.comb(first, x) * math.comb(second, working - x) for x in range(low, high + 1)}
    observed = weights[a]
    return float(Fraction(sum(w for w in weights.values() if w <= observed), sum(weights.values())))


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """
    The one-sided p value of Fisher's exact test of the table [[a, b], [c, d]] (rows: the arms;
    columns: working, not), the alternative that the first arm works more often: the probability,
    under the margins fixed, of `a` or more working in the first arm, exactly as integers.
    """
    assert min(a, b, c, d) >= 0
    first, second, working = a + b, c + d, a + c
    low, high = max(0, working - second), min(working, first)
    weights = {x: math.comb(first, x) * math.comb(second, working - x) for x in range(low, high + 1)}
    return float(Fraction(sum(w for x, w in weights.items() if x >= a), sum(weights.values())))


@dataclass(frozen=True)
class Trend:
    z: float  # positive: the rate rises with the score
    p: float  # two-sided


def cochran_armitage(successes: Sequence[int], trials: Sequence[int], scores: Sequence[float]) -> Trend:
    """
    The Cochran-Armitage test of a linear trend in a rate over ordered groups (each with its
    successes, trials and score: the code count, say), two-sided, by the normal approximation.
    """
    assert len(successes) == len(trials) == len(scores) >= 2
    total, rate = sum(trials), sum(successes) / sum(trials)
    statistic = sum(t * (x - n * rate) for x, n, t in zip(successes, trials, scores, strict=True))
    spread = sum(n * t * t for n, t in zip(trials, scores, strict=True)) - (
        sum(n * t for n, t in zip(trials, scores, strict=True)) ** 2 / total
    )
    variance = rate * (1 - rate) * spread
    if variance == 0:  # every group at 0 or every one at 1, or a single score: no evidence of a trend
        return Trend(0.0, 1.0)
    z = statistic / math.sqrt(variance)
    return Trend(z, _two_sided(z))


@dataclass(frozen=True)
class RankTest:
    u: float  # the first sample's U: pairs where it is larger, a tie counting a half
    p: float  # two-sided
    exact: bool


def _u_counts(m: int, n: int) -> list[int]:
    """How many of the C(m + n, m) orderings of two samples without ties give each U, 0 to m n."""
    # counts[i][j][u]: arrangements of i of the first and j of the second with statistic u
    previous = [[1] for _ in range(n + 1)]  # i = 0: U is 0
    for i in range(1, m + 1):
        current: list[list[int]] = [[1]]  # j = 0: U is 0
        for j in range(1, n + 1):
            # the largest value is from the first sample (it beats all j) or from the second
            row = [0] * (i * j + 1)
            for u, count in enumerate(previous[j]):
                row[u + j] += count
            for u, count in enumerate(current[j - 1]):
                row[u] += count
            current.append(row)
        previous = current
    return previous[n]


def mann_whitney(first: Sequence[float], second: Sequence[float]) -> RankTest:
    """
    The Mann-Whitney U test, two-sided: exact (the U distribution counted) when no value is shared
    between or within the samples; with ties, the normal approximation with the tie correction and
    a continuity correction of a half.
    """
    m, n = len(first), len(second)
    assert m > 0 and n > 0
    u = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in first for y in second)
    values = [*first, *second]
    if len(set(values)) == len(values):
        counts = _u_counts(m, n)
        total = sum(counts)
        below = sum(counts[: int(u) + 1])  # P(U <= u)
        above = sum(counts[int(u) :])  # P(U >= u)
        return RankTest(u, min(1.0, float(Fraction(2 * min(below, above), total))), True)
    mean = m * n / 2
    tied = [values.count(v) for v in set(values)]
    size = m + n
    variance = m * n / 12 * (size + 1 - sum(t**3 - t for t in tied) / (size * (size - 1)))
    if variance == 0:  # every value the same
        return RankTest(u, 1.0, False)
    z = max(0.0, abs(u - mean) - 0.5) / math.sqrt(variance)
    return RankTest(u, _two_sided(z), False)


def _doubled_midranks(values: Sequence[float]) -> list[int]:
    """Each value's midrank among them, doubled (an integer: ties share their ranks' mean)."""
    order = sorted(values)
    first = {v: order.index(v) for v in set(values)}  # the 0-based position of each value's first
    return [2 * first[v] + order.count(v) + 1 for v in values]


def rank_sum_greater(first: Sequence[float], second: Sequence[float]) -> float:
    """
    The exact one-sided p value of the Wilcoxon rank-sum test that `first` tends larger: under every
    split of the pooled values into samples of the two sizes equally likely, the share whose first
    sample's midrank sum is at least the observed one, counted exactly over the tie groups.
    """
    m, n = len(first), len(second)
    assert m > 0 and n > 0
    ranks = _doubled_midranks([*first, *second])
    observed = sum(ranks[:m])
    groups: dict[int, int] = {}  # each doubled midrank, its tie group's size
    for rank in ranks:
        groups[rank] = groups.get(rank, 0) + 1
    ways: dict[tuple[int, int], int] = {(0, 0): 1}  # (values in the first sample, their rank sum): splits
    for rank, size in groups.items():
        following: dict[tuple[int, int], int] = {}
        for (taken, total), count in ways.items():
            for k in range(min(size, m - taken) + 1):
                key = (taken + k, total + k * rank)
                following[key] = following.get(key, 0) + count * math.comb(size, k)
        ways = following
    tail = sum(count for (taken, total), count in ways.items() if taken == m and total >= observed)
    return float(Fraction(tail, math.comb(m + n, m)))


def signed_rank_greater(differences: Sequence[float]) -> float:
    """
    The exact one-sided p value of the Wilcoxon signed-rank test that paired differences tend
    positive: zeros dropped (Wilcoxon's convention), the rest ranked by size with midranks, and the
    share of the 2^n sign assignments whose positive ranks sum to at least the observed; 1 with no
    nonzero difference.
    """
    nonzero = [d for d in differences if d != 0]
    if not nonzero:
        return 1.0
    ranks = _doubled_midranks([abs(d) for d in nonzero])
    observed = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    ways = {0: 1}  # a positive ranks' sum: assignments
    for rank in ranks:
        following = dict(ways)
        for total, count in ways.items():
            following[total + rank] = following.get(total + rank, 0) + count
        ways = following
    return float(Fraction(sum(c for total, c in ways.items() if total >= observed), 2 ** len(nonzero)))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p values, in the order given: reject where adjusted <= alpha."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
