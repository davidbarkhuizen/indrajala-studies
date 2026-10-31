"""
Base-3 addition of three operands, and the catalogue of properties a network that adds must have
(the addition study workplan's Stage 2, retired: git show 8990f74:docs/addition-study-workplan.md).

An addition is a triple (a, b, c) of n-digit base-3 operands, 0 <= a, b, c < 3^n; its sum, under
3^(n+1), has n + 1 digits. Two-operand addition is the case with a zero in one slot. Digits are
least significant first throughout. Column t's digit sum s_t = a_t + b_t + c_t is in 0..6; with the
carry in k_t (0..2) the sum's digit t is (s_t + k_t) mod 3 and the carry out (s_t + k_t) // 3. A
column sum of 0, 3 or 6 is a reset (its carry out is 0, 1 or 2 whatever comes in); 1, 2, 4 and 5
are dependent. The carry into column t is fixed by the nearest reset below it (or column 0's carry
in, 0) composed through the dependent columns between: their count is the carry's distance.

A property is a statement, a generator of cases, a verifier and a coverage (the workplan's
catalogue). An Adder is what a property is verified against: a candidate network's encoder,
forward pass and decoder, or a plain function, taking a batch of triples to their sums. evaluate()
runs the catalogue against one and shrinks each failure to a minimal failing case. Every
property's cases are drawn from a generator seeded by the property and n alone, so every candidate
meets the same cases. The training mixture draws from other seeds and never draws a held-out
triple (D8).
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from random import Random

BASE = 3
COLUMN_SUMS = range(3 * (BASE - 1) + 1)  # 0..6
RESETS = (0, 3, 6)  # the column sums whose carry out doesn't depend on the carry in
DEPENDENT = (1, 2, 4, 5)
HELD_OUT_TENTHS = 1  # D8: a tenth of the triples, by hash
EXHAUSTIVE_LIMIT = 600_000  # a space of at most this many cases is tested whole
SAMPLED = 100_000  # the behavioural properties' sample, over a larger space
RELATION_SAMPLED = 10_000  # A1 and A2
PER_STRATUM = 1_000  # M3 per distance, M4 per final carry
PATTERN_CASES = 10_000  # M5: per carry pattern, max(1, this // 7^n) concrete triples
FAILURES_KEPT = 3  # shrunk failing cases kept per property

Triple = tuple[int, int, int]
# a batch of triples to their sums (each under 3^(n+1)); a network's encoder, forward and decoder
Adder = Callable[[Sequence[Triple]], list[int]]


DIGIT_CHUNK = 8  # digits split off x at a time
_CHUNK_DIGITS = tuple(tuple((v // BASE**t) % BASE for t in range(DIGIT_CHUNK)) for v in range(BASE**DIGIT_CHUNK))


def digits(x: int, n: int) -> tuple[int, ...]:
    """
    x's n base-3 digits, least significant first: DIGIT_CHUNK at a time, each chunk's digits from a
    table (`x // BASE**t` for every t is quadratic in n, slow at the long tests' 1,024 columns).
    """
    assert 0 <= x < BASE**n, f"{x} doesn't have {n} base-{BASE} digits"
    result: list[int] = []
    for _ in range(-(-n // DIGIT_CHUNK)):
        x, low = divmod(x, BASE**DIGIT_CHUNK)
        result += _CHUNK_DIGITS[low]
    return tuple(result[:n])


def from_digits(ds: Sequence[int]) -> int:
    """The number of digits ds, least significant first: Horner's rule, DIGIT_CHUNK at a time (as
    digits; any integers, not only 0-2, give sum(d * BASE**t))."""
    value = 0
    for start in reversed(range(0, len(ds), DIGIT_CHUNK)):
        chunk = ds[start : start + DIGIT_CHUNK]
        low = 0
        for d in reversed(chunk):
            low = low * BASE + d
        value = value * BASE ** len(chunk) + low
    return value


def column_sums(triple: Triple, n: int) -> tuple[int, ...]:
    a, b, c = (digits(x, n) for x in triple)
    return tuple(a[t] + b[t] + c[t] for t in range(n))


def carries(sums: Sequence[int]) -> tuple[int, ...]:
    """The carry into each column 0..n (n + 1 of them, the last the sum's top digit)."""
    k = [0]
    for s in sums:
        k.append((s + k[-1]) // BASE)
    return tuple(k)


def distance(sums: Sequence[int], t: int) -> int:
    """The count of dependent columns directly below column t, down to a reset or column 0."""
    count = 0
    while t - 1 - count >= 0 and sums[t - 1 - count] not in RESETS:
        count += 1
    return count


def sum_digit(total: int, t: int) -> int:
    return (total // BASE**t) % BASE


def triple_hash(triple: Triple, n: int) -> int:
    """A 64-bit hash of a triple at n; every order of a triple alike."""
    x, y, z = sorted(triple)
    return splitmix64(((x * BASE**n + y) * BASE**n + z) * 64 + n)


def held_out(triple: Triple, n: int) -> bool:
    """D8: whether a triple is in the held-out tenth; every order of a triple alike."""
    return triple_hash(triple, n) % 10 < HELD_OUT_TENTHS


def splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


# a column sum's digit triples: 1, 3, 6, 7, 6, 3, 1 of them
_COLUMN_DIGITS = {s: [d for d in itertools.product(range(BASE), repeat=3) if sum(d) == s] for s in COLUMN_SUMS}


def triple_of_sums(sums: Sequence[int], rng: Random) -> Triple:
    """A triple with these column sums, each column's digits uniform among those with its sum."""
    columns = [rng.choice(_COLUMN_DIGITS[s]) for s in sums]
    return (
        from_digits([column[0] for column in columns]),
        from_digits([column[1] for column in columns]),
        from_digits([column[2] for column in columns]),
    )


def uniform_triple(n: int, rng: Random) -> Triple:
    return (rng.randrange(BASE**n), rng.randrange(BASE**n), rng.randrange(BASE**n))


def pattern_triple(n: int, rng: Random) -> Triple:
    """A triple whose carry pattern (column sums) is uniform over the 7^n."""
    return triple_of_sums([rng.choice(COLUMN_SUMS) for _ in range(n)], rng)


def two_operand_triple(n: int, rng: Random) -> Triple:
    """A uniform pair, with the zero in a uniform slot."""
    pair = [rng.randrange(BASE**n), rng.randrange(BASE**n)]
    pair.insert(rng.randrange(3), 0)
    return (pair[0], pair[1], pair[2])


_KINDS = (uniform_triple, pattern_triple, two_operand_triple)


def training_triple(kind: int, n: int, rng: Random, hold_out: bool = True) -> Triple:
    """
    One triple of D7's mixture: kind 0 uniform, 1 uniform over carry patterns, 2 two-operand (kind
    taken mod 3); a held-out triple (D8) is drawn again, unless not hold_out.
    """
    draw = _KINDS[kind % 3]
    triple = draw(n, rng)
    while hold_out and held_out(triple, n):
        triple = draw(n, rng)
    return triple


def training_triples(n: int, count: int, rng: Random) -> list[Triple]:
    """D7's mixture: a third of each kind, in turn; a held-out triple (D8) is drawn again."""
    return [training_triple(i, n, rng) for i in range(count)]


@dataclass(frozen=True)
class Case:
    """One case of a property: a triple, the sum's digit checked (None: the whole sum), a stratum."""

    triple: Triple
    column: int | None = None
    stratum: int | None = None


Verifier = Callable[[Sequence[Case], Adder, int], list[bool]]


@dataclass(frozen=True)
class Property:
    id: str
    statement: str
    generate: Callable[[int, Random], list[Case]]
    verify: Verifier
    coverage: str


@dataclass
class PropertyResult:
    id: str
    cases: int
    passed: int
    # stratum: (passed, cases)
    strata: dict[int, tuple[int, int]] = field(default_factory=dict[int, tuple[int, int]])
    failures: list[Case] = field(default_factory=list[Case])  # shrunk, the first FAILURES_KEPT

    @property
    def holds(self) -> bool:
        return self.passed == self.cases

    @property
    def accuracy(self) -> float:
        return self.passed / self.cases

    def reach(self) -> int:
        """M3's L*: the largest distance L with every case at 1..L right (0 when L = 1 fails)."""
        reach = 0
        for stratum in sorted(self.strata):
            passed, cases = self.strata[stratum]
            if passed < cases:
                break
            reach = stratum
        return reach


# verifiers


def oracle(cases: Sequence[Case], f: Adder, n: int) -> list[bool]:
    """Each case's answer is the sum: the whole of it, or the digit at the case's column."""
    answers = f([case.triple for case in cases])
    return [
        answer == sum(case.triple)
        if case.column is None
        else sum_digit(answer, case.column) == sum_digit(sum(case.triple), case.column)
        for case, answer in zip(cases, answers, strict=True)
    ]


def _orders(triple: Triple) -> list[Triple]:
    return [(x, y, z) for x, y, z in itertools.permutations(triple)]


def commutes(cases: Sequence[Case], f: Adder, n: int) -> list[bool]:
    """A1: the 6 orders of each triple give one answer."""
    answers = f([order for case in cases for order in _orders(case.triple)])
    return [len(set(answers[6 * i : 6 * i + 6])) == 1 for i in range(len(cases))]


def associates(cases: Sequence[Case], f: Adder, n: int) -> list[bool]:
    """
    A2: f(f(a, b, 0), c, 0) = f(a, f(b, c, 0), 0) = f(a, b, c), operands of n - 1 digits; a partial
    sum that doesn't fit n digits fails.
    """
    first = f([q for case in cases for q in _first_stage(case.triple)])
    partials = [(first[3 * i], first[3 * i + 1], first[3 * i + 2]) for i in range(len(cases))]
    fits = [ab < BASE**n and bc < BASE**n for ab, bc, _abc in partials]
    second_queries = [
        q
        for case, (ab, bc, _abc), ok in zip(cases, partials, fits, strict=True)
        if ok
        for q in ((ab, case.triple[2], 0), (case.triple[0], bc, 0))
    ]
    second = iter(f(second_queries)) if second_queries else iter(())
    holds: list[bool] = []
    for (_ab, _bc, abc), ok in zip(partials, fits, strict=True):
        if not ok:
            holds.append(False)
            continue
        left, right = next(second), next(second)
        holds.append(left == right == abc)
    return holds


def _first_stage(triple: Triple) -> tuple[Triple, Triple, Triple]:
    a, b, c = triple
    return ((a, b, 0), (b, c, 0), (a, b, c))


# generators


def _all_triples(n: int) -> Iterator[Triple]:
    r = range(BASE**n)
    return ((a, b, c) for a in r for b in r for c in r)


def _adds_three(n: int, rng: Random, sampled: int) -> list[Case]:
    if BASE ** (3 * n) <= EXHAUSTIVE_LIMIT:
        return [Case(t) for t in _all_triples(n)]
    return [Case(uniform_triple(n, rng)) for _ in range(sampled)]


def _adds_two(n: int, rng: Random, sampled: int) -> list[Case]:
    if 3 * BASE ** (2 * n) <= EXHAUSTIVE_LIMIT:
        r = range(BASE**n)
        return [Case(t) for x in r for y in r for t in ((0, x, y), (x, 0, y), (x, y, 0))]
    return [Case(two_operand_triple(n, rng)) for _ in range(sampled)]


def _held_out(n: int, rng: Random, sampled: int) -> list[Case]:
    if BASE ** (3 * n) <= EXHAUSTIVE_LIMIT:
        return [Case(t) for t in _all_triples(n) if held_out(t, n)]
    cases: list[Case] = []
    while len(cases) < sampled:
        triple = uniform_triple(n, rng)
        if held_out(triple, n):
            cases.append(Case(triple))
    return cases


def _commutativity(n: int, rng: Random, sampled: int) -> list[Case]:
    cases = [Case(uniform_triple(n, rng)) for _ in range(sampled // 2)]
    for _ in range(sampled - len(cases)):
        x, y = rng.randrange(BASE**n), rng.randrange(BASE**n)
        triple = [x, x, y]
        rng.shuffle(triple)
        cases.append(Case((triple[0], triple[1], triple[2])))
    return cases


def _associativity(n: int, rng: Random, sampled: int) -> list[Case]:
    return [Case(uniform_triple(n - 1, rng)) for _ in range(sampled)]


def _identity(n: int, rng: Random, sampled: int) -> list[Case]:
    if 3 * BASE**n <= EXHAUSTIVE_LIMIT:
        return [Case(t) for a in range(BASE**n) for t in ((a, 0, 0), (0, a, 0), (0, 0, a))]
    cases: list[Case] = []
    for _ in range(sampled):
        triple = [rng.randrange(BASE**n), 0, 0]
        rng.shuffle(triple)
        cases.append(Case((triple[0], triple[1], triple[2])))
    return cases


def _column_sum(n: int, rng: Random, sampled: int) -> list[Case]:
    if 4 * BASE**n <= EXHAUSTIVE_LIMIT:
        return [Case(triple_of_sums(sums, rng)) for sums in itertools.product(range(BASE), repeat=n) for _ in range(4)]
    return [Case(triple_of_sums([rng.randrange(BASE) for _ in range(n)], rng)) for _ in range(sampled)]


def _carry_from_reset(n: int, rng: Random) -> list[Case]:
    # a reset at column j, every value; the digit checked is j + 1's
    return [
        Case(triple_of_sums(sums, rng), column=j + 1, stratum=reset // BASE)
        for j in range(n)
        for reset in RESETS
        for _ in range(16)
        for sums in [[reset if t == j else rng.choice(COLUMN_SUMS) for t in range(n)]]
    ]


def doubling_distances(n: int) -> list[int]:
    """1, 2, 4, ... up to n, and n: M3's distances at a long length (the claims workplan's D9)."""
    distances = [2**i for i in range(n.bit_length()) if 2**i <= n]
    return distances if distances[-1] == n else [*distances, n]


def _carry_over_distance(n: int, rng: Random, per_stratum: int, doubling: bool = False) -> list[Case]:
    # for each distance L in 1..n (or doubling_distances(n)): the checked column t, a reset at
    # t - 1 - L (column 0's carry in when that is -1), L dependent columns between, every other
    # column uniform
    cases: list[Case] = []
    for length in doubling_distances(n) if doubling else range(1, n + 1):
        for _ in range(per_stratum):
            t = rng.randrange(length, n + 1)
            r = t - 1 - length
            sums = [
                rng.choice(RESETS) if i == r else rng.choice(DEPENDENT) if r < i < t else rng.choice(COLUMN_SUMS)
                for i in range(n)
            ]
            cases.append(Case(triple_of_sums(sums, rng), column=t, stratum=length))
    return cases


def _final_carry(n: int, rng: Random, per_stratum: int) -> list[Case]:
    cases: list[Case] = []
    for value in range(BASE):
        count = 0
        while count < per_stratum:
            triple = pattern_triple(n, rng)
            if carries(column_sums(triple, n))[n] == value:
                cases.append(Case(triple, column=n, stratum=value))
                count += 1
    return cases


def _every_pattern(n: int, rng: Random, sampled: int) -> list[Case]:
    patterns = len(COLUMN_SUMS) ** n
    if patterns > EXHAUSTIVE_LIMIT:  # past n = 6: sampled patterns (the terminated plan's D8)
        return [Case(pattern_triple(n, rng)) for _ in range(sampled)]
    per_pattern = max(1, PATTERN_CASES // patterns)
    return [
        Case(triple_of_sums(sums, rng)) for sums in itertools.product(COLUMN_SUMS, repeat=n) for _ in range(per_pattern)
    ]


def catalogue(
    sampled: int = SAMPLED,
    relation_sampled: int = RELATION_SAMPLED,
    per_stratum: int = PER_STRATUM,
    doubling: bool = False,
) -> tuple[Property, ...]:
    """
    The catalogue, its samples sized: sampled cases for B1-B3, and past the exhaustive limit for A3,
    M1 and M5 (so any n is affordable: the window-recurrent adder workplan, W17); relation_sampled
    for A1 and A2; per_stratum per M3 distance and M4 final carry; M3 at doubling_distances only when
    doubling. At n <= 6 with the default sizes, every property's cases are CATALOGUE's.
    """
    sized = functools.partial
    return (
        Property(
            "B1",
            "adds three: f(a, b, c) = a + b + c",
            sized(_adds_three, sampled=sampled),
            oracle,
            f"every triple, or {sampled:,} sampled",
        ),
        Property(
            "B2",
            "adds two: a zero in any one slot",
            sized(_adds_two, sampled=sampled),
            oracle,
            f"every pair in every slot, or {sampled:,} sampled",
        ),
        Property(
            "B3",
            "adds three on the held-out tenth (D8)",
            sized(_held_out, sampled=sampled),
            oracle,
            f"the whole tenth, or {sampled:,} sampled",
        ),
        Property(
            "A1",
            "commutativity: the 6 orders give one answer",
            sized(_commutativity, sampled=relation_sampled),
            commutes,
            f"{relation_sampled:,} triples × 6",
        ),
        Property(
            "A2",
            "associativity: f(f(a, b, 0), c, 0) = f(a, f(b, c, 0), 0) = f(a, b, c)",
            sized(_associativity, sampled=relation_sampled),
            associates,
            f"{relation_sampled:,} triples of n - 1 digits",
        ),
        Property(
            "A3",
            "identity: f(a, 0, 0) = f(0, a, 0) = f(0, 0, a) = a",
            sized(_identity, sampled=sampled),
            oracle,
            f"every a, every slot, or {sampled:,} sampled",
        ),
        Property(
            "M1",
            "column sum: with no carries, digit t is s_t",
            sized(_column_sum, sampled=sampled),
            oracle,
            f"every pattern over 0..2 × 4, or {sampled:,} sampled",
        ),
        Property(
            "M2", "carry from a reset: the next column's digit", _carry_from_reset, oracle, "every column × reset × 16"
        ),
        Property(
            "M3",
            "carry over distance L: the digit at distance L above a reset",
            sized(_carry_over_distance, per_stratum=per_stratum, doubling=doubling),
            oracle,
            f"{per_stratum:,} per L in " + ("1, 2, 4, ..., n" if doubling else "1..n"),
        ),
        Property(
            "M4",
            "final carry: the sum's top digit, 0, 1 or 2",
            sized(_final_carry, per_stratum=per_stratum),
            oracle,
            f"{per_stratum:,} per value",
        ),
        Property(
            "M5",
            "every carry pattern: every one of the 7^n",
            sized(_every_pattern, sampled=sampled),
            oracle,
            f"every pattern × max(1, 10,000 // 7^n), or {sampled:,} sampled",
        ),
    )


CATALOGUE: tuple[Property, ...] = catalogue()


def property_cases(prop: Property, n: int) -> list[Case]:
    """A property's cases at n: the same for every candidate and every run."""
    return prop.generate(n, Random(f"addition-{prop.id}-{n}"))


@functools.cache
def screen_cases(prop: Property, n: int, limit: int) -> list[Case]:
    """Every k-th of a property's cases, k the smallest that leaves at most limit; held per process."""
    cases = property_cases(prop, n)
    return cases[:: -(-len(cases) // limit)] if len(cases) > limit else cases


def evaluate(
    f: Adder, n: int, properties: Sequence[Property] = CATALOGUE, limit: int | None = None, shrink: bool = True
) -> list[PropertyResult]:
    """
    Each property verified against f, its failures shrunk (unless not shrink). limit takes every
    k-th case (k the smallest that leaves at most limit): a fast screen of the same cases.
    """
    results: list[PropertyResult] = []
    for prop in properties:
        cases = property_cases(prop, n) if limit is None else screen_cases(prop, n, limit)
        holds = prop.verify(cases, f, n)
        result = PropertyResult(prop.id, len(cases), sum(holds))
        for case, ok in zip(cases, holds, strict=True):
            if case.stratum is not None:
                passed, count = result.strata.get(case.stratum, (0, 0))
                result.strata[case.stratum] = (passed + ok, count + 1)
            if not ok and len(result.failures) < FAILURES_KEPT:
                result.failures.append(minimal(case, prop.verify, f, n) if shrink else case)
        results.append(result)
    return results


def minimal(case: Case, verify: Verifier, f: Adder, n: int) -> Case:
    """
    A failing case made minimal: whole columns zeroed from the top while it still fails, then
    single digits lowered; the column checked and the stratum kept.
    """

    def fails(triple: Triple) -> bool:
        return not verify([Case(triple, case.column, case.stratum)], f, n)[0]

    operands: list[list[int]] = [list(digits(x, n)) for x in case.triple]
    changed = True
    while changed:
        changed = False
        for t in reversed(range(n)):
            if any(o[t] for o in operands):
                trial = [[0 if i == t else d for i, d in enumerate(o)] for o in operands]
                if fails(_triple(trial)):
                    operands, changed = trial, True
        for o_index in range(3):
            for t in range(n):
                while operands[o_index][t] > 0:
                    trial = [list(o) for o in operands]
                    trial[o_index][t] -= 1
                    if not fails(_triple(trial)):
                        break
                    operands, changed = trial, True
    return Case(_triple(operands), case.column, case.stratum)


def _triple(operands: Sequence[Sequence[int]]) -> Triple:
    return (from_digits(operands[0]), from_digits(operands[1]), from_digits(operands[2]))
