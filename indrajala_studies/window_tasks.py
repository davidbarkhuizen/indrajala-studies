"""
The window-recurrent study's tasks (indrajala-research/docs/research-notes/window-recurrent-report.md, stage 5, D14-D18):
what a window-recurrent network learns, as the harness (window_recurrent_study) and the
certificate (window_recurrent_certificate) read it.

A task is a target Mealy machine over column tokens (its start, its transition and its output for
each state and token), an encoding of its cases as column tokens in its reading order, a training
mixture and a catalogue of tests. Its labels are the machine's outputs: an example's tokens (the
alignment's zero columns, the case's columns and zero columns through the final window) run
through the machine give one label a column, so the certificate checks a network against the very
machine its labels come from. Zero columns leave each task's start state where it is (no carry; A
and B both 0), so an alignment's zero columns are part of the word like any other.

- **addition:** three base-3 operands, least significant first, a column's token 9 a_t + 3 b_t +
  c_t (27); the carry automaton, 3 states, its output the sum's digit (3 classes); addition_data's
  catalogue.
- **modular-division-p:** two base-3 numbers, most significant first, a column's token 3 a_i + b_i
  (9); the modular automaton, p^2 states (A mod p, B mod p), its output the running quotient or
  undefined (p + 1 classes); modular_data's catalogue.
- **modular-sum-q:** the same pairs and tokens; the automaton of (A + B) mod q, q states, its output
  the running sum (q classes); modular_data's sum catalogue (the claims workplan's E8).

The general algebras (indrajala-research's docs/workplans/general-algebras-workplan.md, D1, D2): three machines over the n!
orderings of n items, n + 1 tokens, token 0 the identity, each output the label of the state
after the token in n classes, hiding most of the state; a case is a word of tokens (MachineTask).

- **word-problem-s<n>:** S_n's word problem, its tokens the star transpositions (positions 1 and
  i) and the n-cycle acting on the ordering's positions; its output the item at position 1.
- **move-to-front-<n>:** an LRU's recency order, token i moving item i - 1 to the front (a left
  regular band: no group part); its output the last item, the one an eviction takes.
- **random-<n>-<seed>:** n random maps of n! states, redrawn until every state is reachable, its
  output a random label per state, redrawn until the machine is minimal.
- **random-perm-<n>-<seed>, random-q<q>-<seed>:** probes of the random class's failure B (the
  code-consolidation workplan's stage 2b): random-n with every token a bijection, and random-4's
  tokens and classes at q states.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from indrajala_studies import addition_data as ad
from indrajala_studies import modular_data as md
from indrajala_studies.addition_data import BASE

IntArray = npt.NDArray[np.int64]
Labels = tuple[int, ...]
# a batch of cases to each one's labels, from its first column (past the alignment) to its end
Predict = Callable[[Sequence[Any]], list[Labels]]


@dataclass(frozen=True)
class Coordinates:
    """
    A machine's states named by numbers mod modulus (the factored-state workplan, D9): values[s] the
    coordinates of state s, one per name (division's A and B), so the certificate can name what a
    factor of a network's state holds, as a linear form of them.
    """

    names: tuple[str, ...]
    values: IntArray  # (S, len(names))
    modulus: int


@dataclass(frozen=True)
class Mealy:
    """
    A Mealy machine over tokens 0..T-1: transition[s, t] the next state, output[s, t] the label;
    its states' coordinates, if named.
    """

    transition: IntArray  # (S, T)
    output: IntArray  # (S, T)
    start: int = 0
    coordinates: Coordinates | None = None

    @property
    def states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def tokens(self) -> int:
        return int(self.transition.shape[1])

    @property
    def classes(self) -> int:
        return int(self.output.max()) + 1

    @functools.cached_property
    def _table(self) -> list[list[tuple[int, int]]]:
        return [
            [(int(r), int(o)) for r, o in zip(row_t, row_o, strict=True)]
            for row_t, row_o in zip(self.transition, self.output, strict=True)
        ]

    def run(self, tokens: Sequence[int]) -> Labels:
        """The labels of a word of tokens from the start."""
        table, state = self._table, self.start
        labels: list[int] = []
        for token in tokens:
            state, label = table[state][token]
            labels.append(label)
        return tuple(labels)

    def windows(self, windows: IntArray) -> tuple[IntArray, IntArray]:
        """From each state (axis 0): each window's labels (S, N, k) and the state after it (S, N)."""
        labels = np.zeros((self.states, *windows.shape), dtype=np.int64)
        after = np.zeros((self.states, windows.shape[0]), dtype=np.int64)
        for s in range(self.states):
            state = np.full(windows.shape[0], s, dtype=np.int64)
            for j in range(windows.shape[1]):
                labels[s, :, j] = self.output[state, windows[:, j]]
                state = self.transition[state, windows[:, j]]
            after[s] = state
        return labels, after

    def reachable(self) -> list[int]:
        """The states reachable from the start, breadth first."""
        seen, queue = {self.start}, [self.start]
        for s in queue:
            for r in self.transition[s]:
                if int(r) not in seen:
                    seen.add(int(r))
                    queue.append(int(r))
        return queue

    def depth(self) -> int:
        """The deepest reachable state's distance from the start: the longest of the shortest words reaching each."""
        distance, queue = {self.start: 0}, [self.start]
        for s in queue:
            for r in self.transition[s]:
                if int(r) not in distance:
                    distance[int(r)] = distance[s] + 1
                    queue.append(int(r))
        return max(distance.values())

    def token_ranks(self) -> list[int]:
        """Each token's rank: the size of its map's image (the states some state goes to on it)."""
        return [len({int(r) for r in self.transition[:, t]}) for t in range(self.tokens)]

    def synchronising(self) -> bool:
        """
        Whether some word (a reset word) takes every state to one: by Cerny's criterion, whether every
        pair of states is taken to one by some word (pairs read back from the merged ones).
        """
        before: dict[int, set[tuple[int, int]]] = {}
        for s, r in itertools.combinations_with_replacement(range(self.states), 2):
            for t in range(self.tokens):
                a, b = sorted((int(self.transition[s, t]), int(self.transition[r, t])))
                before.setdefault(a * self.states + b, set()).add((s, r))
        merged = {(s, s) for s in range(self.states)}
        queue = list(merged)
        for a, b in queue:
            for pair in before.get(a * self.states + b, ()):
                if pair not in merged:
                    merged.add(pair)
                    queue.append(pair)
        return len(merged) == self.states * (self.states + 1) // 2

    def path(self, tokens: Sequence[int]) -> list[int]:
        """The states after each token of a word from the start."""
        table, state = self._table, self.start
        states: list[int] = []
        for token in tokens:
            state = table[state][token][0]
            states.append(state)
        return states

    def distinguishing_length(self) -> int | None:
        """
        The least k such that every pair of states is told apart by some word of k tokens (Chow 1978's
        k-distinguishability: the outputs differ somewhere along it), or None if some pair never is.
        """
        classes, count = [0] * self.states, 1
        for k in itertools.count(1):
            keys = [
                (
                    classes[s],
                    *((int(self.output[s, t]), classes[int(self.transition[s, t])]) for t in range(self.tokens)),
                )
                for s in range(self.states)
            ]
            index = {key: i for i, key in enumerate(dict.fromkeys(keys))}
            classes = [index[key] for key in keys]
            if len(index) == self.states:
                return k
            if len(index) == count:
                return None
            count = len(index)
        raise AssertionError("unreachable")

    def is_congruence(self, labels: IntArray | Sequence[int]) -> bool:
        """
        Whether the partition of the states by `labels` is respected by every transition: states with
        one label go, on every token, to states with one label. Then the labels' machine is a
        homomorphic image of this one (an approximation, in Egri-Nagy and Nehaniv's terms).
        """
        for t in range(self.tokens):
            image: dict[int, int] = {}
            for s in range(self.states):
                after = int(labels[int(self.transition[s, t])])
                if image.setdefault(int(labels[s]), after) != after:
                    return False
        return True

    def congruence_closure(self, labels: IntArray | Sequence[int]) -> IntArray:
        """
        The finest congruence coarser than the partition by `labels`: states with one label joined,
        then the states two joined states go to on any token, until every transition respects it. Its
        labels number the blocks in order of their first state. What `labels` determines of the
        machine exactly: an approximation (is_congruence) is its own closure.
        """
        parent = list(range(self.states))

        def find(s: int) -> int:
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s

        first: dict[int, int] = {}
        pending = [(s, first.setdefault(int(labels[s]), s)) for s in range(self.states)]
        while pending:
            s, r = pending.pop()
            s, r = find(s), find(r)
            if s != r:
                parent[s] = r
                pending.extend((int(self.transition[s, t]), int(self.transition[r, t])) for t in range(self.tokens))
        blocks: dict[int, int] = {}
        return np.array([blocks.setdefault(find(s), len(blocks)) for s in range(self.states)], dtype=np.int64)

    def principal_congruences(self) -> list[tuple[int, ...]]:
        """
        The machine's distinct nontrivial principal congruences (each pair of states joined, closed:
        congruence_closure), finest first, as labels. Every nontrivial congruence holds one, so none
        means the machine is primitive: no approximation but itself and the one-block machine.
        """
        found: set[tuple[int, ...]] = set()
        for s, r in itertools.combinations(range(self.states), 2):
            labels = np.arange(self.states)
            labels[r] = s
            closure = tuple(int(x) for x in self.congruence_closure(labels))
            if max(closure) > 0:
                found.add(closure)
        return sorted(found, key=lambda labels: (-max(labels), labels))

    def congruences(self) -> list[tuple[int, ...]]:
        """
        Every nontrivial congruence, finest first, as labels: the principal ones and their joins, until
        a join gives none new (each congruence is the join of the principal ones it holds; the join of
        two congruences is one: the states either joins, joined).
        """
        atoms = self.principal_congruences()
        found, frontier = set(atoms), set(atoms)
        while frontier:
            joined = {join for a in frontier for b in atoms if max(join := _joined(a, b)) > 0} - found
            found |= joined
            frontier = joined
        return sorted(found, key=lambda labels: (-max(labels), labels))

    def quotient_monoid(self, labels: Sequence[int]) -> int:
        """The size of the transition monoid of the approximation `labels` names (is_congruence): every map its words make of the blocks."""
        blocks = max(labels) + 1
        maps: list[tuple[int, ...]] = []
        for t in range(self.tokens):
            image = [0] * blocks
            for s in range(self.states):
                image[labels[s]] = labels[int(self.transition[s, t])]
            maps.append(tuple(image))
        seen = {tuple(range(blocks))}
        frontier = list(seen)
        while frontier:
            grown: list[tuple[int, ...]] = []
            for x in frontier:
                for g in maps:
                    composed = tuple(g[i] for i in x)
                    if composed not in seen:
                        seen.add(composed)
                        grown.append(composed)
            frontier = grown
        return len(seen)

    def distinguishing(self, s: int, r: int) -> list[int]:
        """The shortest word (first in token order) whose last label differs from s and from r."""
        assert s != r
        parent: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {(s, r): None}
        queue = [(s, r)]
        for pair in queue:
            for t in range(self.tokens):
                if self.output[pair[0], t] != self.output[pair[1], t]:
                    word = [t]
                    at: tuple[int, int] = pair
                    while (step := parent[at]) is not None:
                        at, token = step
                        word.append(token)
                    return word[::-1]
                following = (int(self.transition[pair[0], t]), int(self.transition[pair[1], t]))
                if following not in parent:
                    parent[following] = (pair, t)
                    queue.append(following)
        raise ValueError(f"states {s} and {r} are equivalent: the machine isn't minimal")


def carry_machine() -> Mealy:
    """Base-3 addition of three operands: states the carry 0..2, a column's token 9 a + 3 b + c."""
    sums = np.array([t // (BASE * BASE) + t // BASE % BASE + t % BASE for t in range(BASE**3)])
    totals = np.arange(3)[:, np.newaxis] + sums[np.newaxis, :]
    return Mealy(totals // BASE, totals % BASE, coordinates=Coordinates(("carry",), np.arange(3)[:, np.newaxis], 3))


@functools.cache
def modular_machine(p: int) -> Mealy:
    """Modular division: state A * p + B (each mod p), a column's token 3 a + b, the output the running quotient."""
    transition = np.zeros((p * p, BASE * BASE), dtype=np.int64)
    output = np.zeros((p * p, BASE * BASE), dtype=np.int64)
    for A in range(p):
        for B in range(p):
            for t in range(BASE * BASE):
                A2, B2 = (BASE * A + t // BASE) % p, (BASE * B + t % BASE) % p
                transition[A * p + B, t] = A2 * p + B2
                output[A * p + B, t] = md.quotient(A2, B2, p)
    coordinates = np.array([[A, B] for A in range(p) for B in range(p)], dtype=np.int64)
    return Mealy(transition, output, coordinates=Coordinates(("A", "B"), coordinates, p))


@functools.cache
def modular_sum_machine(q: int) -> Mealy:
    """Modular sum: state (A + B) mod q, a column's token 3 a + b, the output the running sum."""
    transition = np.zeros((q, BASE * BASE), dtype=np.int64)
    for total in range(q):
        for t in range(BASE * BASE):
            transition[total, t] = (BASE * total + t // BASE + t % BASE) % q
    return Mealy(transition, transition.copy(), coordinates=Coordinates(("total",), np.arange(q)[:, np.newaxis], q))


CARRY = carry_machine()


def orderings(n: int) -> list[tuple[int, ...]]:
    """The n! orderings of items 0..n-1, the identity first."""
    return list(itertools.permutations(range(n)))


def order_labels(coordinates: Coordinates | None) -> dict[str, IntArray]:
    """
    An ordering machine's named partitions (its coordinates each position's item): "order {i, j, ...}"
    the relative order of those items, "first k" the first k positions' items, "places {i, ...}" the
    positions those items hold, "ahead {i}" the set of items ahead of item i, "pairings" (four items) the
    permutation of the three pairings of places, "parity" the ordering's sign; none when the
    coordinates aren't orderings. Which are congruences is the machine's (Mealy.is_congruence):
    move-to-front's every "order", "first" and "ahead", each "order" of two items's quotient a
    flip-flop (an LRU's age matrix); a word problem's every "places" (S_n acting on the placements of
    those items), "pairings" (S4's quotient by the Klein four-group) and "parity" (A_n's cosets).
    """
    if coordinates is None:
        return {}
    values = [tuple(int(x) for x in row) for row in coordinates.values]
    n = len(coordinates.names)
    if any(sorted(row) != list(range(n)) for row in values):
        return {}

    def labelled(keys: list[tuple[int, ...]]) -> IntArray:
        index: dict[tuple[int, ...], int] = {}
        return np.array([index.setdefault(key, len(index)) for key in keys], dtype=np.int64)

    named = {
        f"order {{{', '.join(map(str, items))}}}": labelled([tuple(x for x in row if x in items) for row in values])
        for r in range(2, n)
        for items in itertools.combinations(range(n), r)
    }
    named |= {f"first {k}": labelled([row[:k] for row in values]) for k in range(1, n)}
    named |= {
        f"places {{{', '.join(map(str, items))}}}": labelled([tuple(row.index(x) for x in items) for row in values])
        for r in range(1, n - 1)
        for items in itertools.combinations(range(n), r)
    }
    named |= {f"ahead {{{i}}}": above_labels(coordinates, i) for i in range(n)}
    if n == 4:
        named["pairings"] = pairings_labels(coordinates)
    inversions = [sum(a > b for a, b in itertools.combinations(row, 2)) % 2 for row in values]
    return named | {"parity": np.array(inversions, dtype=np.int64)}


def linear_labels(coordinates: Coordinates | None) -> dict[str, IntArray]:
    """
    A machine's linear forms of its coordinates mod a prime modulus, each once up to a scalar (its
    first coefficient 1): division's "A + B" (the sum), "A - B" (the difference), "A", "B" and the
    rest of its p + 1 lines; none when the modulus isn't prime. Division's are its congruences
    (Mealy.is_congruence) and, with the two trivial partitions, all of them: its columns x -> 3x + v
    are permutations whose group holds every translation (one column after another's inverse), so a
    congruence's block through 0 is a subgroup of Z_p^2, and those are the lines.
    """
    if coordinates is None:
        return {}
    m, names = coordinates.modulus, coordinates.names
    if m < 2 or any(m % d == 0 for d in range(2, math.isqrt(m) + 1)):
        return {}

    def term(c: int, name: str, first: bool) -> str:
        if first:
            return name
        sign, size = ("-", m - c) if c > m // 2 else ("+", c)
        return f"{sign} {name}" if size == 1 else f"{sign} {size}{name}"

    named: dict[str, IntArray] = {}
    for coefficients in itertools.product(range(m), repeat=len(names)):
        nonzero = [c for c in coefficients if c]
        if not nonzero or nonzero[0] != 1:
            continue
        terms = [(c, n) for c, n in zip(coefficients, names, strict=True) if c]
        label = " ".join(term(c, n, j == 0) for j, (c, n) in enumerate(terms))
        named[label] = (coordinates.values @ np.array(coefficients, dtype=np.int64)) % m
    return named


def named_labels(coordinates: Coordinates | None) -> dict[str, IntArray]:
    """A machine's named partitions: an ordering machine's (order_labels), else its coordinates' linear forms (linear_labels)."""
    return order_labels(coordinates) or linear_labels(coordinates)


def _ordering_machine(
    n: int, actions: Sequence[Callable[[tuple[int, ...]], tuple[int, ...]]], label: Callable[[tuple[int, ...]], int]
) -> Mealy:
    """A machine over the orderings of n, a token per action, its output the label of the ordering after it;
    its coordinates each position's item."""
    states = orderings(n)
    index = {s: i for i, s in enumerate(states)}
    transition = np.array([[index[act(s)] for act in actions] for s in states], dtype=np.int64)
    output = np.array([[label(states[int(r)]) for r in row] for row in transition], dtype=np.int64)
    names = tuple(f"position-{j + 1}" for j in range(n))
    return Mealy(transition, output, coordinates=Coordinates(names, np.array(states, dtype=np.int64), n))


def _identity(s: tuple[int, ...]) -> tuple[int, ...]:
    return s


@functools.cache
def word_problem_machine(n: int) -> Mealy:
    """S_n's word problem: the identity, the star transpositions of positions 1 and i, the n-cycle; the output the item at position 1."""

    def swap(i: int) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
        return lambda s: (s[i], *s[1:i], s[0], *s[i + 1 :])

    def rotate(s: tuple[int, ...]) -> tuple[int, ...]:
        return (*s[1:], s[0])

    actions: list[Callable[[tuple[int, ...]], tuple[int, ...]]] = [_identity, *(swap(i) for i in range(1, n)), rotate]
    return _ordering_machine(n, actions, lambda s: s[0])


@functools.cache
def move_to_front_machine(n: int) -> Mealy:
    """An LRU's recency order: token 0 no access, token i moving item i - 1 to the front; the output the last item."""

    def access(x: int) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
        return lambda s: (x, *(y for y in s if y != x))

    return _ordering_machine(n, [_identity, *(access(x) for x in range(n))], lambda s: s[-1])


@functools.cache
def random_machine(n: int, seed: int) -> Mealy:
    """
    n random maps of n! states and the identity (token 0), redrawn until every state is reachable from
    state 0; the output a random label in n classes per state (the label of the state after a
    token), redrawn until the machine is minimal.
    """
    rng, q = Random(f"random-machine-{n}-{seed}"), math.factorial(n)
    while True:
        transition = np.array([[s, *(rng.randrange(q) for _ in range(n))] for s in range(q)], dtype=np.int64)
        if len(Mealy(transition, transition).reachable()) == q:
            break
    while True:
        labels = np.array([rng.randrange(n) for _ in range(q)], dtype=np.int64)
        machine = Mealy(transition, labels[transition])
        if machine.distinguishing_length() is not None:
            return machine


def random_labels(transition: IntArray, classes: int, rng: Random) -> Mealy:
    """A random label in `classes` per state (the label of the state after a token), redrawn until minimal."""
    while True:
        labels = np.array([rng.randrange(classes) for _ in range(len(transition))], dtype=np.int64)
        machine = Mealy(transition, labels[transition])
        if machine.distinguishing_length() is not None:
            return machine


@functools.cache
def random_permutation_machine(n: int, seed: int) -> Mealy:
    """
    random-perm-<n>-<seed>, a probe of B's origins (the code-consolidation workplan): random-n's
    machine with every token a bijection, so no two states merge: n random permutations of n! states
    and the identity, redrawn until every state is reachable; labels as random-n's.
    """
    rng, q = Random(f"random-permutation-machine-{n}-{seed}"), math.factorial(n)
    while True:
        maps = [rng.sample(range(q), q) for _ in range(n)]
        transition = np.array([[s, *(m[s] for m in maps)] for s in range(q)], dtype=np.int64)
        if len(Mealy(transition, transition).reachable()) == q:
            return random_labels(transition, n, rng)


@functools.cache
def random_sized_machine(q: int, seed: int) -> Mealy:
    """
    random-q<q>-<seed>, a probe of B's origins (the code-consolidation workplan): random-4's
    machine at q states, its tokens (the identity and 4 random maps) and 4 classes unchanged, so only
    the size differs; redrawn as random-n's.
    """
    rng = Random(f"random-sized-machine-{q}-{seed}")
    while True:
        transition = np.array([[s, *(rng.randrange(q) for _ in range(4))] for s in range(q)], dtype=np.int64)
        if len(Mealy(transition, transition).reachable()) == q:
            return random_labels(transition, 4, rng)


@functools.cache
def random_planted_machine(q: int, blocks: int, seed: int) -> Mealy:
    """
    random-planted-q<q>-b<b>-<seed>, the paper's T10: random-q<q>'s machine (the identity and 4 random
    maps, 4 classes, redrawn until reachable and minimal) with one approximation planted: the states
    in b equal blocks (state s in block s mod b), each token's target block a random b-state machine's
    (the identity on token 0), the state within it random. So `planted_blocks` is a congruence by
    construction, and only that differs from random-q<q>.
    """
    assert q % blocks == 0 and 1 < blocks < q, f"b divides q and 1 < b < q; got q {q}, b {blocks}"
    rng = Random(f"random-planted-machine-{q}-{blocks}-{seed}")
    while True:
        quotient = [[c, *(rng.randrange(blocks) for _ in range(4))] for c in range(blocks)]
        transition = np.array(
            [
                [s, *(rng.randrange(q // blocks) * blocks + quotient[s % blocks][t] for t in range(1, 5))]
                for s in range(q)
            ],
            dtype=np.int64,
        )
        if len(Mealy(transition, transition).reachable()) == q:
            return random_labels(transition, 4, rng)


def planted_blocks(q: int, blocks: int) -> IntArray:
    """random_planted_machine's planted approximation: state s in block s mod b."""
    return np.arange(q, dtype=np.int64) % blocks


def canonical(labels: IntArray | Sequence[int]) -> tuple[int, ...]:
    """A partition's labels renumbered by first state, as congruence_closure numbers its blocks."""
    blocks: dict[int, int] = {}
    return tuple(blocks.setdefault(int(label), len(blocks)) for label in labels)


def _joined(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """The join of two partitions: states either joins, joined (connected); numbered by first state."""
    parent = list(range(len(a)))

    def find(s: int) -> int:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for labels in (a, b):
        first: dict[int, int] = {}
        for s, label in enumerate(labels):
            parent[find(s)] = find(first.setdefault(label, s))
    blocks: dict[int, int] = {}
    return tuple(blocks.setdefault(find(s), len(blocks)) for s in range(len(a)))


def coarsest_cover(machine: Mealy) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    The paper workplan's D20 rule (T11; the protocol's, experiment-design.md section 1): two
    approximations that together determine the state, the coarsest such pair: among every pair of
    congruences (Mealy.congruences) whose meet is the identity, the fewest blocks in total; ties to
    the simplest quotients (Mealy.quotient_monoid, summed), then the first in the machine's state
    order. The coarser first. None for a machine with no such pair (a primitive one).
    """
    found = machine.congruences()
    best: tuple[tuple[int, int, tuple[int, ...], tuple[int, ...]], ...] = ()
    for a, b in itertools.combinations(sorted(found, key=lambda labels: (max(labels), labels)), 2):
        if len(set(zip(a, b, strict=True))) == machine.states:
            best += ((max(a) + max(b) + 2, machine.quotient_monoid(a) + machine.quotient_monoid(b), a, b),)
    assert best, "no pair of approximations determines the state: a primitive machine has no cover to give"
    _, _, a, b = min(best)
    return a, b


def pairings_labels(coordinates: Coordinates) -> IntArray:
    """
    An ordering of four items by the permutation it makes of the three pairings of its places
    ({1, 2 | 3, 4} and the other two): S4's quotient by the Klein four-group, which is S3.
    """
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]

    def pairing(places: tuple[int, int]) -> int:
        return next(i for i, p in enumerate(pairings) if tuple(sorted(places)) in p)

    index: dict[tuple[int, ...], int] = {}
    keys = [
        tuple(pairing((row.index(a), row.index(b))) for (a, b), _ in pairings) for row in coordinates.values.tolist()
    ]
    return np.array([index.setdefault(key, len(index)) for key in keys], dtype=np.int64)


def above_labels(coordinates: Coordinates, item: int) -> IntArray:
    """An ordering by the set of items before `item` (more recent, in move-to-front's recency order)."""
    index: dict[tuple[int, ...], int] = {}
    keys = [tuple(sorted(row[: row.index(item)])) for row in coordinates.values.tolist()]
    return np.array([index.setdefault(key, len(index)) for key in keys], dtype=np.int64)


def _cover(name: str) -> tuple[IntArray, IntArray]:
    """D20's registered pair for a T11 machine (coarsest_cover's, named), the coarser first."""
    coordinates = task_named(name).machine.coordinates
    assert coordinates is not None
    named = order_labels(coordinates)
    return {
        "word-problem-s3": lambda: (named["parity"], named["places {0}"]),  # the sign, a point's place
        "word-problem-s4": lambda: (named["places {0}"], pairings_labels(coordinates)),  # a point's place, S3
        "move-to-front-4": lambda: (named["order {0, 1, 3}"], above_labels(coordinates, 2)),
    }[name]()


COVERED = ("word-problem-s3", "word-problem-s4", "move-to-front-4")  # T11's machines (D20)


def query_partitions(name: str) -> dict[str, IntArray]:
    """
    A task's named approximations an encoding can ask as extra outputs, each its states' labels (a
    congruence): a planted machine's "planted" (planted_blocks; T10's P2, the paper workplan's D19);
    T11's machines' "cover-a" and "cover-b" (their coarsest cover, the coarser first; D20).
    """
    if name in COVERED:
        return dict(zip(("cover-a", "cover-b"), _cover(name), strict=True))
    if found := re.fullmatch(r"random-planted-q(\d+)-b(\d+)-\d+", name):
        return {"planted": planted_blocks(int(found[1]), int(found[2]))}
    return {}


def planted_labels(name: str) -> dict[str, IntArray]:
    """
    A planted machine's planted approximation, named for the decomposition reading (the paper
    workplan's T4', D27): {"planted": planted_blocks}; empty for any other task. T11's machines' covers
    aren't added: their order labels already name them (T11c's registered reading, D25, unchanged).
    """
    known = query_partitions(name)
    return {"planted": known["planted"]} if "planted" in known else {}


Word = tuple[int, ...]


@dataclass(frozen=True)
class WordProperty:
    """A machine task's property: its cases, words of n tokens, each with its stratum (or None)."""

    id: str
    statement: str
    generate: Callable[[int, Random], list[tuple[Word, int | None]]]


@dataclass
class WordResult:
    id: str
    cases: int
    passed: int
    strata: dict[int, tuple[int, int]] = field(default_factory=dict[int, tuple[int, int]])
    failures: list[Word] = field(default_factory=list[Word])  # the first ad.FAILURES_KEPT

    @property
    def holds(self) -> bool:
        return self.passed == self.cases


class Outcome(Protocol):
    """A property's result, as the harness reads it."""

    id: str
    cases: int
    passed: int

    @property
    def holds(self) -> bool: ...


class Task(Protocol):
    """What the harness and the certificate need of a task (the module's docstring)."""

    @property
    def name(self) -> str: ...

    @property
    def machine(self) -> Mealy: ...

    @property
    def held_out_id(self) -> str:
        """The held-out tenth's property: checked from the harness's HOLD_OUT_FROM, never tested."""
        ...

    @property
    def codes_id(self) -> str:
        """The property whose first cases count the codes used."""
        ...

    @property
    def reach_id(self) -> str | None:
        """The stratified property whose L* the test pass records (None: the task has none)."""
        ...

    def column_tokens(self, case: Any, n: int) -> tuple[int, ...]: ...

    def training_case(self, kind: int, n: int, rng: Random, hold_out: bool) -> Any: ...

    def hash(self, case: Any, n: int) -> int: ...

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[Any, ...]: ...

    def property_cases(self, prop: Any, n: int) -> list[Any]: ...

    def evaluate(self, predict: Predict, n: int, properties: Sequence[Any], limit: int | None = None) -> list[Any]: ...

    def reach(self, result: Any) -> int: ...

    def result_json(self, result: Any) -> dict[str, Any]: ...

    def case_json(self, columns: Sequence[int]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Addition:
    """Three-operand base-3 addition, least significant first (the matrix's task)."""

    name: str = "addition"
    machine: Mealy = CARRY
    held_out_id: str = "B3"
    codes_id: str = "B1"
    reach_id: str = "M3"

    def column_tokens(self, case: ad.Triple, n: int) -> tuple[int, ...]:
        a, b, c = (ad.digits(x, n) for x in case)
        return tuple(BASE * BASE * a[t] + BASE * b[t] + c[t] for t in range(n))

    def training_case(self, kind: int, n: int, rng: Random, hold_out: bool) -> ad.Triple:
        return ad.training_triple(kind, n, rng, hold_out=hold_out)

    def hash(self, case: ad.Triple, n: int) -> int:
        return ad.triple_hash(case, n)

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[ad.Property, ...]:
        return ad.catalogue(sampled, relation_sampled, per_stratum, doubling)

    def property_cases(self, prop: ad.Property, n: int) -> list[ad.Triple]:
        return [case.triple for case in ad.property_cases(prop, n)]

    def evaluate(
        self, predict: Predict, n: int, properties: Sequence[ad.Property], limit: int | None = None
    ) -> list[ad.PropertyResult]:
        def add(triples: Sequence[ad.Triple]) -> list[int]:
            return [ad.from_digits(labels) for labels in predict(triples)]

        return ad.evaluate(add, n, properties, limit=limit, shrink=False)

    def reach(self, result: ad.PropertyResult) -> int:
        return result.reach()

    def result_json(self, result: ad.PropertyResult) -> dict[str, Any]:
        return {
            "cases": result.cases,
            "passed": result.passed,
            "strata": {str(k): list(v) for k, v in sorted(result.strata.items())},
            "failures": [{"triple": list(c.triple), "column": c.column} for c in result.failures],
        }

    def case_json(self, columns: Sequence[int]) -> dict[str, Any]:
        """The columns as an addition at alignment 0, least significant first: its triple and n."""
        operands = [0, 0, 0]
        for t, token in enumerate(columns):
            for i, digit in enumerate((token // (BASE * BASE), token // BASE % BASE, token % BASE)):
                operands[i] += digit * BASE**t
        return {"triple": operands, "n": len(columns)}


@dataclass(frozen=True)
class ModularDivision:
    """(a / b) mod p of two base-3 numbers, most significant first, the running quotient at every column."""

    p: int

    @property
    def name(self) -> str:
        return f"modular-division-{self.p}"

    @property
    def machine(self) -> Mealy:
        return modular_machine(self.p)

    held_out_id: str = "U2"
    codes_id: str = "U1"
    reach_id: str | None = "S2"

    def column_tokens(self, case: md.Pair, n: int) -> tuple[int, ...]:
        return md.column_tokens(case, n)

    def training_case(self, kind: int, n: int, rng: Random, hold_out: bool) -> md.Pair:
        return md.training_pair(kind, n, rng, hold_out=hold_out)

    def hash(self, case: md.Pair, n: int) -> int:
        return md.pair_hash(case, n)

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[md.Property, ...]:
        return md.catalogue(self.p, sampled, per_stratum, doubling)

    def property_cases(self, prop: md.Property, n: int) -> list[md.Pair]:
        return [case.pair for case in md.property_cases(prop, n)]

    def evaluate(
        self, predict: Predict, n: int, properties: Sequence[md.Property], limit: int | None = None
    ) -> list[md.PropertyResult]:
        def divide(pairs: Sequence[md.Pair]) -> list[md.Answers]:
            return [labels[:n] for labels in predict(pairs)]

        return md.evaluate(divide, n, properties, limit=limit)

    def reach(self, result: md.PropertyResult) -> int:
        return result.reach()

    def result_json(self, result: md.PropertyResult) -> dict[str, Any]:
        return {
            "cases": result.cases,
            "passed": result.passed,
            "strata": {str(k): list(v) for k, v in sorted(result.strata.items())},
            "failures": [{"pair": list(c.pair)} for c in result.failures],
        }

    def case_json(self, columns: Sequence[int]) -> dict[str, Any]:
        """The columns as a pair at alignment 0, most significant first: the pair and n."""
        a = b = 0
        for token in columns:
            a, b = BASE * a + token // BASE, BASE * b + token % BASE
        return {"pair": [a, b], "n": len(columns)}


@dataclass(frozen=True)
class ModularSum(ModularDivision):
    """(a + b) mod q of two base-3 numbers, most significant first, the running sum at every column (E8)."""

    reach_id: str | None = None  # no undefined class, so no S2

    @property
    def name(self) -> str:
        return f"modular-sum-{self.p}"

    @property
    def machine(self) -> Mealy:
        return modular_sum_machine(self.p)

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[md.Property, ...]:
        return md.sum_catalogue(self.p, sampled, per_stratum)


@dataclass(frozen=True)
class MachineTask:
    """
    Any machine's own task (the general algebras, D6): a case is a word of its tokens, its labels the
    machine's outputs. U1 is uniform words; S1 the shortest word to each reachable state (its
    stratum) then uniform tokens to n, so every state's behaviour is tested whatever the words
    visit. No held-out tenth: the probe draws every word.
    """

    name: str
    machine: Mealy
    held_out_id: str = "U2"  # in no catalogue
    codes_id: str = "U1"
    reach_id: str | None = None

    def column_tokens(self, case: Word, n: int) -> tuple[int, ...]:
        return case

    def training_case(self, kind: int, n: int, rng: Random, hold_out: bool) -> Word:
        return tuple(rng.randrange(self.machine.tokens) for _ in range(n))

    def hash(self, case: Word, n: int) -> int:
        return int.from_bytes(hashlib.sha256(bytes(case) + n.to_bytes(4)).digest()[:8])

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[WordProperty, ...]:
        tokens = self.machine.tokens

        def uniform(n: int, rng: Random) -> list[tuple[Word, int | None]]:
            return [(tuple(rng.randrange(tokens) for _ in range(n)), None) for _ in range(sampled)]

        def every_state(n: int, rng: Random) -> list[tuple[Word, int | None]]:
            cases: list[tuple[Word, int | None]] = []
            for state, prefix in self._prefixes.items():
                if len(prefix) <= n:
                    for _ in range(per_stratum):
                        cases.append(((*prefix, *(rng.randrange(tokens) for _ in range(n - len(prefix)))), state))
            return cases

        return (
            WordProperty("U1", f"every token's answer, {sampled:,} uniform words", uniform),
            WordProperty("S1", f"from every reachable state, {per_stratum:,} uniform words each", every_state),
        )

    @functools.cached_property
    def _prefixes(self) -> dict[int, Word]:
        """The shortest word (first in token order) to each reachable state."""
        prefixes: dict[int, Word] = {self.machine.start: ()}
        for s in self.machine.reachable():
            for t in range(self.machine.tokens):
                prefixes.setdefault(int(self.machine.transition[s, t]), (*prefixes[s], t))
        return prefixes

    def property_cases(self, prop: WordProperty, n: int) -> list[Word]:
        return [word for word, _ in self._cases(prop, n)]

    def _cases(self, prop: WordProperty, n: int) -> list[tuple[Word, int | None]]:
        return prop.generate(n, Random(f"{self.name}-{prop.id}-{n}"))

    def evaluate(
        self, predict: Predict, n: int, properties: Sequence[WordProperty], limit: int | None = None
    ) -> list[WordResult]:
        results: list[WordResult] = []
        for prop in properties:
            cases = self._cases(prop, n)
            if limit is not None and len(cases) > limit:
                cases = cases[:: -(-len(cases) // limit)]
            got = predict([word for word, _ in cases])
            result = WordResult(prop.id, len(cases), 0)
            for (word, stratum), labels in zip(cases, got, strict=True):
                ok = labels[:n] == self.machine.run(word)
                result.passed += ok
                if stratum is not None:
                    passed, count = result.strata.get(stratum, (0, 0))
                    result.strata[stratum] = (passed + ok, count + 1)
                if not ok and len(result.failures) < ad.FAILURES_KEPT:
                    result.failures.append(word)
            results.append(result)
        return results

    def reach(self, result: WordResult) -> int:
        return 0

    def result_json(self, result: WordResult) -> dict[str, Any]:
        return {
            "cases": result.cases,
            "passed": result.passed,
            "strata": {str(k): list(v) for k, v in sorted(result.strata.items())},
            "failures": [{"word": list(word)} for word in result.failures],
        }

    def case_json(self, columns: Sequence[int]) -> dict[str, Any]:
        return {"word": [int(t) for t in columns], "n": len(columns)}


ADDITION = Addition()


MACHINES: dict[str, Callable[..., Mealy]] = {
    r"word-problem-s(\d+)": word_problem_machine,
    r"move-to-front-(\d+)": move_to_front_machine,
    r"random-(\d+)-(\d+)": random_machine,
    r"random-perm-(\d+)-(\d+)": random_permutation_machine,
    r"random-q(\d+)-(\d+)": random_sized_machine,
    r"random-planted-q(\d+)-b(\d+)-(\d+)": random_planted_machine,
}


def task_named(name: str) -> Task:
    """addition, modular-division-p for p in md.PRIMES, modular-sum-q for q >= 2, or a general algebra's machine."""
    if name == ADDITION.name:
        return ADDITION
    for pattern, machine in MACHINES.items():
        if found := re.fullmatch(pattern, name):
            assert int(found[1]) >= 2, f"a machine's n is at least 2; got {found[1]}"
            return MachineTask(name, machine(*(int(x) for x in found.groups())))
    if found := re.fullmatch(r"modular-sum-(\d+)", name):
        assert int(found[1]) >= 2, f"a modular sum's q is at least 2; got {found[1]}"
        return ModularSum(int(found[1]))
    found = re.fullmatch(r"modular-division-(\d+)", name)
    assert found is not None and int(found[1]) in md.PRIMES, (
        f"a task is addition, modular-division-p (p in {md.PRIMES}) or modular-sum-q"
    )
    return ModularDivision(int(found[1]))
