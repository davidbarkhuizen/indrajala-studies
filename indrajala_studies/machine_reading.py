"""
Reading a snapped network's machine against its target, beyond the certificate's verdict (the
code-consolidation workplan's stage 2, the D9 probes): `by_length`, how exact the network is on
every word of each length; `breaks`, which of its transitions break, from codes that are right.

The network's machine is walked as the certificate walks it, one column a window (k 1), but on its
snapped states only (a near tie is taken on its nearest code), together with the target's: the
pairs (code, state) the start reaches. A code is **right** for a state when the network's outputs
from it agree with the target's from the state on every word of up to `d` tokens, `d` the target's
distinguishing length plus one (the workplan's D2): with right outputs, enough to tell every pair
of the target's states apart, with a token's margin. A state's **first code** is the one its
breadth-first shortest word reaches; its other codes are **extra**.

`groups` reads the network alone, with no target (the workplan's A, D1): its reachable codes grouped
by `d`-equivalence (Moore's `k`-equivalence: no word of `d` tokens or fewer tells two apart by the
network's outputs), each group's representative the code its shortest word reaches; `consolidation`
is the table that pulls each duplicate to its representative.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_tasks as wt

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Walk:
    """
    The network's machine alone: its codes the start reaches, breadth first (code 0 the start's, so
    each code's index orders it by its shortest word), each code's next code and outputs per token.
    """

    codes: list[wc.State]
    next: IntArray  # (C, T) the code after each token
    outputs: IntArray  # (C, T, S) the outputs after each token, a label per slot the machine reads


def walk(machine: wc.WindowMachine, tokens: int) -> Walk:
    """The network's machine walked from its start over every token (k 1, snapped states only)."""
    assert machine.columns == 1, "one column a window"
    windows = wc.all_windows(1, tokens)
    start = tuple(int(x) for x in machine.start()[0])
    index: dict[wc.State, int] = {start: 0}
    codes = [start]
    nexts: list[list[int]] = []
    outputs: list[IntArray] = []
    for code in codes:  # breadth first: codes grows as it's walked
        st = machine.step(code, windows)
        following: list[int] = []
        for row in st.states:
            reached = tuple(int(x) for x in row)
            if reached not in index:
                index[reached] = len(codes)
                codes.append(reached)
            following.append(index[reached])
        nexts.append(following)
        outputs.append(np.asarray(st.digits, dtype=np.int64))
    return Walk(codes, np.array(nexts, dtype=np.int64), np.stack(outputs))


def mealy(walk: Walk, slot: int = 0) -> wt.Mealy:
    """The walk as a Mealy machine (its codes the states, code 0 the start), its labels one slot's (the first, as product reads)."""
    return wt.Mealy(walk.next, walk.outputs[:, :, slot])


def groups(walk: Walk, d: int) -> list[list[int]]:
    """
    The walk's codes grouped by d-equivalence (Moore refinement: two codes equivalent to depth k
    when their outputs agree and their next codes are equivalent to depth k - 1), each group's codes
    in order of reach, so its first the representative; groups ordered by their representatives.
    """
    count = len(walk.codes)
    outputs = walk.outputs.reshape(count, -1)
    block = np.zeros(count, dtype=np.int64)
    for _ in range(d):
        _, inverse = np.unique(np.concatenate([outputs, block[walk.next]], axis=1), axis=0, return_inverse=True)
        block = inverse.reshape(-1)
    members: dict[int, list[int]] = {}
    for c in range(count):
        members.setdefault(int(block[c]), []).append(c)
    return sorted(members.values())


def consolidation(walk: Walk, found: list[list[int]]) -> dict[wc.State, wc.State]:
    """Each group's duplicates' codes to its representative's (the consolidation network.consolidate takes)."""
    return {walk.codes[c]: walk.codes[group[0]] for group in found for c in group[1:]}


@dataclass(frozen=True)
class Product:
    """
    The pairs (network code, target state) the start reaches, breadth first, pair 0 the start's:
    each pair's next pair and whether its output is right, per token; its depth (the shortest
    word's length); and the codes in order of reach.
    """

    codes: list[wc.State]  # the network's reachable codes, breadth first
    pairs: list[tuple[int, int]]  # (index into codes, target state)
    next: IntArray  # (P, T) the pair after each token
    right_output: BoolArray  # (P, T)
    depth: IntArray  # (P,)

    @property
    def states(self) -> IntArray:
        return np.array([s for _, s in self.pairs], dtype=np.int64)


def product(machine: wc.WindowMachine, target: wt.Mealy) -> Product:
    """The network's machine walked with the target's from both starts (k 1, snapped states only; its first slot)."""
    network = walk(machine, target.tokens)
    pair_index: dict[tuple[int, int], int] = {(0, target.start): 0}
    pairs: list[tuple[int, int]] = [(0, target.start)]
    depth: list[int] = [0]
    nexts: list[list[int]] = []
    right: list[list[bool]] = []
    for p, (c, s) in enumerate(pairs):  # breadth first: pairs grows as it's walked
        following, outputs = network.next[c], network.outputs[c, :, 0]
        row_next: list[int] = []
        row_right: list[bool] = []
        for t in range(target.tokens):
            pair = (int(following[t]), int(target.transition[s, t]))
            if pair not in pair_index:
                pair_index[pair] = len(pairs)
                pairs.append(pair)
                depth.append(depth[p] + 1)
            row_next.append(pair_index[pair])
            row_right.append(int(outputs[t]) == int(target.output[s, t]))
        nexts.append(row_next)
        right.append(row_right)
    return Product(
        network.codes,
        pairs,
        np.array(nexts, dtype=np.int64),
        np.array(right, dtype=np.bool_),
        np.array(depth, dtype=np.int64),
    )


def word_length(target: wt.Mealy) -> int:
    """`d`: the target's distinguishing length plus one (the workplan's D2)."""
    length = target.distinguishing_length()
    assert length is not None, "a minimal target"
    return length + 1


def right_codes(walk: Product, d: int) -> BoolArray:
    """Each pair's code right for its state: every output agrees on every word of up to d tokens."""
    right = np.ones(len(walk.pairs), dtype=np.bool_)
    for _ in range(d):
        right = (walk.right_output & right[walk.next]).all(axis=1)
    return right


def summary(walk: Product, d: int) -> dict[str, int]:
    """
    The walk in three counts (the one-code-one-state workplan's R2): the codes reached, the pairs
    (code, state) reached and the pairs whose code is right within d. Pairs over codes is how many
    target states a code stands for on average: 1.0 in every certified network.
    """
    return {"codes": len(walk.codes), "pairs": len(walk.pairs), "right": int(right_codes(walk, d).sum())}


def counterexamples(walk: Product, d: int, limit: int) -> list[list[int]]:
    """
    The shortest word (tokens) to each pair whose code isn't right for its state within d (the
    one-code-one-state workplan's R4: every code standing for several states is wrong for at least
    one of them, and a code with wrong outputs is wrong), shortest first, at most limit of them. Each
    pair's word is the one product found it by: its parent the first (pair, token) reaching it, in
    the walk's breadth-first order.
    """
    parent: dict[int, tuple[int, int]] = {}
    for p in range(len(walk.pairs)):
        for t, q in enumerate(walk.next[p].tolist()):
            if q != 0 and q not in parent:
                parent[q] = (p, t)
    right = right_codes(walk, d)
    words: list[list[int]] = []
    for q in range(len(walk.pairs)):  # breadth first, so the shortest words first
        if right[q]:
            continue
        word: list[int] = []
        at = q
        while at != 0:
            at, token = parent[at]
            word.append(token)
        words.append(word[::-1])
        if len(words) == limit:
            break
    return words


def by_length(walk: Product, lengths: list[int], d: int) -> list[dict[str, Any]]:
    """
    Over all words of each length (uniform tokens, the start's pair): the share whose last answer is
    right, the share right at every answer (exact), and the share ending on a code right for its
    state; each from the mass propagated over the pairs, so exact, not sampled.
    """
    right = right_codes(walk, d)
    tokens = walk.next.shape[1]
    mass = np.zeros(len(walk.pairs))
    mass[0] = 1.0
    exact = mass.copy()  # the mass of words right at every answer so far
    rows: list[dict[str, Any]] = []
    for n in range(1, max(lengths) + 1):
        last = float((mass[:, np.newaxis] * walk.right_output).sum() / tokens)
        after, after_exact = np.zeros_like(mass), np.zeros_like(exact)
        for t in range(tokens):
            np.add.at(after, walk.next[:, t], mass / tokens)
            np.add.at(after_exact, walk.next[:, t], exact * walk.right_output[:, t] / tokens)
        mass, exact = after, after_exact
        if n in lengths:
            rows.append(
                {
                    "n": n,
                    "last": round(last, 4),
                    "exact": round(float(exact.sum()), 4),
                    "named": round(float(mass[right].sum()), 4),
                }
            )
    return rows


def breaks(walk: Product, d: int) -> dict[str, Any]:
    """
    Each transition that lands on a code wrong for the next state, split by the code it leaves (each
    state's first code, or an extra one), over every pair and over the pairs whose code is right (a
    break from a right code is the fault itself, not one inherited), and by the pair's depth.
    """
    right = right_codes(walk, d)
    first: dict[int, int] = {}  # state -> its first code
    for c, s in walk.pairs:  # breadth first, so the first seen is the shortest word's
        first.setdefault(s, c)
    tokens = walk.next.shape[1]
    tallies: dict[str, list[int]] = {k: [0, 0] for k in ("first", "extra", "first_right", "extra_right")}
    depths: dict[int, list[int]] = {}
    for p, (c, s) in enumerate(walk.pairs):
        broken = int((~right[walk.next[p]]).sum())
        kind = "first" if first[s] == c else "extra"
        for key in (kind, f"{kind}_right") if right[p] else (kind,):
            tallies[key][0] += broken
            tallies[key][1] += tokens
        if right[p]:
            at = depths.setdefault(int(walk.depth[p]), [0, 0])
            at[0] += broken
            at[1] += tokens
    states = set(first)
    return {
        "d": d,
        "codes": len(walk.codes),
        "pairs": len(walk.pairs),
        "states": len(states),
        "first_codes": len({first[s] for s in states}),
        "right_pairs": int(right.sum()),
        **{k: {"broken": v[0], "transitions": v[1]} for k, v in tallies.items()},
        "depth_right": {str(k): {"broken": v[0], "transitions": v[1]} for k, v in sorted(depths.items())},
    }


def _refines(fine: IntArray, coarse: IntArray, states: list[int]) -> bool:
    """Whether every block of `fine` over the states sits inside one block of `coarse`."""
    image: dict[int, int] = {}
    return all(image.setdefault(int(fine[s]), int(coarse[s])) == int(coarse[s]) for s in states)


def approximations(walk: Product, target: wt.Mealy, extra: Mapping[str, IntArray] | None = None) -> dict[str, Any]:
    """
    What each factor's code determines of the target exactly, over the pairs the start reaches: the
    reached states joined when one code of the factor holds both (the certificate's readout), whether
    that partition is itself a congruence, and its closure (Mealy.congruence_closure), the finest
    approximation (a homomorphic image, Egri-Nagy and Nehaniv) the factor's code determines; the
    target's named congruences that closure refines, the finest only (window_tasks.named_labels, and
    `extra`'s, a task's further names: window_tasks.planted_labels); and
    the share of the factor's (code, token) pairs whose next code is one, 1.0 when the factor's code
    is a machine of its own. Jointly, whether the factors' approximations cover the target: their
    meet separates every reached state (then the target is a subdirect product of them, a parallel
    cover). And the factors on top of a cascade: each factor whose approximation is nontrivial, while
    the joint codes determine the state, so the target is that approximation with the rest's codes
    below it, reading it (the series form, Krohn and Rhodes; a
    nontrivial congruence is what a serial decomposition needs, Hartmanis and Stearns). A cascade's
    lower factor alone is no congruence, so its closure, and the parallel meet, miss it.
    """
    states = sorted({s for _, s in walk.pairs})
    named = {
        k: v for k, v in (wt.named_labels(target.coordinates) | dict(extra or {})).items() if target.is_congruence(v)
    }
    closures: list[IntArray] = []
    factors: list[dict[str, Any]] = []
    for g in range(len(walk.codes[0])):
        holders: dict[int, int] = {}  # the factor's code -> a state it holds, joined below
        labels = np.arange(target.states, dtype=np.int64)
        for c, s in walk.pairs:
            labels[labels == labels[s]] = labels[holders.setdefault(walk.codes[c][g], s)]
        codes_of: dict[int, set[int]] = {}
        following: dict[tuple[int, int], set[int]] = {}
        for p, (c, s) in enumerate(walk.pairs):
            codes_of.setdefault(s, set()).add(walk.codes[c][g])
            for t in range(target.tokens):
                after = walk.codes[walk.pairs[int(walk.next[p, t])][0]][g]
                following.setdefault((walk.codes[c][g], t), set()).add(after)
        closure = target.congruence_closure(labels)
        closures.append(closure)
        determined = [k for k, v in named.items() if _refines(closure, v, states)]
        finest = [
            k
            for k in determined
            if not any(
                j != k and _refines(named[j], named[k], states) and not _refines(named[k], named[j], states)
                for j in determined
            )
        ]
        factors.append(
            {
                "codes": len(holders),
                "function": all(len(c) == 1 for c in codes_of.values()),
                "blocks": len({int(labels[s]) for s in states}),
                "congruence": target.is_congruence(labels),
                "approximation": sorted(np.unique(closure[states], return_counts=True)[1].tolist(), reverse=True),
                "determines": finest,
                "autonomous": sum(len(n) == 1 for n in following.values()) / len(following),
            }
        )
    meet = {tuple(int(c[s]) for c in closures) for s in states}
    held: dict[int, set[int]] = {}  # each joint code -> the states it holds
    for c, s in walk.pairs:
        held.setdefault(c, set()).add(s)
    determined = all(len(h) == 1 for h in held.values())
    top = [g for g, f in enumerate(factors) if determined and 1 < len(f["approximation"]) < len(states)]
    return {
        "states": len(states),
        "factors": factors,
        "meet": len(meet),
        "cover": len(meet) == len(states),
        "cascade": top,
    }
