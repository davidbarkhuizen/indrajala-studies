"""
Chow's test set (Chow 1978, the W-method): an exact, answers-only check of a learned machine against
a black box, under a state bound (machine-discovery.md's check 4; the state-formation workplan's
D42, stage B's first tool; the programme audit's F32).

A minimal Mealy machine M of n states is equivalent to a black box of at most m states over the same
tokens, both deterministic and complete, **if and only if** the two give the same outputs on every
word of the test set P · Z: P a transition cover of M (the empty word, each state's access word,
and each access word followed by every token), Z = W ∪ X W ∪ ... ∪ X^(m - n) W, X the tokens and W
a characterization set of M (words that tell every pair of its states apart). The set comes from M
alone; only its outputs need the black box, so with query access and a state bound a learned
machine is proved equal to an unknown target. Its size grows as |X|^(m - n): the bound must be
tight.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from indrajala_ml.studies import window_tasks as wt

Word = tuple[int, ...]
# a batch of words to each one's labels, a label a token (the black box's queries)
Query = Callable[[Sequence[Word]], list[wt.Labels]]


def minimised(machine: wt.Mealy) -> wt.Mealy:
    """
    The machine's reachable part, minimised by Moore's refinement (states equivalent when they give
    the same labels and move to equivalent states on every token), its states numbered by their
    first reach breadth first, so the start is 0.
    """
    reachable = machine.reachable()
    index = {s: i for i, s in enumerate(reachable)}
    transition = np.array([[index[int(r)] for r in machine.transition[s]] for s in reachable], dtype=np.int64)
    output = machine.output[reachable]
    block = [0] * len(reachable)
    count = 0
    while True:
        keys = [
            (block[s], *((int(output[s, t]), block[int(transition[s, t])]) for t in range(machine.tokens)))
            for s in range(len(reachable))
        ]
        ids = {key: i for i, key in enumerate(dict.fromkeys(keys))}
        block = [ids[key] for key in keys]
        if len(ids) == count:
            break
        count = len(ids)
    # number the blocks by their first state's reach (the start's block first)
    order = {b: i for i, b in enumerate(dict.fromkeys(block))}
    first = {order[block[s]]: s for s in reversed(range(len(reachable)))}
    return wt.Mealy(
        np.array([[order[block[int(transition[first[b], t])]] for t in range(machine.tokens)] for b in range(count)]),
        np.array([[int(output[first[b], t]) for t in range(machine.tokens)] for b in range(count)]),
    )


def access_words(machine: wt.Mealy) -> dict[int, Word]:
    """Each reachable state's shortest word from the start, first in token order (breadth first)."""
    words: dict[int, Word] = {machine.start: ()}
    queue = [machine.start]
    for s in queue:
        for t in range(machine.tokens):
            r = int(machine.transition[s, t])
            if r not in words:
                words[r] = (*words[s], t)
                queue.append(r)
    return words


def transition_cover(machine: wt.Mealy) -> list[Word]:
    """P: the empty word, each state's access word and each followed by every token, without repeats."""
    access = access_words(machine)
    words = [access[s] for s in sorted(access, key=lambda s: (len(access[s]), access[s]))]
    words += [
        (*access[s], t)
        for s in sorted(access, key=lambda s: (len(access[s]), access[s]))
        for t in range(machine.tokens)
    ]
    return list(dict.fromkeys(words))


def characterisation_set(machine: wt.Mealy) -> list[Word]:
    """
    W for a minimal machine: for every pair of states, the shortest word whose labels tell them apart
    (Mealy.distinguishing), without repeats, shortest first. Its one-word {()} for a one-state machine,
    which nothing needs telling apart from (Chow's W is then empty; the empty word keeps P · Z = P).
    """
    pairs = itertools.combinations(range(machine.states), 2)
    words = {tuple(machine.distinguishing(s, r)) for s, r in pairs}
    return sorted(words, key=lambda w: (len(w), w)) if words else [()]


def test_set_size(machine: wt.Mealy, bound: int) -> int:
    """|P| · |W| · (1 + |X| + ... + |X|^(bound - n)): the words test_set yields, before repeats are dropped."""
    extra = bound - machine.states
    assert extra >= 0, "the bound below the machine's states"
    middles = sum(machine.tokens**i for i in range(extra + 1))
    return len(transition_cover(machine)) * len(characterisation_set(machine)) * middles


def test_set(machine: wt.Mealy, bound: int) -> Iterator[Word]:
    """
    P · Z for a minimal `machine` and a black box of at most `bound` states (Chow 1978), each word
    once; shorter middles first, so a counterexample tends to come early.
    """
    extra = bound - machine.states
    assert extra >= 0, "the bound below the machine's states"
    cover, characterising = transition_cover(machine), characterisation_set(machine)
    seen: set[Word] = set()
    for i in range(extra + 1):
        for middle in itertools.product(range(machine.tokens), repeat=i):
            for prefix in cover:
                for suffix in characterising:
                    word = (*prefix, *middle, *suffix)
                    if word not in seen:
                        seen.add(word)
                        yield word


@dataclass(frozen=True)
class Check:
    """A check's verdict: equivalent under the bound, the words queried, and the first disagreement."""

    equivalent: bool
    states: int  # the minimised machine's
    bound: int
    words: int  # queried
    counterexample: Word | None


def check(machine: wt.Mealy, query: Query, bound: int, batch: int = 4096, limit: int | None = None) -> Check:
    """
    The minimised `machine` against a black box answering `query`, under a state bound (Chow's
    theorem: equivalent iff every test word agrees, for a black box of at most `bound` states). A
    machine larger than the bound fails at once (no query). `limit` caps the words queried: a check
    that hits it raises rather than claiming equivalence.
    """
    m = minimised(machine)
    if m.states > bound:
        return Check(False, m.states, bound, 0, None)
    if limit is not None and test_set_size(m, bound) > limit:
        raise ValueError(f"the test set's {test_set_size(m, bound):,} words exceed the limit {limit:,}")
    queried = 0
    words = test_set(m, bound)
    while chunk := list(itertools.islice(words, batch)):
        answers = query(chunk)
        queried += len(chunk)
        for word, labels in zip(chunk, answers, strict=True):
            if m.run(word) != tuple(labels):
                return Check(False, m.states, bound, queried, word)
    return Check(True, m.states, bound, queried, None)


def machine_query(target: wt.Mealy) -> Query:
    """A known machine as the black box (offline: the target answers the queries discovery would make)."""
    return lambda words: [target.run(word) for word in words]
