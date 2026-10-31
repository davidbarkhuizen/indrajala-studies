"""
The modular family (the modular-family workplan): one window-recurrent network answering
several modular operations on the same columns at every window, so a factor of its state gets a
dense signal for a residue (the sum's and difference's counters) where division alone gives only
its zero tests.

A column is the token `3 a + b` of two ternary digits, most significant first (modular_data's
encoding); after each column the prefixes read so far are `A` and `B`. A window is the column
followed by one query token per extra operation (D1): the column's own output is the running
quotient `A / B mod p` (or undefined, class `p`), each query's output its operation's answer:

- sum: `(A + B) mod p`;
- difference: `(A - B) mod p`.

`QUOTIENT` alone (no queries) is division as the window studies train it, the probe's control;
`SUM` the family without the difference (one dense counter).
The probe trains one network by the curriculum (step `j` trains lengths `1..j`) within a
wall-clock budget, diagnosing it as it goes: each operation's accuracy, the quotient's by class,
and each factor's share of `A`, `B`, the sum, the difference and the zero tests
(window_diagnostics.share).

The probe runs any target the same way (the general algebras, indrajala-research's docs/workplans/general-algebras-workplan.md,
D6): a target is an encoding (`Target`), division's `Encoding` one, `MachineEncoding` (any
window_tasks machine, its own task, one token a window) another. The fork, the trigger, the guard
and the readouts read only what a Target gives: its windows and answers, each answer's kind (D7:
a machine's by its token), which kinds are the certified answer's (the quotient's for division,
every kind for a machine), its task and the quantities its diagnosis shares.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from indrajala_ml.model.layers.array.branch_layers import BranchArrayLayer, MergeArrayLayer
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.load_network import load_network, network_from_state
from indrajala_ml.model.specs import window_presets as wp
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import (
    BEFORE_STATE,
    OUTPUT,
    AttentionPooling,
    Codes,
    Factor,
    WindowRecurrence,
)

from indrajala_studies import addition_data as ad
from indrajala_studies import batch as runner
from indrajala_studies import certificate_guard as cg
from indrajala_studies import inference
from indrajala_studies import machine_reading as mr
from indrajala_studies import modular_data as md
from indrajala_studies import window_diagnostics as wd
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt
from indrajala_studies.collapse import Trigger

BASE = md.BASE
COLUMNS = BASE * BASE  # a column's tokens, 3 a + b

# the operations a query token asks for, after the column's own quotient
OPERATIONS: dict[str, Callable[[int, int, int], int]] = {
    "sum": lambda A, B, p: (A + B) % p,
    "difference": lambda A, B, p: (A - B) % p,
    "product": lambda A, B, p: (A * B) % p,  # not an approximation (the approximation claim)
    # the state itself, A p + B: the finest approximation (the identity congruence), T12's arm (the
    # paper workplan's D31); its p^2 answers widen the output layer (Encoding.classes)
    "state": lambda A, B, p: A * p + B,
}
# the paper's T1 controls (paper-workplan.md): outputs of the columns read, not of the state (A, B), so
# no approximation of division's machine: a column's digit sum (its token a + b) and, with the column
# before (none before the first: 0), the last two columns' (a state the network must carry, one column)
COLUMN_OPERATIONS: dict[str, Callable[[int, int, int], int]] = {
    "token": lambda digits, before, p: digits % p,
    "suffix": lambda digits, before, p: (digits + before) % p,
}
QUERIES = (*OPERATIONS, *COLUMN_OPERATIONS)  # every operation a query token can ask for
FAMILY = ("sum", "difference")
SUM = ("sum",)  # the family without the difference (the division workplan's stage 3 option)
PRODUCT = ("product",)
STATE = ("state",)  # T12's: the state as the one extra output
TOKEN = ("token",)  # T1's controls: an extra output that is no approximation, state-free
SUFFIX = ("suffix",)  # and state-carrying
PLANTED = ("planted",)  # a planted machine's approximation asked (T10's P2: wt.query_partitions)
COVER = ("cover-a", "cover-b")  # T11's machines' coarsest cover asked (the paper workplan's D20: wt.query_partitions)
QUOTIENT: tuple[str, ...] = ()
VOCABULARY = ("sum", "difference")  # the operations with a token: every study's before the approximation claim
WIDE = ("sum", "difference", "product")  # the approximation claim's, the same for its every arm

Example = wr.Example

LONG_PROPERTIES = ("U1", "S1")  # the long tests' (the division workplan's D4)


@dataclass(frozen=True)
class Encoding:
    """
    A window's layout: the column, then a query token per operation, each its own answer. The
    vocabulary is the operations with a token (the embedding's rows), so arms compared with each
    other share it whatever they query; the default is every study's before the product's.
    """

    p: int
    queries: tuple[str, ...]
    vocabulary: tuple[str, ...] = VOCABULARY

    def __post_init__(self) -> None:
        assert set(self.queries) <= set(self.vocabulary) <= set(QUERIES), (
            f"queries {self.queries} need tokens in the vocabulary {self.vocabulary}, of {QUERIES}"
        )

    @property
    def width(self) -> int:
        return 1 + len(self.queries)

    @property
    def tokens(self) -> int:
        return COLUMNS + len(self.vocabulary)

    @property
    def classes(self) -> int:
        """
        The output layer's: residues and undefined (p + 1), or the state's p^2 when the vocabulary has
        it, so arms on one vocabulary share the layer as they share the tokens (D31's design (a)).
        """
        return self.p * self.p if "state" in self.vocabulary else self.p + 1

    def query_token(self, operation: str) -> int:
        return COLUMNS + self.vocabulary.index(operation)

    @property
    def query_tokens(self) -> tuple[int, ...]:
        return tuple(self.query_token(operation) for operation in self.queries)

    @property
    def task(self) -> wt.Task:
        return wt.task_named(f"modular-division-{self.p}")

    @property
    def target_states(self) -> int:
        return self.p * self.p

    def draw(self, n: int, rng: random.Random) -> list[int]:
        return columns(n, rng)

    def kinds(self, tokens: Sequence[float], labels: Sequence[int]) -> list[str]:
        """Each answer's kind: the quotient by class (0, undefined, nonzero), each query by its operation."""
        names: list[str] = []
        for i, label in enumerate(labels):
            slot = i % self.width
            if slot == 0:
                names.append(
                    "quotient-0" if label == 0 else "quotient-undefined" if label == self.p else "quotient-nonzero"
                )
            else:
                names.append(self.queries[slot - 1])
        return names

    def certified(self, kind: str) -> bool:
        """Whether a kind is the quotient's, the answer certified."""
        return kind.startswith("quotient")

    @property
    def quantities(self) -> dict[str, Callable[[Any], int]]:
        """What the diagnosis shares each factor with, of a state (A, B)."""
        p = self.p
        return {
            "A": lambda s: s[0],
            "B": lambda s: s[1],
            "S": lambda s: (s[0] + s[1]) % p,
            "D": lambda s: (s[0] - s[1]) % p,
            "A=0": lambda s: int(s[0] == 0),
            "B=0": lambda s: int(s[1] == 0),
        }

    def states(self, columns: Sequence[int]) -> list[tuple[int, int]]:
        """(A, B) mod p after each column."""
        a = b = 0  # the prefixes mod p
        states: list[tuple[int, int]] = []
        for token in columns:
            a, b = (BASE * a + token // BASE) % self.p, (BASE * b + token % BASE) % self.p
            states.append((a, b))
        return states

    def example(self, columns: Sequence[int]) -> Example:
        """The windows' tokens and each token's answer (states' tables: as `states` and the answers per column)."""
        tokens: list[float] = []
        labels: list[int] = []
        windows, answers, successors = self._windows, self._answers, self._successors
        state = 0  # (A, B) mod p as A p + B
        if not any(q in COLUMN_OPERATIONS for q in self.queries):  # every answer a function of the state
            for token in columns:
                state = successors[state][token]
                tokens += windows[token]
                labels += answers[state]
            return tuple(tokens), tuple(labels)
        before = 0  # the column before's digit sum
        for token in columns:
            state = successors[state][token]
            tokens += windows[token]
            digits = token // BASE + token % BASE
            labels += (
                answer if q is None else COLUMN_OPERATIONS[q](digits, before, self.p)
                for q, answer in zip((None, *self.queries), answers[state], strict=True)
            )
            before = digits
        return tuple(tokens), tuple(labels)

    @functools.cached_property
    def _windows(self) -> tuple[tuple[float, ...], ...]:
        """Each column token's window: the token, then a token per query."""
        queries = tuple(float(self.query_token(operation)) for operation in self.queries)
        return tuple((float(token), *queries) for token in range(COLUMNS))

    @functools.cached_property
    def _answers(self) -> tuple[tuple[int, ...], ...]:
        """Each state's answers, A p + B: the quotient's, then each query's (-1 for a column operation's, set per column)."""
        p = self.p
        return tuple(
            (md.quotient(A, B, p), *(OPERATIONS[q](A, B, p) if q in OPERATIONS else -1 for q in self.queries))
            for A in range(p)
            for B in range(p)
        )

    @functools.cached_property
    def _successors(self) -> tuple[tuple[int, ...], ...]:
        """Each state's next, A p + B, after each column token (as `states`)."""
        p = self.p
        return tuple(
            tuple((BASE * A + token // BASE) % p * p + (BASE * B + token % BASE) % p for token in range(COLUMNS))
            for A in range(p)
            for B in range(p)
        )


@dataclass(frozen=True)
class MachineEncoding:
    """
    Any machine's own task as the probe's target (D6): a window is one token, then a query token per
    query (a named approximation, wt.query_partitions: its answer the block of the state after the
    token, as division's queries ask its approximations); the token's answer the machine's output,
    its kind its token (D7), the certified answer's; each query's kind its name. The diagnosis
    shares each factor with each coordinate of the state (window_diagnostics.coordinates).
    """

    task: wt.Task
    queries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        known = wt.query_partitions(self.task.name)
        assert set(self.queries) <= set(known), f"{self.task.name}'s queries are of {sorted(known)}; got {self.queries}"

    @property
    def width(self) -> int:
        return 1 + len(self.queries)

    @property
    def query_tokens(self) -> tuple[int, ...]:
        return tuple(range(self.task.machine.tokens, self.task.machine.tokens + len(self.queries)))

    @property
    def tokens(self) -> int:
        return self.task.machine.tokens + len(self.queries)

    @property
    def classes(self) -> int:
        return max((self.task.machine.classes, *(int(labels.max()) + 1 for labels in self._partitions)))

    @property
    def target_states(self) -> int:
        return self.task.machine.states

    def draw(self, n: int, rng: random.Random) -> list[int]:
        """n uniform tokens, then the identity (token 0), as division's final zero column: its answer
        needs the state the word left, so even one token trains the state (a probe without it,
        2026-10-17: 5 of 6 runs at 6 states kept one code from step 1, where no answer needs the state)."""
        return [rng.randrange(self.task.machine.tokens) for _ in range(n)] + [0]

    def example(self, columns: Sequence[int]) -> Example:
        machine = self.task.machine
        if not self.queries:
            return tuple(float(token) for token in columns), machine.run(columns)
        tokens: list[float] = []
        labels: list[int] = []
        for token, label, state in zip(columns, machine.run(columns), machine.path(columns), strict=True):
            tokens += (float(token), *(float(q) for q in self.query_tokens))
            labels += (label, *(int(blocks[state]) for blocks in self._partitions))
        return tuple(tokens), tuple(labels)

    def kinds(self, tokens: Sequence[float], labels: Sequence[int]) -> list[str]:
        width = self.width
        return [
            f"token-{int(tokens[i])}" if i % width == 0 else self.queries[i % width - 1] for i in range(len(tokens))
        ]

    def certified(self, kind: str) -> bool:
        return kind not in self.queries

    @property
    def quantities(self) -> dict[str, Callable[[Any], int]]:
        return dict(wd.coordinates(self.task.machine))

    def states(self, columns: Sequence[int]) -> list[int]:
        return self.task.machine.path(columns)

    @functools.cached_property
    def _partitions(self) -> tuple[npt.NDArray[np.int64], ...]:
        """Each query's labels of the machine's states."""
        known = wt.query_partitions(self.task.name)
        return tuple(known[q] for q in self.queries)


class Target(Protocol):
    """What the probe needs of a target (D6): Encoding's and MachineEncoding's members, documented there."""

    @property
    def width(self) -> int: ...

    @property
    def tokens(self) -> int: ...

    @property
    def classes(self) -> int: ...

    @property
    def query_tokens(self) -> tuple[int, ...]: ...

    @property
    def task(self) -> wt.Task: ...

    @property
    def target_states(self) -> int: ...

    @property
    def quantities(self) -> Mapping[str, Callable[[Any], int]]: ...

    def draw(self, n: int, rng: random.Random) -> list[int]: ...

    def example(self, columns: Sequence[int]) -> Example: ...

    def kinds(self, tokens: Sequence[float], labels: Sequence[int]) -> list[str]: ...

    def certified(self, kind: str) -> bool: ...

    def states(self, columns: Sequence[int]) -> Sequence[Hashable]: ...


def target_named(name: str, queries: tuple[str, ...] = (), vocabulary: tuple[str, ...] = VOCABULARY) -> Target:
    """modular-division-p's Encoding (with queries and vocabulary), or any other task's MachineEncoding (with queries)."""
    if found := re.fullmatch(r"modular-division-(\d+)", name):
        return Encoding(int(found[1]), queries, vocabulary)
    return MachineEncoding(wt.task_named(name), queries)


def columns(n: int, rng: random.Random) -> list[int]:
    """n uniform columns, then the final zero column (the window studies' final zero window, at
    k 1: its answer needs the state the data left, so even one column trains the state)."""
    return [rng.randrange(COLUMNS) for _ in range(n)] + [0]


LINEAR, CONTINUOUS_STATE = "linear", "continuous"
BASELINES = (LINEAR, CONTINUOUS_STATE)  # the paper's T3: the transformer's and the snap's controls


def network(
    encoding: Target,
    factors: Sequence[tuple[int, int]],
    seed: int,
    reads: str = BEFORE_STATE,
    width: int = wp.D,
    branches: bool = False,
    answer_branch: bool = False,
    baseline: str | None = None,
    recombine: bool = False,
    branch_heads: int = 1,
) -> Any:
    """The transformer window (W11) over the encoding's tokens, a factored state (D3), Adam; the
    poolings reading before the first StateInput (D10) or, with reads OUTPUT, the window's output.
    With branches, the window branches workplan's window (wp.branched_window), each factor's
    pooling reading its own branch's end (reads OUTPUT); with answer_branch, the one-code-one-state
    workplan's R1 (wp.answer_branch_window), every factor reading the shared state branch; with
    recombine, the post-paper workplan's recombining window (wp.recombined_window, D7), a two-block
    trunk, each factor reading its own branch's end, the ends merged into one more block before the
    output. branch_heads
    is each branch's attention heads with branches (the post-paper workplan, D8; 1 the retired
    design's). A baseline (BASELINES, the paper's T3): "linear", E6's window without its blocks
    (wp.linear_window); "continuous", every factor's state continuous, no codes (E2's)."""
    assert not (branches or answer_branch or recombine) or reads == OUTPUT, (
        "with branches each factor reads a branch's end: reads OUTPUT"
    )
    assert sum((branches, answer_branch, recombine)) <= 1, (
        "the window branches, the answers' branch or the recombining window, one at most"
    )
    assert branch_heads == 1 or branches, "branch heads are the window branches' (--branches)"
    assert baseline is None or baseline in BASELINES, f"a baseline: one of {BASELINES}"
    assert baseline is None or not (branches or answer_branch or recombine), (
        "a baseline is W11's window or E6's, unbranched"
    )
    recurrence = WindowRecurrence(
        tuple(Factor(s, None if baseline == CONTINUOUS_STATE else Codes(m)) for s, m in factors),
        False,
        pooling=AttentionPooling(reads),
    )
    if baseline == LINEAR:
        specs = wp.linear_window(encoding.tokens, encoding.classes, width)
    elif branches:
        specs = wp.branched_window(encoding.tokens, encoding.classes, len(factors), width, branch_heads)
    elif recombine:
        specs = wp.recombined_window(encoding.tokens, encoding.classes, len(factors), width)
    elif answer_branch:
        specs = wp.answer_branch_window(encoding.tokens, encoding.classes, len(factors), width)
    else:
        specs = wp.transformer_window(encoding.tokens, encoding.classes, width)
    return WindowRecurrentRustArrayNetwork.randomized((encoding.width,), specs, recurrence, Adam(), seed=seed)


@dataclass(frozen=True)
class Settings:
    """The probe's training (D4)."""

    epoch_examples: int = 2**15
    batch_size: int = 64
    rate: float = 0.003
    patience: int = 10
    max_epochs: int = 60
    screen: int = 200  # cases a length, each epoch's screen
    diagnose_every: int = 5
    lengths: tuple[int, ...] = (3, 6, 12)  # the diagnosis's
    cases: int = 200  # a diagnosis length's
    long: tuple[int, ...] = (24, 48)  # the quotient's, at a step's end


PROBE = Settings()


@dataclass(frozen=True)
class Fork:
    """
    Snapshot and fork on collapse (the collapse-fork workplan, retired:
    `git show 89cf3a3:docs/collapse-fork-workplan.md`): within a step, the network at the step's
    best value of the trigger's signal is kept; an epoch the trigger calls a collapse
    (collapse.Trigger; by default the screen below 0.6 of the step's peak, the workplan's D1) sends
    the run back to the snapshot, cools the step's rate by `cool` and takes its data from a new
    stream; at most `limit` forks a step.
    """

    trigger: Trigger = field(default_factory=Trigger)
    cool: float = 0.5  # D3
    limit: int = 4  # D4


def screen(net: Any, encoding: Target, screens: Mapping[int, Sequence[Example]]) -> tuple[float, dict[str, float]]:
    """
    A step's screen: the share of every answer right over its screens (its training lengths), and
    each answer kind's share over them (the p11-trigger workplan's D4: a trigger's signals).
    """
    tallies = [accuracy(net, encoding, rows) for rows in screens.values()]
    correct = sum(c for t in tallies for c, _ in t.values())
    pooled = correct / sum(a for t in tallies for _, a in t.values())
    kinds: dict[str, list[int]] = {}
    for tally in tallies:
        for name, (right, all_) in tally.items():
            kind = kinds.setdefault(name, [0, 0])
            kind[0] += right
            kind[1] += all_
    return pooled, {name: right / all_ for name, (right, all_) in sorted(kinds.items())}


def certified_right(encoding: Target, kinds: Mapping[str, float]) -> bool:
    """Every kind of the certified answer right on a screen: necessary for its machine to be right."""
    return all(share == 1.0 for name, share in kinds.items() if encoding.certified(name))


def _kinds(kinds: Mapping[str, float]) -> dict[str, float]:
    return {name: round(share, 4) for name, share in kinds.items()}


def _windows(encoding: Target, n: int, cases: int, seed: str) -> tuple[list[Example], list[list[int]]]:
    rng = random.Random(seed)
    drawn = [encoding.draw(n, rng) for _ in range(cases)]
    return [encoding.example(c) for c in drawn], drawn


def windows_drawn(encoding: Target, n: int, cases: int, seed: str) -> list[Example]:
    """The examples of `cases` words of length `n` drawn as a run's screen draws them (its seed's name)."""
    return _windows(encoding, n, cases, seed)[0]


def accuracy(net: Any, encoding: Target, rows: Sequence[Example]) -> dict[str, list[int]]:
    """Each answer kind's [correct, all] (division's quotient by class, 0, undefined, nonzero)."""
    tally: dict[str, list[int]] = {}
    predictions: list[Sequence[int]] = net.classify_rows(net.prepare_dataset(rows))
    for (tokens, labels), predicted in zip(rows, predictions, strict=True):
        for name, label, guess in zip(encoding.kinds(tokens, labels), labels, predicted, strict=True):
            t = tally.setdefault(name, [0, 0])
            t[0] += int(label == guess)
            t[1] += 1
    return tally


def diagnose(net: Any, encoding: Target, settings: Settings) -> dict[str, Any]:
    """
    Each answer's accuracy and each factor's codes used and shares, at the diagnosis's lengths; a
    continuous state (the T3 baseline) has no codes, so its answers' accuracy alone.
    """
    tally: dict[str, list[int]] = {}
    samples: list[tuple[tuple[int, ...], Hashable]] = []  # codes, the target's state per window
    snapped = net.recurrence.snapped
    for n in settings.lengths:
        rows, drawn = _windows(encoding, n, settings.cases, f"family-diagnose-{n}")
        for name, (c, a) in accuracy(net, encoding, rows).items():
            t = tally.setdefault(name, [0, 0])
            t[0] += c
            t[1] += a
        if not snapped:
            continue
        codes: list[list[tuple[int, ...]]] = net.state_codes(net.backend.matrix([tokens for tokens, _ in rows]))
        for states, cols in zip(codes, drawn, strict=True):
            for w, state in enumerate(encoding.states(cols)):
                samples.append((states[w + 1], state))
    if not snapped:
        return {"accuracy": {name: round(c / a, 3) for name, (c, a) in sorted(tally.items())}}
    factors = [
        {
            "codes": len({c[g] for c, _ in samples}),
            "shares": {
                q: round(wd.share([(c[g], f(state)) for c, state in samples]), 3)
                for q, f in encoding.quantities.items()
            },
        }
        for g in range(len(samples[0][0]))
    ]
    return {
        "accuracy": {name: round(c / a, 3) for name, (c, a) in sorted(tally.items())},
        "factors": factors,
        "state_share": round(wd.share([(c, state) for c, state in samples]), 3),
    }


def long_accuracy(net: Any, encoding: Target, settings: Settings) -> dict[str, float]:
    """The certified answer's accuracy (division's quotient) over every window at each long length."""
    result: dict[str, float] = {}
    for n in settings.long:
        rows, _ = _windows(encoding, n, settings.cases, f"family-long-{n}")
        tally = accuracy(net, encoding, rows)
        q = [tally[k] for k in tally if encoding.certified(k)]
        result[str(n)] = round(sum(c for c, _ in q) / sum(a for _, a in q), 4)
    return result


def _codes(row: Mapping[str, Any]) -> list[int]:
    return [int(factor["codes"]) for factor in row["factors"]]


def _collapsed(before: Sequence[int] | None, codes: Sequence[int], drop: float) -> bool:
    """The codes in use below `drop` of the diagnosis before's (a collapse, as `trace` marks it)."""
    return before is not None and sum(codes) < drop * sum(before)


def trace(rows: Sequence[Mapping[str, Any]], drop: float = 0.5) -> list[str]:
    """
    A run's log.jsonl, a line per diagnosis: step, epoch (or the step's end), each factor's codes in
    use, the state's share of the pair, each answer's accuracy; marked where the codes in use fall
    below `drop` of the diagnosis before (a collapse: the division workplan's stage 4 trace); and a
    line per fork and per restored best. An epoch's screen score alone (a row without codes) isn't shown.
    """
    lines: list[str] = []
    before: list[int] | None = None
    for row in rows:
        if "fork" in row:
            lines.append(
                f"step {row['step']:>3} epoch {row['epoch']:>3} FORK {row['fork']}: screen {row['screen']:.3f} "
                f"peak {row['peak']:.3f} restored {row['restored']:.3f} rate {row['rate']:.2g}"
            )
            continue
        if "restore" in row:
            lines.append(
                f"step {row['step']:>3} epoch {row['epoch']:>3} RESTORE the step's best: screen {row['restore']:.3f} "
                f"restored {row['restored']:.3f}"
            )
            continue
        if "restore_machine" in row:  # R3's kept machine, restored at a plateau
            lines.append(
                f"step {row['step']:>3} epoch {row['epoch']:>3} RESTORE the kept machine: pairs/code "
                f"{row['restore_machine']:.2f} screen {row['restored']:.3f}"
            )
            continue
        if "factors" not in row:  # an epoch's screen score alone
            continue
        codes = _codes(row)
        answers = " ".join(f"{name} {value:.3f}" for name, value in sorted(row["accuracy"].items()))
        end = f" END {row['end']} min {row.get('minimised')}" if "end" in row else ""
        if "machine" in row:  # pairs per code (the one-code-one-state workplan's R2)
            walked = row["machine"]
            end += f" pairs/code {walked['pairs'] / walked['codes']:.2f} right {walked['right']}"
        mark = " << collapse" if _collapsed(before, codes, drop) else ""
        epoch = row.get("epoch", "-")
        share = row.get("state_share", 0.0)
        lines.append(f"step {row['step']:>3} epoch {epoch:>3} codes {codes} share {share:.3f} {answers}{end}{mark}")
        before = codes
    return lines


@dataclass(frozen=True)
class Outcome:
    """
    A run's end from its log.jsonl: the last step, how it ended, epochs trained in all, minutes; and
    its course: the best its worst answer reached at any diagnosis (`peak`, the accuracy of the
    answer furthest behind) and at which step, its collapses (as `trace` marks them) and its large
    drops (`large_drops`), with how many it never recovered from.
    """

    step: int
    end: str  # certified, or the last step's end (passed, plateau, budget, spent)
    epochs: int
    minutes: float
    peak: float
    peak_step: int
    collapses: int
    drops: int  # large drops (large_drops)
    unrecovered: int  # of them, never back to DROP_FROM


CENSORED = ("spent", "deadline")  # a run's ends at a budget: what it would have done next is unknown


def outcome(rows: Sequence[Mapping[str, Any]], drop: float = 0.5) -> Outcome:
    """
    A run's Outcome: certified at the first step's end that certified (the run stops there). A step
    stopped by a budget (`spent`, `deadline`) logs as its epochs the loop's index at the stop, an
    epoch not trained, so one less is counted for it. A log holding a re-run counts its last run only
    (the batch runner's last_run).
    """
    rows = runner.last_run(rows)
    ends = [row for row in rows if "end" in row]
    last = next((row for row in ends if row.get("certified")), ends[-1] if ends else rows[-1])
    epochs = sum(int(row["epochs"]) - (row["end"] in CENSORED) for row in ends if row["step"] <= last["step"])
    end = "certified" if last.get("certified") else str(last.get("end", "running"))
    diagnoses = [row for row in rows if "accuracy" in row and row["step"] <= last["step"]]
    best = max(diagnoses, key=_worst, default=last)
    coded = [row for row in diagnoses if "factors" in row]  # a continuous state logs no codes
    collapses = sum(_collapsed(_codes(before), _codes(row), drop) for before, row in itertools.pairwise(coded))
    falls = large_drops(diagnoses)
    return Outcome(
        int(last["step"]),
        end,
        epochs,
        round(float(last["seconds"]) / 60, 1),
        round(min(best.get("accuracy", {"-": 0.0}).values()), 3),  # the log keeps three decimals
        int(best["step"]),
        collapses,
        len(falls),
        sum(not recovered for _, recovered in falls),
    )


def progress(rows: Sequence[Mapping[str, Any]]) -> str:
    """
    A run's progress from its log.jsonl, running or ended, as a line: its Outcome (the last step, how it
    ended, epochs trained in the steps that ended, minutes, the best worst answer and its step), or for
    a run still in a step, that step and its epoch; and the anneal's last phase, if it anneals.
    """
    rows = runner.last_run(rows)
    o, last = outcome(rows), rows[-1]
    phase = next((r["anneal"]["phase"] for r in reversed(rows) if "phase" in r.get("anneal", {})), None)
    if o.end != "certified" and "end" not in last:  # the step in flight's diagnoses count to the peak
        at = f"step {last['step']} running at epoch {last.get('epoch', '-')}, {o.epochs} epochs before it"
        minutes = round(float(last.get("seconds", 0.0)) / 60, 1)
        best = max((r for r in rows if "accuracy" in r), key=_worst, default=None)
        peak, peak_step = (o.peak, o.peak_step) if best is None else (min(best["accuracy"].values()), best["step"])
    else:
        at, minutes, peak, peak_step = f"step {o.step} {o.end}, {o.epochs} epochs", o.minutes, o.peak, o.peak_step
    line = f"{at}, {minutes:g} min, peak {peak:.3f} at step {peak_step}"
    return line if phase is None else f"{line}, anneal {phase}"


def _anneal_at(rows: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The anneal at a diagnosis: its own, or the last epoch's before it (a step's end logs none)."""
    at = next(i for i, r in enumerate(rows) if r is row)
    none: Mapping[str, Any] = {}
    return next((r["anneal"] for r in reversed(rows[: at + 1]) if "anneal" in r), none)


def trained_length(rows: Sequence[Mapping[str, Any]], run: Mapping[str, Any], step: int) -> int:
    """
    The longest length a run had trained on by `step` (the state formation workplan's D42), from its
    log and its checkpoint's `run` record: the cap with `--whole` or once a crucible grew (D26); a
    crucible's length after its warm-up; else the curriculum's step.
    """
    grown = next((int(row["step"]) for row in rows if row.get("crucible") == "grown"), None)
    if run.get("whole") or (grown is not None and grown <= step):
        return int(run["cap"])
    fixed = next((row for row in rows if "warm_up_end" in row and "crucible" in row), None)
    return int(fixed["crucible"]) if fixed is not None and step > fixed["step"] else step


def crucible_course(
    rows: Sequence[Mapping[str, Any]], drop: float = 0.5, run: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """
    A seeded crystallisation's run (D26) from its log.jsonl, for a population's reading (D27): the
    warm-up's end (its step, the crucible's length, each factor's codes in use, effective dimension,
    melting point and transitions there; the warm-up's best screen); the crucible's best state share
    in the cool (the heat's are the warm-up's codes, which the melt erases), its step and coupling; the first diagnosis after it whose codes in use fall below `drop` of the
    best's (the loss: its step, phase, coupling, mean temperature); the first grown step; how it ended
    (outcome's). Observables to set against a population's outcomes, early ones first. Up to the
    end: the epochs D29's gate held, the largest rise of a factor's coupling in an epoch (D34, D45),
    and with the checkpoint's `run` record the longest length trained (D42).
    """
    start = next((row for row in rows if "warm_up_end" in row), None)
    warm = [row for row in rows if row.get("warm") and "screen" in row]
    diagnoses = [row for row in rows if "factors" in row and "accuracy" in row]
    grown = next((int(row["step"]) for row in rows if row.get("crucible") == "grown"), None)
    crucible = [row for row in diagnoses if start is not None and row["step"] > start["step"]]
    crucible = [row for row in crucible if grown is None or row["step"] < grown]
    cooled = [row for row in crucible if _anneal_at(rows, row).get("phase") == "cool"] or crucible
    best = max(cooled, key=lambda row: float(row.get("state_share", 0.0)), default=None)
    lost = None
    if best is not None:
        later = cooled[cooled.index(best) + 1 :]
        lost = next((row for row in later if sum(_codes(row)) < drop * sum(_codes(best))), None)

    def anneal(row: Mapping[str, Any] | None) -> dict[str, Any]:
        a: Mapping[str, Any] = {} if row is None else _anneal_at(rows, row)
        if not a:
            return {}
        couplings: list[float] = a["coupling"]
        soft = [t for t in a["temperature"] if t is not None]  # a hard factor has no temperature
        return {
            "phase": a["phase"],
            "coupling": round(sum(couplings) / len(couplings), 3),
            "temperature": float(np.mean(soft)) if soft else None,
        }

    end = outcome(rows, drop)
    profile = [row["anneal"] for row in rows if "anneal" in row and row["step"] <= end.step]
    couplings = [a["coupling"] for a in profile if "coupling" in a]
    rises = [b - a for before, after in itertools.pairwise(couplings) for a, b in zip(before, after, strict=True)]
    return {
        "warm_up_end": None if start is None else int(start["step"]),
        "crucible": None if start is None else start.get("crucible"),  # None with --whole (D22)
        "warm_screen": max((float(row["screen"]) for row in warm), default=None),
        "in_use": [] if start is None else [int(f["in_use"]) for f in start["warm_up_end"]],
        "effective_dimension": [] if start is None else [int(f["effective_dimension"]) for f in start["warm_up_end"]],
        "melting_point": [] if start is None else [float(f["melting_point"]) for f in start["warm_up_end"]],
        "transitions": [] if start is None else [len(f["transitions"]) for f in start["warm_up_end"]],
        "best_share": None if best is None else float(best.get("state_share", 0.0)),
        "best_step": None if best is None else int(best["step"]),
        "best": anneal(best),
        "lost_step": None if lost is None else int(lost["step"]),
        "lost": anneal(lost),
        "grown": grown,
        "end": end.end,
        "step": end.step,
        "minutes": end.minutes,
        "held": max((int(a.get("held", 0)) for a in profile), default=0),
        "rise": round(max(rises, default=0.0), 4),
        "trained": None if run is None else trained_length(rows, run, end.step),
    }


def crucible_summary(course: Mapping[str, Any]) -> str:
    """crucible_course's reading on one line."""

    def at(a: Mapping[str, Any]) -> str:
        if not a:
            return ""
        heat = "hard" if a["temperature"] is None else f"T {a['temperature']:.2g}"
        return f" ({a['phase']}, coupling {a['coupling']:.2f}, {heat})"

    best = (
        "-"
        if course["best_share"] is None
        else f"{course['best_share']:.3f} at {course['best_step']}{at(course['best'])}"
    )
    lost = "-" if course["lost_step"] is None else f"{course['lost_step']}{at(course['lost'])}"
    melting = ", ".join(f"{t:.2f}" for t in course["melting_point"])
    return (
        f"warm-up to {course['warm_up_end']} (crucible {course['crucible']}, screen {course['warm_screen']}, "
        f"in use {course['in_use']}, d_eff {course['effective_dimension']}, melting {melting}, "
        f"transitions {course['transitions']}); best share {best}; lost {lost}; grown {course['grown'] or '-'}; "
        f"{course['end']} at {course['step']} ({course['minutes']} min); held {course['held']}, "
        f"coupling's largest rise {course['rise']}, trained to {course['trained'] or '?'}"
    )


def heat_course(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any] | None]:
    """
    Each factor's cool from a profiled run's log.jsonl (D32's understanding step), its epochs counted
    from the cool's start: its splits (an epoch whose codes' groups rise, D20), the first (nucleation)
    and the largest (the first of the largest rise); its heat capacity's peak (`thermals`, the
    epoch's logged `capacity`), the peak's distance in epochs from each and whether it lies within the
    cascade (from the first split to the last); the capacity's local maxima and its jump across the
    nucleation epoch. None for a factor never soft in the cool, or a log without `capacity`.
    """
    cool = [r["anneal"] for r in rows if r.get("anneal", {}).get("phase") == "cool" and "capacity" in r["anneal"]]
    courses: list[dict[str, Any] | None] = []
    for g in range(len(cool[0]["temperature"]) if cool else 0):
        soft = [a for a in cool if a["temperature"][g] is not None]
        if not soft:
            courses.append(None)
            continue
        T = [float(a["temperature"][g]) for a in soft]
        groups = [int(a["groups"][g]) for a in soft]
        C = [float(a["capacity"][g]) for a in soft]
        splits = [(i, groups[i] - groups[i - 1]) for i in range(1, len(soft)) if groups[i] > groups[i - 1]]
        peak = max(range(len(C)), key=C.__getitem__)
        first = splits[0][0] if splits else None
        largest = max(splits, key=lambda s: s[1])[0] if splits else None
        courses.append(
            {
                "epochs": len(soft),
                "groups": [groups[0], groups[-1]],
                "splits": [[i, rise, round(T[i], 4)] for i, rise in splits],
                "nucleation": first,
                "largest": largest,
                "peak": peak,
                "peak_capacity": C[peak],
                "peak_temperature": round(T[peak], 4),
                "from_nucleation": None if first is None else peak - first,
                "from_largest": None if largest is None else peak - largest,
                "in_cascade": bool(splits) and splits[0][0] <= peak <= splits[-1][0],
                "maxima": sum(1 for i in range(1, len(C) - 1) if C[i - 1] < C[i] > C[i + 1]),
                "jump": None if not first or C[first - 1] <= 0 else round(C[first] / C[first - 1], 2),
            }
        )
    return courses


def heat_summary(courses: Sequence[Mapping[str, Any] | None]) -> str:
    """heat_course's reading on one line."""

    def factor(g: int, c: Mapping[str, Any] | None) -> str:
        if c is None:
            return f"factor {g} -"
        splits = c["splits"]
        span = (
            f"T {splits[0][2]:.3g} to {splits[-1][2]:.3g}, epochs {splits[0][0]}-{splits[-1][0]}" if splits else "none"
        )
        return (
            f"factor {g}: groups {c['groups'][0]} to {c['groups'][1]} in {c['epochs']} epochs, {len(splits)} splits "
            f"({span}; nucleation {c['nucleation']}, largest {c['largest']}, jump x{c['jump']}); capacity peak "
            f"{c['peak_capacity']:.3g} at {c['peak']} (T {c['peak_temperature']:.3g}), {c['from_nucleation']} from "
            f"nucleation, {c['from_largest']} from the largest, {'in' if c['in_cascade'] else 'outside'} the "
            f"cascade; {c['maxima']} local maxima"
        )

    return "; ".join(factor(g, c) for g, c in enumerate(courses))


FORMED_SHARE = 0.95  # a step's end whose state share reaches this formed the state (formation_course)


def formation_course(rows: Sequence[Mapping[str, Any]], by: int = 200) -> dict[str, Any] | None:
    """
    A run's formation and its keeping from its log.jsonl (the scaling conjecture's reading, 2026-10-22):
    the first step's end whose state share reaches FORMED_SHARE (`formed`), the largest share, the first
    step certified at the strict verdict and at the default (`strict`, `certified`), the first whose
    answers-only stop passed (`passed`), `lost` (after `formed`, the first share below 0.6), the seconds
    at step `by` and at the last row (the cost a kill at `by` would save); and the first melting curve
    (D21) per factor: the share of its clusters made at the first transition (`first_jump`) and the
    octaves below the melting point to half and three quarters of them (`octaves_half`,
    `octaves_three_quarters`: how crowded its cascade is). None for a log without step ends.
    """
    steps = [r for r in rows if "end" in r and "state_share" in r]
    if not steps:
        return None

    def first(test: Callable[[Mapping[str, Any]], bool]) -> int | None:
        return next((int(r["step"]) for r in steps if test(r)), None)

    formed = first(lambda r: r["state_share"] >= FORMED_SHARE)
    melting: Sequence[Mapping[str, Any]] | None = next((r["melting"] for r in rows if "melting" in r), None)
    cascades: list[dict[str, Any] | None] = []
    for factor in melting or ():
        # each transition [temperature, groups before, groups after]
        transitions = [(float(t[0]), int(t[2])) for t in factor["transitions"]]
        if not transitions:
            cascades.append(None)
            continue
        foot = max(after for _, after in transitions)

        def reach(share: float, transitions: list[tuple[float, int]] = transitions, foot: int = foot) -> float:
            T = next(T for T, after in transitions if after >= share * foot)
            return round(math.log2(transitions[0][0] / T), 2)

        cascades.append(
            {
                "foot": foot,
                "transitions": len(transitions),
                "first_jump": round(transitions[0][1] / foot, 2),
                "octaves_half": reach(0.5),
                "octaves_three_quarters": reach(0.75),
            }
        )
    return {
        "steps": len(steps),
        "max_share": max(float(r["state_share"]) for r in steps),
        "formed": formed,
        "strict": first(lambda r: bool(r.get("certificates", {}).get("strict", {}).get("certified"))),
        "certified": first(lambda r: bool(r.get("certified"))),
        "passed": first(lambda r: bool(r.get("answers_stop", {}).get("passed"))),
        "lost": None if formed is None else first(lambda r: r["step"] > formed and r["state_share"] < 0.6),
        "seconds_by": next((float(r["seconds"]) for r in steps if r["step"] >= by), None),
        "seconds": float(steps[-1]["seconds"]),
        "cascades": cascades,
    }


def formation_summary(course: Mapping[str, Any] | None) -> str:
    """formation_course's reading on one line."""
    if course is None:
        return "no step ends"
    cascades = "; ".join(
        "-"
        if c is None
        else f"{c['transitions']} transitions to {c['foot']}, first {c['first_jump']}, "
        f"half in {c['octaves_half']} oct, 3/4 in {c['octaves_three_quarters']} oct"
        for c in course["cascades"]
    )
    by = course["seconds_by"]
    return (
        f"max share {course['max_share']:.3f}, formed {course['formed']}, strict {course['strict']}, "
        f"certified {course['certified']}, passed {course['passed']}, lost {course['lost']}; "
        f"{'-' if by is None else round(by / 60, 1)} of {round(course['seconds'] / 60, 1)} min by the kill's step; "
        f"melting: {cascades or '-'}"
    )


def restart_course(rows: Sequence[Mapping[str, Any]], clock: str = "epochs") -> tuple[float | None, float, bool] | None:
    """
    A run as a restart attempt, on `clock` ("epochs" trained in all, the budget's unit and the time
    axis, the owner 2026-10-23; "steps"; or "seconds", money not time), from its log.jsonl: its time
    to success (D41's: the first step's end certified at both verdicts, the default and the strict;
    None if never), its last step's end, and whether it was censored there (ended at a budget,
    "spent", or the wall-clock guard's "deadline": what it would have done next is unknown) rather
    than finished (a stop that fired on a machine not certified is a failed attempt of known cost).
    None for a log without step ends.
    """
    steps = [r for r in rows if "end" in r and "state_share" in r]
    if not steps:
        return None
    trained: dict[int, int] = {}  # each step end's epochs trained in all, counted as outcome counts them
    total = 0
    for r in rows:
        if "end" in r:
            total += int(r.get("epochs", 0)) - (r["end"] in CENSORED)
            trained[id(r)] = total

    def at(r: Mapping[str, Any]) -> float:
        return float(trained[id(r)] if clock == "epochs" else r["step"] if clock == "steps" else r["seconds"])

    hit = next(
        (r for r in steps if r.get("certified") and r.get("certificates", {}).get("strict", {}).get("certified")),
        None,
    )
    return (None if hit is None else at(hit)), at(steps[-1]), hit is None and steps[-1]["end"] in CENSORED


def restart_cost(runs: Sequence[tuple[float | None, float, bool]]) -> dict[str, Any]:
    """
    Luby, Sinclair and Zuckerman's expected cost per success of fresh-seed restarts at a fixed cutoff
    `t` (a certified run is a Las Vegas run: right whenever it stops): `l(t) = E[min(T, t)] / q(t)`,
    with `q` the share of runs certified by `t` and `T` a run's time to success, or to its end when it
    fails. `runs` are restart_course's; a run censored at its budget is read only to the shortest
    censored run's time (`horizon`); a finished failure costs its known time at any cutoff. The optimal
    fixed cutoff `t*` lies at a success time (between them `q` is flat and the cost rises). `run_out`:
    each run to its own end, today's cost per certificate.
    """
    censored = [end for _, end, c in runs if c]
    successes = sorted(s for s, _, _ in runs if s is not None)
    horizon = min(censored) if censored else max((end for _, end, _ in runs), default=0.0)
    curve: list[tuple[float, float, float]] = []  # (t, q(t), l(t))
    for t in sorted({s for s in successes if s <= horizon} | ({horizon} if successes else set())):
        q = sum(1 for s in successes if s <= t) / len(runs)
        cost = sum(min(end if s is None else s, t) for s, end, _ in runs) / len(runs)
        if q > 0:
            curve.append((t, q, cost / q))
    best = min(curve, key=lambda c: c[2]) if curve else None
    return {
        "runs": len(runs),
        "certified": len(successes),
        "censored": len(censored),
        "horizon": horizon,
        "curve": curve,
        "best": best,
        "run_out": sum(end if s is None else s for s, end, _ in runs) / len(successes) if successes else None,
    }


def restart_summary(cost: Mapping[str, Any], unit: str = "steps") -> str:
    """restart_cost's reading on one line."""
    best = cost["best"]
    head = f"{cost['runs']} runs, {cost['certified']} certified, {cost['censored']} censored"
    if best is None:
        return f"{head}; none certified by the horizon {cost['horizon']:g} {unit}: no cutoff pays"
    t, q, l = best
    out = cost["run_out"]
    return (
        f"{head}; read to {cost['horizon']:g} {unit}: t* {t:g} (q {q:.2f}), l(t*) {l:.0f} {unit} per "
        f"certificate against {out:.0f} running each out ({out / l:.1f}x)"
    )


JUMP = 0.1  # a coupling move of this or more in an epoch, any factor, is a jump (cool_course)
SCREEN_FALL = 0.9  # the screen's fall: its first epoch below this after one at or above it (cool_course)


def cool_course(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """
    A run's cool from its log.jsonl, its epochs in the phase "cool" (the workplan's readings of the
    coupling's jumps and the freeze, D61, D63): the coupling's jumps (`jumps`, `largest` move, the first
    jump's step `jump`), the screen's fall (`fall`, the step of its first epoch below SCREEN_FALL after
    one at or above it) and their `order` ("jump first", "the same epoch", "drop first", or None without
    either) with the fall's epochs after the first jump (`gap`; a row's coupling is set before its epoch
    trains, its screen read after); the longest freeze (`frozen`: the clock's rate held at 0, its epochs, its first step, the
    clock, the hottest factor's temperature and the largest coupling there); the reheats (D63 (a):
    stretches of a rate below 0, their count and the first's step). None for a log without a cool.
    """
    cool = [r for r in rows if r.get("anneal", {}).get("phase") == "cool"]
    if not cool:
        return None
    moves = [
        (int(b["step"]), max(abs(y - x) for x, y in zip(a["anneal"]["coupling"], b["anneal"]["coupling"], strict=True)))
        for a, b in itertools.pairwise(cool)
    ]
    jumps = [(i + 1, step) for i, (step, move) in enumerate(moves) if move >= JUMP]
    high = next((i for i, r in enumerate(cool) if r["screen"] >= SCREEN_FALL), None)
    fall = None if high is None else next((i for i in range(high, len(cool)) if cool[i]["screen"] < SCREEN_FALL), None)
    order, gap = None, None
    if jumps and fall is not None:
        gap = fall - jumps[0][0]
        order = "jump first" if gap > 0 else "the same epoch" if gap == 0 else "drop first"
    frozen: dict[str, Any] | None = None
    held, reheats, reheat = 0, 0, None
    for i, r in enumerate(cool):
        rate = float(r["anneal"]["rate"])
        held = held + 1 if rate == 0 else 0
        if held and (frozen is None or held > frozen["epochs"]):
            at = cool[i - held + 1]["anneal"]
            frozen = {
                "epochs": held,
                "step": int(cool[i - held + 1]["step"]),
                "clock": round(float(at["clock"]), 1),
                "temperature": round(max(at["temperature"]), 3),
                "coupling": round(max(at["coupling"]), 2),
            }
        if rate < 0 and (i == 0 or float(cool[i - 1]["anneal"]["rate"]) >= 0):
            reheats += 1
            reheat = int(r["step"]) if reheat is None else reheat
    return {
        "epochs": len(cool),
        "start": int(cool[0]["step"]),
        "jumps": len(jumps),
        "largest": round(max((move for _, move in moves), default=0.0), 2),
        "jump": jumps[0][1] if jumps else None,
        "fall": None if fall is None else int(cool[fall]["step"]),
        "order": order,
        "gap": gap,
        "frozen": frozen,
        "reheats": reheats,
        "reheat": reheat,
    }


def cool_summary(course: Mapping[str, Any] | None) -> str:
    """cool_course's reading on one line."""
    if course is None:
        return "no cool"
    frozen = course["frozen"]
    held = (
        "never frozen"
        if frozen is None
        else f"frozen {frozen['epochs']} epochs from step {frozen['step']} (clock {frozen['clock']:g}, "
        f"T {frozen['temperature']:g}, coupling {frozen['coupling']:g})"
    )
    order = "-" if course["order"] is None else f"{course['order']} ({course['gap']:+d} epochs)"
    return (
        f"cool {course['epochs']} epochs from step {course['start']}; {course['jumps']} jumps (largest "
        f"{course['largest']:g}, first at step {course['jump']}), screen fall at step {course['fall']}: "
        f"{order}; {held}; {course['reheats']} reheats (first at step {course['reheat']})"
    )


def transitions_course(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """
    A run's stages of crystallisation, holds and reheats, read from its log (the incremental reheat
    workplan's stage 0): each stage as `Backtrack.recorded` keeps it (D1: an epoch of the cool at which a
    factor's code groups pass their running maximum), with its step, clock, temperatures, groups and
    the state share last read; each hold (a stretch of the clock stopped, D49), its start, epochs and
    whether the clock moved forward after it; each reheat (a stretch of it running back), how far back,
    the stages it crossed and whether it passed the first. None for a run with no cool.
    """
    cool = [r for r in rows if r.get("anneal", {}).get("phase") == "cool"]
    if not cool:
        return None
    share: dict[int, float] = {}
    for r in rows:
        if "accuracy" in r and "state_share" in r:
            share[int(r["step"])] = float(r["state_share"])
    backtrack = Backtrack()
    stages: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    reheats: list[dict[str, Any]] = []
    hold: dict[str, Any] | None = None
    back: dict[str, Any] | None = None
    last_share = 0.0
    for r in cool:
        a = r["anneal"]
        step, clock, rate = int(r["step"]), float(a["clock"]), float(a["rate"])
        last_share = share.get(step, last_share)
        before = len(backtrack.stages)
        backtrack = backtrack.recorded(a["groups"], clock, True)
        if len(backtrack.stages) > before:
            stages.append(
                {
                    "step": step,
                    "clock": round(clock, 3),
                    "temperature": [None if t is None else round(t, 3) for t in a["temperature"]],
                    "groups": a["groups"],
                    "share": last_share,
                }
            )
        if rate == 0 and hold is None:
            hold = {"step": step, "clock": round(clock, 3), "stages": len(stages), "epochs": 0, "recovered": False}
        if hold is not None:
            if rate > 0:
                holds.append(dict(hold, recovered=True))
                hold = None
            else:
                hold["epochs"] += 1
        if rate < 0 and back is None:
            back = {"step": step, "from": clock, "to": clock}
        if back is not None:
            back["to"] = min(back["to"], clock)
            if rate > 0:
                reheats.append(back)
                back = None
    holds += [hold] if hold is not None else []
    reheats += [back] if back is not None else []
    first = backtrack.stages[0][0] if backtrack.stages else None
    for b in reheats:
        b["crossed"] = sum(1 for st in backtrack.stages if st[0] > b["to"])
        b["past_first"] = first is not None and b["to"] < first
    return {"stages": stages, "holds": holds, "reheats": reheats}


def transitions_summary(course: Mapping[str, Any] | None) -> str:
    """transitions_course's reading on one line."""
    if course is None:
        return "no cool"
    stages, holds, reheats = course["stages"], course["holds"], course["reheats"]
    last = stages[-1] if stages else None
    recovered = [h["epochs"] for h in holds if h["recovered"]]
    unrecovered = [h["epochs"] for h in holds if not h["recovered"]]
    deep = sum(1 for b in reheats if b["past_first"])
    return (
        f"{len(stages)} stages (peak share {max((s['share'] for s in stages), default=0.0):g}"
        + (f", last at step {last['step']} T {last['temperature']}" if last is not None else "")
        + f"); {len(holds)} holds (recovered within {max(recovered, default=0)} epochs; "
        f"{len(unrecovered)} not, longest {max(unrecovered, default=0)}); {len(reheats)} reheats "
        f"({max((b['crossed'] for b in reheats), default=0)} stages crossed at most, {deep} past the first)"
    )


def same_prefix(
    rows: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]], ignore: Collection[str] = ()
) -> dict[str, Any]:
    """
    A paired arm's integrity against its control (D61, D63): how many leading rows are the control's
    (`same`), but for `seconds` and the `anneal` fields named in `ignore` (an arm's own), the rows before
    the run's cool (`before_cool`, its first row in the phase "cool"), the first differing row's step
    (`differs`, None if neither log has one past the other) and whether every row before the cool is
    the control's (`intact`): an arm acting only in the cool keeps it; D63's reheat differs at its first
    reheat at the earliest.
    """

    def kept(r: Mapping[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in r.items() if k != "seconds"}
        if "anneal" in out:
            out["anneal"] = {k: v for k, v in out["anneal"].items() if k not in ignore}
        return out

    same = 0
    for mine, theirs in zip(rows, control):
        if kept(mine) != kept(theirs):
            break
        same += 1
    before_cool = next((i for i, r in enumerate(rows) if r.get("anneal", {}).get("phase") == "cool"), len(rows))
    differs = None if same >= min(len(rows), len(control)) else rows[same].get("step")
    if differs is None and len(rows) != len(control):
        differs = (rows if len(rows) > len(control) else control)[same].get("step")
    return {"same": same, "before_cool": before_cool, "differs": differs, "intact": same >= before_cool}


def answer_profile(
    machine: wt.Mealy, rng: random.Random, depth: int = 6, words: int = 6000, length: int = 40
) -> list[float]:
    """
    The answer's entropy in bits given the last k tokens, k = 0..depth, over uniform words (answers
    only: the words and their answers, nothing of the machine's structure): how fast recent input
    removes the answer's uncertainty (the order note in the state formation workplan).
    """
    counts: list[dict[tuple[int, ...], Counter[int]]] = [defaultdict(Counter) for _ in range(depth + 1)]
    for _ in range(words):
        word = [rng.randrange(machine.tokens) for _ in range(length)]
        answers = machine.run(word)
        for t in range(depth, length):
            for k in range(depth + 1):
                counts[k][tuple(word[t - k + 1 : t + 1]) if k else ()][answers[t]] += 1
    profile: list[float] = []
    for by_context in counts:
        total = sum(sum(c.values()) for c in by_context.values())
        profile.append(
            sum(-sum(n / total * math.log2(n / sum(c.values())) for n in c.values()) for c in by_context.values())
        )
    return profile


def order_against_random(
    machine: wt.Mealy, states: int, ensemble: int = 30, seed: int = 0, depth: int = 6
) -> dict[str, Any]:
    """
    A machine's order as its answer profile's distance from random machines' (a relative entropy
    against randomness): `ensemble` random machines with `states` states (the state bound) and the
    machine's tokens (token 0 the identity) and classes, drawn as random-q's; `z` per k, its root mean
    square (`rms_z`: about 0 for a machine that looks random), and the uncertainty the last `depth`
    tokens remove as a share (`drop`) against the ensemble's (`drop_z`: negative where the past is
    conserved, a group; positive where it is forgotten, a reset machine).
    """
    rng = random.Random(f"order-{seed}")

    def drawn(i: int) -> wt.Mealy:
        draw = random.Random(f"order-{states}-{machine.tokens}-{machine.classes}-{seed}-{i}")
        while True:
            transition = np.array(
                [[s, *(draw.randrange(states) for _ in range(machine.tokens - 1))] for s in range(states)],
                dtype=np.int64,
            )
            if len(wt.Mealy(transition, transition).reachable()) == states:
                return wt.random_labels(transition, machine.classes, draw)

    profile = np.array(answer_profile(machine, rng, depth))
    profiles = np.array([answer_profile(drawn(i), rng, depth) for i in range(ensemble)])
    mean, spread = profiles.mean(0), profiles.std(0) + 1e-9
    z = (profile - mean) / spread
    drop = (profile[0] - profile[-1]) / profile[0] if profile[0] > 0 else 0.0
    drops = (profiles[:, 0] - profiles[:, -1]) / profiles[:, 0]
    return {
        "profile": [round(float(h), 3) for h in profile],
        "ensemble": [round(float(h), 3) for h in mean],
        "z": [round(float(v), 2) for v in z],
        "rms_z": round(float(np.sqrt((z**2).mean())), 2),
        "drop": round(float(drop), 3),
        "drop_z": round(float((drop - drops.mean()) / (drops.std() + 1e-9)), 2),
    }


ANSWERS_LENGTHS = (8, 16, 24, 48)  # a snapped network's answers re-read at these lengths (answers_stop)


def snapped_answers(net: Any, encoding: Target, lengths: Sequence[int] = ANSWERS_LENGTHS) -> dict[str, float]:
    """
    A saved network's certified answer's accuracy at each length (as long_accuracy), read uncoupled: a
    loaded network carries no coupling, so these are its snapped machine's answers, not a soft run's.
    """
    result: dict[str, float] = {}
    for n in lengths:
        tally = accuracy(net, encoding, windows_drawn(encoding, n, Settings().cases, f"answers-stop-{n}"))
        q = [tally[k] for k in tally if encoding.certified(k)]
        result[str(n)] = round(sum(c for c, _ in q) / sum(a for _, a in q), 4)
    return result


def answers_stop(
    rows: Sequence[Mapping[str, Any]],
    bound: float,
    snapped: Callable[[int], Mapping[str, float] | None] | None = None,
    strict: Callable[[int], bool | None] | None = None,
) -> dict[str, Any]:
    """
    An answers-only stop, read offline from a run's log.jsonl (the reasoning review's item 1): the first
    step whose answers are all right (the step's own and the long lengths') and whose machine, read off
    and minimised, has at most `bound` states (a state bound, as discovery supplies). The certificate is
    read only to judge the stop: "kept" (the stop's network certified), "strict-only" (not certified
    with near ties branched, but exact at snap ratio 1.0, the network as it computes: the margin's
    case, D59; read from the row's `certificates` where logged), "false-stop" (an uncertified
    network, in a run that certified later), "false-discovery" (in a run that never certified),
    "missed" (the run certified before the rule passed) or "silent" (neither). `snapped(step)` replaces
    the logged answers with a saved network's (snapped_answers) for runs whose logged answers are a
    soft network's (option 1's coupling); None where the step's network wasn't saved. `strict(step)`
    the step's saved network's strict verdict where its row logs none (a cell that certifies once, the
    `paper` cell's: without it, "strict-only" read as "false-stop"); None where it wasn't saved.
    """
    ends = [row for row in rows if "end" in row and "certified" in row]
    certified = next((int(row["step"]) for row in ends if row["certified"]), None)
    for row in ends:
        if row["minimised"] <= bound:
            answers = (
                {**row.get("accuracy", {}), **row.get("long", {})} if snapped is None else snapped(int(row["step"]))
            )
            if answers and all(v == 1.0 for v in answers.values()):
                logged = row.get("certificates", {}).get("strict", {}).get("certified")
                verdict = (
                    "kept"
                    if row["certified"]
                    else "strict-only"
                    if (logged if logged is not None or strict is None else strict(int(row["step"])))
                    else "false-stop"
                    if certified is not None
                    else "false-discovery"
                )
                return {"verdict": verdict, "stop": int(row["step"]), "certified": certified}
        if row["certified"]:
            return {"verdict": "missed", "stop": None, "certified": certified}
    return {"verdict": "silent", "stop": None, "certified": certified}


def answers_stopping(
    net: Any,
    encoding: Target,
    task: wt.Task,
    certificate: wc.Certificate,
    bound: int,
    coupled: Sequence[float],
    warm: bool,
) -> tuple[bool, dict[str, Any]]:
    """
    D35's stop at a step's end: whether the run stops (the snapped machine's answers all right at
    ANSWERS_LENGTHS and its minimised machine within `bound` states, never during the warm-up, whose
    factors run continuous), and the row's `answers_stop` (the answers, the verdict and, on a stop,
    the strict certificate at snap ratio 1.0 beside the step's own). A coupled network's answers are
    read with every coupling at 1 (its snapped machine's), then its couplings restored.
    """
    if warm:
        return False, {"answers_stop": {"passed": False, "warm": True}}
    if coupled:
        net.set_couplings([1.0] * len(coupled))
    answers = snapped_answers(net, encoding)
    if coupled:
        net.set_couplings(list(coupled))
    passed = len(certificate.minimised) <= bound and all(v == 1.0 for v in answers.values())
    row: dict[str, Any] = {"answers": answers, "passed": passed}
    if passed:
        row["strict"] = wc.certify(QueryMachine(net, encoding, 0), task, snap_ratio=1.0).certified
    return passed, {"answers_stop": row}


@dataclass
class Margin:
    """
    The margin stop's state (D59, D60): its wait W, D35's first pass with a split or label tie, and
    the pass with the fewest since (its ties).
    """

    wait: int | None
    first_pass: int = -1
    best_pass: int = -1
    best_ties: int = -1
    waited: int = -1  # the step the wait ended at

    def judge(
        self, step: int, stopping: bool, certificate: wc.Certificate, judged: dict[str, Any], net: Any, out: Path
    ) -> bool:
        """
        Whether D35's stop, `stopping`, stops the run at `step` (D59): only with no split or label tie,
        the first such pass's network and each with fewer ties than any before kept (D60); or at the
        wait's end, every step past the first pass counted (the run's last row then the best pass's: `stop_row`).
        """
        if stopping:
            ties = certificate.split_ties + certificate.lead.short
            if ties and self.first_pass < 0:
                self.first_pass = step
                net.save(str(out / f"first-pass-step-{step}.network.json"))
            if ties and (self.best_pass < 0 or ties < self.best_ties) and not self.due(step):
                self.best_pass, self.best_ties = step, ties  # D60: the pass the wait would end on
                net.save(str(out / f"best-pass-step-{step}.network.json"))
            margins = {"split_ties": certificate.split_ties, "lead_ties": certificate.lead.short}
            judged["answers_stop"].update(margins, first_pass=self.first_pass, **({"waited": False} if ties else {}))
            stopping = not ties
        if self.due(step):
            judged["answers_stop"].update(waited=True, best_pass=self.best_pass)
            self.waited = step
            return True
        return stopping

    def due(self, step: int) -> bool:
        """
        Whether the wait ends at `step` (D60): W steps past the first pass, every step counted (no pass
        cleared before it: a clean pass stops the run).
        """
        return self.wait is not None and self.first_pass >= 0 and step - self.first_pass >= self.wait

    def stop_row(
        self,
        out: Path,
        encoding: Target,
        settings: Settings,
        bound: int,
        coupled: Sequence[float],
        readings: bool,
    ) -> dict[str, Any]:
        """
        The wait's stop (D60): the best pass's network restored, kept as the stop's
        (answers-stop-step-S), and its row, judged as a step's end (with D53's `readings`).
        """
        net = load_network(str(out / f"best-pass-step-{self.best_pass}.network.json"))
        net.save(str(out / f"answers-stop-step-{self.waited}.network.json"))
        certificate = wc.certify(QueryMachine(net, encoding, 0), encoding.task)
        judged = answers_stopping(net, encoding, encoding.task, certificate, bound, coupled, False)[1]
        margins = {"split_ties": certificate.split_ties, "lead_ties": certificate.lead.short}
        judged["answers_stop"].update(margins, first_pass=self.first_pass, waited=True, best_pass=self.best_pass)
        if readings:
            judged.update(continuous_readings(net, encoding, encoding.task, certificate, coupled, False))
        return {
            "step": self.waited,
            "end": "waited",
            "restored": self.best_pass,
            **diagnose(net, encoding, settings),
            "long": long_accuracy(net, encoding, settings),
            "certified": certificate.certified,
            "minimised": len(certificate.minimised),
            "machine": machine_summary(net, encoding),
            **judged,
        }


def continuous_readings(
    net: Any,
    encoding: Target,
    task: wt.Task,
    certificate: wc.Certificate,
    coupled: Sequence[float],
    answers: bool,
) -> dict[str, Any]:
    """
    D53's readings at a step's end (the continuous crystallisation's every epoch): the step's
    certificate and the strict one (snap ratio 1.0) each as its verdict, wrong states, wrong outputs,
    those wrong only on a near tie, near ties and states minimised (`certificates`); with `answers`,
    the snapped machine's answers (as answers_stopping reads them: every coupling at 1, then restored).
    """
    strict = wc.certify(QueryMachine(net, encoding, 0), task, snap_ratio=1.0)
    row: dict[str, Any] = {
        "certificates": {
            name: {
                "certified": c.certified,
                "wrong_states": c.wrong_states,
                "wrong_outputs": c.wrong_outputs,
                "wrong_ties": c.wrong_ties,
                "near_ties": c.snap.short,
                "minimised": len(c.minimised),
            }
            for name, c in (("default", certificate), ("strict", strict))
        }
    }
    if answers:
        if coupled:
            net.set_couplings([1.0] * len(coupled))
        row["snapped_answers"] = snapped_answers(net, encoding)
        if coupled:
            net.set_couplings(list(coupled))
    return row


DROP_FROM, DROP_BY = 0.9, 0.3
RECOVERED = 0.05  # a drop recovered: a later point's worst answer back within this of its level before


def _worst(row: Mapping[str, Any]) -> float:
    return min(row["accuracy"].values())


def large_drops(diagnoses: Sequence[Mapping[str, Any]]) -> list[tuple[int, bool]]:
    """
    A run's large drops (division stage 5's trace): its worst answer from at least DROP_FROM down by
    more than DROP_BY between two diagnoses, each as the step it fell in and whether a later diagnosis
    is back at DROP_FROM.
    """
    falls: list[tuple[int, bool]] = []
    for i, (before, after) in enumerate(itertools.pairwise(diagnoses)):
        if _worst(before) >= DROP_FROM and _worst(after) < _worst(before) - DROP_BY:
            falls.append((int(after["step"]), any(_worst(row) >= DROP_FROM for row in diagnoses[i + 2 :])))
    return falls


@dataclass(frozen=True)
class TriggerScore:
    """A trigger read against one run's log (the p11-trigger workplan, D6)."""

    points: int  # the logged points scored: each epoch's screen with its kinds, or else each diagnosis
    drops: int  # large drops: the worst answer kind from DROP_FROM down by more than DROP_BY between points
    caught: int  # drops at whose point the trigger fires
    unrecovered: int  # drops after which no point is back within RECOVERED of the level before it
    unrecovered_caught: int
    false_alarms: int  # firings near a solution (the worst kind at DROP_FROM before) at no large drop
    early: int  # firings while still learning (the worst kind below DROP_FROM before): a restore, a cooler rate


def score_trigger(rows: Sequence[Mapping[str, Any]], trigger: Trigger) -> TriggerScore:
    """
    A trigger against a run's log, offline: at each logged point past a step's first, its signal
    against the step's peak of it over the points before (D6). A log with each epoch's answer kinds
    (D4) is scored every epoch, an older one at its diagnoses (every few epochs, on the diagnosis's
    cases: a proxy). A fork arm's log is altered by its forks: score runs made without one.
    """
    epochs = [r for r in rows if "kinds" in r and "fork" not in r and r["step"] > 0]
    points = (
        [(r, r["kinds"]) for r in epochs]
        if epochs
        else [(r, r["accuracy"]) for r in rows if "accuracy" in r and isinstance(r.get("epoch"), int) and r["step"] > 0]
    )
    worst = [min(kinds.values()) for _, kinds in points]
    drops = caught = unrecovered = unrecovered_caught = alarms = early = 0
    for i in range(1, len(points)):
        row, kinds = points[i]
        prior = [trigger.value(r.get("screen", 0.0), k) for r, k in points[:i] if r["step"] == row["step"]]
        fires = bool(prior) and trigger.fires(trigger.value(row.get("screen", 0.0), kinds), max(prior))
        if worst[i - 1] >= DROP_FROM and worst[i] < worst[i - 1] - DROP_BY:
            lost = not any(w >= worst[i - 1] - RECOVERED for w in worst[i + 1 :])
            drops, caught = drops + 1, caught + fires
            unrecovered, unrecovered_caught = unrecovered + lost, unrecovered_caught + (fires and lost)
        elif worst[i - 1] >= DROP_FROM:
            alarms += fires
        else:
            early += fires
    return TriggerScore(len(points), drops, caught, unrecovered, unrecovered_caught, alarms, early)


NEAR = 0.95  # a run's worst answer at least this at some diagnosis: it came near a solution
VERDICTS = (wc.SNAP_RATIO, 1.0)  # a certified network's two recertifications: the default snap ratio and strict


@dataclass(frozen=True)
class ArmRun:
    """One run of a registered comparison's arm: its seed, its course (Outcome), and whether it counts
    as certified (its saved network recertified in both verdicts), with the verdicts found; and, when
    long tests are read, whether its network passed every case (None: no long test found). A run
    with no codes (a continuous state) has no certificate: its network at its budget's end is long-tested."""

    seed: int
    outcome: Outcome
    certified: bool
    verdicts: dict[float, bool]
    coded: bool = True
    long: bool | None = None

    @property
    def succeeded(self) -> bool:
        """With long tests read: certified (coded) and long-tested, or long-tested alone (no codes)."""
        return (self.certified or not self.coded) and self.long is True


def recertifications(certify: Path) -> dict[str, dict[float, bool]]:
    """The recertifications in `certify` (*.certificate.json): each network's sha256 -> {snap ratio: certified}."""
    verdicts: dict[str, dict[float, bool]] = {}
    for path in sorted(certify.glob("*.certificate.json")):
        result = json.loads(path.read_text())
        verdicts.setdefault(result["sha256"], {})[result["snap_ratio"]] = bool(result["certified"])
    return verdicts


def long_tested(long_tests: Path) -> dict[str, bool]:
    """The long tests under `long_tests` (*/*.long-test.json): each network's sha256 -> every case passed."""
    passed: dict[str, bool] = {}
    for path in sorted(long_tests.glob("*/*.long-test.json")):
        long = json.loads(path.read_text())
        passed[long["sha256"]] = not wr.long_test_failures(long)[1]
    return passed


def arm_runs(
    root: Path,
    arm: str,
    verdicts: Mapping[str, Mapping[float, bool]],
    long: Mapping[str, bool] | None = None,
) -> list[ArmRun]:
    """An arm's runs, root/<arm>/seed<s>/log.jsonl in seed order, each certified only when its saved network
    (matched by sha256) is recertified in both verdicts (VERDICTS); with `long` (long_tested), each
    certified network's long test, or a run without codes its network.json's (its budget's end)."""
    runs: list[ArmRun] = []
    for log in sorted((root / arm).glob("seed*/log.jsonl"), key=lambda p: int(p.parent.name.removeprefix("seed"))):
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        run = outcome(rows)
        coded = any("factors" in row for row in rows)
        found: dict[float, bool] = {}
        tested: Path | None = None if coded else log.parent / "network.json"
        if run.end == "certified":
            tested = log.parent / f"certified-step-{run.step}.network.json"
            found = dict(verdicts.get(hashlib.sha256(tested.read_bytes()).hexdigest(), {}))
        certified = run.end == "certified" and all(found.get(ratio) for ratio in VERDICTS)
        passed = None
        if long is not None and tested is not None and tested.exists():
            passed = long.get(hashlib.sha256(tested.read_bytes()).hexdigest())
        runs.append(ArmRun(int(log.parent.name.removeprefix("seed")), run, certified, found, coded, passed))
    return runs


def readout(
    root: Path,
    arms: Sequence[str],
    certify: Path,
    pairs: Sequence[tuple[str, str]] | None = None,
    long_tests: Path | None = None,
) -> list[str]:
    """
    A registered comparison's readout (the division workplan's stage 5): each run in root/<arm>/seed<s>
    by its log, a run counted as certified only when its saved network (matched by sha256) has a
    certificate in `certify` in both verdicts (VERDICTS); each arm's count with its Wilson 95%
    interval; for each pair (better, other), Fisher's exact test, one-sided, that the first arm
    certifies more often than the second (by default the first two arms of two), and with more than
    one pair each p Holm-adjusted over the pairs (the approximation claim's D5).
    Each run's course too (Outcome), and how many of an arm's failures came near a solution (NEAR):
    a failure near one lost it, one never near never found it. With `long_tests` (long-test's outputs,
    <run>/<network>.long-test.json), a run counts only when it succeeded (ArmRun.succeeded: certified
    and every long-test case passed; a run with no codes, its last network passing every case), and
    a certified network not yet long-tested is marked.
    """
    verdicts = recertifications(certify)
    long = long_tested(long_tests) if long_tests is not None else None
    counted = "certified" if long is None else "succeeded"
    lines: list[str] = []
    counts: dict[str, tuple[int, int]] = {}
    for arm in arms:
        seeds = arm_runs(root, arm, verdicts, long)
        certified, near, collapsed, drops, unrecovered, dropped = 0, 0, 0, 0, 0, 0
        lines.append(f"== {arm}")
        for entry in seeds:
            run = entry.outcome
            note = ""
            if run.end == "certified" and not entry.certified:
                note = f"  NOT RECERTIFIED (verdicts found: {entry.verdicts})"
            if long is not None and (entry.certified or not entry.coded):
                note += {True: "; long tests passed", False: "; LONG TEST FAILED", None: "; NOT LONG-TESTED"}[
                    entry.long
                ]
            certified += entry.certified if long is None else entry.succeeded
            if run.end != "certified":
                near += run.peak >= NEAR
                collapsed += run.collapses > 0
            drops, unrecovered, dropped = drops + run.drops, unrecovered + run.unrecovered, dropped + (run.drops > 0)
            lines.append(
                f"seed{entry.seed}: {run.end} at step {run.step}, {run.epochs} epochs, {run.minutes} min; "
                f"worst answer at best {run.peak:.3f} (step {run.peak_step}), collapses {run.collapses}, "
                f"large drops {run.drops} ({run.unrecovered} unrecovered){note}"
            )
        low, high = inference.wilson(certified, len(seeds))
        lines.append(f"{arm}: {certified} of {len(seeds)} {counted} (Wilson 95%: {low:.2f}-{high:.2f})")
        failures = len(seeds) - certified
        lines.append(
            f"{arm}'s {failures} failures: {near} came near a solution (worst answer >= {NEAR}), {collapsed} collapsed"
        )
        lines.append(f"{arm}: {drops} large drops in {dropped} runs, {drops - unrecovered} recovered")
        counts[arm] = (certified, len(seeds))
    if pairs is None:
        pairs = [(arms[0], arms[1])] if len(arms) == 2 else []
    ps: list[float] = []
    for better, other in pairs:
        (a, n), (c, m) = counts[better], counts[other]
        ps.append(inference.fisher_exact_greater(a, n - a, c, m - c))
    adjusted = inference.holm(ps)
    for (better, other), p, held in zip(pairs, ps, adjusted, strict=True):
        line = f"Fisher's exact, one-sided ({better} more often than {other}): p {p:.3g}"
        lines.append(line if len(pairs) == 1 else f"{line}, Holm-adjusted over {len(pairs)}: {held:.3g}")
    return lines


MACHINE_UNIT = re.compile(r"^(?P<test>.+)-(?P<arm>[^-]+)-(?P<machine>\d+)$")  # <test>-<arm>-<machine>, T10's backdrops


def machine_unit_runs(
    root: Path,
    test: str,
    arms: Sequence[str],
    verdicts: Mapping[str, Mapping[float, bool]],
    long: Mapping[str, bool],
) -> dict[str, dict[int, list[ArmRun]]]:
    """
    A test whose unit is the machine (the paper workplan's D19, D24: T10, T10r): each arm's machines,
    each with its runs (arm_runs of root/<test>-<arm>-<machine>/seed<s>), read with their
    recertifications and long tests; a run counts when it succeeded (ArmRun.succeeded).
    """
    found: dict[str, dict[int, list[ArmRun]]] = {arm: {} for arm in arms}
    for directory in sorted(root.iterdir()):
        named = MACHINE_UNIT.match(directory.name)
        if named is None or named["test"] != test or named["arm"] not in found:
            continue
        found[named["arm"]][int(named["machine"])] = arm_runs(root, directory.name, verdicts, long)
    return found


def machine_successes(machines: Mapping[int, Sequence[ArmRun]], cap: int | None = None) -> list[int]:
    """Each machine's successful runs, in machine order; with `cap`, each count at most cap (D24's sensitivity)."""
    counts = [sum(r.succeeded for r in machines[m]) for m in sorted(machines)]
    return counts if cap is None else [min(c, cap) for c in counts]


def machine_unit_readout(
    runs: Mapping[str, Mapping[int, Sequence[ArmRun]]], pairs: Sequence[tuple[str, str]]
) -> list[str]:
    """
    A machine-unit test's reading (D19, D24): each arm's machines succeeding, its runs succeeding, and
    each machine's count; for each pair (first, second), the exact one-sided rank-sum test over the
    machines' counts that the first tends larger (midranks), Holm-adjusted over the pairs; and each
    p again with every machine's count capped at one (whether a machine succeeding twice decides it).
    """
    lines: list[str] = []
    for arm, machines in runs.items():
        counts = machine_successes(machines)
        every = sum(len(r) for r in machines.values())
        each = ", ".join(f"{m}: {c}" for m, c in zip(sorted(machines), counts, strict=True) if c)
        lines.append(
            f"{arm}: {sum(1 for c in counts if c)} of {len(counts)} machines, {sum(counts)} of {every} runs ({each or 'none'})"
        )
    ps = [inference.rank_sum_greater(machine_successes(runs[a]), machine_successes(runs[b])) for a, b in pairs]
    capped = [
        inference.rank_sum_greater(machine_successes(runs[a], 1), machine_successes(runs[b], 1)) for a, b in pairs
    ]
    for (a, b), p, adjusted, once in zip(pairs, ps, inference.holm(ps), capped, strict=True):
        lines.append(f"{a} > {b}: rank-sum p {p:.4f}, Holm {adjusted:.4f}; each machine at most once: p {once:.4f}")
    return lines


def long_summary(root: Path) -> list[str]:
    """
    A study's long tests (long-test --out root/<run>, a <network>.long-test.json each): a line per
    run (its cases passed, the failures listed, the network's sha256 prefix) and one for all.
    """
    lines: list[str] = []
    passed = cases = 0
    failing: list[str] = []
    for path in sorted(root.glob("*/*.long-test.json")):
        long = json.loads(path.read_text())
        results = [r for length in long["lengths"].values() for r in length["properties"].values()]
        run_cases, run_passed = sum(r["cases"] for r in results), sum(r["passed"] for r in results)
        _, failures = wr.long_test_failures(long)
        listed = sum(len(r.get("failures", [])) for r in results)
        lines.append(
            f"{path.parent.name}: {run_passed:,} of {run_cases:,} passed"
            + (f" ({', '.join(failures)})" if failures else "")
            + f", {listed} failures listed, sha256 {long['sha256'][:12]}"
        )
        cases, passed = cases + run_cases, passed + run_passed
        if failures:
            failing.append(path.parent.name)
    lines.append(f"all: {passed:,} of {cases:,} passed; networks with a failure: {', '.join(failing) or 'none'}")
    return lines


STALL_SHARE, STALL_EPOCHS = 0.95, 10  # a guarded run frozen (the guard workplan's D9)


def scaled_factors(q: int, ratio: Fraction, count: int, width: int = 16) -> list[tuple[int, int]]:
    """
    count factors of width, each ceil(ratio sqrt(q)) codes for a target of q states, computed exactly:
    the surplus held at ratio (the division ladder's D2, ceil(ratio p) for division's p * p states;
    the general algebras' D3).
    """
    bound = ratio * ratio * q
    codes = math.isqrt(math.floor(bound))
    while codes * codes < bound:
        codes += 1
    return [(width, codes)] * count


def exclusion_bound(q: int, factors: Sequence[tuple[int, int]]) -> int:
    """`m + q - 1` (experiment-design.md, section 1): the joint codes plus the target's q states, less one."""
    return math.prod(m for _, m in factors) + q - 1


def first_length(machine: wt.Mealy) -> int:
    """
    `s0`, the curriculum's first step (the code-consolidation workplan's D3): the least n at which the
    answers of every training word of lengths 1..n (n tokens, then token 0, as `draw` makes them)
    aren't a function of the token alone, so a network with no state can't pass the first step.
    Exact, from the states reached at each depth: a word's i-th token leaves a state reached in i - 1
    tokens, its final token 0 one reached in n. 1 for a machine whose answers never need the state.
    """
    seen: list[set[int]] = [set() for _ in range(machine.tokens)]  # each token's answers so far
    level = {machine.start}  # the states n - 1 tokens reach
    for n in range(1, machine.states + 2):  # by n = q + 1 every reachable state has been a level
        for s in level:
            for t in range(machine.tokens):
                seen[t].add(int(machine.output[s, t]))
        level = {int(machine.transition[s, t]) for s in level for t in range(machine.tokens)}
        seen[0].update(int(machine.output[s, 0]) for s in level)
        if any(len(answers) > 1 for answers in seen):
            return n
    return 1


def epoch_lengths(step: int, examples: int, long_tail: int | None = None, cap: int | None = None) -> list[int]:
    """
    Each of an epoch's examples' lengths at a step: 1..step in turn; with `long_tail` F (the
    workplan's D4, B), every other example's in turn over step + 1..F step instead, capped at `cap`
    (all of them 1..step when that's empty), so half the examples are the curriculum's as before.
    """
    short = range(1, step + 1)
    if long_tail is None:
        return [short[i % len(short)] for i in range(examples)]
    long = range(step + 1, min(long_tail * step, cap if cap is not None else long_tail * step) + 1)
    if not long:
        return [short[i % len(short)] for i in range(examples)]
    return [short[i // 2 % len(short)] if i % 2 == 0 else long[i // 2 % len(long)] for i in range(examples)]


def whole_lengths(cap: int, points: int = 10) -> list[int]:
    """
    The screen's lengths when every epoch trains on the whole distribution (run's `whole`): 1 to 4 and
    `points` more spaced geometrically to `cap`, its last.
    """
    spread = {1, 2, 3, 4, cap} | {round(cap ** (i / points)) for i in range(points + 1)}
    return sorted(n for n in spread if n <= cap)


def consolidation(net: Any, encoding: Target, d: int) -> tuple[dict[tuple[int, ...], tuple[int, ...]], int, int]:
    """
    A's table from the network alone (the workplan's D1): its reachable joint codes grouped by
    d-equivalence over every slot's outputs, each duplicate to its group's representative (the code
    its shortest word reaches; machine_reading.groups); with the codes reached and the groups.
    """
    walk = mr.walk(QueryMachine(net, encoding, None), encoding.task.machine.tokens)
    found = mr.groups(walk, d)
    return mr.consolidation(walk, found), len(walk.codes), len(found)


def machine_summary(net: Any, encoding: Target) -> dict[str, int]:
    """The network's machine walked beside the target's, in three counts (machine_reading.summary), d the task's."""
    machine = encoding.task.machine
    return mr.summary(mr.product(QueryMachine(net, encoding, 0), machine), mr.word_length(machine))


CHECKPOINT = "checkpoint.json"


@dataclass(frozen=True)
class Thermostat:
    """
    The commitment-control workplan's usage thermostat (stage 2, D1 (a), D2'): each factor's
    commitment off while its effective codes (effective_codes) are below `floor`; once they reach
    `restore` of its codebook, ramped back to its specs' over `ramp` epochs; off again only below
    `floor` (the gap between the two is the hysteresis). A factor's level is 0 (off) to `ramp` (its
    specs' commitment). With `restore_screen` (D2''), a factor off is restored once the step's best
    screen reaches it, in place of half its codebook: the answers right, so a state has formed.
    """

    floor: float = 3.0
    restore: float = 0.5
    ramp: int = 20
    restore_screen: float | None = None

    def levels(
        self, levels: Sequence[int], effective: Sequence[float], counts: Sequence[int], screen: float = 0.0
    ) -> list[int]:
        """Each factor's next level from its level, its effective codes, its codebook's size and the step's best screen."""
        following: list[int] = []
        for level, codes, count in zip(levels, effective, counts, strict=True):
            if codes < self.floor:
                following.append(0)
            elif level > 0:
                following.append(min(level + 1, self.ramp))
            elif self.restore_screen is not None:
                following.append(1 if screen >= self.restore_screen else 0)
            else:
                following.append(1 if codes >= self.restore * count else 0)
        return following

    def commitments(self, levels: Sequence[int], full: Sequence[float]) -> list[float]:
        """Each factor's commitment at its level, a share of its specs'."""
        return [c * level / self.ramp for level, c in zip(levels, full, strict=True)]


def effective_codes(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]]) -> list[float]:
    """
    Each factor's effective codes, 2 ** H of its codes' use over every window's next state of the
    rows given (lists of windows, one length each): the thermostat's signal (the network's own,
    never the target's).
    """
    counts: list[dict[int, int]] = []
    for batch in rows:
        codes: list[list[tuple[int, ...]]] = net.state_codes(net.backend.matrix([tokens for tokens, _ in batch]))
        for states in codes:
            for state in states[1:]:
                if not counts:
                    counts = [{} for _ in state]
                for g, code in enumerate(state):
                    counts[g][code] = counts[g].get(code, 0) + 1
    effective: list[float] = []
    for used in counts:
        total = sum(used.values())
        entropy = -sum(n / total * math.log2(n / total) for n in used.values())
        effective.append(2**entropy)
    return effective


@dataclass(frozen=True)
class Anneal:
    """
    The commitment-control workplan's annealing (stage 3, the soft-to-hard quantiser; D6-D8): each
    snapped factor's codes soft in training at `start` halved every `halving` epochs trained, its
    commitment 0, until `share` of the sample's windows weight one code above `hard`; then soft for
    `tail` epochs more, cooling on (the state formation workplan's D19: the crystal settles), then
    hard for good, its commitment ramped back to its specs' over `ramp` epochs (never, without
    `commit`). A factor's level is 0
    (soft, not yet hard), 1 to `tail` (soft, in its tail), to `tail + ramp` (hard, its specs'
    commitment).
    """

    start: float = 1.0
    halving: int = 10
    hard: float = 0.99
    share: float = 0.99
    ramp: int = 20
    tail: int = 0
    commit: bool = True

    @property
    def top(self) -> int:
        """The last level: hard, at its specs' commitment."""
        return self.tail + self.ramp

    def soft(self, level: int) -> bool:
        """Whether a factor at this level trains soft."""
        return level <= self.tail

    def temperature(self, spent: int) -> float:
        """The soft factors' temperature after `spent` epochs trained in all."""
        return self.start * 0.5 ** (spent // self.halving)

    def levels(self, levels: Sequence[int], hardness: Sequence[float | None]) -> list[int]:
        """Each factor's next level from its level and its hardness (None: continuous, or not read)."""
        following: list[int] = []
        for level, hard in zip(levels, hardness, strict=True):
            if level > 0:
                following.append(min(level + 1, self.top))
            else:
                following.append(1 if hard is not None and hard >= self.share else 0)
        return following

    def commitments(self, levels: Sequence[int], full: Sequence[float]) -> list[float]:
        """Each factor's commitment at its level, a share of its specs' (0 throughout without `commit`)."""
        if not self.commit:
            return [0.0] * len(levels)
        return [c * max(level - self.tail, 0) / self.ramp for level, c in zip(levels, full, strict=True)]


def _soft_assignments(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], temperatures: Sequence[float | None]
) -> Iterator[tuple[int, list[tuple[int, float, np.ndarray, np.ndarray]]]]:
    """
    Every window's cell outputs over the rows given (lists of windows, one length each), a step at a
    time: the step's window count and, for each soft factor g at its temperature t (one not at None),
    (g, t, D, A), D each output's squared distances to the factor's codes and A the codes' softmax
    weights at t before normalising (training's, shifted by each row's least distance). hardness,
    sharpness and thermals read them (the DRY third pass workplan, D9).
    """
    starts = [sum(f.state_size for f in net.recurrence.factors[:g]) for g in range(len(net.recurrence.factors))]
    for batch in rows:
        _, _, cells = net.window_states(net.backend.matrix([tokens for tokens, _ in batch]))
        for cell in cells[1:]:
            H = np.array(cell.tolist(), dtype=float)
            soft: list[tuple[int, float, np.ndarray, np.ndarray]] = []
            for g, (factor, t) in enumerate(zip(net.recurrence_layer.factors, temperatures, strict=True)):
                if t is None:
                    continue
                E = np.array(factor.codes.E.tolist(), dtype=float)
                h = H[:, starts[g] : starts[g] + E.shape[1]]
                D = np.sum((h[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2)
                soft.append((g, t, D, np.exp(-(D - D.min(axis=1)[:, np.newaxis]) / t)))
            yield len(H), soft


def hardness(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], temperatures: Sequence[float | None], hard: float
) -> list[float | None]:
    """
    Each soft factor's share of every window's cell output, over the rows given (lists of windows, one
    length each), whose largest weight at its temperature (the codes' softmax, as training's) is above
    `hard`; None for a factor at None (annealing's test, D8: the network's own, never the target's).
    """
    above = [0] * len(temperatures)
    total = 0
    for windows, soft in _soft_assignments(net, rows, temperatures):
        total += windows
        for g, _, _, A in soft:
            above[g] += int(np.sum(A.max(axis=1) / A.sum(axis=1) > hard))
    return [None if t is None else above[g] / max(total, 1) for g, t in enumerate(temperatures)]


def sharpness(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], temperatures: Sequence[float | None], hard: float
) -> list[tuple[float, float] | None]:
    """
    Each soft factor's codes' sharpness over the rows given (the continuous crystallisation's D50): the
    mean over every window's cell output of its largest weight at its temperature (the codes' softmax,
    as training's), and D8's share of them above `hard` (hardness's, read every epoch, D53); None for
    a factor at None.
    """
    largest = [0.0] * len(temperatures)
    above = [0] * len(temperatures)
    total = 0
    for windows, soft in _soft_assignments(net, rows, temperatures):
        total += windows
        for g, _, _, A in soft:
            weights = A.max(axis=1) / A.sum(axis=1)
            largest[g] += float(np.sum(weights))
            above[g] += int(np.sum(weights > hard))
    n = max(total, 1)
    return [None if t is None else (largest[g] / n, above[g] / n) for g, t in enumerate(temperatures)]


def thermals(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], temperatures: Sequence[float | None]
) -> list[tuple[float, float] | None]:
    """
    Each soft factor's energy and heat capacity at its temperature over the rows given (D32's
    understanding step): the mean over windows of the soft assignment's distortion, `<D>` (Rose 1998's
    energy), and the mean of its variance under the assignment over `T^2`, `d<D>/dT` (its specific heat,
    peaking at a phase transition); None for a factor at None. Exact, no sampling; the target unread.
    """
    energy = [0.0] * len(temperatures)
    spread = [0.0] * len(temperatures)
    total = 0
    for windows, soft in _soft_assignments(net, rows, temperatures):
        total += windows
        for g, _, D, A in soft:
            P = A / A.sum(axis=1)[:, np.newaxis]
            mean = np.sum(P * D, axis=1)
            energy[g] += float(mean.sum())
            spread[g] += float(np.sum(np.sum(P * D**2, axis=1) - mean**2))
    n = max(total, 1)
    return [None if t is None else (energy[g] / n, spread[g] / n / (t * t)) for g, t in enumerate(temperatures)]


def _groups(close: np.ndarray) -> int:
    """How many groups a symmetric closeness matrix joins its items into (transitively)."""
    group = list(range(len(close)))

    def root(i: int) -> int:
        while group[i] != i:
            i = group[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(close, 1)), strict=True):
        group[root(int(i))] = root(int(j))
    return len({root(i) for i in range(len(close))})


def factor_states(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], live: bool = False) -> list[np.ndarray]:
    """
    Each factor's cell outputs `h` (windows by its state size) over every window of the rows given
    (`live`: continuous factors read unsnapped).
    """
    starts = [sum(f.state_size for f in net.recurrence.factors[:g]) for g in range(len(net.recurrence.factors))]
    blocks: list[list[np.ndarray]] = [[] for _ in starts]
    for batch in rows:
        _, _, cells = net.window_states(net.backend.matrix([tokens for tokens, _ in batch]), live=live)
        for cell in cells[1:]:
            H = np.array(cell.tolist(), dtype=float)
            for g, f in enumerate(net.recurrence.factors):
                blocks[g].append(H[:, starts[g] : starts[g] + f.state_size])
    return [np.concatenate(b) for b in blocks]


def melting_curve(
    h: np.ndarray,
    k: int,
    rng: np.random.Generator,
    per_octave: int = 4,
    octaves: int = 20,
    iterations: int = 30,
    sample: int = 2000,
) -> dict[str, Any]:
    """
    The states' melting curve (the state formation workplan's D21): Rose's deterministic annealing of
    `k` codes on the fixed outputs `h` (at most `sample` of them, drawn), from twice the whole's
    critical temperature `T_c = 2 lambda_max` down `octaves` octaves, `per_octave` temperatures an
    octave, `iterations` soft-assignment updates at each, the codes perturbed by `10^-6 sqrt(T_c)` so
    identical ones can split; at each temperature the codes' groups closer than `T / 100` squared
    (D20's). Returns `T_c`, the curve (temperature, groups) and its transitions (temperature, groups
    before, after), the highest the melting point: where the codes leave the melt. A split is seen
    late, up to about an octave below its temperature: just below a transition the codes part slowly
    (critical slowing down), within the updates a temperature gets.
    """
    if len(h) > sample:
        h = h[rng.choice(len(h), sample, replace=False)]
    critical = 2.0 * float(np.linalg.eigvalsh(np.cov(h, rowvar=False))[-1])
    scale = 1e-6 * math.sqrt(max(critical, 1e-300))
    codes = np.repeat(h.mean(axis=0)[np.newaxis, :], k, axis=0)
    curve: list[list[float]] = []
    transitions: list[list[float]] = []
    for i in range(per_octave * octaves + 1):
        t = 2.0 * critical * 2.0 ** (-i / per_octave)
        if t <= 0.0:
            break
        for _ in range(iterations):
            codes = codes + rng.normal(0.0, scale, size=codes.shape)
            D = np.sum((h[:, np.newaxis, :] - codes[np.newaxis, :, :]) ** 2, axis=2)
            weights = np.exp(-(D - D.min(axis=1)[:, np.newaxis]) / t)
            weights /= weights.sum(axis=1)[:, np.newaxis]
            mass = weights.sum(axis=0)
            codes = np.where(mass[:, np.newaxis] > 0, (weights.T @ h) / np.maximum(mass, 1e-300)[:, np.newaxis], codes)
        groups = _groups(np.sum((codes[:, np.newaxis, :] - codes[np.newaxis, :, :]) ** 2, axis=2) < t / 100)
        if curve and groups != curve[-1][1]:
            transitions.append([t, curve[-1][1], groups])
        curve.append([t, groups])
        if groups == k:
            break
    return {
        "critical_temperature": critical,
        "melting_point": transitions[0][0] if transitions else None,
        "curve": curve,
        "transitions": transitions,
    }


def melting_curves(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]]) -> list[dict[str, Any] | None]:
    """
    Each snapped factor's melting curve over every window's cell output of the rows given (D21; factor
    `g`'s draws from seed `g`, as at the warm-up's end); None unsnapped.
    """
    return [
        None if f.codes is None else melting_curve(h, f.codes.count, np.random.default_rng([g]))
        for g, (f, h) in enumerate(zip(net.recurrence_layer.factors, factor_states(net, rows), strict=True))
    ]


def target_states(machine: wt.Mealy, rows: Sequence[Sequence[tuple[Any, Any]]]) -> np.ndarray:
    """Each window's target state, the machine's after the word's tokens to it, in factor_states' order."""
    blocks: list[np.ndarray] = []
    for batch in rows:
        words = np.array([[int(t) for t in tokens] for tokens, _ in batch], dtype=np.int64)
        state = np.full(len(words), machine.start, dtype=np.int64)
        for j in range(words.shape[1]):
            state = machine.transition[state, words[:, j]]
            blocks.append(state.copy())
    return np.concatenate(blocks)


def state_separation(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], machine: wt.Mealy, sample: int = 3000, seed: int = 0
) -> dict[str, Any]:
    """
    Whether the network's cell outputs `h` determine the target machine's state (the state formation
    workplan's feasibility check: annealing clusters `h`, so it can find the target's states only where
    `h` already separates them). Over every window of the rows given (at most `sample`, drawn), for
    the joint `h` and each factor's: the leave-one-out nearest neighbour's and the nearest target
    centroid's accuracy at naming the window's target state, against the most common state's share
    (chance). Reads the network as set: continuous factors live (set_continuous first) for a warm-up's.
    The target is read, never trained on: a diagnosis.
    """
    states = factor_states(net, rows, live=True)
    targets = target_states(machine, rows)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(targets), min(sample, len(targets)), replace=False)
    targets = np.asarray(targets[chosen], dtype=np.int64)
    blocks = {"joint": np.concatenate(states, axis=1)[chosen]} | {
        f"factor-{g}": h[chosen] for g, h in enumerate(states)
    }
    found, counts = np.unique(targets, return_counts=True)
    values = np.asarray(found, dtype=np.int64)
    separation: dict[str, Any] = {
        "windows": len(targets),
        "states_seen": len(values),
        "chance": float(counts.max() / len(targets)),
    }
    for name, h in blocks.items():
        D = np.sum((h[:, np.newaxis, :] - h[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(D, np.inf)
        nearest: np.ndarray = np.argmin(D, axis=1)
        neighbour = np.count_nonzero(targets[nearest] == targets) / len(targets)
        centroids = np.array([h[targets == v].mean(axis=0) for v in values])
        to_centroid = np.sum((h[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        closest: np.ndarray = np.argmin(to_centroid, axis=1)
        centroid = np.count_nonzero(values[closest] == targets) / len(targets)
        separation[name] = {"neighbour": neighbour, "centroid": centroid}
    return separation


def state_spread(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], merged: float = 1e-6, share: float = 0.99
) -> list[dict[str, Any]]:
    """
    Each factor's spread over every window's cell output `h` of the rows given (the state formation
    workplan's D3: the network's own, never the target's): the top eigenvalue of `h`'s covariance and
    deterministic annealing's critical temperature, twice it (Rose 1998, Theorem 1); `h`'s effective
    dimension, the fewest principal components holding `share` of its variance (Zhao et al. 2026); the
    codes' pairwise squared distances and their groups closer than `merged` (identical codes); and,
    over the windows' nearest codes, the codes in use and the pooled within-cluster covariance's top
    eigenvalue (D7's `tau_0` is twice it). A factor without codes has its codes' entries None.
    """
    states = factor_states(net, rows)
    spread: list[dict[str, Any]] = []
    for g, factor in enumerate(net.recurrence_layer.factors):
        h = states[g]
        eigen = np.sort(np.linalg.eigvalsh(np.cov(h, rowvar=False)))[::-1].clip(min=0.0)
        held = np.cumsum(eigen) / max(float(eigen.sum()), 1e-300)
        row: dict[str, Any] = {
            "windows": len(h),
            "lambda_max": float(eigen[0]),
            "critical_temperature": 2.0 * float(eigen[0]),
            "effective_dimension": int(np.searchsorted(held, share) + 1),
            "codes": None,
            "groups": None,
            "nearest_squared": None,
            "median_squared": None,
            "in_use": None,
            "within_lambda_max": None,
        }
        if factor.codes is not None:
            E = np.array(factor.codes.E.tolist(), dtype=float)
            pairs = np.sum((E[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2)
            upper = pairs[np.triu_indices(len(E), 1)]
            nearest = np.argmin(np.sum((h[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2), axis=1)
            within = np.zeros((h.shape[1], h.shape[1]))
            for code in np.unique(nearest):
                members = h[nearest == code]
                centred = members - members.mean(axis=0)
                within += centred.T @ centred
            row |= {
                "codes": len(E),
                "groups": _groups(pairs < merged),
                "nearest_squared": float(upper.min()),
                "median_squared": float(np.median(upper)),
                "in_use": len(np.unique(nearest)),
                "within_lambda_max": float(np.linalg.eigvalsh(within / max(len(h) - 1, 1))[-1]),
            }
        spread.append(row)
    return spread


def weight_norms(net: Any) -> dict[str, Any]:
    """
    A saved network's weight norms (Frobenius), for plasticity's correlates and the slingshot (the
    2026-10-22 literature review: Dohare et al.'s growing weights; Thilak et al.'s last-layer norm
    growth before a loss spike): every trained parameter (an attribute with a `grad_` twin) of every
    layer and part (`total`), the readout's weights and bias (`readout`, the output layer), and per
    factor its cell's (`cells`) and its codes' (`codes`, None without codes).
    """

    def norm2(layer: Any) -> float:
        own: dict[str, Any] = layer.__dict__
        trained = [k for k in own if not k.startswith("grad_") and f"grad_{k}" in own]
        return sum(float(np.sum(np.square(np.array(own[k].tolist(), dtype=float)))) for k in trained)

    layers = list(net.layers) + list(net.parts)
    factors = net.recurrence_layer.factors
    return {
        "total": math.sqrt(sum(norm2(layer) for layer in layers)),
        "readout": math.sqrt(norm2(net.output_layer)),
        "cells": [math.sqrt(norm2(f.cell)) for f in factors],
        "codes": [None if f.codes is None else math.sqrt(norm2(f.codes)) for f in factors],
    }


COOLS: dict[str, Callable[[float], float]] = {  # the shaped cools' octaves fallen, a share of all, by share of time
    "concave": lambda u: u**2,
    "convex": lambda u: 1.0 - (1.0 - u) ** 2,
    "s": lambda u: 3 * u**2 - 2 * u**3,
}


@dataclass(frozen=True)
class WarmUp:
    """
    The state formation workplan's option 1 (D4, D6, D10, D14-D18): every snapped factor run continuous
    (its cell's output passed on, its codes unread, its commitment 0) from the run's start until the
    end of the first step whose best screen reaches `screen` and where every snapped factor's distinct
    states seen number `distinct` times its codes, or of its `cap`th step; then each factor's codes set
    by k-means of its cells' outputs on the step's screen (`iterations` Lloyd iterations, k-means++
    seeding from the run's seed), and its temperature profiled (the owner's, from crystallisation):
    heated from `T_hot / 2 ** heat` doubling each epoch, held at `T_hot` for `hold` epochs, then
    cooled exponentially, halved every `halving` epochs, with no floor; `T_hot` is `hot` times the
    whole's critical temperature (Rose 1998: above it every code is at the mean, a melt). Each soft
    epoch moves every soft code by thermal noise of variance `kappa T` (D17). The run's Anneal reads
    hardness from the cool's start and holds the tail and the commitment's ramp. With `gate` (D29) the
    cool is quasi-static: its clock (the profile's epochs) holds through an epoch whose screen before it
    is below `gate` times the hold's best, so the temperature and the coupling wait for the network;
    with `window` (D30) each factor's coupling window spans at least that many octaves below its melting
    point. With `shape` `adaptive` (D32, D33) the cool isn't precomputed: each factor's temperature
    steps once an equilibrium test passes (`within`) or after `settle` epochs at it, by a step sized
    from its heat capacity (`pace`), as Cooling sets it.
    """

    screen: float = 0.95
    cap: int = 8
    iterations: int = 25
    distinct: float = 4.0
    hot: float = 2.0
    heat: int = 6
    hold: int = 10
    halving: int = 15
    kappa: float = 1e-2
    shape: str = "exponential"
    cool: int = 600
    octaves: float = 20.0
    handover: bool = False
    gate: float = 0.0
    window: float = 0.0
    pace: float = 0.03
    settle: int = 10
    within: float = 0.75
    rise: float = 0.0

    def holds(self, n: float, screen: float, level: float) -> bool:
        """
        Whether the clock holds at `n` (the profile's epochs so far) for the next epoch (D29): within the
        cool, the screen before it (`screen`) below `gate` times the hold's best (`level`).
        """
        return self.gate > 0 and self.phase(n - 1) == "cool" and screen < self.gate * level

    def coupling(self, n: float, temperature: float, melting: float, final: float) -> float:
        """
        A soft factor's coupling after `n` epochs from the warm-up's end at `temperature` (D25, with
        `handover`; else 1): 0 through the heat and the hold (the state the cells' outputs, the codes
        forming on them), then in the cool log-linear in T from 0 at its melting point `melting` to 1
        at its last transition `final` (the melting curve's, D21), clipped to [0, 1].
        """
        if not self.handover:
            return 1.0
        if self.phase(n) != "cool":
            return 0.0
        share = math.log(melting / temperature) / math.log(melting / final)
        return min(max(share, 0.0), 1.0)

    def floor(self, temperature: float, melting: float, final: float) -> float:
        """
        The coldest temperature the adaptive cool may step to from `temperature` (D34): the one where the
        coupling (log-linear in T, D25) has risen by `rise` at most; 0 (no bound) without `rise` or the
        handover, or where that rise would reach the window's end.
        """
        if not (self.rise > 0 and self.handover):
            return 0.0
        width = math.log(melting / final)
        share = min(max(math.log(melting / temperature) / width, 0.0), 1.0) + self.rise
        return melting * math.exp(-share * width) if share < 1.0 else 0.0

    def ends(self, step: int, first: int, best: float, distinct: Sequence[int], counts: Sequence[int]) -> bool:
        """Whether the warm-up ends at this step's end (`distinct`, `counts`: each snapped factor's states and codes)."""
        filled = all(n >= self.distinct * c for n, c in zip(distinct, counts, strict=True))
        return (best >= self.screen and filled) or step - first + 1 >= self.cap

    def reads_hardness(self, n: float) -> bool:
        """
        Whether the hardness test is read after `n` epochs from the warm-up's end: from the cool's start
        (exponential), or once a shaped cool has run its `cool` epochs, so its whole curve trains soft.
        """
        cooled = n - self.heat - self.hold
        return cooled >= 0 and (self.shape in ("exponential", "adaptive") or cooled >= self.cool)

    def phase(self, n: float) -> str:
        """The profile's phase after `n` epochs trained from the warm-up's end."""
        return "heat" if n < self.heat else "hold" if n < self.heat + self.hold else "cool"

    def temperature(self, n: float, hot: float) -> float:
        """
        A factor's temperature after `n` epochs trained from the warm-up's end, at its `T_hot` `hot`. The
        cool's shape on a log-temperature scale (D24): `exponential` a straight line, halved every
        `halving` epochs; `concave` (the octaves fallen go as `u ** 2`, `u` the share of `cool` epochs
        passed: it stays near the melt, then falls fast), `convex` (as `1 - (1 - u) ** 2`: it falls
        fast, then a long slow tail) and `s` (as `3 u ** 2 - 2 u ** 3`: slow near the melt and in the
        tail, fast between), each falling `octaves` in `cool` epochs, then halved every `halving` as
        the exponential; `adaptive` (D33) `hot` here, its cool Cooling's.
        """
        if n < self.heat:
            return hot * 2.0 ** (n - self.heat)
        m = max(n - self.heat - self.hold, 0)
        if self.shape == "adaptive":
            return hot  # the cool's start; Cooling's from there (D33)
        if self.shape == "exponential":
            return hot * 0.5 ** (m / self.halving)
        if m >= self.cool:
            return hot * 2.0**-self.octaves * 0.5 ** ((m - self.cool) / self.halving)
        u = m / self.cool
        return hot * 2.0 ** (-self.octaves * COOLS[self.shape](u))


@dataclass
class Cooling:
    """
    The adaptive cool's state (D32, D33: Huang, Romeo and Sangiovanni-Vincentelli 1986), each soft
    factor's: its temperature, the energies and heat capacities measured at it (`thermals`, an epoch
    each), and the energy predicted for it from the last (None: none, the cool's first). WarmUp's
    `pace` is lambda, `within` the equilibrium test's share of the spread, `settle` the cap.
    """

    temperature: list[float]
    energies: list[list[float]]
    capacities: list[list[float]]
    predicted: list[float | None]

    @staticmethod
    def start(hot: Sequence[float]) -> Cooling:
        """The cool at its start, each factor at its T_hot."""
        return Cooling(list(hot), [[] for _ in hot], [[] for _ in hot], [None] * len(hot))

    def advance(self, warm_up: WarmUp, melting: Sequence[float] = (), final: Sequence[float] = ()) -> None:
        """
        Each factor's temperature for the next epoch (D33): it steps once the epochs at it pass the
        equilibrium test or number `settle`, to `T exp(-pace / sqrt(C))` (Huang et al.'s `T exp(-lambda
        T / sigma)`, sigma `T sqrt(C)`, C the capacity's mean at T), at most an octave, and no colder
        than WarmUp.floor in its coupling window (`melting`, `final`: D34); the energy predicted there
        `E - C dT`. The test: the first epoch's energy within `within` sigma of its
        prediction, or a later epoch's within `within` sigma of the epoch's before (stationary at T).
        """
        for g, (t, energies, capacities) in enumerate(
            zip(self.temperature, self.energies, self.capacities, strict=True)
        ):
            if not energies:
                continue
            capacity = sum(capacities) / len(capacities)
            sigma = t * math.sqrt(max(capacity, 0.0))
            last = energies[-1]
            before = self.predicted[g] if len(energies) == 1 else energies[-2]
            settled = before is not None and abs(last - before) <= warm_up.within * sigma
            if not (settled or len(energies) >= warm_up.settle):
                continue
            ratio = math.exp(-warm_up.pace / math.sqrt(capacity)) if capacity > 0 else 0.0
            cooler = t * max(ratio, 0.5)
            if g < len(melting):
                cooler = max(cooler, warm_up.floor(t, melting[g], final[g]))
            self.temperature[g] = cooler
            self.predicted[g] = last - capacity * (t - cooler)
            self.energies[g], self.capacities[g] = [], []

    def record(self, heat: Sequence[tuple[float, float] | None]) -> None:
        """Each soft factor's energy and capacity this epoch, at its temperature (thermals'; None: hard)."""
        for g, e in enumerate(heat):
            if e is not None:
                self.energies[g].append(e[0])
                self.capacities[g].append(e[1])


@dataclass(frozen=True)
class Continuous:
    """
    The continuous crystallisation (the state formation workplan's D48-D53, with WarmUp's exponential
    cool, from the crucible): every switch of the profile continuous. The cool's clock advances an
    epoch by `rate` (D49: 1 at `high` times the hold's level, 0 at `low`, linear between); each
    snapped factor's coupling from its codes' sharpness (D50, `coupling`), 0 through the heat and the
    hold; soft to the end, its gradient straight through, its temperature held from where its
    sharpness first reaches `sharp` (D51); the crucible's longest length grown in `ln L` by `growth`
    times the rate and the smallest coupling an epoch, to the cap (D52, `grow`). After the warm-up a
    step is one epoch, its readings every epoch; the melting curve every `curve_every`, the step's
    network saved every `save_every` (D53). Two ways to keep the coupling from jumping when the
    sharpness does (D61), neither with a scale of its own: `pace`, each factor's coupling moving
    toward its sharpness's at most its allowance an epoch, the allowance its first rise and then
    times `2 ** (2 rate - 1)` an epoch (doubled while the screen holds, halved when the clock stops:
    a trust region the network's response sets); `smooth`, the coupling read from the sharpness
    smoothed by a one-sided Savitzky-Golay filter (`smoothed`: its least-squares line over the
    readings of the last e-folding time of the cool's clock, `halving / ln 2`, read at the clock's
    end): no lag on a steady rise, a jump spread over the time the temperature takes to move.
    `reheat` (the state formation workplan's D63 (a)) extends the rate's line below `low` to -1 (at
    `2 low - high` of the level): a screen fallen that far runs the clock back, never past the cool's
    start, so the temperature rises, the codes soften and the coupling read from their sharpness falls;
    a held temperature the clock passes back over is released; growth only forward. `slowing` (D66,
    with `reheat`) halves the forward rate after each reheat (a stretch of the rate below 0, counted
    when it next rises above 0): the re-cool through the zone at half the pace of the pass before, the
    run back unchanged; a run that never reheats is unchanged. `incremental` (the incremental reheat
    workplan's D1-D7, apart from `reheat`) undoes the cool one stage at a time (`Backtrack`).
    """

    low: float = 0.95
    high: float = 0.99
    sharp: float = 0.999
    growth: float = 0.02
    curve_every: int = 25
    save_every: int = 10
    pace: bool = field(default=False, repr=False)  # D61 (b); in the run's identity apart
    smooth: bool = field(default=False, repr=False)  # D61 (c); in the run's identity apart
    reheat: bool = field(default=False, repr=False)  # D63 (a); in the run's identity apart
    slowing: bool = field(default=False, repr=False)  # D66; in the run's identity apart
    incremental: bool = field(default=False, repr=False)  # the incremental reheat; in the run's identity apart

    def rate(self, screen: float, level: float) -> float:
        """The share of an epoch the cool's clock advances (D49), from the screen before it against the hold's level."""
        least = -1.0 if self.reheat else 0.0  # D63 (a): below `low`, the clock runs back
        return min(max((screen / level - self.low) / (self.high - self.low), least), 1.0) if level > 0 else 1.0

    @staticmethod
    def slowed(rate: float, reheats: int, back: bool) -> tuple[float, int, bool]:
        """
        D66: the rate after `reheats` reheats, with whether the clock was running back: a reheat counted
        as the clock turns forward again, the forward rate halved for each, the run back unchanged.
        """
        if back and rate > 0:
            reheats, back = reheats + 1, False
        return (rate * 2.0**-reheats if rate > 0 else rate), reheats, back or rate < 0

    @staticmethod
    def coupling(sharp: float, start: float) -> float:
        """A snapped factor's coupling in the cool (D50): its sharpness's rise from the cool's start, as a share of what's left."""
        return min(max((sharp - start) / max(1.0 - start, 1e-12), 0.0), 1.0)

    def grow(self, length: float, rate: float, coupling: float, cap: int) -> float:
        """The crucible's longest length after an epoch (D52), from its rate and the smallest coupling."""
        return min(float(cap), length * math.exp(self.growth * rate * coupling))


def savitzky_golay(history: Sequence[Sequence[float]], clock: float) -> list[float]:
    """
    Each factor's sharpness smoothed (D61 (c)): the least-squares line through its readings in
    `history` (each the clock and every factor's sharpness), read at `clock`, a one-sided
    Savitzky-Golay filter of degree 1 on the clock's (uneven) times; their mean where the readings
    share one time.
    """
    xs = [h[0] for h in history]
    x0 = sum(xs) / len(xs)
    spread = sum((x - x0) ** 2 for x in xs)
    smoothed: list[float] = []
    for g in range(len(history[0]) - 1):
        ys = [h[1 + g] for h in history]
        y0 = sum(ys) / len(ys)
        slope = sum((x - x0) * (y - y0) for x, y in zip(xs, ys, strict=True)) / spread if spread > 1e-12 else 0.0
        smoothed.append(y0 + slope * (clock - x0))
    return smoothed


def growth_lengths(length: float, examples: int) -> list[int]:
    """
    An epoch's examples' lengths at the longest `length` (D52; real): the mixture `(1 - f) U(1..n) + f
    U(1..n+1)`, `n` its floor and `f` the rest, each part's lengths in turn (as epoch_lengths), the
    `f` share first.
    """
    n = int(length)
    longer = round((length - n) * examples)
    return [i % (n + 1) + 1 if i < longer else (i - longer) % n + 1 for i in range(examples)]


def growth_screen(crucible: int, length: float) -> list[int]:
    """The screen's lengths at the longest `length` (D52): the crucible's every length and whole_lengths's to its floor."""
    return sorted(set(range(1, crucible + 1)) | set(whole_lengths(int(length))))


@dataclass(frozen=True)
class Backtrack:
    """
    The incremental reheat (the incremental reheat workplan, D1-D7): the cool undone one stage of
    crystallisation at a time. A stage is a transition, an epoch of the forward cool at which a soft
    factor's code groups pass their running maximum (D1); each is kept as the clock of the epoch
    before it, where the temperatures were just above the split, and every factor's maximum then. A
    hold (the clock stopped, D49) lasting `wait` epochs starts a reheat (D2): the clock runs back along
    the cool's own profile at most an epoch of clock an epoch (D4, quasi-static) to the latest stage's
    clock and no further (D3); there the stage is popped, the maxima reset to its, and the clock held
    for `wait` epochs (D5). A screen recovered resumes the cool; not recovered, the stage before is the
    next ceiling (D6); none left, the run is exhausted and ends (D7). `mode` is "cool", "back" or
    "wait"; `clock` the last epoch's.
    """

    maxima: list[int] = field(default_factory=list[int])
    stages: list[list[float]] = field(default_factory=list[list[float]])
    clock: float = 0.0
    mode: str = "cool"
    ceiling: float = 0.0
    waited: int = 0
    exhausted: bool = False

    def moved(self, rate: float, clock: float, wait: int) -> tuple[float, Backtrack]:
        """The clock's move this epoch from the last epoch's `clock`, and the state after it: D2-D7 on D49's `rate`."""
        if self.mode == "back":
            return self.toward(self.ceiling, clock)
        if rate > 0:
            return rate, replace(self, mode="cool", waited=0)
        if self.waited + 1 < wait:
            return 0.0, replace(self, waited=self.waited + 1)
        if not self.stages:  # D7: nothing formed is left to return to
            return 0.0, replace(self, waited=self.waited + 1, exhausted=True)
        return replace(self, mode="back", ceiling=self.stages[-1][0], waited=0).toward(self.stages[-1][0], clock)

    def toward(self, ceiling: float, clock: float) -> tuple[float, Backtrack]:
        """D3, D4: an epoch of the run back toward `ceiling`; arrived, the stage popped and the hold at it begun (D5)."""
        move = max(ceiling - clock, -1.0)
        if clock + move > ceiling + 1e-9:
            return move, self
        stage = self.stages[-1]
        return move, replace(self, mode="wait", waited=0, stages=self.stages[:-1], maxima=[int(m) for m in stage[1:]])

    def recorded(self, groups: Sequence[int | None], clock: float, cooling: bool) -> Backtrack:
        """The epoch's `groups` read (D1) at its `clock`: a stage pushed if the forward cool passed a factor's maximum."""
        if not cooling or self.mode != "cool":
            return replace(self, clock=clock)
        maxima = list(self.maxima) or [-1] * len(groups)
        before = list(maxima)
        split = False
        for g, n in enumerate(groups):
            if n is not None and n > maxima[g]:
                split = split or maxima[g] >= 0
                maxima[g] = n
        stages = [*self.stages, [self.clock, *before]] if split else self.stages
        return replace(self, maxima=maxima, stages=stages, clock=clock)


@dataclass(frozen=True)
class ContinuousState:
    """
    What the continuous crystallisation carries from epoch to epoch (D49-D52): the clock's held share,
    each factor's T_hot, held temperature (0: none) and the cool's first sharpness, and the crucible's
    longest length; with D61's `pace` each factor's allowance and its coupling last set, with
    `smooth` the cool's readings within its window, each the clock and every factor's sharpness
    (empty before the cool); with D66's `slowing` the reheats so far and whether the clock is running
    back; with `incremental`, the incremental reheat's stages and mode (`Backtrack`).
    """

    held: float
    hot: list[float]
    floors: list[float]
    sharp_from: list[float]
    length: float
    allowance: list[float] = field(default_factory=list[float])
    last: list[float] = field(default_factory=list[float])
    history: list[list[float]] = field(default_factory=list[list[float]])
    reheats: int = 0
    back: bool = False
    backtrack: Backtrack | None = None


@dataclass(frozen=True)
class ContinuousEpoch:
    """continuous_epoch's result: the state carried on (its length the next epoch's, D52), the epoch's phase, rate and couplings, and its row's `anneal`."""

    state: ContinuousState
    phase: str
    rate: float
    couplings: list[float]
    thermo: dict[str, Any]


def continuous_epoch(
    net: Any,
    continuous: Continuous,
    warm_up: WarmUp,
    rows: Sequence[Sequence[tuple[Any, Any]]],
    snapped: Sequence[bool],
    state: ContinuousState,
    trained: int,
    screen: float,
    level: float,
    hard: float,
    beta: list[float],
    coupling_cap: float,
    rng: np.random.Generator,
    cap: int,
) -> ContinuousEpoch:
    """
    One epoch's setting under the continuous crystallisation (D48-D51), before it trains, `trained`
    epochs after the warm-up's end: the clock advanced by the rate from `screen` against the hold's
    `level` (D49), each snapped factor soft at its profile's temperature or its held one (D51), its
    sharpness read on `rows` (D50, D53), its coupling from it (0 through the heat and the hold), its
    temperature held once its sharpness reaches `sharp`; the soft codes jittered (D17), the thermals
    read, and the network set (temperatures, couplings, its gradient straight through, `beta`); in
    the cool, the longest length grown by the rate and the smallest coupling for the next epoch (D52:
    this epoch's examples and screen were drawn at the state's).
    """
    length = state.length
    held, rate = state.held, 1.0
    reheats, back = state.reheats, state.back
    backtrack = state.backtrack
    if warm_up.phase(trained - held - 1) == "cool":
        rate = continuous.rate(screen, level)
        if continuous.slowing:
            rate, reheats, back = continuous.slowed(rate, reheats, back)
        if backtrack is not None:  # the incremental reheat: the clock's move from the last epoch's
            rate, backtrack = backtrack.moved(rate, trained - 1 - held, round(warm_up.halving / math.log(2)))
        held = min(held + 1.0 - rate, trained - warm_up.heat - warm_up.hold)  # D63 (a): never back past the cool
    clock = trained - held
    phase = warm_up.phase(clock)
    floors = list(state.floors)
    if rate < 0:  # D63 (a): a held temperature the clock passes back over released
        floors = [
            0.0 if f > 0 and warm_up.temperature(clock, h) > f else f for f, h in zip(floors, state.hot, strict=True)
        ]
    temperatures: list[float | None] = [
        max(warm_up.temperature(clock, t), f) if s else None for t, f, s in zip(state.hot, floors, snapped, strict=True)
    ]
    sharp = sharpness(net, rows, temperatures, hard)
    levels = [0.0 if v is None else v[0] for v in sharp]
    sharp_from = list(levels) if phase == "cool" and not state.sharp_from else list(state.sharp_from)
    history, smoothed = [list(h) for h in state.history], []
    if continuous.smooth and phase == "cool":  # D61 (c): the sharpness's line over the cool's e-folding time
        span = warm_up.halving / math.log(2)
        history = [h for h in history if h[0] > clock - span] + [[clock, *levels]]
        smoothed = savitzky_golay(history, clock)
    couplings = [
        1.0 if not s else continuous.coupling(v, v0) if phase == "cool" else 0.0
        for s, v, v0 in zip(snapped, smoothed or levels, sharp_from or levels, strict=True)
    ]
    targets = list(couplings)
    allowance, last = list(state.allowance), list(state.last)
    if continuous.pace and phase == "cool":  # D61 (b): each coupling's move at most its allowance
        before = last if allowance else [0.0] * len(couplings)
        allowance = [
            a * 2.0 ** (2 * rate - 1) if a > 0 else max(c - b, 0.0)
            for c, b, a in zip(couplings, before, allowance or [0.0] * len(couplings), strict=True)
        ]
        couplings = [
            c if not s else b + min(max(c - b, -a), a)
            for s, c, b, a in zip(snapped, couplings, before, allowance, strict=True)
        ]
        last = list(couplings)
    if coupling_cap < 1.0:  # D38's cap, as without
        couplings = [min(c, coupling_cap) for c in couplings]
    floors = [
        f
        if f > 0 or t is None or phase != "cool" or v < continuous.sharp or rate < 0
        else t  # D63: none held running back
        for f, t, v in zip(floors, temperatures, levels, strict=True)
    ]
    jitter(net, temperatures, warm_up.kappa, rng)
    heat = thermals(net, rows, temperatures)
    net.set_temperatures(temperatures)
    net.set_couplings(couplings)
    net.set_straight(True)
    net.set_commitments(beta)
    groups = code_groups(net, temperatures, warm_up.kappa)
    if backtrack is not None:
        backtrack = backtrack.recorded(groups, clock, phase == "cool")
    thermo = {
        "anneal": {
            "temperature": temperatures,
            "sharpness": [None if v is None else round(v[0], 6) for v in sharp],
            "hard": [None if v is None else round(v[1], 4) for v in sharp],
            "beta": [round(b, 4) for b in beta],
            "phase": phase,
            "groups": groups,
            "energy": [None if e is None else round(e[0], 6) for e in heat],
            "capacity": [None if e is None else round(e[1], 6) for e in heat],
            "coupling": [round(c, 4) for c in couplings],
            "rate": round(rate, 4),
            "clock": round(clock, 3),
            "length": round(length, 3),
            "floors": floors,
            **({"target": [round(c, 4) for c in targets]} if continuous.pace or continuous.smooth else {}),
            **({"allowance": [round(a, 6) for a in allowance]} if continuous.pace else {}),
            **({"smoothed": [round(m, 6) for m in smoothed]} if continuous.smooth else {}),
            **({"reheats": reheats} if continuous.slowing else {}),
            **(
                {
                    "backtrack": {
                        "mode": backtrack.mode,
                        "stages": len(backtrack.stages),
                        "ceiling": round(backtrack.ceiling, 3),
                        "waited": backtrack.waited,
                        **({"exhausted": True} if backtrack.exhausted else {}),
                    }
                }
                if backtrack is not None
                else {}
            ),
        }
    }
    grown = (
        continuous.grow(length, max(rate, 0.0), min(c for c, s in zip(couplings, snapped, strict=True) if s), cap)
        if phase == "cool"
        else length
    )
    return ContinuousEpoch(
        ContinuousState(
            held, list(state.hot), floors, sharp_from, grown, allowance, last, history, reheats, back, backtrack
        ),
        phase,
        rate,
        couplings,
        thermo,
    )


def step_screens(
    encoding: Target,
    settings: Settings,
    cap: int,
    longest: int,
    flat: bool,
    crucible: int,
    length: float,
    cached: dict[int, list[Example]],
) -> tuple[dict[int, list[Example]], list[int]]:
    """
    A step's screen (its windows by length) and the lengths its readings take: the whole
    distribution's (`flat`), the curriculum's to `longest`, or, growing from a `crucible` (D52; 0:
    not), the crucible's and the growth's to `length`, their windows kept in `cached` between epochs.
    """
    if crucible:
        lengths = growth_screen(crucible, length)
        for n in lengths:
            if n not in cached:
                cached[n] = _windows(encoding, n, settings.screen, f"family-screen-continuous-{n}")[0]
        return {n: cached[n] for n in lengths}, sorted({1, (crucible + 1) // 2, crucible, int(length)})
    lengths = whole_lengths(cap) if flat else list(range(1, longest + 1))
    named = "whole" if flat else longest
    screens = {n: _windows(encoding, n, settings.screen, f"family-screen-{named}-{n}")[0] for n in lengths}
    read_at = (  # the step's screen at its shortest, middle and longest
        sorted({lengths[0], lengths[len(lengths) // 2], lengths[-1]})
        if flat
        else sorted({1, (longest + 1) // 2, longest})
    )
    return screens, read_at


def handover_window(made: Sequence[Mapping[str, Any]], window: float = 0.0) -> tuple[list[float], list[float]]:
    """
    Each factor's coupling window from the warm-up end's rows (codes_from_states; D25): its melting
    point and its last transition's temperature, the whole's critical temperature and a thousandth of
    the melting point where its curve has none; at least `window` octaves wide (D30: a curve whose
    codes part in one transition would otherwise hand over within an octave); a factor without codes 1
    and 1 (never coupled).
    """
    melting: list[float] = []
    final: list[float] = []
    for m in made:
        if not m:
            melting.append(1.0)
            final.append(1.0)
            continue
        top = float(m.get("melting_point") or m["critical_temperature"])
        transitions: list[list[float]] = m.get("transitions") or []
        last = float(transitions[-1][0]) if transitions else top / 1000
        melting.append(top)
        final.append(min(last if last < top else top / 2, top * 2.0**-window))
    return melting, final


def distinct_states(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]]) -> list[int]:
    """Each snapped factor's distinct cell outputs (to 6 decimals) over every window of the rows given (WarmUp, D14)."""
    return [
        len(np.unique(np.round(h, 6), axis=0))
        for h, f in zip(factor_states(net, rows), net.recurrence.factors, strict=True)
        if f.codes is not None
    ]


def jitter(net: Any, temperatures: Sequence[float | None], kappa: float, rng: np.random.Generator) -> None:
    """Every soft factor's codes moved by Gaussian noise of variance `kappa T` in all (D17: thermal noise)."""
    for factor, t in zip(net.recurrence_layer.factors, temperatures, strict=True):
        if t is None:
            continue
        codes = np.array(factor.codes.E.tolist(), dtype=float)
        moved = codes + rng.normal(0.0, math.sqrt(kappa * t / codes.shape[1]), size=codes.shape)
        factor.codes.E = net.backend.matrix(moved.tolist())


def code_groups(net: Any, temperatures: Sequence[float | None], kappa: float = 0.0) -> list[int | None]:
    """
    Each soft factor's codes' groups closer than `T max(1 / 100, 10 kappa)` squared (D20: the melt and
    its splits, the distance above thermal noise's `kappa T`); None if hard.
    """
    groups: list[int | None] = []
    for factor, t in zip(net.recurrence_layer.factors, temperatures, strict=True):
        if t is None:
            groups.append(None)
            continue
        E = np.array(factor.codes.E.tolist(), dtype=float)
        close = t * max(1 / 100, 10 * kappa)
        groups.append(_groups(np.sum((E[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2) < close))
    return groups


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """
    k centres of the points (n, d), k-means++ seeding then Lloyd's iterations, and each point's nearest
    centre. A centre with no points (fewer distinct points than k) is a drawn point moved by a small
    random step, so no two codes start equal (deterministic annealing's perturbation).
    """
    n = len(points)
    centres = [points[int(rng.integers(n))]]
    for _ in range(1, k):
        d = np.min(np.sum((points[:, np.newaxis, :] - np.array(centres)[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        total = float(d.sum())
        centres.append(points[int(rng.choice(n, p=d / total))] if total > 0 else points[int(rng.integers(n))])
    C = np.array(centres, dtype=float)
    scale = max(float(np.sqrt(np.var(points, axis=0).sum())), 1e-6)
    for _ in range(iterations + 1):
        nearest = np.argmin(np.sum((points[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        for j in range(k):
            members = points[nearest == j]
            C[j] = (
                members.mean(axis=0)
                if len(members)
                else points[int(rng.integers(n))] + rng.normal(0.0, 1e-3 * scale, size=points.shape[1])
            )
    nearest = np.argmin(np.sum((points[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2), axis=1)
    return C, nearest


def codes_from_states(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], rng: np.random.Generator, iterations: int, hot: float
) -> list[dict[str, Any]]:
    """
    Each snapped factor's codes set by k-means of its cells' outputs over the rows given (WarmUp, D6),
    and its `T_hot`, `hot` times the whole's critical temperature (D16); a row each, with the clusters'
    own critical temperature (twice the pooled within-cluster top eigenvalue), the whole's, and the
    states' melting curve's melting point and transitions (D21) beside it.
    """
    spread = state_spread(net, rows)
    states = factor_states(net, rows)
    made: list[dict[str, Any]] = []
    for g, factor in enumerate(net.recurrence_layer.factors):
        if factor.codes is None:
            made.append({})
            continue
        h = states[g]
        melting = melting_curve(h, factor.codes.count, np.random.default_rng([g]))  # k-means' draws untouched
        E, nearest = kmeans(h, factor.codes.count, rng, iterations)
        within = np.zeros((h.shape[1], h.shape[1]))
        for code in np.unique(nearest):
            centred = h[nearest == code] - h[nearest == code].mean(axis=0)
            within += centred.T @ centred
        lam = float(np.linalg.eigvalsh(within / max(len(h) - 1, 1))[-1])
        factor.codes.E = net.backend.matrix(E.tolist())
        made.append(
            {
                "hot": hot * spread[g]["critical_temperature"],
                "within_temperature": 2.0 * lam,
                "critical_temperature": spread[g]["critical_temperature"],
                "in_use": len(np.unique(nearest)),
                "effective_dimension": spread[g]["effective_dimension"],
                "melting_point": melting["melting_point"],
                "transitions": melting["transitions"],
            }
        )
    return made


@dataclass
class Place:
    """
    Where a run is, as its checkpoint keeps it (run's resume): the step and the last epoch trained
    in it (0: the step just begun), the epochs trained in all, the step's best screen and epochs
    since, its rate, forks and the fork's peak, the last screen's answer kinds, the step's undone
    updates and the guard's frozen epochs; the seconds trained, and the log's length in bytes; and
    A's table, (duplicate, representative) pairs (training state, not saved with the network); and
    each factor's levels under the thermostat or annealing.
    """

    step: int
    epoch: int
    spent: int
    best: float
    since: int
    rate: float
    forks: int
    peak: float
    last_kinds: dict[str, float]
    step_rollbacks: int
    frozen: int
    seconds: float
    log_bytes: int
    consolidation: list[tuple[list[int], list[int]]] = field(default_factory=list[tuple[list[int], list[int]]])
    thermostat: list[int] = field(default_factory=list[int])  # each factor's level (Thermostat), with one
    machine_best: list[float] = field(default_factory=list[float])  # R3: the step's kept machine's key, with it
    words: list[list[int]] = field(default_factory=list[list[int]])  # R4: the last diagnosis's counterexamples
    anneal: list[int] = field(default_factory=list[int])  # each factor's level (Anneal), with it
    warm: bool = False  # WarmUp: still continuous
    hot: list[float] = field(default_factory=list[float])  # WarmUp: each factor's T_hot, once set
    anneal_from: int = 0  # WarmUp: the epochs trained when the warm-up ended
    melting: list[float] = field(default_factory=list[float])  # WarmUp: each factor's melting point (D25)
    final: list[float] = field(default_factory=list[float])  # WarmUp: each factor's last transition (D25)
    crucible: int = 0  # the crucible's longest length, once its warm-up ended (D26)
    grown: int = 0  # the crucible's first block of growth, once begun (D26)
    held: float = 0  # WarmUp: the epochs its clock held (D29; with Continuous, the shares of epochs, D49)
    level: float = -1.0  # WarmUp: the hold's best screen (D29)
    gate_screen: float = -1.0  # WarmUp: the last screen, which the gate reads (D29)
    cooling: dict[str, Any] = field(default_factory=dict[str, Any])  # WarmUp: the adaptive cool's state (D33)
    couplings: list[float] = field(default_factory=list[float])  # WarmUp: the network's last set (D25)
    length: float = 0.0  # Continuous: the crucible's longest length, once its warm-up ended (D52)
    sharp_from: list[float] = field(default_factory=list[float])  # Continuous: the cool's first sharpness (D50)
    floors: list[float] = field(default_factory=list[float])  # Continuous: each factor's held temperature (D51)
    first_pass: int = -1  # the margin stop: the step D35's stop first passed with a split tie (D59)
    best_pass: int = -1  # the margin stop: the pass with the fewest ties since (D60)
    best_ties: int = -1  # its split and label ties (D60)
    allowance: list[float] = field(default_factory=list[float])  # Continuous: each coupling's allowance (D61)
    history: list[list[float]] = field(default_factory=list[list[float]])  # Continuous: the window's readings (D61)
    reheats: int = 0  # Continuous: the reheats so far (D66)
    back: bool = False  # Continuous: the clock running back (D66)
    succeeded: int = -1  # the cutoff: the first step certified at both verdicts (D41), -1 before one
    backtrack: dict[str, Any] = field(default_factory=dict[str, Any])  # Continuous: the incremental reheat's


COUNTEREXAMPLE_WORDS = 256  # R4's words at most (D6)
BEGUN, ENDED = "begun", "ended"  # a checkpoint's place before the run's first row, and after its last
GUARD = "budget_seconds"  # the wall-clock guard, in identities from before it left them (ignored on resume)
STEP_FILE = re.compile(r"-step-(\d+)\.network\.json$")  # a step end's networks (step-S, certified-step-S, ...)


def _write_checkpoint(out: Path, identity: dict[str, Any], place: Place | str, net: Any) -> None:
    # the run's place (or BEGUN, ENDED) and its network in one file, replaced whole, so a stop at any
    # moment leaves the last checkpoint or the new one
    state = {
        "run": identity,
        "place": place if isinstance(place, str) else vars(place),
        "network": network_to_json(net),
    }
    partial = out / (CHECKPOINT + ".partial")
    partial.write_text(json.dumps(state))
    partial.replace(out / CHECKPOINT)


@dataclass(frozen=True)
class RunConfig:
    """run's settings, the cap resolved (its default the cell's exclusion bound): see run."""

    encoding: Target
    factors: Sequence[tuple[int, int]]
    seed: int
    budget_seconds: float | None
    out: Path
    settings: Settings
    cap: int
    reads: str
    budget_epochs: int | None
    fork: Fork | None
    guard: bool
    long_tail: int | None
    consolidate: bool
    restore_best: bool
    commitment_from: int | None
    width: int
    branches: bool
    answer_branch: bool
    thermostat: Thermostat | None
    keep_machine: bool
    counterexamples: float | None
    anneal: Anneal | None
    warm_up: WarmUp | None
    whole: bool
    crucible: bool
    state_bound: int | None
    coupling_cap: float
    continuous: Continuous | None
    margin_wait: int | None
    cutoff_epochs: int | None = None
    baseline: str | None = None
    recombine: bool = False
    branch_heads: int = 1

    def identity(self, first: int) -> dict[str, Any]:
        """
        What a resumed run must share with the run it resumes: the consolidation workplan's settings
        only when not their defaults, so a checkpoint from before them still resumes; not the
        wall-clock guard (a runaway guard alone since epochs bind), so a run its deadline stopped
        resumes with a larger one.
        """
        return {
            "target_states": self.encoding.target_states,
            "tokens": self.encoding.tokens,
            "factors": [list(f) for f in self.factors],
            "seed": self.seed,
            "settings": repr(self.settings),
            "cap": self.cap,
            "reads": self.reads,
            "budget_epochs": self.budget_epochs,
            "fork": None if self.fork is None else repr(self.fork),
            "guard": self.guard,
            **({"first": first} if first != 1 else {}),
            **({"long_tail": self.long_tail} if self.long_tail is not None else {}),
            **({"consolidate": True} if self.consolidate else {}),
            **({"restore_best": True} if self.restore_best else {}),
            **({"commitment_from": self.commitment_from} if self.commitment_from is not None else {}),
            **({"width": self.width} if self.width != wp.D else {}),
            **({"branches": True} if self.branches else {}),
            **({"answer_branch": True} if self.answer_branch else {}),
            **({"thermostat": repr(self.thermostat)} if self.thermostat is not None else {}),
            **({"keep_machine": True} if self.keep_machine else {}),
            **({"counterexamples": self.counterexamples} if self.counterexamples is not None else {}),
            **({"anneal": repr(self.anneal)} if self.anneal is not None else {}),
            **({"warm_up": repr(self.warm_up)} if self.warm_up is not None else {}),
            **({"whole": True} if self.whole else {}),
            **({"crucible": True} if self.crucible else {}),
            **({"state_bound": self.state_bound} if self.state_bound is not None else {}),
            **({"coupling_cap": self.coupling_cap} if self.coupling_cap < 1.0 else {}),
            **({"continuous": repr(self.continuous)} if self.continuous is not None else {}),
            **({"coupling_pace": True} if self.continuous is not None and self.continuous.pace else {}),
            **({"sharpness_smoothing": True} if self.continuous is not None and self.continuous.smooth else {}),
            **({"reheat": True} if self.continuous is not None and self.continuous.reheat else {}),
            **({"reheat_slowing": True} if self.continuous is not None and self.continuous.slowing else {}),
            **({"incremental_reheat": True} if self.continuous is not None and self.continuous.incremental else {}),
            **({"margin_wait": self.margin_wait} if self.margin_wait is not None else {}),
            **({"cutoff_epochs": self.cutoff_epochs} if self.cutoff_epochs is not None else {}),
            **({"baseline": self.baseline} if self.baseline is not None else {}),
            **({"recombine": True} if self.recombine else {}),
            **({"branch_heads": self.branch_heads} if self.branch_heads != 1 else {}),
        }

    def check(self) -> None:
        """The settings' combinations a run allows (an AssertionError naming the first it doesn't)."""
        assert self.baseline != CONTINUOUS_STATE or not (
            self.state_bound is not None or self.anneal or self.warm_up or self.continuous or self.guard
        ), "a continuous state has no codes: no state bound, anneal, warm-up, continuous profile or guard"
        assert self.margin_wait is None or self.state_bound is not None, (
            "the margin stop waits on D35's: with a state bound"
        )
        assert self.cutoff_epochs is None or (self.continuous is not None and self.cutoff_epochs >= 1), (
            "the cutoff reads D41's success at both verdicts, a continuous run's step ends: with --continuous, at least 1"
        )
        assert self.continuous is None or not self.continuous.pace or self.coupling_cap == 1.0, (
            "D61's pace or D38's cap, not both"
        )
        assert self.warm_up is None or self.anneal is not None, "the warm-up's end is an annealed snap: with an Anneal"
        assert not (self.whole and self.long_tail is not None), "the whole distribution has no long tail beyond it"
        assert not self.crucible or (self.warm_up is not None and not self.whole and self.long_tail is None), (
            "a crucible holds a warm-up's material, then grows on the whole distribution: with a warm-up, not --whole"
        )
        assert (
            self.warm_up is None or self.warm_up.shape in ("exponential", "adaptive") or self.warm_up.shape in COOLS
        ), f"a cool's shape: exponential, adaptive or one of {sorted(COOLS)}"
        assert self.warm_up is None or (self.fork is None and not self.restore_best and not self.keep_machine), (
            "a crystallising run never goes back (D18): the warm-up not with the fork, restore_best or keep_machine"
        )
        assert not (self.keep_machine and self.restore_best), (
            "the step's kept machine (R3) or its best screen, not both"
        )
        assert self.continuous is None or (
            self.warm_up is not None
            and self.crucible
            and self.warm_up.shape == "exponential"
            and not self.warm_up.handover
            and self.warm_up.gate == 0.0
        ), "the continuous crystallisation (D48) is a crucible's, its exponential cool with its own coupling and rate"
        assert self.counterexamples is None or 0 < self.counterexamples < 1, (
            f"a share of each epoch's examples; got {self.counterexamples}"
        )
        assert self.thermostat is None or (self.commitment_from is None and not self.consolidate), (
            "the thermostat sets the commitment's weight itself: not with --commitment-from, nor with A, whose target "
            "it moves (the commitment-control workplan)"
        )
        assert self.anneal is None or (
            self.thermostat is None and self.commitment_from is None and not self.consolidate
        ), "annealing sets the commitment's weight itself, and its soft codes take no consolidation's target"


@dataclass
class RunState:
    """
    What a run carries from epoch to epoch and step to step (the state formation workplan's D62), as
    its objects; Place is its checkpoint's form (`place`, `resumed`). The step's own (the best screen
    and the epochs since, the rate, forks and the fork's peak, the last screen's answer kinds, the
    undone updates, R3's kept machine) set at each step's start; the rest carried over.
    """

    spent: int  # epochs trained in all
    frozen: int  # consecutive armed epochs with STALL_SHARE of their updates undone
    table: dict[tuple[int, ...], tuple[int, ...]]  # A's, from the last diagnosis
    levels: list[int]  # each factor's level (Thermostat)
    words: list[list[int]]  # R4: the last diagnosis's counterexamples
    annealed: list[int]  # Anneal's levels: soft at the start (a continuous factor's at the ramp's end, never soft)
    warm: bool  # WarmUp: still continuous
    hot: list[float]  # WarmUp: each factor's T_hot, once set
    anneal_from: int  # WarmUp: the epochs trained at its end
    melting_points: list[float]  # WarmUp: each factor's melting point (D25)
    last_splits: list[float]  # WarmUp: each factor's last transition (D25)
    fixed: int  # D26: the crucible's longest length, once set
    grown_from: int  # D26: the first block of growth, once begun
    held: int  # D29: the epochs the profile's clock held (the continuous clock's, D49, in cstate)
    level: float  # D29: the hold's best screen
    gate_screen: float  # D29: the last screen
    cooling: Cooling | None  # D33: the adaptive cool
    coupled: list[float]  # D25: the couplings last set, training state
    cstate: ContinuousState  # D49-D52, D61
    margin: Margin  # D59-D60
    best: float = -1.0
    since: int = 0
    rate: float = 0.0
    forks: int = 0
    peak: float = 0.0
    last_kinds: dict[str, float] = field(default_factory=dict[str, float])
    step_rollbacks: int = 0
    machine_best: list[float] = field(default_factory=list[float])
    score: float = -1.0  # the last epoch's screen (WarmUp's end reads it with the step's best)
    succeeded: int = -1  # the first step certified at both verdicts (D41), for the cutoff

    @staticmethod
    def begun(config: RunConfig, snapped: Sequence[bool]) -> RunState:
        """A new run's state."""
        anneal = config.anneal
        return RunState(
            spent=0,
            frozen=0,
            table={},
            levels=[0] * len(snapped),
            words=[],
            annealed=[0 if s else (anneal.top if anneal is not None else 0) for s in snapped],
            warm=config.warm_up is not None,
            hot=[],
            anneal_from=0,
            melting_points=[],
            last_splits=[],
            fixed=0,
            grown_from=0,
            held=0,
            level=-1.0,
            gate_screen=-1.0,
            cooling=None,
            coupled=[],
            cstate=ContinuousState(0.0, [], [], [], 0.0),
            margin=Margin(config.margin_wait),
            rate=config.settings.rate,
        )

    @staticmethod
    def resumed(place: Place, config: RunConfig, snapped: Sequence[bool]) -> RunState:
        """A stopped run's state, from its checkpoint's place."""
        state = RunState.begun(config, snapped)
        state.spent, state.frozen = place.spent, place.frozen
        state.table = {tuple(duplicate): tuple(representative) for duplicate, representative in place.consolidation}
        state.levels = list(place.thermostat) if place.thermostat else state.levels
        state.words = [list(w) for w in place.words]
        state.annealed = list(place.anneal) if place.anneal else state.annealed
        state.warm, state.hot, state.anneal_from = place.warm, list(place.hot), place.anneal_from
        state.melting_points, state.last_splits = list(place.melting), list(place.final)
        state.fixed, state.grown_from, state.held = place.crucible, place.grown, int(place.held)
        state.level, state.gate_screen = place.level, place.gate_screen
        state.cooling = Cooling(**place.cooling) if place.cooling else None
        state.coupled = list(place.couplings)
        state.cstate = ContinuousState(
            place.held,
            list(place.hot),
            list(place.floors),
            list(place.sharp_from),
            place.length,
            list(place.allowance),
            list(place.couplings) if place.allowance else [],
            [list(h) for h in place.history],
            place.reheats,
            place.back,
            Backtrack(**place.backtrack) if place.backtrack else None,
        )
        state.margin = Margin(config.margin_wait, place.first_pass, place.best_pass, place.best_ties)
        state.succeeded = place.succeeded
        return state

    def place(self, step: int, epoch: int, seconds: float, log_bytes: int, config: RunConfig) -> Place:
        """The checkpoint's place after `epoch` of `step` (0: its start), every count as it stands."""
        return Place(
            step,
            epoch,
            self.spent,
            self.best,
            self.since,
            self.rate,
            self.forks,
            self.peak,
            dict(self.last_kinds),
            self.step_rollbacks,
            self.frozen,
            seconds,
            log_bytes,
            [(list(duplicate), list(representative)) for duplicate, representative in self.table.items()],
            list(self.levels) if config.thermostat is not None else [],
            list(self.machine_best),
            [list(w) for w in self.words],
            list(self.annealed) if config.anneal is not None else [],
            self.warm,
            list(self.hot),
            self.anneal_from,
            list(self.melting_points),
            list(self.last_splits),
            self.fixed,
            self.grown_from,
            self.cstate.held if config.continuous is not None else self.held,
            self.level,
            self.gate_screen,
            asdict(self.cooling) if self.cooling is not None else {},
            list(self.coupled),
            self.cstate.length,
            list(self.cstate.sharp_from),
            list(self.cstate.floors),
            self.margin.first_pass,
            self.margin.best_pass,
            self.margin.best_ties,
            list(self.cstate.allowance),
            [list(h) for h in self.cstate.history],
            self.cstate.reheats,
            self.cstate.back,
            self.succeeded,
            asdict(self.cstate.backtrack) if self.cstate.backtrack is not None else {},
        )


@dataclass(frozen=True)
class StepPlan:
    """A step's setting (run's): its screen's windows and the lengths read, and the curriculum's place in it."""

    step: int
    screens: dict[int, list[Example]]
    read_at: list[int]
    flat: bool  # every epoch over every length to the cap (the whole distribution, or a crucible grown)
    longest: int  # the curriculum's longest length (the crucible's, held)
    grown: bool  # a crucible's growth begun (D26)
    growing: bool  # D52: after the warm-up of a continuous run, a step an epoch

    @property
    def sample(self) -> list[list[Example]]:
        """The screen's windows at the lengths read (hardness's, sharpness's, the melting curve's)."""
        return [self.screens[n] for n in self.read_at]


@dataclass(frozen=True)
class EpochSetting:
    """An epoch's setting before it trains (_Run.set_epoch): its row's thermodynamics, its phase, whether it melts."""

    thermo: dict[str, Any]
    phase: str | None
    melting: bool


class _Run:
    """
    One run (run's), its settings (RunConfig), its state (RunState), its network and its log apart
    (the state formation workplan's D62): each phase of a step a method of them.
    """

    def __init__(self, config: RunConfig, resume: bool) -> None:
        c = config
        c.out.mkdir(parents=True, exist_ok=True)
        self.c = c
        self.first = first_length(c.encoding.task.machine)
        self.identity = c.identity(self.first)
        c.check()
        self.d = mr.word_length(c.encoding.task.machine) if c.consolidate else 0
        self.place: Place | None = None  # the checkpoint resumed from
        self.ended = False
        self.deadline = False  # the wall-clock guard stopped it: its last checkpoint kept, so it resumes
        saved = json.loads((c.out / CHECKPOINT).read_text()) if resume and (c.out / CHECKPOINT).exists() else None
        if saved is not None:
            kept = {k: v for k, v in saved["run"].items() if k != GUARD}
            assert kept == self.identity, f"{c.out} holds another run: {saved['run']} against {self.identity}"
            if saved["place"] == ENDED:
                self.ended = True
                return
        table: dict[tuple[int, ...], tuple[int, ...]] = {}
        if saved is not None and saved["place"] != BEGUN:
            self.place = Place(**saved["place"])
            self.net: Any = network_from_state(saved["network"])
            table = {tuple(duplicate): tuple(representative) for duplicate, representative in self.place.consolidation}
            self.net.consolidate(table)
            with (c.out / "log.jsonl").open("r+") as log:
                log.truncate(self.place.log_bytes)
            for path in c.out.glob("*-step-*.network.json"):  # a step's end from this step on came after the
                found = STEP_FILE.search(path.name)  # checkpoint (a deadline's cut step's): the run writes its own
                if found and int(found.group(1)) >= self.place.step:
                    path.unlink()
        else:
            assert saved is not None or not (resume and (c.out / "log.jsonl").exists()), (
                f"{c.out} has a log but no checkpoint to resume from"
            )
            self.net = network(
                c.encoding,
                c.factors,
                c.seed,
                c.reads,
                c.width,
                c.branches,
                c.answer_branch,
                c.baseline,
                recombine=c.recombine,
                branch_heads=c.branch_heads,
            )
            if saved is not None:
                (c.out / "log.jsonl").write_text("")  # stopped before its first checkpoint: begun again from the seed
            _write_checkpoint(c.out, self.identity, BEGUN, self.net)
        self.start = time.perf_counter() - (self.place.seconds if self.place is not None else 0.0)
        self.log = (c.out / "log.jsonl").open("a")
        if self.place is None:
            self.record({"step": 0, "epoch": 0, **diagnose(self.net, c.encoding, c.settings)})
        self.task = c.encoding.task
        self.guarding = cg.Guard(lambda n: QueryMachine(n, c.encoding, 0), self.task) if c.guard else None
        self.snapshot = c.out / "snapshot.network.json"
        self.step_best = c.out / "best.network.json"  # restore_best's
        factors = self.net.recurrence.factors
        self.commitments = [0.0 if f.codes is None else f.codes.commitment for f in factors]  # the specs'
        self.counts = [0 if f.codes is None else f.codes.count for f in factors]
        self.snapped = [f.codes is not None for f in factors]
        self.counts_snapped = [n for n, s in zip(self.counts, self.snapped, strict=True) if s]
        self.s = (
            RunState.resumed(self.place, c, self.snapped) if self.place is not None else RunState.begun(c, self.snapped)
        )
        self.cached: dict[int, list[Example]] = {}  # D52: the growing screen's windows, by length
        if self.s.coupled:
            self.net.set_couplings(
                self.s.coupled
            )  # the loaded network's, as the stopped run's was (its screens read them)
        self.walked_d = mr.word_length(c.encoding.task.machine)  # R3 and R4's walks' d, the task's
        self.machine_kept = c.out / "machine-best.network.json"
        self.continuing = c.continuous is not None  # D48: after the warm-up, each epoch continuous_epoch's

    def record(self, row: dict[str, Any]) -> None:
        row["seconds"] = round(time.perf_counter() - self.start, 1)
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()

    def warming(self) -> None:
        # the network about to train or screen (a fork's or a restore's loaded one included) continuous
        # while warming, snapped after (WarmUp)
        if self.c.warm_up is not None:
            self.net.set_temperatures([None] * len(self.snapped))
            self.net.set_continuous(list(self.snapped) if self.s.warm else [False] * len(self.snapped))

    def checkpoint(self, step: int, epoch: int) -> None:
        # the run's place after epoch of step (0: its start), every count read as it stands
        self.log.flush()
        place = self.s.place(step, epoch, time.perf_counter() - self.start, self.log.tell(), self.c)
        _write_checkpoint(self.c.out, self.identity, place, self.net)

    def regroup(self) -> dict[str, int]:
        # A's table read from the network as it stands and held until the next; the row's record of it
        self.s.table, codes, groups = consolidation(self.net, self.c.encoding, self.d)
        self.net.consolidate(self.s.table)
        return {"codes": codes, "groups": groups, "duplicates": len(self.s.table)}

    def go(self) -> None:
        """The run, from its first step (or its checkpoint's) to its stop."""
        c, s = self.c, self.s
        # the curriculum's steps end at the cap; the whole distribution's blocks go on until the budget (D22:
        # a step is then a block of epochs, not a length)
        begin = self.place.step if self.place is not None else self.first
        for step in itertools.count(begin) if c.whole or c.crucible else range(begin, c.cap + 1):
            if self.run_step(self.plan(step)):
                break
        if c.state_bound is not None and s.margin.waited >= 0:  # D60: the wait's stop, on the best pass's network
            growing = c.continuous is not None and s.fixed > 0
            self.record(s.margin.stop_row(c.out, c.encoding, c.settings, c.state_bound, s.coupled, growing))
        self.log.close()
        if not self.deadline:  # a deadline's stop keeps the last epoch's checkpoint: the run is censored, not ended
            _write_checkpoint(c.out, self.identity, ENDED, self.net)

    def plan(self, step: int) -> StepPlan:
        # the curriculum's lengths, or the whole distribution's every step (a step then a block of epochs);
        # a crucible's (D26) its warm-up's last step's every block, then, every factor hard, the whole's
        # growth begins at the first block to start with every factor hard, and is kept (a resumed block
        # within it reads it from its checkpoint, not from the levels the block's epochs moved)
        c, s = self.c, self.s
        if (
            not s.grown_from
            and not (self.place is not None and self.place.step == step)
            and s.fixed > 0
            and c.anneal is not None
            and all(not c.anneal.soft(level) for level, sn in zip(s.annealed, self.snapped, strict=True) if sn)
        ):
            s.grown_from = step
        grown = s.grown_from > 0
        flat = c.whole or grown
        longest = s.fixed or step
        growing = c.continuous is not None and s.fixed > 0
        screens, read_at = step_screens(
            c.encoding, c.settings, c.cap, longest, flat, s.fixed if growing else 0, s.cstate.length, self.cached
        )
        return StepPlan(step, screens, read_at, flat, longest, grown, growing)

    def begin_step(self, p: StepPlan) -> int:
        """The step's start: its own state set (or resumed), its first row; the epoch it goes on from."""
        c, s = self.c, self.s
        epoch = 0
        if self.place is not None and self.place.step == p.step:  # resumed within this step
            self.warming()  # the loaded network continuous while warming, as the stopped run's was
            s.best, s.since, epoch = self.place.best, self.place.since, self.place.epoch
            s.rate, s.forks, s.peak = self.place.rate, self.place.forks, self.place.peak
            s.last_kinds, s.step_rollbacks = self.place.last_kinds, self.place.step_rollbacks
            s.machine_best = list(self.place.machine_best)
        else:
            s.best, s.since = -1.0, 0
            s.machine_best = []
            s.rate, s.forks = c.settings.rate, 0
            self.warming()
            start_score, start_kinds = screen(
                self.net, c.encoding, p.screens
            )  # the step before's network on this step's lengths
            self.record(
                {
                    "step": p.step,
                    "epoch": 0,
                    "screen": round(start_score, 4),
                    "kinds": _kinds(start_kinds),
                    **({"crucible": "grown" if p.grown else s.fixed} if c.crucible and s.fixed else {}),
                }
            )
            s.last_kinds, s.step_rollbacks = start_kinds, 0
            s.gate_screen = start_score  # D29: the gate reads the screen before the epoch
            s.peak = 0.0
            if c.fork is not None:
                s.peak = c.fork.trigger.value(start_score, start_kinds)
                self.net.save(str(self.snapshot))
        if epoch == 0:
            self.checkpoint(p.step, 0)
        return epoch

    def run_step(self, p: StepPlan) -> bool:
        """One step, its epochs and its end; whether the run stops after it."""
        c, s = self.c, self.s
        stop = "budget"
        s.score = -1.0  # the last epoch's screen (WarmUp's end reads it with the step's best)
        epoch = resumed = self.begin_step(p)
        for epoch in range(resumed + 1, c.settings.max_epochs + 1):
            if c.budget_seconds is not None and time.perf_counter() - self.start > c.budget_seconds:
                stop = "deadline"
                self.deadline = True
                break
            if c.budget_epochs is not None and s.spent >= c.budget_epochs:
                stop = "spent"
                break
            if s.cstate.backtrack is not None and s.cstate.backtrack.exhausted:
                stop = "spent"  # the incremental reheat's D7: back to the first stage, not recovered
                break
            ended = self.run_epoch(p, epoch)
            if ended == "fork":
                continue
            if ended is not None:
                stop = ended
                break
            self.checkpoint(p.step, epoch)
        if self.end_step(p, stop, epoch):
            return True
        self.warm_up_end(p)
        return stop in ("deadline", "spent", "stalled")

    def set_epoch(self, p: StepPlan, epoch: int) -> EpochSetting:
        """The epoch's setting before it trains: the thermostat's, the warm-up's, the continuous or annealed profile's."""
        c, s, net = self.c, self.s, self.net
        thermo: dict[str, Any] = {}
        melting = False  # WarmUp: in the profile's heat or hold (D18)
        phase: str | None = None  # WarmUp: the profile's phase this epoch trains in
        if c.thermostat is not None:
            # the network about to train (a fork's restored one included), read on the step's screen at its
            # shortest, middle and longest lengths
            effective = effective_codes(net, p.sample)
            s.levels = c.thermostat.levels(s.levels, effective, self.counts, s.best)  # best: the step's best so far
            beta = c.thermostat.commitments(s.levels, self.commitments)
            net.set_commitments(beta)
            thermo = {
                "thermostat": {"effective": [round(e, 2) for e in effective], "beta": [round(b, 4) for b in beta]}
            }
        if s.warm:
            # WarmUp: every snapped factor continuous, its commitment 0 (D4, D9)
            self.warming()
            net.set_commitments([0.0] * len(self.snapped))
            thermo = {"warm": True}
        elif self.continuing:
            # D48-D51: the epoch's clock, temperatures, couplings and readings (continuous_epoch)
            assert c.anneal is not None and c.continuous is not None and c.warm_up is not None
            self.warming()
            moved = continuous_epoch(
                net,
                c.continuous,
                c.warm_up,
                p.sample,
                self.snapped,
                s.cstate,
                s.spent - s.anneal_from,
                s.gate_screen,
                s.level,
                c.anneal.hard,
                c.anneal.commitments(s.annealed, self.commitments),
                c.coupling_cap,
                np.random.default_rng([c.seed, p.step, epoch, s.forks]),
                c.cap,
            )
            s.cstate = moved.state
            phase, s.coupled, thermo = moved.phase, moved.couplings, moved.thermo
            melting = phase in ("heat", "hold")
        elif c.anneal is not None:
            phase, thermo = self.anneal_epoch(p, epoch, c.anneal)
            melting = phase in ("heat", "hold")
        return EpochSetting(thermo, phase, melting)

    def anneal_epoch(self, p: StepPlan, epoch: int, anneal: Anneal) -> tuple[str | None, dict[str, Any]]:
        # the network about to train (a fork's restored one included), its soft factors' hardness read
        # at the temperature they trained at, on the step's screen at its shortest, middle and longest;
        # after a warm-up, each factor's temperature its profile's from its T_hot (D15), hardness read
        # from the cool's start or a shaped cool's end (D24), the soft codes moved by thermal noise (D17)
        c, s, net, warm_up = self.c, self.s, self.net, self.c.warm_up
        self.warming()
        holding = (
            warm_up is not None
            and bool(s.hot)
            and warm_up.holds(s.spent - s.anneal_from - s.held, s.gate_screen, s.level)
        )
        if holding:
            s.held += 1  # D29: this epoch trains at the last one's clock
        clock = s.spent - s.anneal_from - s.held  # the profile's epochs (D29: less those held)
        phase = warm_up.phase(clock) if warm_up is not None and s.hot else None
        reading = phase is None or (warm_up is not None and warm_up.reads_hardness(clock))
        each = (
            [warm_up.temperature(clock, t) for t in s.hot]
            if warm_up is not None and s.hot
            else [anneal.temperature(s.spent)] * len(self.snapped)
        )
        if s.cooling is not None and warm_up is not None and phase == "cool":
            if not holding:  # D33: the adaptive cool steps from its measurements; D29's gate holds it too
                s.cooling.advance(warm_up, s.melting_points, s.last_splits)
            each = list(s.cooling.temperature)
        soft = [
            t if sn and level == 0 and reading else None
            for t, sn, level in zip(each, self.snapped, s.annealed, strict=True)
        ]
        hard = hardness(net, p.sample, soft, anneal.hard)
        s.annealed = anneal.levels(s.annealed, hard)
        temperatures = [
            t if sn and anneal.soft(level) else None
            for t, sn, level in zip(each, self.snapped, s.annealed, strict=True)
        ]
        beta = anneal.commitments(s.annealed, self.commitments)
        profiled: dict[str, Any] = {}
        couplings = [1.0] * len(self.snapped)
        if phase is not None and warm_up is not None:
            jitter(net, temperatures, warm_up.kappa, np.random.default_rng([c.seed, p.step, epoch, s.forks]))
            profiled = {"phase": phase, "groups": code_groups(net, temperatures, warm_up.kappa)}
            # D32's logging, and the adaptive cool's reading (D33)
            heat = thermals(net, p.sample, temperatures)
            profiled["energy"] = [None if e is None else round(e[0], 6) for e in heat]
            profiled["capacity"] = [None if e is None else round(e[1], 6) for e in heat]
            if s.cooling is not None and phase == "cool":  # D33: the measurements the next step reads
                s.cooling.record(heat)
                profiled["at"] = [len(e) for e in s.cooling.energies]
            if warm_up.handover:  # D25: each soft factor's coupling from its temperature
                couplings = [
                    1.0 if t is None else warm_up.coupling(clock, t, m, f)
                    for t, m, f in zip(temperatures, s.melting_points, s.last_splits, strict=True)
                ]
                profiled["coupling"] = [round(x, 4) for x in couplings]
            if warm_up.gate:
                profiled["held"] = s.held
        if c.coupling_cap < 1.0:  # D38: the coupling probe's cap, every factor, every phase
            couplings = [min(x, c.coupling_cap) for x in couplings]
            profiled["coupling"] = [round(x, 4) for x in couplings]
        net.set_temperatures(temperatures)
        net.set_couplings(couplings)
        s.coupled = couplings
        net.set_commitments(beta)
        thermo = {
            "anneal": {
                "temperature": temperatures,
                "hard": [None if h is None else round(h, 4) for h in hard],
                "beta": [round(b, 4) for b in beta],
                **profiled,
            }
        }
        return phase, thermo

    def run_epoch(self, p: StepPlan, epoch: int) -> str | None:
        """One epoch, trained, screened and logged; how the step ends after it ("fork": goes on from its snapshot), or None."""
        c, s = self.c, self.s
        rng = random.Random(f"family-train-{c.seed}-{p.step}-{epoch}" + (f"-fork{s.forks}" if s.forks else ""))
        drawn = (
            growth_lengths(s.cstate.length, c.settings.epoch_examples)
            if p.growing
            else epoch_lengths(c.cap if p.flat else p.longest, c.settings.epoch_examples, c.long_tail, c.cap)
        )
        examples = [c.encoding.example(c.encoding.draw(n, rng)) for n in drawn]
        if c.counterexamples is not None and s.words:
            # R4 (D6): a share of the epoch from the last diagnosis's counterexamples, each word then d
            # uniform tokens and the final zero column (encoding.draw), so its answers tell its states apart
            taken = round(c.counterexamples * len(examples))
            examples[:taken] = [
                c.encoding.example(s.words[i % len(s.words)] + c.encoding.draw(self.walked_d, rng))
                for i in range(taken)
            ]
        if c.commitment_from is not None:
            self.net.set_commitments([0.0 if p.step < c.commitment_from else x for x in self.commitments])
        setting = self.set_epoch(p, epoch)
        guarded: dict[str, Any] = {}
        if self.guarding is not None:
            if certified_right(c.encoding, s.last_kinds):
                self.guarding.arm(self.net)
            else:
                self.guarding.disarm()
            updates, rollbacks = self.guarding.updates, self.guarding.rollbacks
            wr.train_epoch(self.net, examples, c.settings.batch_size, s.rate, rng, self.guarding)
            updates, rollbacks = self.guarding.updates - updates, self.guarding.rollbacks - rollbacks
            guarded = {"armed": self.guarding.armed, "updates": updates, "rollbacks": rollbacks}
            s.step_rollbacks += rollbacks
            s.frozen = s.frozen + 1 if updates and rollbacks >= STALL_SHARE * updates else 0
        else:
            wr.train_epoch(self.net, examples, c.settings.batch_size, s.rate, rng)
        s.spent += 1
        s.score, kinds = screen(self.net, c.encoding, p.screens)
        s.gate_screen = s.score
        if setting.phase == "hold":
            s.level = max(s.level, s.score)  # D29: the liquid's level, the gate's reference
        s.last_kinds = kinds
        if c.fork is not None and self.forked(p, epoch, kinds, guarded, setting.thermo):
            return "fork"
        self.record_epoch(p, epoch, kinds, guarded, setting.thermo)
        if s.frozen >= STALL_EPOCHS:
            return "stalled"
        if s.score == 1.0:
            return "passed"
        if p.growing:  # D53: after the warm-up a step is one epoch, its readings each one's
            return "epoch"
        if s.score > s.best or setting.melting:  # the melt's screen is its best (D18)
            s.best, s.since = s.score, 0
            if c.restore_best:
                self.net.save(str(self.step_best))
        else:
            s.since += 1
            if s.since >= c.settings.patience:
                return "plateau"
        return None

    def forked(
        self, p: StepPlan, epoch: int, kinds: dict[str, float], guarded: dict[str, Any], thermo: dict[str, Any]
    ) -> bool:
        """The fork (Fork) after an epoch: the peak kept, or, on a collapse, the snapshot restored and the rate cooled; whether it forked."""
        c, s = self.c, self.s
        assert c.fork is not None
        value = c.fork.trigger.value(s.score, kinds)
        if value > s.peak:
            s.peak = value
            self.net.save(str(self.snapshot))
            return False
        if not (c.fork.trigger.fires(value, s.peak) and s.forks < c.fork.limit):
            return False
        s.forks += 1
        s.rate *= c.fork.cool
        self.net = load_network(str(self.snapshot))
        restored, s.last_kinds = screen(self.net, c.encoding, p.screens)
        grouped = {"consolidation": self.regroup()} if c.consolidate else {}
        self.record(
            {
                "step": p.step,
                "epoch": epoch,
                "fork": s.forks,
                "screen": round(s.score, 4),
                "kinds": _kinds(kinds),
                "peak": round(s.peak, 4),
                "restored": round(restored, 4),
                "rate": s.rate,
                **guarded,
                **thermo,
                **grouped,
            }
        )
        self.checkpoint(p.step, epoch)
        return True

    def record_epoch(
        self, p: StepPlan, epoch: int, kinds: dict[str, float], guarded: dict[str, Any], thermo: dict[str, Any]
    ) -> None:
        """The epoch's row; every `diagnose_every`, with the diagnosis and R3's and R4's walks."""
        c, s, net = self.c, self.s, self.net
        if epoch % c.settings.diagnose_every != 0:
            self.record(
                {
                    "step": p.step,
                    "epoch": epoch,
                    "screen": round(s.score, 4),
                    "kinds": _kinds(kinds),
                    **guarded,
                    **thermo,
                }
            )
            return
        walked: dict[str, Any] = {}
        if c.keep_machine or c.counterexamples is not None:
            walk = mr.product(QueryMachine(net, c.encoding, 0), c.encoding.task.machine)
            summary = mr.summary(walk, self.walked_d)
            walked["machine"] = summary
            if c.keep_machine:  # R3 (D5): the fewest pairs per code, ties to more right pairs
                key = [summary["pairs"] / summary["codes"], -float(summary["right"])]
                if not s.machine_best or key < s.machine_best:
                    s.machine_best = key
                    net.save(str(self.machine_kept))
                    walked["kept"] = True
            if c.counterexamples is not None:
                s.words = mr.counterexamples(walk, self.walked_d, COUNTEREXAMPLE_WORDS)
                walked["counterexamples"] = len(s.words)
        self.record(
            {
                "step": p.step,
                "epoch": epoch,
                "screen": round(s.score, 4),
                "kinds": _kinds(kinds),
                **guarded,
                **thermo,
                **diagnose(net, c.encoding, c.settings),
                **({"consolidation": self.regroup()} if c.consolidate else {}),
                **walked,
            }
        )

    def end_step(self, p: StepPlan, stop: str, epoch: int) -> bool:
        """The step's end: a plateau's restores, the network saved and certified, its row; whether the run stops."""
        c, s = self.c, self.s
        if c.keep_machine and stop == "plateau" and s.machine_best:
            self.net = load_network(str(self.machine_kept))
            restored = screen(self.net, c.encoding, p.screens)[0]
            self.record(
                {
                    "step": p.step,
                    "epoch": epoch,
                    "restore_machine": round(s.machine_best[0], 3),
                    "restored": round(restored, 4),
                }
            )
        if c.restore_best and stop == "plateau":
            self.net = load_network(str(self.step_best))
            restored = screen(self.net, c.encoding, p.screens)[0]
            grouped = {"consolidation": self.regroup()} if c.consolidate else {}
            self.record(
                {"step": p.step, "epoch": epoch, "restore": round(s.best, 4), "restored": round(restored, 4), **grouped}
            )
        net = self.net
        net.save(str(c.out / "network.json"))
        if not p.growing or (c.continuous is not None and p.step % c.continuous.save_every == 0):  # D53: every 10th
            net.save(str(c.out / f"step-{p.step}.network.json"))  # every step's kept, to inspect what a run had
        # each step's certified answer (the quotient) certified (under a second at one column a window); the run stops at
        # its first certified step: training past it can destroy the network (E8b; this probe's
        # first runs, both certified networks lost within 3 steps)
        if not net.recurrence.snapped:  # T3's continuous state: no machine to certify; judged by long tests
            return self.end_continuous_step(p, epoch, stop)
        certificate = wc.certify(QueryMachine(net, c.encoding, 0), self.task)
        stopping = certificate.certified
        judged: dict[str, Any] = {}
        if c.state_bound is not None:  # D35: the answers-only stop; the certificate only judges it
            stopping, judged = answers_stopping(
                net, c.encoding, self.task, certificate, c.state_bound, s.coupled, s.warm
            )
            if s.margin.wait is not None:  # D59-D60: the stop waits until no near tie splits the machine
                stopping = s.margin.judge(p.step, stopping, certificate, judged, net, c.out)
        if (
            p.growing and c.continuous is not None
        ):  # D53: both certificates' breakdowns, the snapped answers, the melting curve
            judged.update(
                continuous_readings(net, c.encoding, self.task, certificate, s.coupled, c.state_bound is None)
            )
            if p.step % c.continuous.curve_every == 0:
                judged["melting"] = melting_curves(net, p.sample)
        if (
            s.succeeded < 0
            and certificate.certified
            and judged.get("certificates", {}).get("strict", {}).get("certified")
        ):
            s.succeeded = p.step  # D41's success: both verdicts
        # the batch runner's cutoff (its D1, in epochs: the time axis): an attempt not certified at both
        # verdicts by the first step end at `cutoff_epochs` epochs trained stops there, "cut" (a run
        # already certified goes on to its own stop)
        cut = c.cutoff_epochs is not None and s.spent >= c.cutoff_epochs and s.succeeded < 0 and not stopping
        self.record(
            {
                "step": p.step,
                "end": "cut" if cut else stop,
                "epochs": epoch,
                **({"forks": s.forks} if c.fork is not None else {}),
                **({"rollbacks": s.step_rollbacks} if self.guarding is not None else {}),
                **diagnose(net, c.encoding, c.settings),
                "long": long_accuracy(net, c.encoding, c.settings),
                "certified": certificate.certified,
                "minimised": len(certificate.minimised),
                "machine": machine_summary(net, c.encoding),
                **judged,
            }
        )
        if certificate.certified:
            net.save(str(c.out / f"certified-step-{p.step}.network.json"))
        if stopping and c.state_bound is not None:
            net.save(str(c.out / f"answers-stop-step-{p.step}.network.json"))
        return stopping or cut

    def end_continuous_step(self, p: StepPlan, epoch: int, stop: str) -> bool:
        """A continuous state's step end (the T3 baseline): its answers and long accuracy, never certified."""
        c, s = self.c, self.s
        cut = c.cutoff_epochs is not None and s.spent >= c.cutoff_epochs
        self.record(
            {
                "step": p.step,
                "end": "cut" if cut else stop,
                "epochs": epoch,
                **({"forks": s.forks} if c.fork is not None else {}),
                **diagnose(self.net, c.encoding, c.settings),
                "long": long_accuracy(self.net, c.encoding, c.settings),
                "certified": False,
            }
        )
        return cut

    def warm_up_end(self, p: StepPlan) -> None:
        """A warm step's distinct states and, at the warm-up's end (D14-D16), the codes from the states seen and the profile's start."""
        c, s, warm_up = self.c, self.s, self.c.warm_up
        if not (s.warm and warm_up is not None):
            return
        sample = p.sample
        distinct = distinct_states(self.net, sample)
        self.record({"step": p.step, "distinct": distinct})
        if not warm_up.ends(p.step, self.first, max(s.best, s.score), distinct, self.counts_snapped):
            return
        # WarmUp's end (D14-D16): the codes from the states seen, each factor's T_hot, the profile from here
        s.warm = False
        self.warming()
        rng_codes = np.random.default_rng([c.seed, p.step])
        made = codes_from_states(self.net, sample, rng_codes, warm_up.iterations, warm_up.hot)
        s.hot = [m.get("hot", 1.0) for m in made]
        s.melting_points, s.last_splits = handover_window(made, warm_up.window)
        s.anneal_from = s.spent
        s.cooling = Cooling.start(s.hot) if warm_up.shape == "adaptive" else None
        s.fixed = p.step if c.crucible else 0
        if c.continuous is not None:  # D51, D52: no temperature held yet; growth from the crucible's length
            s.cstate = ContinuousState(
                0.0,
                list(s.hot),
                [0.0] * len(self.snapped),
                [],
                float(s.fixed),
                backtrack=Backtrack() if c.continuous.incremental else None,
            )
        self.record({"step": p.step, "warm_up_end": made, **({"crucible": s.fixed} if c.crucible else {})})
        self.net.save(str(c.out / "network.json"))


def run(
    encoding: Target,
    factors: Sequence[tuple[int, int]],
    seed: int,
    budget_seconds: float | None,
    out: Path,
    settings: Settings = PROBE,
    cap: int | None = None,
    reads: str = BEFORE_STATE,
    budget_epochs: int | None = None,
    fork: Fork | None = None,
    guard: bool = False,
    resume: bool = False,
    long_tail: int | None = None,
    consolidate: bool = False,
    restore_best: bool = False,
    commitment_from: int | None = None,
    width: int = wp.D,
    branches: bool = False,
    answer_branch: bool = False,
    thermostat: Thermostat | None = None,
    keep_machine: bool = False,
    counterexamples: float | None = None,
    anneal: Anneal | None = None,
    warm_up: WarmUp | None = None,
    whole: bool = False,
    crucible: bool = False,
    state_bound: int | None = None,
    coupling_cap: float = 1.0,
    continuous: Continuous | None = None,
    margin_wait: int | None = None,
    cutoff_epochs: int | None = None,
    baseline: str | None = None,
    recombine: bool = False,
    branch_heads: int = 1,
) -> None:
    """
    One probe run: the curriculum from its first step `s0` (first_length: 1 but where a stateless
    network could pass it; a step named by its longest length) until the budget (epochs trained in
    all, `budget_epochs`, the binding budget and the run's time axis: they don't depend on the machine
    or its load; `budget_seconds`, wall-clock, only a runaway guard, None for none; whichever is spent
    first, a "deadline" stop censored like a "spent" one) or the cap (by default the cell's exclusion bound), a line of JSON per diagnosis to
    out/log.jsonl, the network saved at each step's end to out/network.json. With `fork`, a collapse
    within a step goes back to the step's best network and forks (Fork; a log line each). With
    `guard`, the certificate guard on the certified answer against its task's machine
    (certificate_guard), armed for an epoch whose screen before it got every one right (the guard
    workplan's D5);
    each epoch's row logs whether it was armed, its updates and those undone, each step's end the
    undone's sum. A run whose guard undoes STALL_SHARE of the updates of STALL_EPOCHS armed epochs in
    a row ends "stalled" (D9), its step's network certified as any step's.

    The code-consolidation workplan's arms: `long_tail` F (B, D4) trains every other example at
    lengths step + 1..F step, capped at `cap` (epoch_lengths); `consolidate` (A, D1) reads the
    network's groups at each diagnosis, at D2's `d`, and holds each duplicate code's states to its
    representative's codes until the next (network.consolidate), a fork's restored network
    regrouped; each such row logs `consolidation`, the codes reached, the groups and the duplicates
    pulled. `restore_best` (stage 2c's P2) saves the network at each new best screen of a step
    (out/best.network.json) and, when the step ends on a plateau, restores it before the step's end
    (its save and certificate) and the next step, regrouped with A, as a fork restores its snapshot;
    a row logs the step's best screen (`restore`) and the restored network's (`restored`).
    `commitment_from` J (the commitment-control workplan's diagnostic, D1 (b)) trains every step
    before J with each factor's commitment 0 and from J with its specs' (set before each epoch, so a
    fork's or a restore's loaded network has it too). `anneal` (Anneal) sets each snapped factor's
    temperature and commitment before each epoch from its level, read on the step's screen at its
    shortest, middle and longest lengths, a row's `anneal` logging them. `warm_up` (WarmUp, with `anneal`; the
    state formation workplan's option 1) runs every snapped factor continuous from the start, each
    row logging `warm`, each warm step's end its distinct states (`distinct`), until a step's end
    where it ends: its codes set from the states seen, a row logging `warm_up_end`, then each factor's
    temperature profiled from its T_hot (heat, hold, cool), its soft codes moved by thermal noise,
    each row's `anneal` logging the phase and the codes' groups; no step's best is kept during the
    heat and the hold, and nothing goes back (no fork, restore_best or keep_machine: D18). The
    machine's readings, its certificate included, read the snapped network throughout.
    `whole` (the state formation workplan's D22) drops the curriculum: every epoch's examples drawn
    over every length 1..cap in turn, every step's screen over whole_lengths(cap); a step is then a
    block of epochs ending, as any, in its certificate, and blocks go on past the cap until the budget
    (wall-clock or epochs) or a certificate ends the run.
    `crucible` (with `warm_up`; the state formation workplan's D26, a seed grown in its own crucible)
    holds the material at the warm-up's end: every block after it trains and screens the warm-up's
    last step's lengths, so the whole profile crystallises the state the curriculum formed on one
    material; once every snapped factor is hard, the seed grows: every block from the next on is the
    whole distribution's (as `whole`), its codes hard, the forward path coupled to them.
    `state_bound` N (the state formation workplan's D35) stops the run by its answers, not its
    certificate: at the first step's end (after the warm-up) whose snapped machine answers every
    word of ANSWERS_LENGTHS right and has, minimised, at most N states (answers_stopping), saving
    out/answers-stop-step-S.network.json; the certificate is still read at every step's end, only to
    judge the stop (a run certified before it trains on). `coupling_cap` (D38) caps every factor's
    coupling, every phase, so at 0 the codes anneal on the cells' outputs and never enter the
    forward path; the certificate reads the snapped network as ever.
    `continuous` (Continuous, with a crucible's exponential cool, no handover and no gate; the state
    formation workplan's D48) makes every switch after the warm-up continuous: the clock's rate (D49),
    the coupling from the codes' sharpness (D50), soft to the end with a straight-through gradient
    (D51; the levels never move), growth from the crucible (D52); each step then one epoch, its
    readings each one's (D53): each row's `anneal` logging the sharpness, D8's share above `hard`, the
    rate, the clock and the longest length; each step's end the strict certificate (snap ratio 1.0),
    both certificates' wrong states, outputs and near ties, the snapped answers (without a state bound,
    whose stop reads them) and, every `curve_every`, the melting curve.
    `margin_wait` W (with `state_bound`; the state formation workplan's D59) makes D35's stop wait for its
    margins: a step D35's stop passes stops the run only when no near tie splits the snapped machine
    (the certificate's split_ties: a tie's branch into another of its minimised states) and no label is
    a near tie; the first such pass's network kept (out/first-pass-step-S.network.json), and each pass
    with fewer ties than any before it (out/best-pass-step-S.network.json). W steps after the first
    pass, each step counted whether it passes or not (D60), a run none cleared stops ("waited"), its
    last row the best pass's network restored and judged as its stop's (Margin).
    `cutoff_epochs` T (with `continuous`; the batch runner workplan's D1, in epochs since stage 5) ends
    an attempt at the first step's end with T epochs trained in all ("cut") unless some step by then
    was certified at both verdicts (D41), so restarts at a fixed
    cutoff (Luby et al. 1993) need no runner to stop a run; a run already certified goes on to its stop.

    Each step's start and each epoch's end that the step goes on from leave out/checkpoint.json:
    the run's place (Place) and its network, with its optimizer's and generator's state. With
    `resume`, a run stopped part way goes on from its checkpoint (the code-consolidation workplan's
    stage 3): its log cut back to the checkpoint's rows, an epoch begun after it trained again, so
    every network and row but the seconds is the uninterrupted run's; the wall-clock guard counts
    the seconds trained. A run that ended isn't trained again; one without a checkpoint starts. A
    deadline's stop (the guard) logs its step's end but keeps the last epoch's checkpoint, so the
    same run resumed with a larger guard goes on to its epoch budget as if never stopped (its rows
    and step files after that checkpoint dropped); the guard isn't part of the run's identity.
    """
    config = RunConfig(
        encoding,
        factors,
        seed,
        budget_seconds,
        out,
        settings,
        exclusion_bound(encoding.target_states, factors) if cap is None else cap,
        reads,
        budget_epochs,
        fork,
        guard,
        long_tail,
        consolidate,
        restore_best,
        commitment_from,
        width,
        branches,
        answer_branch,
        thermostat,
        keep_machine,
        counterexamples,
        anneal,
        warm_up,
        whole,
        crucible,
        state_bound,
        coupling_cap,
        continuous,
        margin_wait,
        cutoff_epochs,
        baseline,
        recombine=recombine,
        branch_heads=branch_heads,
    )
    running = _Run(config, resume)
    if not running.ended:
        running.go()


def guard_epoch(
    path: Path, encoding: Target, step: int, settings: Settings = PROBE, generator: str = "guard-epoch"
) -> dict[str, Any]:
    """
    One epoch of a step's training from a saved network, unguarded and with the guard armed, on the
    same data: whether its certified answer's machine is right before, and for each its seconds, whether
    right after and the guard's updates and rollbacks (the guard workplan's budget probe).
    """
    task = encoding.task
    result: dict[str, Any] = {
        "network": str(path),
        "step": step,
        "right_before": cg.right(QueryMachine(load_network(str(path)), encoding, 0), task),
    }
    for arm in ("unguarded", "guarded"):
        net = load_network(str(path))
        rng = random.Random(f"{generator}-{step}")
        examples = [encoding.example(encoding.draw(n, rng)) for n in epoch_lengths(step, settings.epoch_examples)]
        guard = cg.Guard(lambda n: QueryMachine(n, encoding, 0), task) if arm == "guarded" else None
        start = time.perf_counter()
        if guard is not None:
            guard.arm(net)
        wr.train_epoch(net, examples, settings.batch_size, settings.rate, rng, guard)
        result[arm] = {
            "seconds": round(time.perf_counter() - start, 1),
            "right_after": cg.right(QueryMachine(net, encoding, 0), task),
            "updates": None if guard is None else guard.updates,
            "rollbacks": None if guard is None else guard.rollbacks,
        }
    return result


def readout_only(net: Any) -> wr.FactorRates:
    """
    The network's optimizer with every index at a rate of 0 but the answers' path: the output layer,
    and with branches the answers' whole branch (the window branches workplan), the layers after the
    last factor's branch's end. The rest keep their weights, so the state (each pooling reads the
    output layer's input, or its branch's end) is the saved network's, and only the readout of its
    answers learns.
    """
    ends = [i for i, layer in enumerate(net.layers) if isinstance(layer, BranchArrayLayer | MergeArrayLayer)]
    learning = range(ends[-1] + 1 if ends else len(net.layers) - 1, len(net.layers))
    return wr.FactorRates(net.optimizer, {i: 0.0 for i in range(len(net.layers) + len(net.parts)) if i not in learning})


def tune_readout(
    path: Path,
    encoding: Target,
    task_name: str,
    step: int,
    out: Path,
    epochs: int,
    settings: Settings = PROBE,
    generator: str = "readout-tune",
) -> list[dict[str, Any]]:
    """
    A saved network's output layer trained alone (readout_only) on step's lengths, an epoch at a time,
    until its certificate (the task's slot, the default snap ratio) passes or epochs are spent: a row
    an epoch (0: before), each with the certificate's counts, out/log.jsonl. The network is saved at
    the end (out/network.json) and, if certified, out/certified.network.json.
    """
    out.mkdir(parents=True, exist_ok=True)
    net = load_network(str(path))
    task, index = wt.task_named(task_name), slot(task_name)
    rows: list[dict[str, Any]] = []
    with (out / "log.jsonl").open("w") as log:
        for epoch in range(epochs + 1):
            if epoch:
                rng = random.Random(f"{generator}-{step}-{epoch}")
                examples = [
                    encoding.example(encoding.draw(n, rng)) for n in epoch_lengths(step, settings.epoch_examples)
                ]
                inner, net.optimizer = net.optimizer, readout_only(net)
                try:
                    wr.train_epoch(net, examples, settings.batch_size, settings.rate, rng)
                finally:
                    net.optimizer = inner
            certificate = wc.certify(QueryMachine(net, encoding, index), task)
            rows.append(
                {
                    "epoch": epoch,
                    "certified": certificate.certified,
                    "wrong_outputs": certificate.wrong_outputs,
                    "wrong_states": certificate.wrong_states,
                    "minimised": len(certificate.minimised),
                    "lead": round(certificate.lead.smallest, 4),
                    "lead_short": certificate.lead.short,
                }
            )
            log.write(json.dumps(rows[-1]) + "\n")
            log.flush()
            if certificate.certified:
                net.save(str(out / "certified.network.json"))
                break
    net.save(str(out / "network.json"))
    return rows


def separability(
    path: Path, encoding: Target, task_name: str, steps: int = 60_000, passes: int = 20_000, seed: int = 0
) -> dict[str, Any]:
    """
    Whether a saved network's answers (its task's slot) are linearly separable over the features its
    output layer reads (the window branches workplan's check): the features for every reachable code
    and every window, an unregularised softmax readout fitted to them all (full batch, Adam, steps
    steps, its fewest wrong at each 5,000), then a multiclass perceptron (0 errors in finitely many
    passes iff separable) for at most passes passes; for each pair the fit leaves wrong, the distance
    to the nearest feature with another answer and with its own.
    """
    net = load_network(str(path))
    machine, task = QueryMachine(net, encoding, slot(task_name)), wt.task_named(task_name).machine
    windows = wc.all_windows(1, task.tokens)
    want_labels, want_state = task.windows(windows)
    start, _ = machine.start()
    phi = {start: task.start}
    queue = [start]
    for q in queue:  # every reachable code, breadth first, its target state
        states = machine.step(q, windows).states
        for window in range(len(windows)):
            r = tuple(int(c) for c in states[window])
            if r not in phi:
                phi[r] = int(want_state[phi[q]][window])
                queue.append(r)
    codes = list(phi)
    queries = np.array(encoding.query_tokens, dtype=np.int64)
    full = np.concatenate([windows, np.repeat(queries[np.newaxis, :], len(windows), axis=0)], axis=1)
    books = machine.inner.E
    vectors = np.array([np.concatenate([books[g][i] for g, i in enumerate(q)]) for q in codes])
    net._window_forward(
        net.backend.matrix(np.tile(full, (len(codes), 1)).astype(np.float64).tolist()),
        net.backend.matrix(np.repeat(vectors, len(windows), axis=0).tolist()),
    )
    features = np.array(net._activations[-2].tolist(), dtype=np.float64)
    width = features.shape[1] // full.shape[1]
    index = slot(task_name)
    feats = features[:, index * width : (index + 1) * width]
    y = np.array([int(want_labels[phi[q]][w][0]) for q in codes for w in range(len(windows))])
    outputs = np.array(net._activations[-1].tolist())
    k = net.class_count
    network_wrong = int((outputs[:, index * k : (index + 1) * k].argmax(axis=1) != y).sum())
    design: npt.NDArray[np.float64] = np.hstack([feats, np.ones((len(feats), 1))])
    onehot = np.eye(k)[y]
    weights: npt.NDArray[np.float64] = np.zeros((design.shape[1], k))
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    fit: list[int] = []
    for step in range(1, steps + 1):
        z = design @ weights
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        g = design.T @ (p - onehot) / len(y)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        weights -= 0.01 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-12)
        if step % 5000 == 0:
            fit.append(int(((design @ weights).argmax(axis=1) != y).sum()))
            if fit[-1] == 0:
                break
    rng = np.random.default_rng(seed)
    u: npt.NDArray[np.float64] = np.zeros((design.shape[1], k))
    fewest, used = len(y), 0
    for used in range(1, passes + 1):
        errors = 0
        for i in rng.permutation(len(y)):
            guess = int(np.argmax(design[i] @ u))
            if guess != y[i]:
                u[:, y[i]] += design[i]
                u[:, guess] -= design[i]
                errors += 1
        fewest = min(fewest, errors)
        if errors == 0:
            break
    wrong: list[dict[str, Any]] = []
    for i in np.flatnonzero((design @ weights).argmax(axis=1) != y):
        distances = np.sqrt(((feats - feats[i]) ** 2).sum(axis=1))
        own = np.sort(distances[y == y[i]])
        other = distances[y != y[i]]
        q = codes[i // len(windows)]
        wrong.append(
            {
                "code": list(q),
                "state": phi[q],
                "window": int(i % len(windows)),
                "answer": int(y[i]),
                "other": round(float(other.min()), 4),
                "own": round(float(own[1]), 4) if len(own) > 1 else None,
            }
        )
    return {
        "network": str(path),
        "codes": len(codes),
        "states": len(set(phi.values())),
        "pairs": len(y),
        "features": width,
        "network_wrong": network_wrong,
        "fit_wrong": fit,
        "perceptron": {"passes": used, "fewest": fewest, "separable": fewest == 0},
        "wrong": wrong,
    }


class QueryMachine:
    """
    A family network as the certificate's WindowMachine over one column a window (stage 3): each
    window c is run as the network's window [c, queries...], and one slot's label read, slot 0 the
    quotient (against division's machine) or a machine's own output, slot i the i-th query's operation
    (the sum against the modular sum's), or with slot None every slot's (A's groups, read on all
    the network's outputs). The states, their snaps and margins are the network's own, as NetworkMachine's.
    """

    def __init__(self, network: Any, encoding: Target, slot: int | None) -> None:
        assert network.columns == encoding.width, "the network's window is the encoding's"
        self.inner = wc.NetworkMachine(network)
        self.encoding, self.slot = encoding, slot

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return self.inner.codes

    def start(self) -> tuple[wc.State, list[wc.FloatArray]]:
        return self.inner.start()

    def step(self, state: wc.State, windows: wc.IntArray) -> wc.Steps:
        return self.step_states([state], windows)

    def step_states(self, states: Sequence[wc.State], windows: wc.IntArray) -> wc.Steps:
        queries = np.array(self.encoding.query_tokens, dtype=np.int64)
        full = np.concatenate([windows, np.repeat(queries[np.newaxis, :], len(windows), axis=0)], axis=1)
        steps = self.inner.step_states(states, full)
        s = self.slot
        if s is None:
            return steps
        return wc.Steps(steps.digits[:, s : s + 1], steps.leads[:, s : s + 1], steps.states, steps.distances)


SLOTS = {"modular-sum": 1}  # the task each slot is certified against; any other task, slot 0


def slot(task_name: str) -> int:
    """The window slot a task is certified and long-tested on: the modular sum's 1, any other task's 0."""
    return SLOTS.get(task_name.rsplit("-", 1)[0], 0)


def certify_file(path: Path, encoding: Target, task_name: str, snap_ratio: float = wc.SNAP_RATIO) -> dict[str, Any]:
    """A saved family network's certificate, one slot against its task's machine, as certify_file's JSON."""
    net = load_network(str(path))
    index = slot(task_name)
    certificate = wc.certify(QueryMachine(net, encoding, index), wt.task_named(task_name), snap_ratio)
    return {
        "network": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "slot": index,
        "snap_ratio": snap_ratio,
        **certificate.to_json(),
    }


class QueryModel:
    """
    A family network as a task's catalogue reads it at a length n (the harness's WindowModel at k
    1, so every alignment 0): each case's columns and the final zero column as the encoding's
    windows, one slot's labels read, in runs of at most the harness's CHUNK_TOKENS tokens.
    """

    def __init__(self, network: Any, encoding: Target, task: wt.Task, slot: int) -> None:
        assert network.columns == encoding.width, "the network's window is the encoding's"
        self.network, self.encoding, self.task, self.slot = network, encoding, task, slot

    def predict(self, n: int) -> wt.Predict:
        width = self.encoding.width
        rows = max(1, wr.CHUNK_TOKENS // ((n + 1) * width))

        def labels(cases: Sequence[Any]) -> list[wt.Labels]:
            result: list[wt.Labels] = []
            for j in range(0, len(cases), rows):
                chunk = [self.encoding.example((*self.task.column_tokens(c, n), 0)) for c in cases[j : j + rows]]
                predictions = self.network.classify_rows(self.network.prepare_dataset(chunk))
                result.extend(tuple(prediction[self.slot :: width]) for prediction in predictions)
            return result

        return labels


def long_test(
    path: Path,
    encoding: Target,
    task_name: str,
    lengths: Sequence[int] = wr.LONG_LENGTHS,
    properties: Sequence[str] = LONG_PROPERTIES,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    A saved family network's long tests (the division workplan's D4), one slot against its task's
    catalogue at each length, sized as the harness's long_test, as its JSON (so
    window_recurrent_study.long_test_failures reads it), with the network's sha256 and the slot.
    """
    task = wt.task_named(task_name)
    index = slot(task_name)
    catalogue = task.catalogue(ad.SAMPLED, ad.RELATION_SAMPLED, ad.PER_STRATUM, doubling=True)
    tested = [p for p in wr.test_properties(task, catalogue) if p.id in properties]
    model = QueryModel(load_network(str(path)), encoding, task, index)
    results: dict[str, Any] = {}
    for n in lengths:
        start = time.perf_counter()
        checked = task.evaluate(model.predict(n), n, tested)
        if log is not None:
            for result in checked:
                failure = "" if result.holds else f", e.g. {result.failures[0]}"
                log(f"n {n}, {result.id}: {result.passed}/{result.cases}{failure}")
        results[str(n)] = {
            "properties": {r.id: task.result_json(r) for r in checked},
            "seconds": time.perf_counter() - start,
        }
    return {
        "network": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task": task_name,
        "slot": index,
        "lengths": results,
    }
