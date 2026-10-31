"""
Where a window-recurrent network's answers go wrong along a long input, and how far its state
moves from where it sat early on: the drift profile (the window-recurrent claims workplan, E2b,
where the continuous state's wrong digits were absent early and grew with the column).

On the first cases of the task's codes property (B1 for addition) at one length n, each at its
alignment (W23):

- **wrong digits by position**: in equal bins of the columns 0..n-1, and those past n (the final
  window's);
- **wrong digits by context**: the target machine's state entering the column and leaving it,
  against how often that context occurs;
- **the state's drift**: in each bin, for each state of the target machine, the network's states
  after each window that ends in the bin with the target in that state: their centroid's distance
  from the first bin's (the shift) and their RMS distance from their own centroid (the spread).
  Beside them the separation, the least distance between two of the target's states' first-bin
  centroids, the scale a shift is read against.

The inputs' columns are drawn independently, so past the first few columns a network whose state
keeps to a fixed set of regions has a shift near zero in every bin (a snapped one exactly, its
codes' mix aside); a shift growing with the bin is a state moving along the input. The first bin
holds the first columns' transient, a few columns of its n / bins at a long n.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from indrajala_studies import addition_data as ad
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt

CASES = 3_000
BINS = 8


@dataclass
class _Moments:
    """A set of state vectors' count, sum and sum of squared norms: its centroid and spread."""

    count: int = 0
    total: Any = 0.0
    square: float = 0.0

    def add(self, vectors: Any) -> None:
        self.count += len(vectors)
        self.total = self.total + vectors.sum(axis=0)
        self.square += float((vectors * vectors).sum())

    def centroid(self) -> Any:
        return self.total / self.count

    def spread(self) -> float:
        """The RMS distance from the centroid: sqrt(E||x||^2 - ||E[x]||^2)."""
        mean = self.centroid()
        return math.sqrt(max(0.0, self.square / self.count - float(mean @ mean)))


@dataclass
class _Tally:
    wrong: list[int]
    past: int = 0
    cases_wrong: int = 0
    contexts: Counter[tuple[int, int]] = field(default_factory=Counter[tuple[int, int]])
    wrong_contexts: Counter[tuple[int, int]] = field(default_factory=Counter[tuple[int, int]])
    moments: dict[tuple[int, int], _Moments] = field(default_factory=dict[tuple[int, int], _Moments])


def profile(network: Any, task: wt.Task, n: int, cases: int = CASES, bins: int = BINS) -> dict[str, Any]:
    """The drift profile of a saved network at length n (see the module's docstring)."""
    k: int = network.columns
    machine = task.machine
    catalogue = task.catalogue(cases, ad.RELATION_SAMPLED, ad.PER_STRATUM)
    codes_property = next(p for p in catalogue if p.id == task.codes_id)
    drawn = task.property_cases(codes_property, n)[:cases]
    aligned = [wr.alignment(task, case, n, k) for case in drawn]
    tally = _Tally(wrong=[0] * bins)
    for width in sorted({wr.data_windows(n, a, k) for a in aligned}):
        chosen = [i for i, a in enumerate(aligned) if wr.data_windows(n, a, k) == width]
        rows_per_pass = max(1, wr.CHUNK_TOKENS // ((width + 1) * k))
        for start in range(0, len(chosen), rows_per_pass):
            part = chosen[start : start + rows_per_pass]
            rows = [wr.example(task, drawn[i], n, k, aligned[i]) for i in part]
            _tally_rows(network, machine, rows, [aligned[i] for i in part], n, bins, tally)
    return _summary(tally, len(drawn), n, bins, machine.states)


def _tally_rows(
    network: Any,
    machine: wt.Mealy,
    rows: Sequence[wr.Example],
    alignments: Sequence[int],
    n: int,
    bins: int,
    tally: _Tally,
) -> None:
    k: int = network.columns
    outputs, states, _cells = network.window_states(network.backend.matrix([tokens for tokens, _ in rows]))
    predictions: list[tuple[int, ...]] = network.backend.argmax_token_rows(outputs, network.class_count)
    # the target's state before each token, each row's
    targets: list[list[int]] = []
    for (tokens, labels), predicted, a in zip(rows, predictions, alignments, strict=True):
        state, before = machine.start, list[int]()
        any_wrong = False
        for t, token in enumerate(tokens):
            after = int(machine.transition[state, int(token)])
            before.append(state)
            j = t - a
            if j >= 0:
                context = (state, after)
                tally.contexts[context] += 1
                if predicted[t] != labels[t]:
                    any_wrong = True
                    tally.wrong_contexts[context] += 1
                    if j < n:
                        tally.wrong[j * bins // n] += 1
                    else:
                        tally.past += 1
            state = after
        before.append(state)
        targets.append(before)
        tally.cases_wrong += any_wrong
    for w in range(1, len(states)):
        read = [w * k - a for a in alignments]  # the data columns read when this state is left
        vectors = np.array(states[w].tolist(), dtype=float)
        target = np.array([row[w * k] for row in targets])
        position = np.array(read)
        for b in range(bins):
            in_bin = (position >= 1) & (position <= n) & ((position - 1) * bins // n == b)
            if not in_bin.any():
                continue
            for s in np.unique(target[in_bin]):
                chosen = in_bin & (target == s)
                tally.moments.setdefault((b, int(s)), _Moments()).add(vectors[chosen])


def _distance(x: Any, y: Any) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)))


def _summary(tally: _Tally, cases: int, n: int, bins: int, machine_states: int) -> dict[str, Any]:
    first = {s: tally.moments[(0, s)].centroid() for s in range(machine_states) if (0, s) in tally.moments}
    pairs: list[tuple[Any, Any]] = list(itertools.combinations(first.values(), 2))
    separation = min((_distance(x, y) for x, y in pairs), default=0.0)
    drift = [
        {
            str(s): {
                "shift": round(_distance(tally.moments[(b, s)].centroid(), first[s]), 6),
                "spread": round(tally.moments[(b, s)].spread(), 6),
            }
            for s in sorted(first)
            if (b, s) in tally.moments
        }
        for b in range(bins)
    ]
    return {
        "n": n,
        "cases": cases,
        "cases_wrong": tally.cases_wrong,
        "wrong_by_bin": tally.wrong,
        "wrong_past_n": tally.past,
        "contexts": [
            {"entering": a, "leaving": b, "wrong": wrong, "occurrences": tally.contexts[(a, b)]}
            for (a, b), wrong in tally.wrong_contexts.most_common()
        ],
        "separation": round(separation, 6),
        "drift": drift,
    }


def lines(name: str, result: dict[str, Any]) -> list[str]:
    """A profile as text: the wrong digits by bin and context, the drift by bin against the separation."""
    n, bins = result["n"], len(result["wrong_by_bin"])
    out = [
        (
            f"{name} at {n}: {result['cases_wrong']}/{result['cases']} cases wrong, "
            f"{sum(result['wrong_by_bin']) + result['wrong_past_n']} wrong digits"
        ),
        f"  wrong digits by {n // bins}-column bin: {result['wrong_by_bin']}, past {n}: {result['wrong_past_n']}",
    ]
    out += [
        f"  entering {c['entering']}, leaving {c['leaving']}: {c['wrong']} wrong of {c['occurrences']}"
        for c in result["contexts"][:8]
    ]
    out.append(
        f"  state by bin, per target state: shift from the first bin / spread; separation {result['separation']:.4g}"
    )
    out += [
        f"   bin {b}: " + ", ".join(f"{s} {d['shift']:.3g}/{d['spread']:.3g}" for s, d in drift.items())
        for b, drift in enumerate(result["drift"])
    ]
    return out
