"""
The certificate note's fault injection (indrajala-research docs/workplans/certificate-note-workplan.md,
stage 2, D8): planted single-entry faults, the certificate's catch rate, and the exact chance that
a sampled long test finds each fault.

A certified network's Table (stage 1's witness) is a machine: TableMachine reads it as certify's
WindowMachine, each entry's labels and snapped next state, every margin wide. A fault changes one
(state, window) entry:

- "label": the window's first label moved on by one (mod the target's classes);
- "next": the snapped next state moved to the first other reachable state, in breadth-first order,
  whose target state differs (so the fault is a real defect, not a copy of the same state).

Entries are chosen by how often the long tests visit them: each entry's expected visits in one
uniform word of the shortest long length (the visit weight), the rarest, the median and the
commonest of the entries visited at all, per kind. Each faulty machine is certified (certify, the
instrument under test) against the task: caught when not certified, with the counterexample's
windows (its length).

The sampled tests' chance is exact, over their sampling distribution (window_tasks' catalogue: U1,
`sampled` uniform words of n tokens; S1, from each reachable target state's shortest prefix,
`per_stratum` uniform words each), not over one seed's draw. A case finds the fault when the faulty
machine's labels differ from the target's somewhere in it, the final zero window included (the
harness's). Under uniform tokens the pair (target state, faulty machine's state) is a Markov chain,
with "found" absorbing; its distribution after n / k windows and the final window gives the chance
p per case exactly, and a property's chance over its cases 1 - prod (1 - p_case). The catalogue's
chance is over every property and length.

    python scripts/certnote/certificate_faults.py TABLE.npz [...] --out faults.jsonl \\
        [--lengths 128 256 1024] [--sampled 100000] [--per-stratum 1000]

One JSON line per fault: the table, the task, the entry (its state's row, the window, the visit
weight and its rank), the kind, caught and the counterexample's length, and per property and length
the chance per case and over the property's cases, and the catalogue's chance of missing the fault.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_tasks as wt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certificate_check as cc  # scripts/ isn't a package

LENGTHS = (128, 256, 1024)  # the long tests' (window_recurrent_study.LONG_LENGTHS)
SAMPLED, PER_STRATUM = 100_000, 1_000  # U1's words and S1's per state (addition_data's SAMPLED, PER_STRATUM)
KINDS = ("label", "next")
RANKS = ("rarest", "median", "commonest")

IntArray = npt.NDArray[np.int64]


class TableMachine:
    """A Table as certify's WindowMachine: its rows' states, each window's recorded labels and snapped next state."""

    def __init__(self, meta: dict[str, Any], a: cc.Arrays) -> None:
        self.meta, self.a = meta, a
        self.row = {tuple(int(c) for c in s): i for i, s in enumerate(a["states"])}
        self.W = cc.windows(meta["columns"], meta["tokens"])
        self.radix: IntArray = int(meta["tokens"]) ** np.arange(int(meta["columns"]) - 1, -1, -1, dtype=np.int64)

    @property
    def columns(self) -> int:
        return int(self.meta["columns"])

    @property
    def codes(self) -> list[int]:
        return list(self.meta["codes"])

    def _distances(self, states: IntArray) -> list[wc.FloatArray]:
        out: list[wc.FloatArray] = []
        for g, m in enumerate(self.codes):
            d = np.full((len(states), m), 2.0)
            d[np.arange(len(states)), states[:, g]] = 1.0
            out.append(d)
        return out

    def start(self) -> tuple[wc.State, list[wc.FloatArray]]:
        start = self.a["states"][0].astype(np.int64)
        return tuple(int(c) for c in start), [d[0] for d in self._distances(start[np.newaxis, :])]

    def step(self, state: wc.State, windows: IntArray) -> wc.Steps:
        r = self.row[state]
        w: IntArray = windows @ self.radix
        labels = self.a["labels"][r, w].astype(np.int64)
        states = self.a["next"][r, w].astype(np.int64)
        return wc.Steps(labels, np.full(labels.shape, 50.0), states, self._distances(states))


def next_rows(a: cc.Arrays, codes: Sequence[int]) -> IntArray:
    """Each entry's snapped next state as its row, (R, W)."""
    R, G = a["states"].shape
    lookup = np.full(int(np.prod(codes)), -1, dtype=np.int64)
    lookup[cc.keys(a["states"], codes)] = np.arange(R)
    return lookup[cc.keys(a["next"].reshape(-1, G), codes)].reshape(R, -1)


def visit_weights(nxt: IntArray, n_windows: int) -> npt.NDArray[np.float64]:
    """Each entry's expected visits in one uniform word of n_windows windows from the start (row 0), (R, W)."""
    R, W = nxt.shape
    occupancy = np.zeros(R)
    p = np.zeros(R)
    p[0] = 1.0
    for _ in range(n_windows):
        occupancy += p
        moved = np.zeros(R)
        np.add.at(moved, nxt.reshape(-1), np.repeat(p / W, W))
        p = moved
    return np.repeat(occupancy[:, np.newaxis] / W, W, axis=1)


def chosen_entries(weights: npt.NDArray[np.float64]) -> dict[str, tuple[int, int, float]]:
    """The rarest, median and commonest visited entries (row, window, weight), ties to the first in order."""
    flat = weights.reshape(-1)
    visited = np.flatnonzero(flat > 0)
    order = visited[np.argsort(flat[visited], kind="stable")]
    picks = {"rarest": order[0], "median": order[len(order) // 2], "commonest": order[-1]}
    W = weights.shape[1]
    return {rank: (int(e // W), int(e % W), float(flat[e])) for rank, e in picks.items()}


def plant(a: cc.Arrays, nxt: IntArray, kind: str, row: int, window: int, classes: int) -> cc.Arrays | None:
    """A copy of the table's arrays with one fault planted (None: no other target state to move to)."""
    out = {name: v.copy() for name, v in a.items()}
    if kind == "label":
        out["labels"][row, window, 0] = (out["labels"][row, window, 0] + 1) % classes
        return out
    want = a["phi"][nxt[row, window]]
    others = np.flatnonzero(a["phi"] != want)
    if len(others) == 0:
        return None
    out["next"][row, window] = a["states"][others[0]]
    return out


def pair_chain(
    meta: dict[str, Any], a: cc.Arrays, target: wt.Mealy
) -> tuple[dict[tuple[int, int], int], list[npt.NDArray[np.int64]], list[npt.NDArray[np.float64]], Any, Any]:
    """
    The (target state, machine row) pairs reachable from (start, row 0) under every window, and each
    pair's transitions as (next pairs, probabilities), "found" (index -1) where the labels differ.
    """
    W = cc.windows(meta["columns"], meta["tokens"])
    nxt = next_rows(a, meta["codes"])
    want_labels, want_after = cc.target_windows(
        np.asarray(target.transition, dtype=np.int64), np.asarray(target.output, dtype=np.int64), W
    )
    pairs = {(int(target.start), 0): 0}
    queue = [(int(target.start), 0)]
    targets: list[npt.NDArray[np.int64]] = []
    probabilities: list[npt.NDArray[np.float64]] = []
    for s, r in queue:
        differs = (a["labels"][r] != want_labels[s]).any(axis=1)
        following = np.where(differs, -1, 0)
        keys = want_after[s] * len(nxt) + nxt[r]
        for key in np.unique(keys[~differs]):
            pair = (int(key // len(nxt)), int(key % len(nxt)))
            if pair not in pairs:
                pairs[pair] = len(pairs)
                queue.append(pair)
        index = np.array([pairs[(int(k // len(nxt)), int(k % len(nxt)))] for k in keys[~differs]], dtype=np.int64)
        following[~differs] = index
        values, counts = np.unique(following, return_counts=True)
        targets.append(values)
        probabilities.append(counts / len(W))
    # the final zero window's labels: found when they differ
    zero = 0
    final = np.array([(a["labels"][r, zero] != want_labels[s, zero]).any() for s, r in pairs], dtype=bool)
    return pairs, targets, probabilities, final, (want_after, nxt)


def found_by(
    targets: Sequence[npt.NDArray[np.int64]],
    probabilities: Sequence[npt.NDArray[np.float64]],
    final: npt.NDArray[np.bool_],
    horizons: Sequence[int],
) -> dict[int, npt.NDArray[np.float64]]:
    """
    For each horizon t, the chance from every pair that t uniform windows, then the zero window,
    find the fault (P,): one backward recursion, f_0 the final window's, f_(t+1)(i) the chance of
    finding it now plus sum_j p_ij f_t(j).
    """
    P = len(targets)
    rows = np.concatenate([np.full(len(t), i, dtype=np.int64) for i, t in enumerate(targets)])
    cols = np.concatenate([np.where(t >= 0, t, P) for t in targets])
    vals = np.concatenate(list(probabilities))
    wanted = set(horizons)
    f = final.astype(np.float64)
    out: dict[int, npt.NDArray[np.float64]] = {}
    for t in range(max(wanted) + 1):
        if t in wanted:
            out[t] = f
        extended = np.append(f, 1.0)  # "found" found
        f = np.minimum(np.bincount(rows, weights=vals * extended[cols], minlength=P), 1.0)
    return out


def found_chance(
    start: npt.NDArray[np.float64],
    targets: Sequence[npt.NDArray[np.int64]],
    probabilities: Sequence[npt.NDArray[np.float64]],
    final: npt.NDArray[np.bool_],
    n_windows: int,
) -> float:
    """The chance a word of n_windows uniform windows, then the zero window, from start (over pairs) finds the fault."""
    return float(start @ found_by(targets, probabilities, final, [n_windows])[n_windows])


def _log_miss(p: float, cases: int) -> float:
    """The log chance that cases independent cases, each finding the fault with chance p, all miss it."""
    return -np.inf if p >= 1.0 else cases * float(np.log1p(-p))


def fault_report(path: Path, lengths: Sequence[int], sampled: int, per_stratum: int) -> list[dict[str, Any]]:
    """Every planted fault of one table (the module's docstring)."""
    meta, a = cc.load(path)
    task = wt.task_named(meta["task"])
    target = task.machine
    k = int(meta["columns"])
    nxt = next_rows(a, meta["codes"])
    weights = visit_weights(nxt, min(lengths) // k)
    reports: list[dict[str, Any]] = []
    clean = wc.certify(TableMachine(meta, a), task, snap_ratio=1.0)
    for kind in KINDS:
        for rank, (row, window, weight) in chosen_entries(weights).items():
            faulty = plant(a, nxt, kind, row, window, target.classes)
            if faulty is None:
                continue
            begun = time.perf_counter()
            certificate = wc.certify(TableMachine(meta, faulty), task, snap_ratio=1.0)
            counterexample = certificate.counterexample
            pairs, targets, probabilities, final, _ = pair_chain(meta, faulty, target)
            chances: dict[str, dict[str, Any]] = {}
            log_miss = 0.0  # the catalogue's log chance of missing the fault, over every property and length
            prefixes = list(_prefixes(target).values()) if k == 1 else []
            starts = {tuple(prefix): _after_prefix(prefix, faulty, target, meta, pairs) for prefix in prefixes}
            horizons = {n // k for n in lengths} | {n - len(p) for n in lengths for p in prefixes if len(p) <= n}
            f = found_by(targets, probabilities, final, sorted(horizons))
            for n in lengths:
                p_u1 = float(f[n // k][0])
                log_u1 = _log_miss(p_u1, sampled)
                chances[str(n)] = {"U1": {"case": p_u1, "property": -float(np.expm1(log_u1))}}
                log_miss += log_u1
                if k == 1:
                    # S1: each reachable target state's shortest prefix, deterministically, then uniform
                    per_state = [
                        1.0 if starts[tuple(p)] is None else float(f[n - len(p)][starts[tuple(p)]])
                        for p in prefixes
                        if len(p) <= n
                    ]
                    log_s1 = float(sum(_log_miss(p, per_stratum) for p in per_state))
                    chances[str(n)]["S1"] = {
                        "case_min": min(per_state),
                        "case_max": max(per_state),
                        "property": -float(np.expm1(log_s1)),
                    }
                    log_miss += log_s1
            reports.append(
                {
                    "table": str(path),
                    "task": meta["task"],
                    "columns": k,
                    "rows": int(a["states"].shape[0]),
                    "clean_certified": clean.certified,
                    "kind": kind,
                    "rank": rank,
                    "row": row,
                    "window": window,
                    "visit_weight": weight,
                    "caught": not certificate.certified,
                    "counterexample_windows": None if counterexample is None else len(counterexample.windows),
                    "chances": chances,
                    "catalogue_miss": float(np.exp(log_miss)),
                    "catalogue_log10_miss": log_miss / float(np.log(10)) if np.isfinite(log_miss) else None,
                    "seconds": time.perf_counter() - begun,
                }
            )
    return reports


def _prefixes(target: wt.Mealy) -> dict[int, tuple[int, ...]]:
    """The shortest word (first in token order) to each reachable state (window_tasks' MachineTask._prefixes)."""
    prefixes: dict[int, tuple[int, ...]] = {target.start: ()}
    for s in target.reachable():
        for t in range(target.tokens):
            prefixes.setdefault(int(target.transition[s, t]), (*prefixes[s], t))
    return prefixes


def _after_prefix(
    prefix: Sequence[int], a: cc.Arrays, target: wt.Mealy, meta: dict[str, Any], pairs: dict[tuple[int, int], int]
) -> int | None:
    """The pair an S1 case's prefix leads to (k 1), or None when the prefix itself finds the fault."""
    nxt = next_rows(a, meta["codes"])
    s, r = int(target.start), 0
    for t in prefix:
        if (a["labels"][r, t] != target.output[s, t]).any():
            return None
        s, r = int(target.transition[s, t]), int(nxt[r, t])
    return pairs[(s, r)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("tables", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--sampled", type=int, default=SAMPLED)
    parser.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    args = parser.parse_args(argv)
    with args.out.open("a") as f:
        for path in args.tables:
            for report in fault_report(path, args.lengths, args.sampled, args.per_stratum):
                f.write(json.dumps(report) + "\n")
                print(
                    f"{path.name} {report['kind']} {report['rank']}: caught {report['caught']} "
                    f"({report['counterexample_windows']} windows), weight {report['visit_weight']:.3g}, "
                    f"catalogue miss {report['catalogue_miss']:.3g}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
