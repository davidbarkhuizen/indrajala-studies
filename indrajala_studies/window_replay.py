"""
A curriculum step replayed from a saved network (the snap wall workplan, retired: git show 53d7fef:docs/workplans/snap-wall-workplan.md): what further training
does to it, batch by batch, and whether its proof survives. From a saved step's network (optionally
converted to numpy, its weights, Adam's moments and generator kept; optionally with the snap wall
on), some epochs of a step's data drawn by a named generator, as a run's step draws them:

- **per batch**: the batch's loss before and after its update, and the update's norm (every
  parameter's change, the codebook's included), so a jump of the loss on an ordinary step (a state
  crossing a code boundary) and a run of large updates after it (Adam amplifying it) show;
- **per epoch**: the screen at the step's lengths, the certificate in both verdicts (near ties
  branched, and strict), and the code margins over the screen's first cases' states: the least
  snap ratio (every other code's distance over the snapped code's, per state; the certificate
  asks 1.05, the wall keeps 1.5) and per snapped factor the codes in use, the closest two of them
  and the farthest state from its code;
- **at the start**: the code margins of the network as loaded (with 0 epochs, only these);
- **per batch, with machines on**: the machine the network is after the update, read off by
  breadth first search over every window from every reachable state (snapped, no near ties
  branched), against the last batch's: the (state, window) pairs whose labels or next state
  changed, the states gained and lost, and the pairs wrong for the target (the certificate's
  count without its near ties). Whether a collapse starts as one transition flipped and then
  spreads, or as many at once. A change is logged as a line.
- **with the guard on** (machines implied): an update that makes a right machine (no wrong pair)
  wrong is undone, the network's weights, its optimizer's state and its generator restored to
  before the batch, and the batch skipped; per epoch the rollbacks counted. A remedy for the
  collapse acting where it starts: the certificate as a training control at the grain of a batch.
  Its training state is certificate_guard's; its machine is read by machine_table, a call a state,
  as its records were (certificate_guard.Guard is the guard for training, read by levels).

E2b' seed 6's replay from step 7 found the first: an ordinary step at batch 977 of step 8 took its
loss from 0.001 to 2.2, and the updates after it ran 6-8 times their size (the workplan's Why).
P1 (the workplan's stage 2) replayed 2 epochs of step 8 from that network's step 7, 10 times with
the snap wall off and 10 on: the proof was lost in 9 and in 8 (C1 refuted), the wall cutting the
loss's jumps about 3-fold but not the loss of the proof. The same replay, wall off, from the step-7
networks of 4 seeds that never lost a proof (1, 3, 8, 9), 5 times each, lost it in none of 20:
seed 6's collapse is its network's, not further training's in general, and neither its least snap
ratio nor its closest codes in use set it apart (the workplan's stage 2). With machines on, P1's
10 wall-off replays show every collapse starting as 1-3 (state, window) pairs flipped by an
ordinary update, silent to its batch's loss in 8, then Adam's amplified updates within 1-13
batches in 9 (the workplan's stage 2c). With the guard on, the same 10 replays all kept the proof
(registered: works if 8 of 10), 174 of 20,480 updates undone, 142 of them silent to their batch's
loss (the workplan's stage 2d).
"""

from __future__ import annotations

import math
import random
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.studies import certificate_guard as cg
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt

RATIO_CASES = 200  # the screen's first cases a length, for the least snap ratio


class Table(NamedTuple):
    """A machine read off a network: per reachable state each window's labels (N, k) and next state (N, G)."""

    start: wc.State
    rows: dict[wc.State, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
    wrong: int  # (state, window) pairs whose labels or next state is wrong for the target


def machine_table(machine: wc.WindowMachine, task: wt.Task) -> Table:
    """
    The machine reachable from machine's start over every window, each state's target state fixed
    where it is first reached (as the certificate fixes it), and its wrong (state, window) pairs.
    """
    target = task.machine
    windows = wc.all_windows(machine.columns, target.tokens)
    want_labels, want_state = target.windows(windows)
    start, _ = machine.start()
    phi = {start: target.start}
    queue = [start]
    rows: dict[wc.State, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = {}
    wrong = 0
    for q in queue:  # breadth first: the queue grows as states are reached
        parts = [machine.step(q, windows[i : i + wc.BATCH]) for i in range(0, len(windows), wc.BATCH)]
        digits = np.concatenate([p.digits for p in parts]).astype(np.int64)
        nexts = np.concatenate([p.states for p in parts]).astype(np.int64)
        rows[q] = (digits, nexts)
        unique, firsts, inverse = np.unique(nexts, axis=0, return_index=True, return_inverse=True)
        for j in np.argsort(firsts):
            r = tuple(int(c) for c in unique[j])
            if r not in phi:
                phi[r] = int(want_state[phi[q]][firsts[j]])
                queue.append(r)
        mapped = np.array([phi[tuple(int(c) for c in row)] for row in unique])[np.ravel(inverse)]
        wrong += int(((digits != want_labels[phi[q]]).any(axis=1) | (mapped != want_state[phi[q]])).sum())
    return Table(start, rows, wrong)


def table_changes(before: Table, after: Table) -> dict[str, int]:
    """The (state, window) pairs of the states both reach whose labels or next state differ, and the states gained and lost."""
    shared = before.rows.keys() & after.rows.keys()
    changed = 0
    for q in shared:
        (d0, n0), (d1, n1) = before.rows[q], after.rows[q]
        changed += int(((d0 != d1).any(axis=1) | (n0 != n1).any(axis=1)).sum())
    return {
        "pairs_changed": changed,
        "states_gained": len(after.rows.keys() - shared),
        "states_lost": len(before.rows.keys() - shared),
    }


def load(path: Path, numpy: bool = False) -> Any:
    """A saved network, as saved or converted to numpy (the same weights, optimizer state and generator)."""
    if not numpy:
        return load_network(str(path))
    text = path.read_text()
    marker = '"implementation": "'
    start = text.index(marker) + len(marker)
    converted = text[:start] + NUMPY.name + text[text.index('"', start) :]
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / path.name
        target.write_text(converted)
        return load_network(str(target))


def set_wall(network: Any, wall: float, ratio: float) -> None:
    """Every snapped factor's wall (numpy's codes only until the crate has it, the workplan's D8)."""
    for factor in network.recurrence_layer.factors:
        if factor.codes is not None:
            assert hasattr(factor.codes, "wall"), "the snap wall is numpy's only: convert the network (numpy=True)"
            factor.codes.wall, factor.codes.wall_ratio = wall, ratio


def _parameters(network: Any) -> npt.NDArray[np.float64]:
    # every parameter, flattened, the codebook's included (numpy's or the crate's arrays)
    parts: list[Any] = []
    for part in network.snapshot():
        for value in part:
            values: list[Any] = value.tolist()  # numpy's arrays and the crate's alike
            parts.append(np.ravel(np.asarray(values, dtype=float)))
    return np.concatenate(parts)


def code_margins(
    network: Any, task: wt.Task, lengths: Sequence[int], cases: int = RATIO_CASES
) -> dict[str, Any] | None:
    """
    Over the states of each length's codes-property's first cases: the least ratio of the next
    nearest code's distance to the snapped code's, over every snapped factor (None if no factor has
    two codes), and per factor (None if not snapped) the codes the states snap to, the least
    distance between two of them (None if one) and the farthest state from its code; None if no
    factor is snapped.
    """
    factors = network.recurrence_layer.factors
    sizes = [f.state_size for f in network.recurrence.factors]
    offsets = np.cumsum([0, *sizes])
    codes = [None if f.codes is None else np.asarray(f.codes.E.tolist(), dtype=float) for f in factors]
    if all(E is None for E in codes):
        return None
    k: int = network.columns
    property_ = next(p for p in task.catalogue(cases, cases, 1) if p.id == task.codes_id)
    least = math.inf
    used: list[set[int]] = [set() for _ in codes]
    farthest = [0.0 for _ in codes]
    for n in lengths:
        drawn = task.property_cases(property_, n)[:cases]
        rows = [wr.example(task, case, n, k, wr.alignment(task, case, n, k)) for case in drawn]
        for width in sorted({len(tokens) for tokens, _ in rows}):
            X = network.backend.matrix([tokens for tokens, _ in rows if len(tokens) == width])
            _, _, cells = network.window_states(X)
            for cell in cells:
                states = np.asarray(cell.tolist(), dtype=float)
                for g, E in enumerate(codes):
                    if E is None:
                        continue
                    block = states[:, offsets[g] : offsets[g + 1]]
                    q = ((block[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2).sum(axis=2)
                    used[g] |= {int(i) for i in q.argmin(axis=1)}
                    q.sort(axis=1)
                    farthest[g] = max(farthest[g], float(np.sqrt(q[:, 0].max())))
                    if len(E) >= 2:
                        least = min(least, float(np.min(np.sqrt(q[:, 1] / np.maximum(q[:, 0], 1e-300)))))
    per_factor: list[dict[str, Any] | None] = []
    for g, codebook in enumerate(codes):
        if codebook is None:
            per_factor.append(None)
            continue
        in_use = codebook[sorted(used[g])]
        d = np.sqrt(((in_use[:, np.newaxis, :] - in_use[np.newaxis, :, :]) ** 2).sum(axis=2))[
            np.triu_indices(len(in_use), 1)
        ]
        per_factor.append(
            {
                "codes_used": sorted(used[g]),
                "closest_used": float(d.min()) if len(d) else None,
                "farthest_state": farthest[g],
            }
        )
    return {"least_snap_ratio": None if least == math.inf else least, "factors": per_factor}


def replay(
    network: Any,
    task: wt.Task,
    counts: Sequence[int],
    settings: wr.Settings,
    epochs: int,
    generator: str,
    log: Callable[[str], None] | None = None,
    machines: bool = False,
    guard: bool = False,
) -> dict[str, Any]:
    """
    epochs of the step whose data window counts are counts, from network (trained in place), each
    epoch's data and order drawn by random.Random(f"{generator}-{epoch}"); the batches' and epochs'
    figures (see the module's docstring), with machines each batch's machine against the last's;
    with guard, an update that makes a right machine wrong undone and its batch skipped.
    """
    machines = machines or guard
    k: int = network.columns
    model = wr.WindowModel(network, k, task)
    catalogue = settings.catalogue(task)
    lengths = [w * k for w in counts]
    checks = {n: wr.check_properties(task, n, catalogue) for n in lengths}
    result: dict[str, Any] = {"generator": generator, "counts": list(counts), "epochs": []}
    result["start"] = code_margins(network, task, lengths)
    if log is not None:
        log(f"{generator} start: {margins_text(result['start'])}")
    table = machine_table(wc.NetworkMachine(network), task) if machines else None
    if table is not None:
        result["start_machine"] = {"reachable": len(table.rows), "wrong_pairs": table.wrong}
        if log is not None:
            log(f"{generator} start machine: {len(table.rows)} states reached, {table.wrong} wrong pairs")
    for epoch in range(epochs):
        start = time.perf_counter()
        rng = random.Random(f"{generator}-{epoch}")
        batches = wr.epoch_batches(
            wr.training_examples(task, counts, k, settings.epoch_examples, rng), settings.batch_size, rng
        )
        rows: list[dict[str, float]] = []
        rollbacks = 0
        for b, batch in enumerate(batches):
            X = network.backend.matrix([tokens for tokens, _ in batch])
            labels = [labels for _, labels in batch]
            before: float = network.loss_batch(X, labels)
            kept = _parameters(network)
            saved = cg.training_state(network) if guard else None
            network.learn_batch(settings.rate, batch)
            after: float = network.loss_batch(X, labels)
            rows.append(
                {
                    "width": len(batch[0][0]),
                    "loss_before": before,
                    "loss_after": after,
                    "update": float(np.linalg.norm(_parameters(network) - kept)),
                }
            )
            if table is not None:
                previous, table = table, machine_table(wc.NetworkMachine(network), task)
                if saved is not None and previous.wrong == 0 and table.wrong > 0:
                    rollbacks += 1
                    if log is not None:
                        log(
                            f"{generator} epoch {epoch} batch {b}: rolled back ({table.wrong} wrong pairs; loss "
                            f"{before:.3g} -> {after:.3g}, update {rows[-1]['update']:.3g})"
                        )
                    cg.restore_training_state(network, saved)
                    rows[-1] |= {"rolled_back": 1.0}
                    table = previous
                    continue
                changes = table_changes(previous, table)
                rows[-1] |= changes | {"reachable": len(table.rows), "wrong_pairs": table.wrong}
                if log is not None and any(changes.values()):
                    log(
                        f"{generator} epoch {epoch} batch {b}: {changes['pairs_changed']} pairs changed, "
                        f"+{changes['states_gained']} -{changes['states_lost']} states, {len(table.rows)} reached, "
                        f"{table.wrong} wrong pairs; loss {before:.3g} -> {after:.3g}, update {rows[-1]['update']:.3g}"
                    )
        screen = [r for n in lengths for r in model.evaluate(n, checks[n], settings.screen)]
        branched = wc.certify(wc.NetworkMachine(network), task)
        strict = wc.certify(wc.NetworkMachine(network), task, snap_ratio=1.0)
        updates = sorted(row["update"] for row in rows)
        margins = code_margins(network, task, lengths)
        summary = {
            "epoch": epoch,
            "screen": sum(r.passed for r in screen) / sum(r.cases for r in screen),
            "certified": branched.certified,
            "certified_strict": strict.certified,
            "minimised": len(branched.minimised),
            "least_snap_ratio": None if margins is None else margins["least_snap_ratio"],
            "margins": None if margins is None else margins["factors"],
            "jumps": sum(row["loss_after"] > 1.0 >= row["loss_before"] for row in rows),
            "machine_changes": None
            if table is None
            else sum(
                bool(row.get("pairs_changed") or row.get("states_gained") or row.get("states_lost")) for row in rows
            ),
            "rollbacks": rollbacks,
            "median_update": updates[len(updates) // 2],
            "max_update": updates[-1],
            "seconds": time.perf_counter() - start,
            "batches": rows,
        }
        result["epochs"].append(summary)
        if log is not None:
            log(line(generator, summary))
    return result


def margins_text(margins: dict[str, Any] | None) -> str:
    """code_margins' result as text."""
    if margins is None:
        return "no snapped factor"
    ratio = margins["least_snap_ratio"]
    parts = [f"least snap ratio {'-' if ratio is None else f'{ratio:.3f}'}"]
    for g, factor in enumerate(margins["factors"]):
        if factor is not None:
            closest = factor["closest_used"]
            parts.append(
                f"factor {g}: {len(factor['codes_used'])} codes used, closest pair "
                f"{'-' if closest is None else f'{closest:.3f}'}, farthest state {factor['farthest_state']:.3f}"
            )
    return ", ".join(parts)


def line(name: str, epoch: dict[str, Any]) -> str:
    """An epoch's summary as a line."""
    margins = (
        None
        if epoch["margins"] is None
        else {"least_snap_ratio": epoch["least_snap_ratio"], "factors": epoch["margins"]}
    )
    return (
        f"{name} epoch {epoch['epoch']}: screen {epoch['screen']:.3f}, "
        f"certified {epoch['certified']} (strict {epoch['certified_strict']}, {epoch['minimised']} states), "
        f"{margins_text(margins)}, {epoch['jumps']} jumps past loss 1, "
        f"update median {epoch['median_update']:.3g} max {epoch['max_update']:.3g}; {epoch['seconds']:.0f} s"
    )
