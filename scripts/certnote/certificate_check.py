"""
The certificate's independent checker (the certificate note's D5, indrajala-research
docs/workplans/certificate-note-workplan.md, stage 1).

A certifying computation emits a witness that a small checker verifies (McConnell, Mehlhorn, Näher
& Schweitzer 2011). certify's witness is its Table (window_recurrent_certificate.Table.save: an npz,
its claim as JSON under "meta"): every reachable state's labels, snapped next state and margins under
every window, each near tie's other next states, the start's, phi and the minimised machine's blocks.
This checker shares no code with certify. It reads the target from window_tasks' transition and
output tables only, and the network through the library's loader and its window_step, so what must
be trusted is this file, the forward pass and the target's tables. It checks two things.

The structure (no network needed):

- shape: the arrays agree with each other and with the meta (W = tokens^columns windows);
- duplicate: no state is two rows;
- closure: every snapped next state, near-tie branch and start is a row;
- start: the first row is the start, and phi maps every start to the target's start;
- labels: each entry's labels are the target's from phi(state) under the window;
- next: phi of each next state (snapped and branched) is the target's state after the window;
- lead: every entry's logit lead is at least the meta's;
- onto: phi reaches every target state reachable by whole windows (the target is minimal, so the minimised
  machine is the target's size);
- blocks: the recorded minimised machine is phi's kernel (phi a homomorphism onto a minimal target
  makes the kernel the coarsest congruence).

Fidelity (the network's file, its sha256 checked): every (state, window) entry is run again through
the network's forward pass, in shuffled batches that mix states (so in an order and batching that
certify never used), and the snap redone here from the cells' outputs and each factor's codebook:

- labels: the forward pass's labels equal the table's;
- next: the forward pass's snapped next state is one of the table's next states for the entry
  (the snapped one or a branch);
- ties: every code within the meta's snap ratio of the snapped one (each near tie's combinations
  over the factors) is one of the table's next states;
- lead: the forward pass's logit lead is at least the meta's;
- start: the network's start, and its near ties, are among the table's starts.

A table passes when every count is zero; the report also gives the margins' largest differences
from the table's (the batch order's effect, C4) and how many snaps moved within a tie.

    python scripts/certnote/certificate_check.py TABLE.npz [...] [--network PATH] [--json OUT]
        [--structure-only] [--batch 8192] [--seed 0]

--network overrides the meta's network path (a moved file; the sha256 is still checked).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT = "certificate-table-1"
ARRAYS = (
    "states",
    "labels",
    "next",
    "leads",
    "ratios",
    "branch_rows",
    "branch_windows",
    "branch_next",
    "starts",
    "start_ratios",
    "phi",
    "blocks",
)
STRUCTURE: tuple[str, ...] = ("shape", "duplicate", "closure", "start", "labels", "next", "lead", "onto", "blocks")
FIDELITY: tuple[str, ...] = ("labels", "next", "ties", "lead", "start")

Arrays = dict[str, npt.NDArray[Any]]


def load(path: Path) -> tuple[dict[str, Any], Arrays]:
    """A table's meta and arrays."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {name: data[name] for name in ARRAYS}
    assert meta.get("format") == FORMAT, f"{path}: not a {FORMAT} table"
    return meta, arrays


def windows(columns: int, tokens: int) -> npt.NDArray[np.int64]:
    """Every window, (tokens^columns, columns): window i's columns are i's base-tokens digits, the first the most significant."""
    i = np.arange(tokens**columns, dtype=np.int64)
    return np.stack([(i // tokens ** (columns - 1 - j)) % tokens for j in range(columns)], axis=1)


def target_windows(
    transition: npt.NDArray[np.int64], output: npt.NDArray[np.int64], W: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """From each target state, each window's labels (S, N, k) and the state after it (S, N)."""
    S = transition.shape[0]
    labels = np.zeros((S, *W.shape), dtype=np.int64)
    after = np.zeros((S, W.shape[0]), dtype=np.int64)
    for s in range(S):
        state = np.full(W.shape[0], s, dtype=np.int64)
        for j in range(W.shape[1]):
            labels[s, :, j] = output[state, W[:, j]]
            state = transition[state, W[:, j]]
        after[s] = state
    return labels, after


def target_reachable(after: npt.NDArray[np.int64], start: int) -> set[int]:
    """The states a table of next states (a state's row, per token or per window) reaches from the start."""
    seen, queue = {start}, [start]
    for s in queue:
        for r in np.unique(after[s]).tolist():
            if r not in seen:
                seen.add(r)
                queue.append(r)
    return seen


def keys(rows: npt.NDArray[Any], codes: Sequence[int]) -> npt.NDArray[np.int64]:
    """Each state (a code per factor) as one integer, mixed radix over the factors' code counts."""
    key = np.zeros(len(rows), dtype=np.int64)
    for g, m in enumerate(codes):
        key = key * m + rows[:, g].astype(np.int64)
    return key


def check_structure(meta: dict[str, Any], a: Arrays, transition: Any, output: Any, start: int) -> dict[str, int]:
    """The structure's failure counts by kind (STRUCTURE), each 0 for a valid witness."""
    fails = {name: 0 for name in STRUCTURE}
    codes = list(meta["codes"])
    R, G = a["states"].shape
    W = windows(meta["columns"], meta["tokens"])
    shapes_ok = (
        G == len(codes)
        and a["labels"].shape[:2] == (R, len(W))
        and a["next"].shape == (R, len(W), G)
        and a["leads"].shape == (R, len(W))
        and a["ratios"].shape == (R, len(W))
        and a["phi"].shape == (R,)
        and a["blocks"].shape == (R,)
        and a["branch_next"].shape == (len(a["branch_rows"]), G)
        and len(a["branch_windows"]) == len(a["branch_rows"])
        and a["starts"].shape[1:] == (G,)
        and len(a["starts"]) >= 1
        and transition.shape[1] == meta["tokens"]
    )
    if not shapes_ok:
        fails["shape"] = 1
        return fails
    for name in ("states", "next", "branch_next", "starts"):
        if len(a[name]) and (a[name].min() < 0 or (a[name].reshape(-1, G) >= np.array(codes)).any()):
            fails["shape"] += 1
    if len(a["branch_rows"]) and (
        a["branch_rows"].min() < 0
        or a["branch_rows"].max() >= R
        or a["branch_windows"].min() < 0
        or a["branch_windows"].max() >= len(W)
    ):
        fails["shape"] += 1
    if fails["shape"]:
        return fails
    row_keys = keys(a["states"], codes)
    fails["duplicate"] = R - len(np.unique(row_keys))
    lookup = np.full(int(np.prod(codes)), -1, dtype=np.int64)
    lookup[row_keys] = np.arange(R)
    next_rows = lookup[keys(a["next"].reshape(-1, G), codes)].reshape(R, len(W))
    branch_rows = lookup[keys(a["branch_next"], codes)]
    start_rows = lookup[keys(a["starts"], codes)]
    fails["closure"] = int((next_rows < 0).sum() + (branch_rows < 0).sum() + (start_rows < 0).sum())
    phi = a["phi"]
    if phi.min() < 0 or phi.max() >= transition.shape[0]:
        fails["shape"] += 1
        return fails
    fails["start"] = int(start_rows[0] != 0) + int(sum(1 for r in start_rows if r >= 0 and phi[r] != start))
    want_labels, want_after = target_windows(transition, output, W)
    fails["labels"] = int((a["labels"] != want_labels[phi]).any(axis=2).sum())
    known = next_rows >= 0
    fails["next"] = int((known & (phi[np.where(known, next_rows, 0)] != want_after[phi])).sum())
    known_b = branch_rows >= 0
    fails["next"] += int(
        (
            known_b & (phi[np.where(known_b, branch_rows, 0)] != want_after[phi[a["branch_rows"]], a["branch_windows"]])
        ).sum()
    )
    fails["lead"] = int((~(a["leads"] >= meta["logit_lead"])).sum())  # so a NaN fails
    fails["onto"] = len(target_reachable(want_after, start) - {int(s) for s in phi})  # over whole windows
    # the blocks as a partition against phi's kernel: the same partition when each block has one phi
    # value and each phi value one block
    pairs = set(zip(a["blocks"].tolist(), phi.tolist(), strict=True))
    fails["blocks"] = (len(pairs) - len(set(a["blocks"].tolist()))) + (len(pairs) - len(set(phi.tolist())))
    return fails


def _labels_leads(probabilities: npt.NDArray[np.float64], classes: int, slot: int | None) -> tuple[Any, Any]:
    by_column = probabilities.reshape(len(probabilities), -1, classes)
    if slot is not None:
        by_column = by_column[:, slot : slot + 1, :]
    ordered = np.sort(by_column, axis=2)
    first, second = ordered[:, :, -1], ordered[:, :, -2]
    with np.errstate(divide="ignore"):
        lead = np.where(first == second, 0.0, np.log(first) - np.log(second))
    return np.argmax(by_column, axis=2), lead.min(axis=1)


def _snap(H: npt.NDArray[np.float64], E: npt.NDArray[np.float64], ratio: float) -> tuple[Any, Any, Any]:
    """Each row's nearest code, its snap ratio and the other codes within ratio of it (its near ties)."""
    d = np.sqrt(((H[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2).sum(axis=2))
    code = np.argmin(d, axis=1)
    rows = np.arange(len(H))
    chosen = d[rows, code]
    others = d.copy()
    others[rows, code] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(chosen > 0, others.min(axis=1) / chosen, np.where(others.min(axis=1) > 0, np.inf, 1.0))
        near = others < ratio * chosen[:, np.newaxis]
    near |= (others == 0) & (chosen == 0)[:, np.newaxis]
    return code, ratios, near


class Forward:
    """The network's window, from a state's codes and a window's columns, as the meta's adapter reads it."""

    def __init__(self, network: Any, meta: dict[str, Any]) -> None:
        self.network = network
        self.E = [np.array(f.codes.E.tolist(), dtype=np.float64) for f in network.recurrence_layer.factors]
        sizes = [f.state_size for f in network.recurrence.factors]
        self.blocks = [(sum(sizes[:g]), sum(sizes[: g + 1])) for g in range(len(sizes))]
        adapter = meta["adapter"]
        self.queries = list(adapter.get("query_tokens", [])) if adapter["kind"] == "query" else []
        self.slot = adapter.get("slot") if adapter["kind"] == "query" else None
        assert network.columns == meta["columns"] + len(self.queries), "the network's window isn't the table's"
        self.ratio = float(meta["snap_ratio"])

    def run(self, states: npt.NDArray[Any], W: npt.NDArray[np.int64]) -> tuple[Any, Any, Any, Any, Any]:
        """Each (state, window) row's labels, lead, snapped codes (N, G), ratios (N,) and near ties per factor."""
        full = np.concatenate([W, np.tile(np.array(self.queries, dtype=np.int64), (len(W), 1))], axis=1)
        C = np.concatenate([E[states[:, g]] for g, E in enumerate(self.E)], axis=1)
        backend = self.network.backend
        output, H, _ = self.network.window_step(
            backend.matrix(full.astype(np.float64).tolist()), backend.matrix(C.tolist())
        )
        labels, lead = _labels_leads(np.array(output.tolist(), dtype=np.float64), self.network.class_count, self.slot)
        values = np.array(H.tolist(), dtype=np.float64)
        snaps = [_snap(values[:, a:b], E, self.ratio) for (a, b), E in zip(self.blocks, self.E, strict=True)]
        codes = np.stack([s[0] for s in snaps], axis=1)
        ratios = np.min(np.stack([s[1] for s in snaps], axis=1), axis=1)
        return labels, lead, codes, ratios, [s[2] for s in snaps]

    def start(self) -> list[tuple[int, ...]]:
        """The network's start state and its near ties' states."""
        H, _ = self.network.first_step()
        values = np.array(H.tolist(), dtype=np.float64)
        options: list[list[int]] = []
        for (a, b), E in zip(self.blocks, self.E, strict=True):
            code, _, near = _snap(values[:, a:b], E, self.ratio)
            options.append([int(code[0]), *(int(c) for c in np.flatnonzero(near[0]))])
        return [tuple(s) for s in itertools.product(*options)]


def check_fidelity(
    meta: dict[str, Any], a: Arrays, forward: Forward, batch: int = 8_192, seed: int = 0
) -> tuple[dict[str, int], dict[str, Any]]:
    """Fidelity's failure counts by kind (FIDELITY), and the margins' differences from the table's."""
    fails = {name: 0 for name in FIDELITY}
    R, G = a["states"].shape
    W = windows(meta["columns"], meta["tokens"])
    entries = R * len(W)
    # each entry's recorded next states beyond the snapped one: (entry, state key) pairs
    codes = list(meta["codes"])
    recorded_branches = set(
        zip(
            (a["branch_rows"] * len(W) + a["branch_windows"]).tolist(),
            keys(a["branch_next"], codes).tolist(),
            strict=True,
        )
    )
    snapped_keys = keys(a["next"].reshape(-1, G), codes)
    order = np.random.default_rng(seed).permutation(entries)
    lead_diff = ratio_diff = 0.0
    moved = leads_bits = 0
    for i in range(0, entries, batch):
        e = order[i : i + batch]
        r, w = e // len(W), e % len(W)
        labels, lead, snapped, ratios, near = forward.run(a["states"][r], W[w])
        fails["labels"] += int((labels != a["labels"][r, w]).any(axis=1).sum())
        fails["lead"] += int((~(lead >= meta["logit_lead"])).sum())  # so a NaN fails
        finite = np.isfinite(lead) & np.isfinite(a["leads"][r, w])
        lead_diff = max(lead_diff, float(np.abs(lead - a["leads"][r, w])[finite].max(initial=0.0)))
        finite = np.isfinite(ratios) & np.isfinite(a["ratios"][r, w])
        ratio_diff = max(ratio_diff, float(np.abs(ratios - a["ratios"][r, w])[finite].max(initial=0.0)))
        leads_bits += int((lead != a["leads"][r, w]).sum())
        got = keys(snapped, codes)
        same = got == snapped_keys[e]
        moved += int((~same).sum())
        for j in np.flatnonzero(~same):
            if (int(e[j]), int(got[j])) not in recorded_branches:
                fails["next"] += 1
        tied = np.flatnonzero(np.any([n.any(axis=1) for n in near], axis=0))
        for j in tied:
            options = [[int(snapped[j, g]), *(int(c) for c in np.flatnonzero(near[g][j]))] for g in range(G)]
            for state in itertools.product(*options):
                key = int(keys(np.array([state]), codes)[0])
                if key != int(snapped_keys[e[j]]) and (int(e[j]), key) not in recorded_branches:
                    fails["ties"] += 1
    starts = {tuple(int(c) for c in s) for s in a["starts"]}
    fails["start"] = sum(1 for s in forward.start() if s not in starts)
    return fails, {
        "lead_difference": lead_diff,
        "ratio_difference": ratio_diff,
        "leads_not_bitwise": leads_bits,
        "snaps_moved": moved,
    }


def check(
    path: Path, network_path: Path | None = None, structure_only: bool = False, batch: int = 8_192, seed: int = 0
) -> dict[str, Any]:
    """A table's report: its sizes, each check's failures and seconds, and the verdict."""
    from indrajala_studies import window_tasks as wt  # the target's tables only

    begun = time.perf_counter()
    meta, a = load(path)
    target = wt.task_named(meta["task"]).machine
    transition = np.asarray(target.transition, dtype=np.int64)
    output = np.asarray(target.output, dtype=np.int64)
    structure = check_structure(meta, a, transition, output, int(target.start))
    report: dict[str, Any] = {
        "table": str(path),
        "bytes": path.stat().st_size,
        "task": meta["task"],
        "snap_ratio": meta["snap_ratio"],
        "rows": int(a["states"].shape[0]),
        "windows": int(a["labels"].shape[1]),
        "entries": int(a["labels"].shape[0] * a["labels"].shape[1]),
        "branches": len(a["branch_rows"]),
        "claimed": meta["certified"],
        "structure": structure,
        "structure_seconds": time.perf_counter() - begun,
    }
    valid = not any(structure.values())
    if not structure_only:
        from indrajala_ml.model.persistence.load_network import load_network

        network_file = network_path or Path(meta["network"])
        sha_ok = hashlib.sha256(network_file.read_bytes()).hexdigest() == meta["sha256"]
        begun = time.perf_counter()
        fidelity, differences = (
            check_fidelity(meta, a, Forward(load_network(str(network_file)), meta), batch, seed)
            if sha_ok and not report["structure"]["shape"]
            else ({name: -1 for name in FIDELITY}, {})
        )
        report.update(
            network=str(network_file),
            sha256_matches=sha_ok,
            fidelity=fidelity,
            differences=differences,
            fidelity_seconds=time.perf_counter() - begun,
        )
        valid = valid and sha_ok and not any(fidelity.values())
    report["valid"] = valid
    report["agrees"] = valid == bool(meta["certified"])
    return report


def summary(report: dict[str, Any]) -> str:
    failed = [f"structure {k} {v}" for k, v in report["structure"].items() if v]
    failed += [f"fidelity {k} {v}" for k, v in report.get("fidelity", {}).items() if v]
    if report.get("sha256_matches") is False:
        failed.append("the network's sha256 differs")
    verdict = "valid" if report["valid"] else "invalid: " + ", ".join(failed)
    seconds = report["structure_seconds"] + report.get("fidelity_seconds", 0.0)
    return (
        f"{report['table']}: {verdict}; {report['rows']} states x {report['windows']} windows, "
        f"{report['branches']} branches, {report['bytes']} bytes; {seconds:.1f} s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("tables", type=Path, nargs="+")
    parser.add_argument("--network", type=Path, help="the network's file, if moved (one table only)")
    parser.add_argument("--structure-only", action="store_true", help="the structure only, no forward pass")
    parser.add_argument("--batch", type=int, default=8_192, help="entries a forward pass")
    parser.add_argument("--seed", type=int, default=0, help="the entries' shuffle")
    parser.add_argument("--json", type=Path, help="write the reports here (a JSON list)")
    args = parser.parse_args(argv)
    if args.network is not None and len(args.tables) != 1:
        parser.error("--network names one table's network")
    reports: list[dict[str, Any]] = []
    for path in args.tables:
        report = check(path, args.network, args.structure_only, args.batch, args.seed)
        print(summary(report), flush=True)
        reports.append(report)
    if args.json is not None:
        args.json.write_text(json.dumps(reports, indent=1))
    return 0 if all(r["valid"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
