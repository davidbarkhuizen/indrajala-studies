"""
What a window-recurrent network's state carries, read against its task's target machine: a
diagnosis of one saved step's network, cheap enough to run on every step while a study trains
(the factored-state workplan's C9', where it showed division's factors tracking only "is it zero").

On cases drawn as training draws them (the task's mixture, at its alignment), at a few lengths:

- **the answer per class**: each output class's columns, correct and in all, so a network that
  learned some classes and stalls on others shows which;
- **each factor's share** of each quantity the target's state carries: I(code; x) / H(x), the
  fraction of x's entropy the factor's code alone removes (1 when the code determines x, 0 when
  independent of it), for each of the machine's coordinates (division's A and B; a machine without
  coordinates, its state), each coordinate's zero test ("A=0") and the answer itself;
- **the state's share** of the target's state: the whole tuple of codes against the machine's state.

A factor sharing all of a coordinate holds it; one sharing a coordinate's zero test but little of
the coordinate tracks only whether it is zero. Shares are measured on samples, so a code used by
few columns can overstate one a little; the cases and lengths are fixed for every diagnosis, so
steps and runs compare.

Each diagnosis is written beside its network as step-<j>.diagnosis.json, so a study's steps are
diagnosed once, and a run being trained can be followed: each new step diagnosed as it is saved,
until the machine's lock is free (the study ended).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt

LENGTHS = (3, 6, 12)
CASES = 600  # a length's cases
SEED = 0


@dataclass(frozen=True)
class Window:
    """One window of one case: its last column's label and prediction, the target's state after it, the codes."""

    label: int
    predicted: int
    state: int
    codes: tuple[int, ...]


def entropy(counts: Counter[Hashable]) -> float:
    total = sum(counts.values())
    return -sum(v / total * math.log2(v / total) for v in counts.values() if v)


def share(pairs: Sequence[tuple[Hashable, Hashable]]) -> float:
    """I(code; x) / H(x) over (code, x) samples; 1 when x doesn't vary."""
    hx = entropy(Counter(x for _, x in pairs))
    if hx == 0:
        return 1.0
    by_code: dict[Hashable, Counter[Hashable]] = {}
    for code, x in pairs:
        by_code.setdefault(code, Counter())[x] += 1
    conditional = sum(sum(c.values()) / len(pairs) * entropy(c) for c in by_code.values())
    return max(0.0, (hx - conditional) / hx)


def coordinates(machine: wt.Mealy) -> dict[str, Callable[[int], int]]:
    """Each coordinate of the machine's states by name; a machine without coordinates, its state."""
    if machine.coordinates is None:
        return {"state": lambda s: s}
    values = machine.coordinates.values
    return {name: (lambda s, g=g: int(values[s, g])) for g, name in enumerate(machine.coordinates.names)}


def samples(network: Any, task: wt.Task, lengths: Sequence[int] = LENGTHS, cases: int = CASES) -> list[Window]:
    """Each window of cases drawn as training draws them (held-out cases excluded), the same for every network."""
    k: int = network.columns
    rng = random.Random(SEED)
    machine = task.machine
    windows: list[Window] = []
    for n in lengths:
        drawn = [task.training_case(i, n, rng, False) for i in range(cases)]
        aligned = [wr.alignment(task, case, n, k) for case in drawn]
        for width in sorted({wr.data_windows(n, a, k) for a in aligned}):
            chosen = [i for i, a in enumerate(aligned) if wr.data_windows(n, a, k) == width]
            rows = [wr.example(task, drawn[i], n, k, aligned[i]) for i in chosen]
            predictions: list[Sequence[int]] = network.classify_rows(network.prepare_dataset(rows))
            codes: list[list[tuple[int, ...]]] = network.state_codes(
                network.backend.matrix([tokens for tokens, _ in rows])
            )
            for (tokens, labels), predicted, states in zip(rows, predictions, codes, strict=True):
                state = machine.start
                for w in range(len(tokens) // k):
                    for token in tokens[w * k : (w + 1) * k]:
                        state = int(machine.transition[state, int(token)])
                    last = (w + 1) * k - 1
                    windows.append(Window(int(labels[last]), int(predicted[last]), state, tuple(states[w + 1])))
    return windows


def diagnose(network: Any, task: wt.Task, lengths: Sequence[int] = LENGTHS, cases: int = CASES) -> dict[str, Any]:
    """The network's diagnosis (the module's docstring), as JSON."""
    windows = samples(network, task, lengths, cases)
    classes: dict[int, list[int]] = {}
    for w in windows:
        tally = classes.setdefault(w.label, [0, 0])
        tally[0] += w.label == w.predicted
        tally[1] += 1
    quantities: dict[str, Callable[[Window], Hashable]] = {}
    for name, value in coordinates(task.machine).items():
        quantities[name] = lambda w, value=value: value(w.state)
        quantities[f"{name}=0"] = lambda w, value=value: value(w.state) == 0
    quantities["answer"] = lambda w: w.label
    factors = [
        {
            "codes_used": len({w.codes[g] for w in windows}),
            "shares": {name: share([(w.codes[g], of(w)) for w in windows]) for name, of in quantities.items()},
        }
        for g in range(len(windows[0].codes))
    ]
    return {
        "task": task.name,
        "lengths": list(lengths),
        "cases": cases,
        "windows": len(windows),
        "classes": {str(label): tally for label, tally in sorted(classes.items())},
        "factors": factors,
        "state": {
            "codes_used": len({w.codes for w in windows}),
            "share": share([(w.codes, w.state) for w in windows]),
        },
    }


def congruence_shares(windows: Sequence[Window], machine: wt.Mealy) -> dict[str, Any]:
    """
    Each factor's share of each of the machine's named congruences (window_tasks.named_labels, those
    the machine respects) and of its state, and the state's share of the whole tuple: I(code; x) / H(x)
    over the windows, so weighted as training draws them. Valid on any network, exact or not, where
    machine_reading.approximations' reachable pairs, each counted once, read an inexact network's rare
    wrong codes as loudly as its common right ones.
    """
    named = {k: v for k, v in wt.named_labels(machine.coordinates).items() if machine.is_congruence(v)}
    return {
        "state": share([(w.codes, w.state) for w in windows]),
        "factors": [
            {
                "state": share([(w.codes[g], w.state) for w in windows]),
                "named": {k: share([(w.codes[g], int(v[w.state])) for w in windows]) for k, v in named.items()},
            }
            for g in range(len(windows[0].codes))
        ],
    }


def line(name: str, diagnosis: dict[str, Any]) -> str:
    """One line: accuracy per class, each factor's shares, the state's share."""
    classes = " ".join(f"{label} {right / total:.2f}" for label, (right, total) in diagnosis["classes"].items())
    factors = " | ".join(
        f"f{g} " + " ".join(f"{name} {value:.2f}" for name, value in factor["shares"].items())
        for g, factor in enumerate(diagnosis["factors"])
    )
    return f"{name}: classes {classes} | {factors} | state {diagnosis['state']['share']:.2f}"


def diagnosis_file(network_file: Path) -> Path:
    """step-<j>.network.json's diagnosis, step-<j>.diagnosis.json beside it."""
    return network_file.with_name(network_file.name.replace(".network.json", ".diagnosis.json"))


def step_networks(run_directory: Path) -> list[Path]:
    """A run's saved step networks, in step order."""
    return sorted(run_directory.glob("step-*.network.json"), key=lambda p: int(p.name.split(".")[0].split("-")[1]))


def diagnose_new(
    run_directory: Path, task: wt.Task, load: Callable[[str], Any], lengths: Sequence[int] = LENGTHS, cases: int = CASES
) -> list[str]:
    """Each of a run's step networks without a diagnosis, diagnosed and written; a line each."""
    lines: list[str] = []
    for network_file in step_networks(run_directory):
        target = diagnosis_file(network_file)
        if target.exists():
            continue
        step = json.loads(network_file.with_name(network_file.name.replace(".network.json", ".json")).read_text())
        result = diagnose(load(str(network_file)), task, lengths, cases)
        result["step"] = {key: step.get(key) for key in ("step", "stop", "epochs", "break")}
        certificate = step.get("certificate", {})
        result["step"]["certified"] = certificate.get("certified")
        target.write_text(json.dumps(result, indent=1) + "\n")
        name = f"{run_directory.name} {network_file.name.split('.')[0]}"
        lines.append(f"{line(name, result)} || {result['step']['stop']}, certified {result['step']['certified']}")
    return lines
