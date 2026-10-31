#!/usr/bin/env python3
"""
The paper's tables regenerated from the result files (the paper workplan's T7): every number in a
table comes from the runs' logs and recertifications, never typed by hand.

`paper_tables.py show NAME...` prints each table's LaTeX rows (the lines between `\\midrule` and
`\\bottomrule`). `paper_tables.py check TEX...` regenerates every table whose label a manuscript
holds and compares its rows with the manuscript's, a line per table, and every number the prose
holds as `\\pn{key}{value}` (the manuscript defines `\\pn` to print its value) against the key's
regeneration; it exits 1 on any difference or a key with no source. `paper_tables.py numbers`
prints every key and its value; `paper_tables.py fill TEX...` writes each `\\pn`'s regenerated
value into the manuscript. The numbers come from NUMBER_SOURCES, each a function of the data
directory returning its keys' values as the manuscript writes them (addition's, for Appendix A).

Each table is a function of the data directory (`--data`, default data/window-recurrent), named by
its LaTeX label. Begun with P2's; T1's, T3's, T2's and T5's rates from their batches (`paper/<test>/`:
runs, recertifications, long tests), a run counted only when it succeeded (certified in both
verdicts and every long-test case passed), as their findings count; T4's decomposition readings (`approximations`' JSON or text) and T6's
answers-proof results.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from indrajala_ml.batch import BACKDROPS
from indrajala_ml.data import addition_data as ad
from indrajala_ml.studies import inference
from indrajala_ml.studies import modular_family as mf
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt

Table = Callable[[Path], list[str]]


def span(values: list[int]) -> str:
    """A range as the manuscripts write it: "lo--hi", one value alone, "--" for none."""
    if not values:
        return "--"
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low}--{high}"


def arm_row(label: str, runs: list[mf.ArmRun]) -> str:
    """An arm's row: certified of runs, the Wilson 95% interval, and the successes' steps and epochs."""
    certified = [r.outcome for r in runs if r.certified]
    low, high = inference.wilson(len(certified), len(runs))
    successes = f"{span([o.step for o in certified])}, {span([o.epochs for o in certified])}" if certified else "--"
    return f"{label} & {len(certified)} of {len(runs)} & {low:.2f}--{high:.2f} & {successes} \\\\"


def approximation(data: Path) -> list[str]:
    """P2's approximation claim at p 5 (tab:approximation): four arms, 20 seeds each, both verdicts."""
    root = data / "approximation-claim"
    verdicts = mf.recertifications(root / "certify")
    arms = [
        ("family", "sum, difference (cover)"),
        ("sum", "sum (an approximation)"),
        ("quotient", "none"),
        ("product", "product (not an approximation)"),
    ]
    return [arm_row(label, mf.arm_runs(root, arm, verdicts)) for arm, label in arms]


def median(values: list[int]) -> str:
    """The median as the findings write it: whole, or to a half."""
    middle = statistics.median(values)
    return str(int(middle)) if middle == int(middle) else f"{middle:.1f}"


def failures(runs: list[mf.ArmRun]) -> tuple[int, int, int]:
    """An arm's failures (runs not succeeded), how many of the uncertified came near a solution, and collapsed (readout's)."""
    stopped = [r.outcome for r in runs if r.outcome.end != "certified"]
    return (
        sum(not r.succeeded for r in runs),
        sum(o.peak >= mf.NEAR for o in stopped),
        sum(o.collapses > 0 for o in stopped),
    )


def succeeded(runs: list[mf.ArmRun]) -> tuple[list[mf.ArmRun], str]:
    """An arm's successes (certified in both verdicts and long-tested, ArmRun.succeeded) and its rate's cells."""
    won = [r for r in runs if r.succeeded]
    low, high = inference.wilson(len(won), len(runs))
    return won, f"{len(won)} of {len(runs)} & {low:.2f}--{high:.2f}"


def batch_runs(data: Path, test: str, arm: str) -> list[mf.ArmRun]:
    """A paper test's arm (its batch's runs/<arm>), each run read with its recertifications and long tests."""
    root = data / "paper" / test
    return mf.arm_runs(root / "runs", arm, mf.recertifications(root / "certify"), mf.long_tested(root / "long-tests"))


def arms_table(test: str, arms: list[tuple[str, str]]) -> Table:
    """A comparison's arms (T1, T3): succeeded of runs, the Wilson 95% interval, failures near a solution, collapsed."""

    def table(data: Path) -> list[str]:
        lines: list[str] = []
        for arm, label in arms:
            runs = batch_runs(data, test, arm)
            _, rate = succeeded(runs)
            failed, near, collapsed = failures(runs)
            lines.append(f"{label} & {rate} & {near} of {failed} & {collapsed} \\\\")
        return lines

    return table


def rates_table(backdrops: list[tuple[str, int]]) -> Table:
    """T2's and T5's rates (one batch, D17): q, succeeded of runs, Wilson 95%, the successes' epochs, failures near."""

    def table(data: Path) -> list[str]:
        lines: list[str] = []
        for backdrop, q in backdrops:
            runs = batch_runs(data, "t2-t5", backdrop)
            won, rate = succeeded(runs)
            epochs = [r.outcome.epochs for r in won]
            course = f"{span(epochs)}, {median(epochs)}" if len(epochs) > 1 else span(epochs)
            failed, near, _ = failures(runs)
            lines.append(f"\\texttt{{{backdrop}}} & {q} & {rate} & {course} & {near} of {failed} \\\\")
        return lines

    return table


@dataclass(frozen=True)
class Reading:
    """One network's decomposition reading (`approximations`): its target's states, whether its factors
    cover the target, and each factor's approximation's blocks with the congruences it determines."""

    states: int
    cover: bool
    factors: list[tuple[int, list[str]]]


NET = re.compile(r": (\d+) states, meet \d+, cover (True|False)")
FACTOR = re.compile(r"factor \d+: \d+ codes, approximation (\d+) blocks \([^)]*\), determines ([^;]*)")


def readings(path: Path) -> list[Reading]:
    """A reading's networks: a directory of JSON (T4's first reading) or a text file, a line a network (T2's and T5's)."""
    if path.is_dir():
        found: list[Reading] = []
        for file in sorted(path.glob("*.json")):
            net = json.loads(file.read_text())
            factors = [(len(f["approximation"]), list(f["determines"])) for f in net["factors"]]
            found.append(Reading(net["states"], net["cover"], factors))
        return found
    found = []
    for line in path.read_text().splitlines():
        net = NET.search(line)
        assert net is not None, line
        factors = [
            (int(blocks), [] if names.strip() == "-" else [n.strip() for n in re.split(r", (?=[A-Za-z])", names)])
            for blocks, names in FACTOR.findall(line)
        ]
        found.append(Reading(int(net[1]), net[2] == "True", factors))
    return found


def tex(name: str) -> str:
    """A congruence's name in LaTeX: its braces escaped (`places {1}`)."""
    return name.replace("{", "\\{").replace("}", "\\}")


def natural(name: str) -> list[int | str]:
    """A sort key reading a name's numbers as numbers (`d7-family` before `d11-family`)."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def decomposition_row(label: str, nets: list[Reading]) -> str:
    """A set's row (T4, D10's rules): networks; factors nontrivial (1 < blocks < states); networks naming a
    congruence in a nontrivial factor (in both factors); covers; the congruences determined, most first."""
    held = [[1 < blocks < n.states and bool(names) for blocks, names in n.factors] for n in nets]
    nontrivial = sum(1 < blocks < n.states for n in nets for blocks, _ in n.factors)
    factors = sum(len(n.factors) for n in nets)
    named, both = sum(any(h) for h in held), sum(all(h) for h in held)
    counts = Counter(name for n in nets for _, names in n.factors for name in names)
    determines = ", ".join(
        f"\\texttt{{{tex(k)}}} {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return (
        f"{label} & {len(nets)} & {nontrivial}/{factors} & {named} ({both}) & {sum(n.cover for n in nets)} & "
        f"{determines or '--'} \\\\"
    )


def decomposition_table(sets: list[tuple[str, list[str]]]) -> Table:
    """T4's sets, each the readings under paper/ named (directories or text files) pooled into one row."""

    def table(data: Path) -> list[str]:
        return [
            decomposition_row(label, [n for path in paths for n in readings(data / "paper" / path)])
            for label, paths in sets
        ]

    return table


def proofs(data: Path) -> list[dict[str, Any]]:
    """T6's readings (D11): every answers-proof result under paper/<test>/proof, a bound each."""
    return [
        bound
        for path in sorted((data / "paper").glob("*/proof/*/*.answers-proof.json"))
        for bound in json.loads(path.read_text())["bounds"]
    ]


def proof_outcome(bound: dict[str, Any]) -> str:
    """A bound's outcome: no stop (silent), or the stop (kept, strict-only, false-stop) with Chow's verdict."""
    if bound["verdict"] == "silent":
        return "silent"
    chow = bound["chow"]
    verdict = "out of reach" if "refused" in chow else ("proved" if chow["equivalent"] else "refuted")
    return f"{bound['verdict']} {verdict}"


PROOF_ROWS = [
    ("silent", "no stop (the run never certified)"),
    ("kept proved", "kept, Chow proved"),
    ("kept out of reach", "kept, out of Chow's reach"),
    ("strict-only proved", "stop certified, not robustly; Chow proved"),
    ("strict-only out of reach", "stop certified, not robustly; out of Chow's reach"),
    ("false-stop proved", "false stop, Chow proved the machine equal to the target"),
    ("false-stop refuted", "false stop, Chow refuted"),
    ("false-stop out of reach", "false stop, out of Chow's reach"),
]


def proof(data: Path) -> list[str]:
    """T6's outcomes (tab:proof): each outcome's count at the exact bound and at 1.1 times it."""
    counts = Counter((b["slack"], proof_outcome(b)) for b in proofs(data))
    assert {o for _, o in counts} <= {o for o, _ in PROOF_ROWS}, counts
    return [f"{label} & {counts[(1.0, o)]} & {counts[(1.1, o)]} \\\\" for o, label in PROOF_ROWS]


def refuted(data: Path) -> list[str]:
    """T6b's refuted stops (tab:refuted): the run, the bound, the stop and certified steps, the counterexample's length."""
    lines: list[str] = []
    for path in sorted((data / "paper").glob("*/proof/*/*.answers-proof.json"), key=lambda p: natural(p.name)):
        for b in json.loads(path.read_text())["bounds"]:
            if proof_outcome(b) == "false-stop refuted":
                run = path.name.removesuffix(".answers-proof.json")
                lines.append(
                    f"\\texttt{{{run}}} & {b['bound']} & {b['stop']}, {b['certified']} & {len(b['chow']['counterexample'])} \\\\"
                )
    return lines


GROUPS = [("s3", 6), ("s4", 24), ("mtf4", 24)]  # T11's backdrops (D20), each against T5's rate


def group_tests(data: Path) -> list[tuple[str, int, list[mf.ArmRun], list[mf.ArmRun], float]]:
    """T11a (D20): each algebra's runs given its cover, against T5's on its answers alone; Fisher's exact, one-sided
    (Holm over the paper's family, `paper_family`)."""
    pairs = [(b, q, batch_runs(data, "t11/batch", f"t11-{b}"), batch_runs(data, "t2-t5", b)) for b, q in GROUPS]
    p = [
        inference.fisher_exact_greater(
            sum(r.succeeded for r in given),
            sum(not r.succeeded for r in given),
            sum(r.succeeded for r in alone),
            sum(not r.succeeded for r in alone),
        )
        for _, _, given, alone in pairs
    ]
    return [(b, q, given, alone, pv) for (b, q, given, alone), pv in zip(pairs, p, strict=True)]


def groups(data: Path) -> list[str]:
    """T11's rates (tab:groups): each algebra on its answers alone (T5) and given its cover, Wilson 95%, Fisher's p and Holm."""
    lines: list[str] = []
    family = paper_family(data)
    for backdrop, q, given, alone, p in group_tests(data):
        holm = family[f"t11-{backdrop}"][1]
        _, rate = succeeded(given)
        lines.append(
            f"\\texttt{{{backdrop}}} & {q} & {sum(r.succeeded for r in alone)} of {len(alone)} & {rate}"
            f" & ${p_math(p)}$ & ${p_math(holm)}$ \\\\"
        )
    return lines


PLANTED_ARMS = [  # T10's arms (D19), the machine the unit
    ("p0", "P0, primitive"),
    ("p1", "P1, planted, the target alone"),
    ("p2", "P2, planted, its congruence quotient an output"),
]
PLANTED_TESTS = [("t10", "T10", ["p0", "p1", "p2"]), ("t10r", "T10r", ["p0", "p2"])]  # T10 and its replication (D24)


def machine_runs(data: Path, test: str, arms: list[str]) -> dict[str, dict[int, list[mf.ArmRun]]]:
    """A machine-unit test's runs (D19, D24): its batch's runs/<test>-<arm>-<machine>, read with its recertifications
    and long tests."""
    root = data / "paper" / test / "batch"
    verdicts, long = mf.recertifications(root / "certify"), mf.long_tested(root / "long-tests")
    return mf.machine_unit_runs(root / "runs", test, arms, verdicts, long)


def planted(data: Path) -> list[str]:
    """T10's and T10r's arms, each test's rows apart: machines succeeding, runs succeeding, the successes' epochs to
    certify (range, median)."""
    labels = dict(PLANTED_ARMS)
    lines: list[str] = []
    for test, name, arms in PLANTED_TESTS:
        runs = machine_runs(data, test, arms)
        for arm in arms:
            counts = mf.machine_successes(runs[arm])
            every = [r for machine in runs[arm].values() for r in machine]
            epochs = [r.outcome.epochs for r in every if r.succeeded]
            course = f"{span(epochs)}, {median(epochs)}" if len(epochs) > 1 else span(epochs)
            machines = f"{sum(1 for c in counts if c)} of {len(counts)}"
            lines.append(f"{name} {labels[arm]} & {machines} & {sum(counts)} of {len(every)} & {course} \\\\")
    return lines


T12_ARMS = [  # T12's arms (D31): division at p 5, the same vocabulary and seeds, the extra outputs differing
    ("d5-cover", "the sum and the difference (cover)"),
    ("d5-state", "the state, $A p + B$"),
]


def state_test(data: Path) -> tuple[list[list[mf.ArmRun]], float]:
    """T12 (D31): each arm's runs, and Fisher's exact test of the cover against the state, two-sided."""
    cover, state = (batch_runs(data, "t12/batch", arm) for arm, _ in T12_ARMS)
    a, c = sum(r.succeeded for r in cover), sum(r.succeeded for r in state)
    return [cover, state], inference.fisher_exact(a, len(cover) - a, c, len(state) - c)


def state(data: Path) -> list[str]:
    """T12's rates (tab:state): each arm's successes of its runs, Wilson 95%, and its successes' epochs, their span and median."""
    lines: list[str] = []
    for (_, label), runs in zip(T12_ARMS, state_test(data)[0], strict=True):
        won, rate = succeeded(runs)
        epochs = [r.outcome.epochs for r in won]
        lines.append(f"{label} & {rate} & {span(epochs)} & {median(epochs) if epochs else '--'} \\\\")
    return lines


T12_LINES = {"d5-cover": [("A + B", 5), ("A - B", 5)], "d5-state": [("A", 5), ("B", 5)]}  # each arm's lines read


def paper_state(data: Path) -> Numbers:
    """A2's numbers from T12 (D31): each arm's successes, the two-sided p, the long tests, and (exploratory) the
    certified networks with a factor holding one of the trained lines (the cover's) or a coordinate (the state's)."""
    (cover, state_runs), p = state_test(data)
    made: Numbers = {
        "t12.cover": count(sum(r.succeeded for r in cover)),
        "t12.state": count(sum(r.succeeded for r in state_runs)),
        "t12.p": p_value(p),
        "t12.networks": count(len(mf.long_tested(data / "paper" / "t12" / "batch" / "long-tests"))),
    }
    for arm, lines in T12_LINES.items():
        found = readings(data / "paper" / "t12" / "batch" / "approximations" / f"{arm}.txt")
        key = arm.removeprefix("d5-")
        made[f"t12.{key}-networks"] = count(len(found))
        made[f"t12.{key}-lines"] = none_or(sum(any(holds(r, n, b) for n, b in lines) for r in found))
    return made


TABLES: dict[str, Table] = {
    "tab:approximation": approximation,
    "tab:extra-outputs": arms_table(
        "t1",
        [
            ("sum", "sum (an approximation)"),
            ("suffix", "suffix (state-carrying, no approximation)"),
            ("token", "token (stateless)"),
        ],
    ),
    "tab:baselines": arms_table(
        "t3",
        [
            ("approximation", "snapped codes, transformer window (control)"),
            ("approximation-continuous", "continuous state (no codes)"),
            ("approximation-linear", "linear window (no blocks)"),
        ],
    ),
    "tab:scale": rates_table([("d7-family", 49), ("d11-family", 121), ("d13-family", 169), ("d11-quotient", 121)]),
    "tab:breadth": rates_table([("s3", 6), ("s4", 24), ("mtf4", 24), ("rq12", 12), ("rq16", 16)]),
    "tab:decomposition": decomposition_table(
        [
            ("$p$ 5 division claim, family", ["t4/p5-division-claim-family"]),
            ("$p$ 5 division claim, modular quotient", ["t4/p5-division-claim-quotient"]),
            ("$p$ 5 approximation claim, family", ["t4/p5-approximation-claim-family"]),
            ("$p$ 5 approximation claim, sum", ["t4/p5-approximation-claim-sum"]),
            ("$p$ 5 approximation claim, modular quotient", ["t4/p5-approximation-claim-quotient"]),
            ("$p$ 5 probe 2a (apart)", ["t4/p5-probe-2a"]),
            ("$p$ 7 probe 4 and fork probe", ["t4/p7-probe-4", "t4/p7-fork"]),
            ("$p$ 11 ladder", ["t4/p11-ladder"]),
            ("$p$ 13 W1", ["t4/p13-w1"]),
            ("word-problem-s3", ["t4/s3-round1"]),
            ("word-problem-s4", ["t4/s4-round1"]),
            ("move-to-front-3", ["t4/mtf3-round1"]),
        ]
    ),
    "tab:decomposition-fresh": decomposition_table(
        [
            (f"\\texttt{{{b}}}", [f"t2-t5/approximations/{b}.txt"])
            for b in ("d7-family", "d11-family", "d13-family", "s3", "s4", "mtf4", "rq12")
        ]
    ),
    "tab:planted": planted,
    "tab:proof": proof,
    "tab:refuted": refuted,
    "tab:groups": groups,
    "tab:state": state,
}


# the numbers in the prose: each \pn{key}{value} in a manuscript, its value regenerated


Numbers = dict[str, str]


def count(n: int) -> str:
    """A whole number as the manuscript writes it: thousands split by {,}."""
    return f"{n:,}".replace(",", "{,}")


def words(low: float, high: float, digits: int = 0) -> str:
    """A range in words, "1 to 3"; one value alone when both ends agree."""
    a, b = f"{low:.{digits}f}", f"{high:.{digits}f}"
    return a if a == b else f"{a} to {b}"


def none_or(n: int) -> str:
    """A count that is expected to be nothing, written "none" when it is."""
    return "none" if n == 0 else count(n)


def every(k: int, n: int) -> str:
    """k of n, written "every" when it is all of them."""
    return "every" if k == n else f"{k} of {n}"


def listed(values: list[int]) -> str:
    """Counts in a list, "0, 1, 2, 4 and 5"."""
    shown = [count(v) for v in values]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def p_value(p: float) -> str:
    """A p value to two significant figures, as the manuscript writes them."""
    return f"{p:#.2g}"


ADDITION_STUDIES = (("matrix", 3), ("control", 1))  # each study and the step its networks end at


def addition_networks(data: Path) -> list[dict[str, Any]]:
    """
    Addition's 105 networks (the matrix's and the all-at-once control's last steps): each run's
    result, its two certificates (branched, strict) and its long test where it had one.
    """
    networks: list[dict[str, Any]] = []
    for study, step in ADDITION_STUDIES:
        for path in sorted((data / study / "results").glob("*.json")):
            result = json.loads(path.read_text())
            name = f"{result['key']}-step-{step}"
            long_test = data / "long-tests" / f"{name}.json"
            networks.append(
                {
                    "study": study,
                    "name": name,
                    "network": data / study / "models" / result["key"] / f"step-{step}.network.json",
                    "result": result,
                    "branched": json.loads((data / "certificates" / f"{name}.json").read_text()),
                    "strict": json.loads((data / "certificates-strict" / f"{name}.json").read_text()),
                    "long": json.loads(long_test.read_text()) if long_test.exists() else None,
                }
            )
    return networks


def base3_digits(x: int) -> int:
    """How many base-3 digits x has (1 for 0)."""
    digits = 1
    while x >= 3:
        x, digits = x // 3, digits + 1
    return digits


def first_wrong(network: Path, triple: tuple[int, int, int]) -> tuple[int, int]:
    """
    The shortest length, in whole windows from the operands' digits to 64 columns, at which the
    test harness reads a network's sum of this triple wrong, and its answer there.
    """

    net = wr.load_network(str(network))
    model = wr.WindowModel(net, net.columns, wt.task_named("addition"))
    digits = max(base3_digits(x) for x in triple)
    start = -(-digits // net.columns) * net.columns
    for n in range(start, 65, net.columns):
        [labels] = model.predict(n)([triple])
        answer = ad.from_digits(labels)
        if answer != sum(triple):
            return n, answer
    raise ValueError(f"{network}: right at every length to 64")


def addition_certificates(data: Path) -> Numbers:
    """K2's addition numbers (Appendix A): the tests against the certificate and the long tests."""

    networks = addition_networks(data)
    working = [n for n in networks if n["result"]["works"]]
    failing = [n for n in networks if not n["result"]["works"]]
    certified = [n for n in networks if n["strict"]["certified"]]
    reached = [len(n["strict"]["reachable"]) for n in certified]
    minimised = sorted({len(n["strict"]["minimised"]) for n in certified})
    uncertified = [n for n in working if not n["strict"]["certified"]]
    wrong = [n for n in uncertified if n["result"]["break"] == 64]
    cases = [(first_wrong(n["network"], tuple(n["strict"]["counterexample"]["addition"]["triple"])), n) for n in wrong]
    cases.sort(key=lambda case: (case[0][0], sum(case[1]["strict"]["counterexample"]["addition"]["triple"])))
    tested = [(n, *wr.long_test_failures(n["long"])) for n in networks if n["long"] is not None]
    each = {cases_ for _, cases_, _ in tested}
    if len(each) != 1:
        raise ValueError(f"addition's long tests: {sorted(each)} cases a network, not one count")
    failed = [n for n, _, failures in tested if failures]
    if len(cases) != 2:
        raise ValueError(f"addition: {len(cases)} wrong networks passing every test, not 2")
    (first, one), (second, other) = cases
    triple = one["strict"]["counterexample"]["addition"]["triple"]
    long_wrong = {n["name"]: sum(int(f.rsplit(" ", 1)[1]) for f in failures) for n, _, failures in tested}
    return {
        "addition.matrix-runs": count(sum(n["study"] == "matrix" for n in networks)),
        "addition.control-runs": count(sum(n["study"] == "control" for n in networks)),
        "addition.networks": count(len(networks)),
        "addition.failing": count(len(failing)),
        "addition.failing-certified": none_or(
            sum(n["strict"]["certified"] or n["branched"]["certified"] for n in failing)
        ),
        "addition.working": count(len(working)),
        "addition.certified": count(sum(n["strict"]["certified"] for n in working)),
        "addition.robust": count(sum(n["branched"]["certified"] for n in working)),
        "addition.minimised": ", ".join(str(m) for m in minimised),
        "addition.codes-reached": words(min(reached), max(reached)),
        "addition.codes-median": median(reached),
        "addition.working-uncertified": count(len(uncertified)),
        "addition.break-early": count(sum(n["result"]["break"] < 64 for n in uncertified)),
        "addition.wrong-passing": count(len(wrong)),
        "addition.counterexample": " + ".join(count(x) for x in triple),
        "addition.counterexample-answer": count(first[1]),
        "addition.counterexample-sum": count(sum(triple)),
        "addition.counterexample-columns": count(first[0]),
        "addition.counterexample-other-columns": count(second[0]),
        "addition.long-each": count(each.pop()),
        "addition.long-cases": f"{sum(c for _, c, _ in tested) / 1e6:.2f}",
        "addition.long-certified-passing": every(
            sum(not f for n, _, f in tested if n["strict"]["certified"]),
            sum(1 for n, _, _ in tested if n["strict"]["certified"]),
        ),
        "addition.long-failing-rejected": every(sum(not n["strict"]["certified"] for n in failed), len(failed)),
        "addition.counterexample-long-wrong": count(long_wrong[one["name"]]),
        "addition.counterexample-other-long-wrong": none_or(long_wrong[other["name"]]),
    }


CODE_COUNTS = (3, 4, 6, 8, 16)  # the code-count comparison's arms, k 4, s 4, the curriculum


def addition_code_counts(data: Path) -> Numbers:
    """
    K2's registered comparison of the code count (Appendix A), with the Holm correction over its
    family as registered: the snap (C2), the code count (C3), the curriculum (C4) and unexplored
    states (C6).
    """

    root = data / "e3-e4"
    results = [json.loads(p.read_text()) for p in sorted((root / "results").glob("*.json"))]

    def cell(arm: str, m: int) -> list[dict[str, Any]]:
        return [r for r in results if r["arm"] == arm and r["k"] == 4 and r["s"] == 4 and r["m"] == m]

    def strict(r: dict[str, Any]) -> bool:
        step = len(r["steps"])
        return json.loads((root / "certificates-strict" / f"{r['key']}-step-{step}.json").read_text())["certified"]

    arms = [cell("curriculum", m) for m in CODE_COUNTS]
    seeds = {len(runs) for runs in arms}
    if len(seeds) != 1:
        raise ValueError(f"the code-count arms: {sorted(seeds)} runs, not one count")
    trials = [len(runs) for runs in arms]
    works = [sum(r["works"] for r in runs) for runs in arms]
    certified = [sum(strict(r) for r in runs) for runs in arms]
    working_trend = inference.cochran_armitage(works, trials, CODE_COUNTS)
    certified_trend = inference.cochran_armitage(certified, trials, CODE_COUNTS)
    # C2: E3's snapped curriculum at m 16 against E2's continuous state, working at 1,024 columns
    # (long-tested when its break is 64)
    snapped = cell("curriculum", 16)

    def at_1024(r: dict[str, Any]) -> bool:
        path = root / "long-tests" / f"{r['key']}-step-{len(r['steps'])}.json"
        return r["break"] == 64 and path.exists() and not wr.long_test_failures(json.loads(path.read_text()))[1]

    continuous = [json.loads(p.read_text()) for p in sorted((data / "e2" / "results").glob("*.json"))]
    s, c = sum(at_1024(r) for r in snapped), sum(r["break"] == 64 for r in continuous)
    snap = inference.fisher_exact(s, len(snapped) - s, c, len(continuous) - c)
    # C4: all at once against the curriculum at m 16, working
    a, b = sum(r["works"] for r in cell("all-at-once", 16)), works[-1]
    curriculum = inference.fisher_exact(a, len(cell("all-at-once", 16)) - a, b, trials[-1] - b)
    # C6: among runs whose check passed, failing against working runs reaching a new code at 16 columns
    explored = [r for r in json.loads((data / "e5" / "e5.json").read_text()) if r["passes"]]
    fail = [bool(r["new"]["16"]) for r in explored if not r["works"]]
    work = [bool(r["new"]["16"]) for r in explored if r["works"]]
    unexplored = inference.fisher_exact(sum(fail), len(fail) - sum(fail), sum(work), len(work) - sum(work))
    holm = inference.holm([snap, working_trend.p, curriculum, unexplored])
    return {
        "addition.codes-seeds": count(seeds.pop()),
        "addition.codes-m": ", ".join(str(m) for m in CODE_COUNTS[:-1]) + f" and {CODE_COUNTS[-1]}",
        "addition.codes-working-z": f"{working_trend.z:.2f}",
        "addition.codes-working-p": p_value(working_trend.p),
        "addition.codes-working-holm": p_value(holm[1]),
        "addition.codes-certified-z": f"{certified_trend.z:.2f}",
        "addition.codes-certified-p": p_value(certified_trend.p),
        "addition.codes-uncertified": listed([w - c for w, c in zip(works, certified, strict=True)]),
    }


def replays(directory: Path) -> list[dict[str, Any]]:
    """A replay probe's runs in order (name-0, name-1, ..., name-9)."""
    paths = [p for p in directory.glob("*.json") if p.stem.rsplit("-", 1)[-1].isdigit()]
    return [
        json.loads(p.read_text())
        for p in sorted(paths, key=lambda p: (p.stem.rsplit("-", 1)[0], int(p.stem.rsplit("-", 1)[1])))
    ]


def kept(replay: dict[str, Any]) -> bool:
    """A replay kept its proof: certified in both verdicts after every epoch."""
    return all(e["certified"] and e["certified_strict"] for e in replay["epochs"])


AMPLIFIED = 0.1  # an update five times the epochs' median (about 0.02): the cascade's start
JUMP = 1.0  # a batch's loss past 1 after its update: a jump the loss sees (the snap-wall workplan's)


def addition_replays(data: Path) -> Numbers:
    """
    K10's addition numbers (Appendix A): the 10 replays read off after every batch (`machine-diff`,
    the same runs as P1's wall-off arm), the certificate guard on them (`replay-guard`), and four
    networks that kept their proofs (`stable-replay`).
    """
    runs = replays(data / "machine-diff")
    firsts: list[dict[str, Any]] = []
    amplified: list[int] = []  # batches from the first flip to the first large update
    silent = 0
    for run in runs:
        batches = [b for e in run["epochs"] for b in e["batches"]]
        i = next(i for i, b in enumerate(batches) if b["pairs_changed"] > 0)
        first = batches[i]
        if first["wrong_pairs"] != first["pairs_changed"]:
            raise ValueError("a first change with a right pair")
        firsts.append(first)
        silent += first["loss_after"] <= JUMP
        amplified.append(next(j - i for j in range(i, len(batches)) if batches[j]["update"] > AMPLIFIED))
    *soon, last = sorted(amplified)
    worst = [max(b["wrong_pairs"] for e in run["epochs"] for b in e["batches"]) for run in runs]
    guarded = replays(data / "replay-guard")
    stable = replays(data / "stable-replay")
    starts = {r["network"] for r in stable}
    return {
        "replay.count": count(len(runs)),
        "replay.lost": count(sum(not kept(r) for r in runs)),
        "replay.kept": count(sum(kept(r) for r in runs)),
        "replay.first-pairs": words(min(f["pairs_changed"] for f in firsts), max(f["pairs_changed"] for f in firsts)),
        "replay.first-norms": f"{min(f['update'] for f in firsts):.3f}--{max(f['update'] for f in firsts):.3f}",
        "replay.silent": count(silent),
        "replay.amplified": count(len(soon)),
        "replay.amplified-within": words(min(soon), max(soon)),
        "replay.amplified-last": count(last),
        "replay.worst-wrong": words(min(worst), max(worst)),
        "guard.count": count(len(guarded)),
        "guard.kept": count(sum(kept(r) for r in guarded)),
        "guard.undone": count(sum(e["rollbacks"] for r in guarded for e in r["epochs"])),
        "guard.batches": count(sum(len(e["batches"]) for r in guarded for e in r["epochs"])),
        "stable.networks": count(len(starts)),
        "stable.each": count(len(stable) // len(starts)),
        "stable.kept": every(sum(kept(r) for r in stable), len(stable)),
    }


def p_math(p: float) -> str:
    """A p value for math mode: two significant figures, in powers of ten below 0.001."""
    if p >= 0.001:
        return p_value(p)
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


def near(runs: list[mf.ArmRun]) -> int:
    """How many of the runs came near a solution (every answer at least NEAR at some diagnosis)."""
    return sum(r.outcome.peak >= mf.NEAR for r in runs)


def wins(data: Path, test: str, arm: str) -> tuple[int, list[mf.ArmRun]]:
    """An arm's successes and its runs."""
    runs = batch_runs(data, test, arm)
    return sum(r.succeeded for r in runs), runs


def greater(first: tuple[int, list[mf.ArmRun]], second: tuple[int, list[mf.ArmRun]]) -> float:
    """The registered comparisons' test: one-sided Fisher, the first arm succeeding more often."""
    (a, ra), (b, rb) = first, second
    return inference.fisher_exact_greater(a, len(ra) - a, b, len(rb) - b)


@cache
def paper_family(data: Path) -> dict[str, tuple[float, float]]:
    """The paper's registered family (D5; R7's C1, T10r in it): each registered test's p, as registered (one- or
    two-sided), and its Holm-adjusted p over the whole family."""
    t1 = wins(data, "t1", "sum")
    t3 = wins(data, "t3", "approximation")
    planted = machine_runs(data, "t10", ["p0", "p1", "p2"])
    t10 = {arm: mf.machine_successes(runs) for arm, runs in planted.items()}
    fresh = {arm: mf.machine_successes(runs) for arm, runs in machine_runs(data, "t10r", ["p0", "p2"]).items()}
    p = {
        "t1a": greater(t1, wins(data, "t1", "suffix")),
        "t1b": greater(t1, wins(data, "t1", "token")),
        "t2b": greater(wins(data, "t2-t5", "d11-family"), wins(data, "t2-t5", "d11-quotient")),
        "t3a": greater(t3, wins(data, "t3", "approximation-continuous")),
        "t3b": greater(t3, wins(data, "t3", "approximation-linear")),
        "t10a": inference.rank_sum_greater(t10["p1"], t10["p0"]),
        "t10b": inference.rank_sum_greater(t10["p2"], t10["p0"]),
        "t10r": inference.rank_sum_greater(fresh["p2"], fresh["p0"]),
        "t12": state_test(data)[1],
    } | {f"t11-{backdrop}": pv for backdrop, _, _, _, pv in group_tests(data)}
    return dict(zip(p, zip(p.values(), inference.holm(list(p.values())), strict=True), strict=True))


def family_holm(data: Path) -> Numbers:
    """Each registered test's Holm-adjusted p over the paper's family."""
    return {f"family.{test}": p_math(holm) for test, (_, holm) in paper_family(data).items()}


PAPER_TESTS = {  # each test's batch and arms (T2 and T5 one batch, D17)
    "t1": ["sum", "suffix", "token"],
    "t3": ["approximation", "approximation-continuous", "approximation-linear"],
    "t2-t5": ["d7-family", "d11-family", "d13-family", "d11-quotient", "s3", "s4", "mtf4", "rq12", "rq16"],
}


def long_tests(data: Path, test: str) -> list[tuple[bool, int, list[str], dict[str, Any]]]:
    """A test's long tests: each network's certification (both verdicts), its cases, failures and lengths."""
    verdicts = mf.recertifications(data / "paper" / test / "certify")
    found: list[tuple[bool, int, list[str], dict[str, Any]]] = []
    for path in sorted((data / "paper" / test / "long-tests").glob("*/*.long-test.json")):
        long = json.loads(path.read_text())
        found_verdicts = verdicts.get(long["sha256"], {})
        certified = set(found_verdicts) == set(mf.VERDICTS) and all(found_verdicts.values())
        found.append((certified, *wr.long_test_failures(long), long))
    return found


def paper_rates(data: Path) -> Numbers:
    """K1's, K3's, K4's, K6's and K5's numbers from T1, T2, T3 and T5's batches and the approximation claim."""
    made: Numbers = {}
    coded = certified = 0
    every_cases = 0
    failing: list[bool] = []  # each long-test failure: whether its network has codes
    for test, arms in PAPER_TESTS.items():
        runs = [r for arm in arms for r in batch_runs(data, test, arm)]
        coded += sum(r.coded for r in runs)
        won = sum(r.succeeded for r in runs)
        certified += won
        tested = [t for t in long_tests(data, test) if t[0]]
        if len(tested) != won or any(t[2] for t in tested):
            raise ValueError(f"{test}: {len(tested)} certified networks long-tested against {won} successes")
        cases = sum(t[1] for t in tested)
        every_cases += cases
        made[f"k1.{test}-networks"] = count(len(tested))
        made[f"k1.{test}-cases"] = count(cases)
        failing += [run.coded for run in runs if run.long is False]
    made |= {
        "k1.networks": count(105 + coded),  # with addition's 105 (Appendix A)
        "k1.coded": count(coded),
        "k1.certified": count(certified),
        "k1.cases": count(every_cases),
        "k1.failing-uncoded": count(sum(not coded_ for coded_ in failing)),
        "k1.failing-coded": none_or(sum(failing)),
    }
    # K3 (T3): the control against the continuous state and the linear window, Holm over the paper's family
    continuous = wins(data, "t3", "approximation-continuous")
    family = paper_family(data)
    p, holm = zip(family["t3a"], family["t3b"], strict=True)
    uncertified = [long for certified_, _, _, long in long_tests(data, "t3") if not certified_]  # the continuous
    drifted = [
        all(r["passed"] == 0 for length in long["lengths"].values() for r in length["properties"].values())
        for long in uncertified
    ]
    made |= {
        "k3.continuous-p": p_math(p[0]),
        "k3.continuous-holm": p_math(holm[0]),
        "k3.linear-same": "the same" if (p[1], holm[1]) == (p[0], holm[0]) else f"$p = {p_math(p[1])}$",
        "k3.continuous-near": every(near(continuous[1]), len(continuous[1])),
        "k3.continuous-failed": every(sum(drifted), len(continuous[1])),
        "k3.continuous-from": count(min(int(n) for long in uncertified for n in long["lengths"])),
    }
    # K6 and K4 (T2): the rates at scale, and the cover against the quotient alone at 121 states
    scale = [wins(data, "t2-t5", b) for b in ("d7-family", "d11-family", "d13-family")]
    quotient = wins(data, "t2-t5", "d11-quotient")
    made |= {
        "k6.rates": ", ".join(str(w) for w, _ in scale[:-1]) + f" and {scale[-1][0]}",
        "k6.seeds": count(len(scale[0][1])),
        "k4.cover": count(scale[1][0]),
        "k4.quotient": none_or(quotient[0]),
        "k4.p": p_math(greater(scale[1], quotient)),
        "k4.quotient-near": none_or(near(quotient[1])),
    }
    # K4 at 25 states and K5's earlier rate: the approximation claim at p 5
    root = data / "approximation-claim"
    claim = mf.recertifications(root / "certify")
    rate = {arm: sum(r.certified for r in mf.arm_runs(root, arm, claim)) for arm in ("family", "sum", "quotient")}
    made |= {
        "k4.claim-cover": count(rate["family"]),
        "k4.claim-quotient": count(rate["quotient"]),
        "k5.claim-sum": count(rate["sum"]),
    }
    # K5 (T1): the sum against the suffix and the token, Holm over the paper's family
    total, suffix, token = (wins(data, "t1", arm) for arm in ("sum", "suffix", "token"))
    p, holm = zip(family["t1a"], family["t1b"], strict=True)
    failed, came_near, _ = failures(total[1])
    made |= {
        "k5.sum": count(total[0]),
        "k5.suffix": count(suffix[0]),
        "k5.token": count(token[0]),
        "k5.suffix-p": p_math(p[0]),
        "k5.suffix-holm": p_math(holm[0]),
        "k5.token-p": p_math(p[1]),
        "k5.token-holm": p_math(holm[1]),
        "k5.suffix-certified": none_or(suffix[0]),
        "k5.suffix-near": none_or(near(suffix[1])),
        "k5.sum-near": f"{came_near} of {failed}",
    }
    return made


DIVISION_READINGS = (
    ["t4/p5-division-claim-family", "t4/p5-division-claim-quotient", "t4/p5-approximation-claim-family"]
    + ["t4/p5-approximation-claim-sum", "t4/p5-approximation-claim-quotient", "t4/p5-probe-2a", "t4/p7-probe-4"]
    + ["t4/p7-fork", "t4/p11-ladder", "t4/p13-w1"],
    [f"t2-t5/approximations/{b}.txt" for b in ("d7-family", "d11-family", "d13-family")],
)
ALGEBRA_READINGS = (  # the algebras: the groups and move-to-front (not the random machines)
    ["t4/s3-round1", "t4/s4-round1", "t4/mtf3-round1"],
    [f"t2-t5/approximations/{b}.txt" for b in ("s3", "s4", "mtf4")],
)
GROUP_READINGS = ["t4/s3-round1", "t4/s4-round1", "t2-t5/approximations/s3.txt", "t2-t5/approximations/s4.txt"]
TRAINED_LINES = {"A + B", "A - B"}  # division's approximations trained as outputs


def paper_readings(data: Path) -> Numbers:
    """K8's numbers (T4): the decomposition readings, the earlier networks' and the fresh ones'."""

    def nets(paths: list[str]) -> list[Reading]:
        return [n for path in paths for n in readings(data / "paper" / path)]

    def nontrivial(found: list[Reading]) -> str:
        held = sum(1 < blocks < n.states for n in found for blocks, _ in n.factors)
        return f"{held} of {sum(len(n.factors) for n in found)}"

    made: Numbers = {}
    for name, paths in (("first", 0), ("fresh", 1)):
        made[f"k8.algebras-{name}"] = nontrivial(nets(ALGEBRA_READINGS[paths]))
        made[f"k8.division-{name}"] = nontrivial(nets(DIVISION_READINGS[paths]))
    groups = [name for n in nets(GROUP_READINGS) for _, names in n.factors for name in names]
    division = [name for paths in DIVISION_READINGS for n in nets(paths) for _, names in n.factors for name in names]
    lined = [  # the congruences named by networks trained on lines (not on the quotient alone)
        name
        for paths in DIVISION_READINGS
        for path in paths
        if "quotient" not in path
        for n in readings(data / "paper" / path)
        for _, names in n.factors
        for name in names
    ]
    made |= {
        "k8.groups-places": every(sum(name.startswith("places") for name in groups), len(groups)),
        "k8.groups-sign": none_or(sum("sign" in name for name in groups)),
        "k8.division-untrained": count(sum(name not in TRAINED_LINES for name in division)),
        "k8.division-trained-lines": every(sum(name in TRAINED_LINES for name in lined), len(lined)),
        "k8.division-one-line": every(
            sum(len(names) == 1 for paths in DIVISION_READINGS for n in nets(paths) for _, names in n.factors if names),
            sum(1 for paths in DIVISION_READINGS for n in nets(paths) for _, names in n.factors if names),
        ),
    }
    return made


def paper_proofs(data: Path) -> Numbers:
    """K9's numbers (T6): the target-free proof at the exact bound and at 1.1 times it."""
    bounds = proofs(data)
    exact = Counter(proof_outcome(b) for b in bounds if b["slack"] == 1.0)
    slack = Counter(proof_outcome(b) for b in bounds if b["slack"] == 1.1)
    lengths = [len(b["chow"]["counterexample"]) for b in bounds if proof_outcome(b) == "false-stop refuted"]
    false_stops = sum(v for k, v in slack.items() if k.startswith("false-stop"))
    strict_only = sum(v for k, v in slack.items() if k.startswith("strict-only"))
    return {
        "k9.runs": count(sum(exact.values())),
        "k9.exact-false": none_or(sum(v for k, v in exact.items() if k.startswith("false-stop"))),
        "k9.exact-proved": count(exact["kept proved"]),
        "k9.exact-unproved": none_or(exact["kept out of reach"]),
        "k9.slack-early": count(false_stops + strict_only),
        "k9.slack-strict": count(strict_only),
        "k9.slack-strict-proved": count(slack["strict-only proved"]),
        "k9.slack-strict-unreached": count(slack["strict-only out of reach"]),
        "k9.slack-false": count(false_stops),
        "k9.slack-proved": none_or(slack["false-stop proved"]),
        "k9.slack-refuted": count(slack["false-stop refuted"]),
        "k9.counterexamples": words(min(lengths), max(lengths)),
        "k9.slack-false-unreached": count(slack["false-stop out of reach"]),
        "k9.slack-kept-unreached": count(slack["kept out of reach"]),
    }


def paper_planted(data: Path) -> Numbers:
    """A4's numbers from T10 (D19): each arm's machines succeeding, and T10a's and T10b's rank-sum tests over the
    machines' counts, Holm over the two, and T10b with each machine counted at most once; and T10r's (D24), its
    replication read alone: its rank-sum (no Holm), each machine at most once, and the pooled Fisher's over runs."""
    runs = machine_runs(data, "t10", [arm for arm, _ in PLANTED_ARMS])
    counts = {arm: mf.machine_successes(runs[arm]) for arm, _ in PLANTED_ARMS}
    p, holm = zip(paper_family(data)["t10a"], paper_family(data)["t10b"], strict=True)
    once = inference.rank_sum_greater(mf.machine_successes(runs["p2"], 1), mf.machine_successes(runs["p0"], 1))
    made: Numbers = {f"t10.{arm}-machines": count(sum(1 for c in counts[arm] if c)) for arm, _ in PLANTED_ARMS}
    made |= {
        "t10.machines": count(len(counts["p0"])),
        "t10.a-p": p_value(p[0]),
        "t10.a-holm": p_value(holm[0]),
        "t10.b-p": f"{p[1]:.4f}",  # four places: T10b passed within T10 by no margin (0.0206 against 0.025)
        "t10.b-holm": p_value(holm[1]),
        "t10.b-once-p": f"{once:.4f}",
    }
    replication = machine_runs(data, "t10r", ["p0", "p2"])  # T10r (D24): one test, no Holm, read alone
    fresh = {arm: mf.machine_successes(replication[arm]) for arm in ("p0", "p2")}
    runs = {arm: [r.succeeded for machine in replication[arm].values() for r in machine] for arm in ("p0", "p2")}
    made |= {f"t10r.{arm}-machines": count(sum(1 for c in fresh[arm] if c)) for arm in ("p0", "p2")}
    made |= {f"t10r.{arm}-runs": count(sum(runs[arm])) for arm in ("p0", "p2")}
    made |= {
        "t10r.machines": count(len(fresh["p0"])),
        "t10r.runs": count(len(runs["p0"])),
        "t10r.p": f"{inference.rank_sum_greater(fresh['p2'], fresh['p0']):.4f}",
        "t10r.once-p": f"{inference.rank_sum_greater(mf.machine_successes(replication['p2'], 1), mf.machine_successes(replication['p0'], 1)):.4f}",
        "t10r.fisher-p": f"{inference.fisher_exact_greater(sum(runs['p2']), len(runs['p2']) - sum(runs['p2']), sum(runs['p0']), len(runs['p0']) - sum(runs['p0'])):.4f}",
    }
    return made


def paper_groups(data: Path) -> Numbers:
    """A5's numbers from T11 (D20): each algebra's successes given its cover, T5's, the Holm-adjusted p, and the long tests."""
    made: Numbers = {}
    family = paper_family(data)
    for backdrop, _, given, alone, _ in group_tests(data):
        holm = family[f"t11-{backdrop}"][1]
        made[f"t11.{backdrop}"] = count(sum(r.succeeded for r in given))
        made[f"t11.{backdrop}-alone"] = count(sum(r.succeeded for r in alone))
        made[f"t11.{backdrop}-holm"] = p_math(holm)
    long = mf.long_tested(data / "paper" / "t11" / "batch" / "long-tests")
    made["t11.networks"] = count(len(long))
    return made


def holds(reading: Reading, name: str, blocks: int) -> bool:
    """Whether a factor of the network holds exactly the named congruence: it determines it with its blocks."""
    return any(n == blocks and name in names for n, names in reading.factors)


T11C_GIVEN = {
    "s3": [("parity", 2)],
    "s4": [("places {0}", 4), ("pairings", 6)],
    "mtf4": [("order {0, 1, 3}", 6), ("ahead {2}", 8)],
}


def paper_factors(data: Path) -> Numbers:
    """A6's numbers: T11c (D25), the certified T11 networks whose factors hold each given approximation exactly, and
    the given pair; T4' (D30), the planted machines' certified networks with a factor holding the planted one."""
    made: Numbers = {}
    for backdrop, given in T11C_GIVEN.items():
        found = readings(data / "paper" / "t11" / "batch" / "approximations" / f"t11-{backdrop}.txt")
        made[f"t11c.{backdrop}-networks"] = count(len(found))
        for name, blocks in given:
            made[f"t11c.{backdrop}-{name.split()[0]}"] = count(sum(holds(r, name, blocks) for r in found))
        if len(given) > 1:
            made[f"t11c.{backdrop}-pair"] = none_or(sum(all(holds(r, n, b) for n, b in given) for r in found))
    planted: dict[str, list[bool]] = {"p1": [], "p2": []}
    for test in ("t10", "t10r"):
        for path in sorted((data / "paper" / test / "batch" / "approximations").glob(f"{test}-p[12]-*.txt")):
            blocks = len(set(wt.planted_labels(BACKDROPS[path.stem].task)["planted"].tolist()))
            planted[path.stem.split("-")[1]] += [holds(r, "planted", blocks) for r in readings(path)]
    for arm, found in planted.items():
        made[f"t4p.{arm}"] = none_or(sum(found))
        made[f"t4p.{arm}-networks"] = count(len(found))
    return made


NUMBER_SOURCES: list[Callable[[Path], Numbers]] = [
    addition_certificates,
    addition_code_counts,
    addition_replays,
    paper_rates,
    paper_readings,
    paper_proofs,
    paper_planted,
    paper_groups,
    paper_state,
    paper_factors,
    family_holm,
]
PN = re.compile(r"\\pn\{([^{}]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}")


def numbers(data: Path) -> Numbers:
    """Every regenerated number, by its key."""
    made: Numbers = {}
    for source in NUMBER_SOURCES:
        for key, value in source(data).items():
            if key in made:
                raise ValueError(f"{key}: two sources")
            made[key] = value
    return made


def manuscript_rows(tex: str, label: str) -> list[str] | None:
    """The rows of the table labelled `label` in a manuscript: its lines between \\midrule and \\bottomrule."""
    for table in re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, re.DOTALL):
        if f"\\label{{{label}}}" in table:
            body = table.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
            return [line.strip() for line in body.splitlines() if line.strip()]
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/window-recurrent"))
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="tables' rows, regenerated")
    show.add_argument("names", nargs="+", choices=sorted(TABLES))
    check = commands.add_parser("check", help="each table and number a manuscript holds against its regeneration")
    check.add_argument("manuscripts", type=Path, nargs="+")
    commands.add_parser("numbers", help="every number, regenerated, by its key")
    fill = commands.add_parser("fill", help="each \\pn's value in a manuscript replaced by its regeneration")
    fill.add_argument("manuscripts", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "show":
        for name in args.names:
            print(f"% {name}")
            print("\n".join(TABLES[name](args.data)))
        return 0
    if args.command == "numbers":
        for key, value in numbers(args.data).items():
            print(f"{key} = {value}")
        return 0
    if args.command == "fill":
        made = numbers(args.data)
        for tex in args.manuscripts:
            text = tex.read_text()
            unknown = sorted({key for key, _ in PN.findall(text)} - made.keys())
            if unknown:
                print(f"{tex}: no source for {', '.join(unknown)}")
                return 1
            filled = PN.sub(lambda m: f"\\pn{{{m[1]}}}{{{made[m[1]]}}}", text)
            changed = sum(old != made[key] for key, old in PN.findall(text))
            tex.write_text(filled)
            print(f"{tex}: {changed} numbers changed")
        return 0
    differs = False
    made_numbers: Numbers | None = None  # computed only when a manuscript holds a number
    for tex in args.manuscripts:
        text = tex.read_text()
        held_numbers = PN.findall(text)
        if held_numbers and made_numbers is None:
            made_numbers = numbers(args.data)
        wrong = [(key, old) for key, old in held_numbers if made_numbers is None or made_numbers.get(key) != old]
        if held_numbers:
            print(f"{tex}: numbers: {len(held_numbers)}, " + (f"{len(wrong)} DIFFER" if wrong else "the same"))
        for key, old in wrong:
            differs = True
            print(f"  {key}: manuscript {old}, regenerated {(made_numbers or {}).get(key, '(no source)')}")
        for name, table in TABLES.items():
            held = manuscript_rows(text, name)
            if held is None:
                continue
            made = table(args.data)
            if held == made:
                print(f"{tex}: {name}: {len(made)} rows, the same")
                continue
            differs = True
            print(f"{tex}: {name}: DIFFERS")
            for old, new in zip(held, made, strict=False):
                if old != new:
                    print(f"  manuscript: {old}\n  regenerated: {new}")
            if len(held) != len(made):
                print(f"  {len(held)} rows in the manuscript, {len(made)} regenerated")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
