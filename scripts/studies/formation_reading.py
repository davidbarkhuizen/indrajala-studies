"""
The formation reading (indrajala_studies.formation): when runs form their states, on the clock of the step
over the target's depth plus Chow's k, and what a stopping rule would have done.

- `survey ROOT ...`: every curriculum run's log under the roots (a run whose steps end every epoch, a
  state-formation or commitment-control probe, is another method's and left out), its machine read from its
  path; for each kind, the runs not formed by `ratio` times depth plus k and how many certified later, then
  each rule's effect over all of them.
- `stages ROOT ...`: the survey's runs with their networks' architectures (formation.Architecture) and
  targets: for each cell (machine, target states, arm, width W, blocks, factor widths, codes) the runs
  formed, separated (one target state a code, from the logged machine summary) and certified, with the
  load C/W^2; then each signal's flags at ratios of depth plus k and the certificates they would lose (the
  share against states per code), and the steps from each stage to the certificate (the stages reading,
  2026-11-23).
- `batch DATA --rules R:T ...`: a runner batch's attempts (DATA/attempts, the machine from the backdrop): for
  each rule, each run it would have stopped, where, and whether the run certified; then its effect. The
  shadow reading of a batch run without stopping (the told plan's width probe, registered 2026-11-22).

A rule `R:T` stops a run at the first step end at or past R times depth plus k, before a certificate, with a
share below T (formation.Rule).
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from indrajala_studies import formation as fm
from indrajala_studies import window_tasks as wt
from indrajala_studies.batch import BACKDROPS

RULES = ["6:0.999", "4:0.9", "2.5:0.9"]
RATIOS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

# a survey's path conventions, in order: (pattern, kind, task); a division run's p from its path, else 13
PATHS = [
    (r"word-problem-s(\d)|/t11-s(\d)-|/s(\d)-\d", "s", "word-problem-s{}"),
    (r"move-to-front-(\d)|mtf(\d)", "mtf", "move-to-front-{}"),
    (r"random-(\d+)-(\d+)/", "random", "random-{}-{}"),
    (r"/rq(\d+)-", "random", "random-q{}-0"),
    (r"b-q(\d+)/", "random", "random-q{}-0"),
    (
        r"told-(?:alone|planted|matched)-(\d+)|/p(\d+)-(?:alone|planted|matched)/|t10r?-p\d-(\d+)",
        "random",
        "random-planted-q16-b4-{}",
    ),
]
OTHER_METHOD = re.compile(r"state-formation|commitment-control|bits|controls|r34|random-perm")


def survey_machine(path: str) -> tuple[str, wt.Mealy]:
    """A survey path's machine and its kind's label, from the path conventions (PATHS)."""
    for pattern, kind, task in PATHS:
        if found := re.search(pattern, path):
            groups = [g for g in found.groups() if g is not None]
            return (f"s{groups[0]}" if kind == "s" else kind), wt.task_named(task.format(*groups)).machine
    p = re.search(r"[/.]d(\d+)-|/p(\d+)\b|division-(\d+)|d(\d+)q?/", path)
    prime = int(next(g for g in p.groups() if g)) if p else 13
    return "division", wt.modular_machine(prime if prime in (5, 7, 11, 13, 17, 19, 23) else 13)


def survey_run(log: Path, root: Path) -> tuple[str, fm.Run] | None:
    path = "/" + str(log.parent.relative_to(root)) + "/"
    if OTHER_METHOD.search(path):
        return None
    label, machine = survey_machine(path)
    run = fm.read_run(log, fm.depth_plus_k(machine), path)
    if not run.ends or max(e.share for e in run.ends) <= 0:  # no codes: a continuous cell
        return None
    return label, run


def rules(texts: list[str]) -> list[fm.Rule]:
    return [fm.Rule(float(r), float(t)) for r, t in (text.split(":") for text in texts)]


def survey(roots: list[Path], ratios: list[float], chosen: list[fm.Rule]) -> list[str]:
    kinds: dict[str, list[fm.Run]] = defaultdict(list)
    for root in roots:
        for log in sorted(root.rglob("log.jsonl")):
            found = survey_run(log, root)
            if found:
                kinds[found[0]].append(found[1])
    everything = [r for runs in kinds.values() for r in runs]
    lines = [f"{len(everything)} curriculum runs, {sum(r.certified for r in everything)} certified"]
    for kind, runs in sorted(kinds.items()) + [("all", everything)]:
        cells = "  ".join(f"{x:g}: {a}/{b}" for x in ratios for a, b in [fm.unformed_later(runs, x)])
        lines.append(f"{kind} ({len(runs)}), not formed by x(d+k), certified later: {cells}")
    lines += [fm.effect_summary(rule, fm.effect(everything, rule)) for rule in chosen]
    return lines


ATTEMPT = re.compile(r"(?:[\w-]+\.)?(.+)-(\d+)(?:-words\d+)?")


def arm(run_directory: Path) -> str:
    """The run's arm and vocabulary, from its attempt's backdrop (its name, `<backdrop>-<seed>`), else '-'."""
    for part in reversed(run_directory.parts):
        if (found := ATTEMPT.fullmatch(part)) and found[1] in BACKDROPS:
            b = BACKDROPS[found[1]]
            return b.arm + ("+" + "/".join(b.vocabulary) if b.vocabulary else "")
    return "-"


@dataclass(frozen=True)
class Surveyed:
    """A survey's run with its kind, target states, arm and network's architecture."""

    kind: str
    states: int
    arm: str
    shape: fm.Architecture
    run: fm.Run


def told_states(run_directory: Path) -> int | None:
    """The target's states as the run recorded them (its checkpoint's run arguments), if it did."""
    checkpoint = run_directory / "checkpoint.json"
    if not checkpoint.exists():
        return None
    try:
        return json.loads(checkpoint.read_text()).get("run", {}).get("target_states")
    except json.JSONDecodeError:
        return None


def surveyed(roots: list[Path]) -> list[Surveyed]:
    found = list[Surveyed]()
    for root in roots:
        for log in sorted(root.rglob("log.jsonl")):
            made = survey_run(log, root)
            network = fm.network_file(log.parent)
            if made is None or network is None:
                continue
            shape = fm.architecture(json.loads(network.read_text()))
            if shape is None:
                continue
            machine = survey_machine("/" + str(log.parent.relative_to(root)) + "/")[1]
            if told_states(log.parent) not in (None, machine.states):  # the path's convention misread the task
                continue
            found.append(Surveyed(made[0], machine.states, arm(log.parent), shape, made[1]))
    return found


SIGNALS: dict[str, Callable[[fm.StepEnd], bool]] = {
    "share < 0.9": lambda e: e.share < 0.9,
    "share < 0.999": lambda e: e.share < fm.FULL,
    "states per code > 1.2": lambda e: (e.states_per_code or 0) > 1.2,
}


def stage_lines(runs: list[Surveyed], ratios: list[float]) -> list[str]:
    cells: dict[tuple[str, int, str, int, int, tuple[int, ...], tuple[int, ...]], list[Surveyed]] = defaultdict(list)
    for s in runs:
        a = s.shape
        cells[(s.kind, s.states, s.arm, a.width, a.blocks, a.factor_widths, a.factor_codes)].append(s)
    traced = [s.run for s in runs if s.run.traced]
    lines = [
        f"{len(runs)} curriculum runs with networks, {len(traced)} with machine summaries at every step end",
        "kind q arm | W blocks factor-widths codes C/W^2 | runs formed separated certified (separated '-': untraced)",
    ]
    for key in sorted(cells, key=str):
        group = cells[key]
        kind, q, arm_, w, blocks, widths, codes = key
        st = fm.stages([s.run for s in group])
        sep = str(st.separated) if all(s.run.traced for s in group) else "-"
        lines.append(
            f"{kind} {q} {arm_} | {w} {blocks} {'x'.join(map(str, widths))} {'x'.join(map(str, codes))} "
            f"{group[0].shape.load:.2f} | {st.runs} {st.formed} {sep} {st.certified}"
        )
    lines.append("signals at x(d+k), traced runs: flagged (certified later)")
    for x in ratios:
        cells_ = "  ".join(
            f"{name}: {b} ({a})" for name, flag in SIGNALS.items() for a, b in [fm.flagged_later(traced, x, flag)]
        )
        lines.append(f"  {x:g}: {cells_}")
    certified = [r for r in traced if r.certified and fm.separated_step(r) is not None]
    for name, step in (("formed", fm.formed_step), ("separated", fm.separated_step)):
        lags = [(fm.certified_step(r) or 0) - (step(r) or 0) for r in certified]
        q1, q2, q3 = statistics.quantiles(lags, n=4) if len(lags) > 1 else (0, 0, 0)
        same = sum(lag == 0 for lag in lags)
        lines.append(
            f"steps from {name} to the certificate ({len(lags)} runs): quartiles {q1:g}, {q2:g}, {q3:g}; at it {same}"
        )
    return lines


def batch(data: Path, chosen: list[fm.Rule]) -> list[str]:
    runs: list[fm.Run] = []
    for directory in sorted((data / "attempts").iterdir()):
        found = re.fullmatch(r"(?:[\w-]+\.)?(.+)-(\d+)(?:-words\d+)?", directory.name)
        assert found, directory.name
        backdrop = BACKDROPS[found[1]]
        log = backdrop.log(directory, int(found[2]))
        if log.exists():
            runs.append(fm.read_run(log, fm.depth_plus_k(wt.task_named(backdrop.task).machine), directory.name))
    lines = [f"{len(runs)} runs, {sum(r.certified for r in runs)} certified"]
    for rule in chosen:
        for r in runs:
            if (at := fm.stop(r, rule)) is not None:
                lines.append(
                    f"  {rule}: {r.name} at step {at.step}, epoch {at.epochs} of {r.epochs}, share {at.share:.2f}"
                    + (" (certified later)" if r.certified else "")
                )
        lines.append(fm.effect_summary(rule, fm.effect(runs, rule)))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("survey")
    s.add_argument("roots", nargs="+", type=Path)
    s.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    s.add_argument("--rules", nargs="+", default=RULES, metavar="R:T")
    t = sub.add_parser("stages")
    t.add_argument("roots", nargs="+", type=Path)
    t.add_argument("--ratios", nargs="+", type=float, default=[2.0, 2.5, 3.0, 4.0, 6.0])
    b = sub.add_parser("batch")
    b.add_argument("data", type=Path)
    b.add_argument("--rules", nargs="+", default=RULES, metavar="R:T")
    args = parser.parse_args()
    if args.command == "survey":
        lines = survey(args.roots, args.ratios, rules(args.rules))
    elif args.command == "stages":
        lines = stage_lines(surveyed(args.roots), args.ratios)
    else:
        lines = batch(args.data, rules(args.rules))
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
