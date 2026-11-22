"""
The formation reading (indrajala_studies.formation): when runs form their states, on the clock of the step
over the target's depth plus Chow's k, and what a stopping rule would have done.

- `survey ROOT ...`: every curriculum run's log under the roots (a run whose steps end every epoch, a
  state-formation or commitment-control probe, is another method's and left out), its machine read from its
  path; for each kind, the runs not formed by `ratio` times depth plus k and how many certified later, then
  each rule's effect over all of them.
- `batch DATA --rules R:T ...`: a runner batch's attempts (DATA/attempts, the machine from the backdrop): for
  each rule, each run it would have stopped, where, and whether the run certified; then its effect. The
  shadow reading of a batch run without stopping (the told plan's width probe, registered 2026-11-22).

A rule `R:T` stops a run at the first step end at or past R times depth plus k, before a certificate, with a
share below T (formation.Rule).
"""

import argparse
import re
from collections import defaultdict
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


def survey_run(log: Path, root: Path) -> tuple[str, fm.Run] | None:
    path = "/" + str(log.parent.relative_to(root)) + "/"
    if OTHER_METHOD.search(path):
        return None
    for pattern, kind, task in PATHS:
        if found := re.search(pattern, path):
            groups = [g for g in found.groups() if g is not None]
            name = task.format(*groups)
            machine = wt.task_named(name).machine
            label = f"s{groups[0]}" if kind == "s" else kind
            break
    else:
        p = re.search(r"/d(\d+)-|/p(\d+)\b|division-(\d+)|d(\d+)q?/", path)
        prime = int(next(g for g in p.groups() if g)) if p else 13
        if prime not in (5, 7, 11, 13, 17, 19, 23):
            prime = 13
        label, machine = "division", wt.modular_machine(prime)
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
    b = sub.add_parser("batch")
    b.add_argument("data", type=Path)
    b.add_argument("--rules", nargs="+", default=RULES, metavar="R:T")
    args = parser.parse_args()
    lines = (
        survey(args.roots, args.ratios, rules(args.rules))
        if args.command == "survey"
        else batch(args.data, rules(args.rules))
    )
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
