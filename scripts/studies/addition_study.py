"""
The addition study (its workplan retired: git show 8990f74:docs/addition-study-workplan.md): the
smallest network, of eight candidates built from the package's layers, that adds base-3 numbers
with every property of indrajala_studies/addition_data.py's catalogue, under
indrajala_studies/addition_study.py's harness.

    python scripts/studies/addition_study.py run CONFIG.json
    python scripts/studies/addition_study.py status CONFIG.json
    python scripts/studies/addition_study.py report CONFIG.json [--md FILE]
    python scripts/studies/addition_study.py check MODEL_DIRECTORY
    python scripts/studies/addition_study.py candidates

A config names the sweep's directory (results/, one JSON file per run, and models/, each run's
trained network(s) in format 2 beside a manifest), its workers, its training settings and its
studies (load_config's docstring). `run` resumes from the results on disk and prints a line per
finished run, then the report; on jebel, under nohup in the background, it needs nobody watching.
`check` loads a saved model and runs the whole catalogue against it.

OPENBLAS_NUM_THREADS=1, as every study.

Findings (stage 5), on jebel (i7-9700K), indrajala-ml aedde48, `scripts/studies/configs/addition_sweep.json`:
candidates 1-3 only (D12), `n` 4 and 6, every shape (1-3 layers, 1-2 heads), `d` bisected per
ladder, rate 0.003, at most 200 epochs, patience 30, 5 seeds, 6 workers. 619 runs, 401 passing,
none raised; 211 stopped on the plateau and 7 at the budget. Wall time 2026-10-11 04:16-09:15
(5.0 h; 30 h of runs). A failing run's epoch took 1.7 s at `n = 4` and 3.2 s at `n = 6` (medians,
6 at once). Every result and the passing models at each threshold and the next width are archived
in davidbarkhuizen/indrajala-models (`7748416`, `addition/2026-10-11-sweep/`), with this table.

- The smallest reliable adders are 2 layers at `n = 4` and 3 layers at `n = 6`, at `d` 6: at
  `n = 4`, 831 parameters (column-unordered) and 933 (causal, noncausal), every seed run passing,
  in 5-11 epochs; at `n = 6`, 1,197 (unordered) and 1,299 (causal, noncausal). Two layers at
  `n = 6` need `d` 8 (1,379 unordered, 1,515 causal and noncausal) or, causal with one head, 12
  (3,039).
- `d` 4 fails for every candidate and shape at both `n` (at most 3/5 seeds).
- One layer is enough at `n = 4` only with more width and many more epochs (a median of 56 against
  3 for deeper passes): column-unordered at `d` 12 (1,527; 477 with two heads at `d` 6, but see
  below), column-noncausal with two heads at `d` 8 (899), with one head at `d` 24 (5,763);
  column-causal with two heads at `d` 12 (1,731), with one head never, up to `d` 32 (2/5 at best).
  At `n = 6` one layer never passes, for any candidate or head count, up to `d` 32: no seed at any
  width.
- The workplan's prediction ("one causal layer reaches some distance L*, and a second extends it to
  n - 1") holds at `n = 6`, where it can be told apart: one causal layer reaches `L*` 2 at best (the
  others 3) up to `d` 32, and every passing run of 2 or 3 layers reaches the whole distance, `L*` 6.
- What fails is the carry over distance. In the 218 failing runs, the column sum (M1), the
  identity (A3) and the carry from a reset (M2) usually pass whole (169, 173 and 168 runs);
  the carry at distance (M3) passes whole in 12 and every carry pattern (M5) in 2. Narrow deep
  failures fail even `L = 1` (`L*` 0 in 53 of the 90 failing runs of 2 or 3 layers).
- Thresholds near the edge aren't monotone in `d` (D7's warning): column-unordered, 1 layer, 2 heads
  passes 4/5 at `d` 6, 3/5 at 8, 5/5 at 12, so its 477 parameters is the smallest model that met
  the rule, not a reliable size; column-noncausal 1 layer 2 heads passes 4/5 at both 8 and 12; and
  column-noncausal 3 layers 2 heads at `n = 6` needs `d` 8 where 1 head passes at 6.
- Heads matter little from 2 layers on (the same threshold either way in 10 of the 12 pairs of
  ladders) and a great deal at 1 layer.
- The three encodings differ little once deep enough: column-unordered's smaller vocabulary (10
  tokens against 27: 17 `d` fewer parameters, 102 at `d` 6) makes it the smallest wherever it
  passes, and its thresholds are the others' or below. The mask neither helps nor hurts: causal and
  noncausal share every threshold from 2 layers on but two (`n = 6`: 2 layers 1 head, 3 layers 2
  heads), one each way.

Per ladder, each width run with its seeds passing ("4/4" when a fifth wasn't needed, "0/3" when the
rung was already lost); the threshold is the smallest `d` with at least 4 of 5 passing; epochs are
the mean over its passing seeds.

| n | candidate | layers | heads | d run (seeds passing) | threshold d | parameters | epochs |
|---|---|---|---|---|---|---|---|
| 4 | causal | 1 | 1 | 12 (1/5), 24 (2/5), 32 (2/5) | none | | |
| 4 | causal | 1 | 2 | 6 (0/3), 8 (1/5), 12 (4/5), 16 (5/5) | 12 | 1,731 | 35.5 |
| 4 | causal | 2 | 1 | 4 (2/5), 6 (4/4), 8 (5/5), 12 (4/4) | 6 | 933 | 5.5 |
| 4 | causal | 2 | 2 | 4 (2/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 933 | 11.2 |
| 4 | causal | 3 | 1 | 4 (3/5), 6 (5/5), 8 (5/5), 12 (4/4) | 6 | 1,287 | 6.4 |
| 4 | causal | 3 | 2 | 4 (3/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 1,287 | 10.4 |
| 4 | unordered | 1 | 1 | 6 (0/5), 8 (1/5), 12 (4/4), 16 (4/5) | 12 | 1,527 | 60.5 |
| 4 | unordered | 1 | 2 | 4 (0/5), 6 (4/5), 8 (3/5), 12 (5/5) | 6 | 477 | 70.2 |
| 4 | unordered | 2 | 1 | 4 (1/4), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 831 | 5.6 |
| 4 | unordered | 2 | 2 | 4 (3/5), 6 (5/5), 8 (5/5), 12 (4/4) | 6 | 831 | 8.0 |
| 4 | unordered | 3 | 1 | 4 (1/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 1,185 | 7.8 |
| 4 | unordered | 3 | 2 | 4 (2/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 1,185 | 4.2 |
| 4 | noncausal | 1 | 1 | 12 (0/3), 16 (1/5), 24 (5/5), 32 (5/5) | 24 | 5,763 | 60.4 |
| 4 | noncausal | 1 | 2 | 6 (0/5), 8 (4/5), 12 (4/5) | 8 | 899 | 75.0 |
| 4 | noncausal | 2 | 1 | 4 (0/5), 6 (5/5), 8 (5/5), 12 (4/4) | 6 | 933 | 5.0 |
| 4 | noncausal | 2 | 2 | 4 (0/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 933 | 7.6 |
| 4 | noncausal | 3 | 1 | 4 (1/5), 6 (5/5), 8 (5/5), 12 (4/4) | 6 | 1,287 | 4.6 |
| 4 | noncausal | 3 | 2 | 4 (1/5), 6 (5/5), 8 (5/5), 12 (4/4) | 6 | 1,287 | 3.2 |
| 6 | causal | 1 | 1 | 12 (0/4), 24 (0/5), 32 (0/5) | none | | |
| 6 | causal | 1 | 2 | 12 (0/2), 24 (0/5), 32 (0/5) | none | | |
| 6 | causal | 2 | 1 | 6 (1/5), 8 (3/5), 12 (5/5), 16 (5/5) | 12 | 3,039 | 10.6 |
| 6 | causal | 2 | 2 | 6 (1/3), 8 (4/5), 12 (4/4) | 8 | 1,515 | 22.8 |
| 6 | causal | 3 | 1 | 4 (0/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 1,299 | 33.2 |
| 6 | causal | 3 | 2 | 4 (0/5), 6 (4/5), 8 (5/5), 12 (5/5) | 6 | 1,299 | 22.2 |
| 6 | unordered | 1 | 1 | 12 (0/2), 24 (0/2), 32 (0/5) | none | | |
| 6 | unordered | 1 | 2 | 12 (0/5), 24 (0/5), 32 (0/5) | none | | |
| 6 | unordered | 2 | 1 | 6 (0/4), 8 (5/5), 12 (5/5) | 8 | 1,379 | 26.2 |
| 6 | unordered | 2 | 2 | 6 (2/5), 8 (5/5), 12 (5/5) | 8 | 1,379 | 19.8 |
| 6 | unordered | 3 | 1 | 4 (0/5), 6 (5/5), 8 (5/5), 12 (4/4) | 6 | 1,197 | 19.2 |
| 6 | unordered | 3 | 2 | 4 (0/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 1,197 | 23.2 |
| 6 | noncausal | 1 | 1 | 12 (0/4), 24 (0/5), 32 (0/5) | none | | |
| 6 | noncausal | 1 | 2 | 12 (0/4), 24 (0/5), 32 (0/5) | none | | |
| 6 | noncausal | 2 | 1 | 6 (1/5), 8 (5/5), 12 (5/5) | 8 | 1,515 | 15.8 |
| 6 | noncausal | 2 | 2 | 6 (3/5), 8 (5/5), 12 (4/4) | 8 | 1,515 | 14.4 |
| 6 | noncausal | 3 | 1 | 4 (0/5), 6 (5/5), 8 (5/5), 12 (5/5) | 6 | 1,299 | 26.6 |
| 6 | noncausal | 3 | 2 | 6 (3/5), 8 (5/5), 12 (5/5) | 8 | 2,115 | 7.6 |
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from indrajala_ml.measurement import machine_lock

from indrajala_studies import addition_data as ad
from indrajala_studies import addition_study as st


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "report":
            command.add_argument("--md", type=Path, help="write the report to this file too")
    commands.add_parser("check").add_argument("model", type=Path)
    commands.add_parser("candidates")
    args = parser.parse_args(argv)

    if args.command == "candidates":
        for candidate in st.CANDIDATES.values():
            print(
                f"{candidate.name}: {candidate.description}; shapes {list(candidate.shapes)}, {candidate.rung} {candidate.rungs}"
            )
        return 0
    if args.command == "check":
        model, n = st.load_model(args.model)
        for result in ad.evaluate(model.add, n):
            print(
                f"{result.id}: {result.passed}/{result.cases}"
                + ("" if result.holds else f", e.g. {result.failures[0]}")
            )
        return 0
    sweep = st.load_config(args.config)
    if args.command == "run":
        with machine_lock.study():
            st.run(sweep, log=lambda line: print(line, flush=True))
        print(st.report(sweep))
    elif args.command == "status":
        print(st.status(sweep))
    else:
        text = st.report(sweep)
        print(text)
        if args.md is not None:
            args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
