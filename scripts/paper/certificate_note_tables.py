#!/usr/bin/env python3
"""
The certificate note's tables and numbers regenerated from its result files (indrajala-research
docs/workplans/certificate-note-workplan.md, D11): no number in the note is typed by hand.

The data directory (`--data`, default data/certificate-note) holds stage 0's inventory.csv and the
stages' JSON lines: recertify/results.jsonl (stage 1), faults/faults.jsonl (stage 2),
bound/bound.jsonl (stage 3) and affine/affine.jsonl (D13), with manifest.sha256 (sha256sum's format, paths relative to the
directory) listing them; every command first checks the files against the manifest.

    certificate_note_tables.py show NAME...     a table's LaTeX rows
    certificate_note_tables.py numbers          every \\pn key and its value
    certificate_note_tables.py check TEX...     every table (by its \\label) and \\pn{key}{value} against its regeneration
    certificate_note_tables.py fill TEX...      each \\pn's value replaced by its regeneration
    certificate_note_tables.py manifest         write manifest.sha256 over the data files

check exits 1 on any difference, a key with no source, or a file that doesn't match the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "certnote"))
import certificate_cost as ccost  # scripts/ isn't a package

FILES = (
    "inventory.csv",
    "recertify/results.jsonl",
    "faults/faults.jsonl",
    "bound/bound.jsonl",
    "affine/affine.jsonl",
)
LINEAR = "curriculum-linear"  # the linear-window networks' files (D9)
PN = re.compile(r"\\pn\{([^{}]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}")
Numbers = dict[str, str]


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def results(data: Path) -> list[dict[str, Any]]:
    return ccost.rows([data / "recertify" / "results.jsonl"])


def by_family(rows: list[dict[str, Any]], task: Callable[[dict[str, Any]], str]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[ccost.family(task(r))].append(r)
    return dict(sorted(groups.items()))


def grouped(n: int) -> str:
    """An integer as the note writes it: thousands with {,} (LaTeX's thin comma)."""
    return f"{n:,}".replace(",", "{,}")


def span(lo: object, hi: object, fmt: str = "{}") -> str:
    a, b = fmt.format(lo), fmt.format(hi)
    return a if a == b else f"{a}--{b}"


def name(family: str) -> str:
    """A family as the note's tables name it."""
    names = {"addition": "addition", "random-planted-q16": "planted ($q$ 16)"}
    if family in names:
        return names[family]
    if family.startswith("modular-division-"):
        return f"division, $p$ {family.rsplit('-', 1)[1]}"
    if family.startswith("modular-sum-"):
        return f"sum mod {family.rsplit('-', 1)[1]}"
    if family.startswith("move-to-front-"):
        return f"move-to-front {family.rsplit('-', 1)[1]}"
    if family.startswith("word-problem-s"):
        return f"$S_{family[-1]}$"
    if family.startswith("random-q"):
        return f"random ($q$ {family.split('q')[1]})"
    return family


# the tables, each a function of the data directory, named by its LaTeX label


def recertify_table(data: Path) -> list[str]:
    """Per family and window: networks, reachable states, entries, seconds, bytes and Chow's words."""
    rows: list[str] = []
    for t in ccost.readings(results(data)):
        words = "--" if t["chow_words"] is None else " / ".join(grouped(w) for w in t["chow_words"])
        rows.append(
            f"{name(t['family'])} & {t['columns']} & {t['jobs']} & {span(*t['reachable'])} & "
            f"{span(grouped(t['entries'][0]), grouped(t['entries'][1]))} & {t['certify_seconds'][1]:.1f} & "
            f"{t['check_seconds'][1]:.1f} & {t['bytes_per_entry']:.0f} & {words} \\\\"
        )
    return rows


def decided(r: dict[str, Any]) -> bool:
    """A bound's table whose every label is sound and every snap sound or among the table's branches."""
    return r["labels_sound"] == r["entries"] and r["snaps_sound"] + r["snaps_covered"] == r["entries"]


def faults_table(data: Path) -> list[str]:
    """Per family: faults planted and caught, counterexample lengths, and the catalogue's worst miss by rank."""
    rows: list[str] = []
    for family, rs in by_family(lines(data / "faults" / "faults.jsonl"), lambda r: r["task"]).items():
        caught = sum(r["caught"] for r in rs)
        windows = [r["counterexample_windows"] for r in rs]
        worst: dict[str, float] = {}
        for rank in ("rarest", "median", "commonest"):
            misses = [r["catalogue_miss"] for r in rs if r["rank"] == rank]
            worst[rank] = max(misses)
        cells = " & ".join(miss_cell(worst[rank]) for rank in ("rarest", "median", "commonest"))
        rows.append(f"{name(family)} & {len(rs)} & {caught} & {span(min(windows), max(windows))} & {cells} \\\\")
    return rows


def miss_cell(p: float) -> str:
    """A miss chance: as a decimal from 0.01, as a power of ten below, $0$ when it underflows."""
    if p >= 0.01:
        return f"{p:.2f}"
    if p <= 0.0:
        return "$0$"
    return f"$10^{{{math.floor(math.log10(p))}}}$"


def bound_table(data: Path) -> list[str]:
    """Per family: the interval bound's tables, whole tables and shares; then the affine bound's on the rest."""
    rows: list[str] = []
    affine = by_family(lines(data / "affine" / "affine.jsonl"), lambda r: r["task"])
    for family, rs in by_family(
        [r for r in lines(data / "bound" / "bound.jsonl") if "unsupported" not in r], lambda r: r["task"]
    ).items():
        entries = sum(r["entries"] for r in rs)
        labels = sum(r["labels_sound"] for r in rs) / entries
        snaps = sum(r["snaps_sound"] for r in rs) / entries
        covered = sum(r["snaps_covered"] for r in rs) / entries
        full = sum(decided(r) for r in rs)
        rest = affine.get(family, [])
        outside = sum(r["outside"] for r in [*rs, *rest])
        rows.append(
            f"{name(family)} & {len(rs)} & {full} & {100 * labels:.1f} & {100 * snaps:.1f} & {100 * covered:.1f} & "
            f"{len(rest)} & {sum(decided(r) for r in rest)} & {outside} \\\\"
        )
    return rows


TABLES: dict[str, Callable[[Path], list[str]]] = {
    "tab:recertify": recertify_table,
    "tab:faults": faults_table,
    "tab:bound": bound_table,
}


# the numbers


def inventory_numbers(data: Path) -> Numbers:
    rows = list(csv.DictReader((data / "inventory.csv").open()))
    networks = {r["sha256"] for r in rows if r["sha256"]}
    return {"inventory-files": grouped(len(rows)), "inventory-networks": grouped(len(networks))}


def recertify_numbers(data: Path) -> Numbers:
    rs = results(data)
    certified = [r for r in rs if r["certified"]]
    reproduced = sum(bool(r["reproduced"]) for r in rs)
    largest = max(rs, key=lambda r: r["check"]["entries"])
    return {
        "recert-jobs": grouped(len(rs)),
        "recert-networks": grouped(len({r["job"]["sha256"] for r in rs})),
        "recert-certified": grouped(len(certified)),
        "recert-certified-networks": grouped(len({r["job"]["sha256"] for r in certified})),
        "recert-valid": grouped(sum(r["check"]["valid"] for r in certified)),
        "recert-uncertified": grouped(len(rs) - len(certified)),
        "recert-uncertified-invalid": grouped(sum(not r["check"]["valid"] for r in rs if not r["certified"])),
        "recert-agree": grouped(sum(r["check"]["agrees"] for r in rs)),
        "recert-reproduced": grouped(reproduced),
        "recert-ambiguous": grouped(sum(r.get("encodings_reproducing", 1) > 1 for r in rs)),
        "recert-moved": grouped(sum(r["check"]["differences"].get("snaps_moved", 0) for r in rs)),
        "recert-not-bitwise": grouped(sum(r["check"]["differences"].get("leads_not_bitwise", 0) for r in rs)),
        "recert-largest-entries": grouped(largest["check"]["entries"]),
        "recert-largest-codes": grouped(largest["reachable"]),
        "recert-largest-check-minutes": f"{(largest['check']['fidelity_seconds'] + largest['check']['structure_seconds']) / 60:.0f}",
        "recert-largest-megabytes": f"{largest['check']['bytes'] / 1e6:.0f}",
        "recert-entries": grouped(sum(r["check"]["entries"] for r in rs)),
    }


def fault_numbers(data: Path) -> Numbers:
    rs = lines(data / "faults" / "faults.jsonl")
    division = [r for r in rs if r["task"].startswith("modular-division-") and r["rank"] == "rarest"]
    likely = sum(r["catalogue_miss"] > 0.5 for r in rs)
    wide = [r for r in rs if r["columns"] > 1]
    return {
        "faults": grouped(len(rs)),
        "fault-tables": grouped(len({r["table"] for r in rs})),
        "faults-caught": grouped(sum(r["caught"] for r in rs)),
        "fault-longest": str(max(r["counterexample_windows"] for r in rs)),
        "faults-likely-missed": grouped(likely),
        "faults-division-rarest": grouped(len(division)),
        "faults-division-rarest-missed": grouped(sum(r["catalogue_miss"] > 0.5 for r in division)),
        "fault-worst-miss": f"{max(r['catalogue_miss'] for r in rs):.5f}",
        "faults-wide-missed": grouped(sum(r["catalogue_miss"] > 0.5 for r in wide)),
        "faults-wide-median-missed": grouped(sum(r["catalogue_miss"] > 0.5 and r["rank"] == "median" for r in wide)),
        "fault-wide-worst-common": miss_cell(
            max((r["catalogue_miss"] for r in wide if r["rank"] == "commonest"), default=0.0)
        ),
    }


def bound_numbers(data: Path) -> Numbers:
    rs = lines(data / "bound" / "bound.jsonl")
    bounded = [r for r in rs if "unsupported" not in r]
    entries = sum(r["entries"] for r in bounded)
    networks = {r["job"]["id"].rsplit("-", 1)[0]: r for r in results(data) if r["certified"]}
    reached = {Path(r["table"]).stem.rsplit("-", 1)[0] for r in rs}
    left = [r for key, r in networks.items() if key not in reached]
    return {
        "bound-networks": grouped(len(rs)),
        "bound-left": grouped(len(left)),
        "bound-left-long": grouped(sum(r["job"]["columns"] == 4 for r in left)),
        "bound-long": grouped(sum(r["job"]["columns"] == 4 for key, r in networks.items() if key in reached)),
        "bound-tables": grouped(len(bounded)),
        "bound-unsupported": grouped(len(rs) - len(bounded)),
        "bound-entries": grouped(entries),
        "bound-outside": grouped(sum(r["outside"] for r in bounded)),
        "bound-contradicted": grouped(sum(r["labels_contradicted"] for r in bounded)),
        "bound-labels-percent": f"{100 * sum(r['labels_sound'] for r in bounded) / entries:.1f}",
        "bound-snaps-percent": f"{100 * sum(r['snaps_sound'] + r['snaps_covered'] for r in bounded) / entries:.1f}",
        "bound-full": grouped(sum(decided(r) for r in bounded)),
        "bound-ulps": str(bounded[0]["ulps"]) if bounded else "--",
    }


def affine_numbers(data: Path) -> Numbers:
    """The affine bound, and how it meets the interval bound: it is run on the tables the interval leaves undecided."""
    rs = lines(data / "affine" / "affine.jsonl")
    bounded = [r for r in lines(data / "bound" / "bound.jsonl") if "unsupported" not in r]
    left = {r["table"] for r in bounded if not decided(r)}
    ran = {r["table"] for r in rs}
    return {
        "affine-tables": grouped(len(rs)),
        "affine-entries": grouped(sum(r["entries"] for r in rs)),
        "affine-full": grouped(sum(decided(r) for r in rs)),
        "affine-labels-undecided": grouped(sum(r["labels_undecided"] for r in rs)),
        "affine-labels-contradicted": grouped(sum(r["labels_contradicted"] for r in rs)),
        "affine-snaps-covered": grouped(sum(r["snaps_covered"] for r in rs)),
        "affine-snaps-uncovered": grouped(sum(r["snaps_uncovered"] for r in rs)),
        "affine-outside": grouped(sum(r["outside"] for r in rs)),
        "affine-cap": grouped(max(r["cap"] for r in rs)),
        "affine-capped-tables": grouped(sum(r["condensed"] > 0 for r in rs)),
        "affine-longest-hours": f"{max(r['seconds'] for r in rs) / 3600:.0f}",
        "affine-mismatch": grouped(len(left ^ ran)),
        "bounds-decided": grouped(sum(decided(r) for r in bounded) + sum(decided(r) for r in rs)),
    }


def second_family_numbers(data: Path) -> Numbers:
    """The linear-window networks (D9), named by their files' curriculum-linear: certificates and the interval bound.

    Not by the job's e6 flag, which marks only the uncertified curriculum steps: the certified networks are
    byte-identical to files of another directory, and the recertification keeps each network once.
    """
    jobs = [r for r in results(data) if LINEAR in r["job"]["network"]]
    bounded = [r for r in lines(data / "bound" / "bound.jsonl") if "unsupported" not in r and LINEAR in r["network"]]
    return {
        "linear-jobs": grouped(len(jobs)),
        "linear-networks": grouped(len({r["job"]["sha256"] for r in jobs})),
        "linear-certified": grouped(sum(r["certified"] for r in jobs)),
        "linear-valid": grouped(sum(r["check"]["valid"] for r in jobs)),
        "linear-uncertified-invalid": grouped(sum(not r["check"]["valid"] for r in jobs if not r["certified"])),
        "linear-bound-tables": grouped(len(bounded)),
        "linear-bound-full": grouped(sum(decided(r) for r in bounded)),
    }


NUMBER_SOURCES: list[Callable[[Path], Numbers]] = [
    inventory_numbers,
    recertify_numbers,
    fault_numbers,
    bound_numbers,
    affine_numbers,
    second_family_numbers,
]


def numbers(data: Path) -> Numbers:
    made: Numbers = {}
    for source in NUMBER_SOURCES:
        for key, value in source(data).items():
            if key in made:
                raise ValueError(f"{key}: two sources")
            made[key] = value
    return made


def manuscript_rows(tex: str, label: str) -> list[str] | None:
    for table in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.DOTALL):
        if f"\\label{{{label}}}" in table:
            body = table.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
            return [line.strip() for line in body.splitlines() if line.strip()]
    return None


def manifest_problems(data: Path) -> list[str]:
    """The data files that don't match manifest.sha256 (or aren't in it)."""
    manifest = data / "manifest.sha256"
    if not manifest.exists():
        return ["no manifest.sha256"]
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            listed[path] = digest
    problems: list[str] = []
    for name in FILES:
        if name not in listed:
            problems.append(f"{name}: not in the manifest")
        elif hashlib.sha256((data / name).read_bytes()).hexdigest() != listed[name]:
            problems.append(f"{name}: its sha256 differs from the manifest's")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/certificate-note"))
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.add_argument("names", nargs="+", choices=sorted(TABLES))
    check = commands.add_parser("check")
    check.add_argument("manuscripts", type=Path, nargs="+")
    commands.add_parser("numbers")
    fill = commands.add_parser("fill")
    fill.add_argument("manuscripts", type=Path, nargs="+")
    commands.add_parser("manifest")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        text = "".join(f"{hashlib.sha256((args.data / f).read_bytes()).hexdigest()}  {f}\n" for f in FILES)
        (args.data / "manifest.sha256").write_text(text)
        print(text, end="")
        return 0
    problems = manifest_problems(args.data)
    if problems:
        print("\n".join(problems))
        return 1
    if args.command == "show":
        for table in args.names:
            print(f"% {table}")
            print("\n".join(TABLES[table](args.data)))
        return 0
    made = numbers(args.data)
    if args.command == "numbers":
        for key, value in made.items():
            print(f"{key} = {value}")
        return 0
    if args.command == "fill":
        for tex in args.manuscripts:
            text = tex.read_text()
            unknown = sorted({key for key, _ in PN.findall(text)} - made.keys())
            if unknown:
                print(f"{tex}: no source for {', '.join(unknown)}")
                return 1
            changed = sum(old != made[key] for key, old in PN.findall(text))
            tex.write_text(PN.sub(lambda m: f"\\pn{{{m[1]}}}{{{made[m[1]]}}}", text))
            print(f"{tex}: {changed} numbers changed")
        return 0
    differs = False
    for tex in args.manuscripts:
        text = tex.read_text()
        held = PN.findall(text)
        wrong = [(key, old) for key, old in held if made.get(key) != old]
        print(f"{tex}: numbers: {len(held)}, " + (f"{len(wrong)} DIFFER" if wrong else "the same"))
        for key, old in wrong:
            differs = True
            print(f"  {key}: manuscript {old}, regenerated {made.get(key, '(no source)')}")
        for label, table in TABLES.items():
            rows = manuscript_rows(text, label)
            if rows is None:
                continue
            regenerated = table(args.data)
            if rows == regenerated:
                print(f"{tex}: {label}: {len(rows)} rows, the same")
                continue
            differs = True
            print(f"{tex}: {label}: DIFFERS")
            for old, new in zip(rows, regenerated, strict=False):
                if old != new:
                    print(f"  manuscript: {old}\n  regenerated: {new}")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
