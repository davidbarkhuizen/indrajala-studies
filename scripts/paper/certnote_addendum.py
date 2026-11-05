#!/usr/bin/env python3
"""
The certificate note's addendum to Zenodo (indrajala-research docs/workplans/certificate-note-workplan.md,
D15): what a reader needs to rerun the note's numbers, the checker and the bounds, with no other record.

Run from the checkout whose data/ holds the runs. Paths inside the archives are that checkout's, so the
archives unpack at the root of a clone of this repository, and the tables' recorded paths resolve.

- certnote-results.tar.gz: the result files certificate_note_tables.py reads, with jobs.jsonl and the manifest;
- certnote-tables.tar.gz: every emitted table (the witnesses);
- certnote-networks.tar.gz: every network the jobs name, and the stored certificate each reproduces;
- MANIFEST.sha256: every archive's hash, then every file's inside.

    certnote_addendum.py build OUT [--readme FILE]     the archives, the README and the manifest
    certnote_addendum.py check OUT                     every archive and every file inside, re-hashed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO

NOTE = Path("data/certificate-note")
RESULTS = (
    "inventory.csv",
    "manifest.sha256",
    "cost.json",
    "numbers.txt",
    "recertify/jobs.jsonl",
    "recertify/results.jsonl",
    "faults/faults.jsonl",
    "bound/bound.jsonl",
    "bound/near-ties.jsonl",
    "affine/affine.jsonl",
)
BLOCK = 1 << 20


def sha256(read: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for block in read:
        digest.update(block)
    return digest.hexdigest()


def blocks(path: Path) -> Iterable[bytes]:
    with path.open("rb") as f:
        yield from stream(f)


def stream(f: IO[bytes]) -> Iterable[bytes]:
    while block := f.read(BLOCK):
        yield block


def contents(root: Path) -> dict[str, list[Path]]:
    """Each archive's files, relative to the checkout; a missing file raises, as an incomplete record must not ship."""
    note = root / NOTE
    jobs = [json.loads(line) for line in (note / "recertify" / "jobs.jsonl").read_text().splitlines() if line.strip()]
    results = [NOTE / name for name in RESULTS]
    tables = [NOTE / "recertify" / "tables" / f"{job['id']}.npz" for job in jobs]
    networks = sorted(
        {Path(job["network"]) for job in jobs} | {Path(job["reproduce"]) for job in jobs if job["reproduce"]}
    )
    made = {"certnote-results.tar.gz": results, "certnote-tables.tar.gz": tables, "certnote-networks.tar.gz": networks}
    for paths in made.values():
        for path in paths:
            if not (root / path).is_file():
                raise FileNotFoundError(root / path)
    return made


def build(root: Path, out: Path, readme: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    inside: list[str] = []
    for name, paths in contents(root).items():
        with tarfile.open(out / name, "w:gz", compresslevel=6) as archive:
            for path in paths:
                archive.add(root / path, arcname=str(path), recursive=False)
                inside.append(f"{sha256(blocks(root / path))}  {path}\n")
        lines.append(f"{sha256(blocks(out / name))}  {name}\n")
        print(f"{name}: {len(paths)} files, {(out / name).stat().st_size / 1e6:.0f} MB", flush=True)
    if readme is not None:
        shutil.copyfile(readme, out / "README.md")
        lines.append(f"{sha256(blocks(out / 'README.md'))}  README.md\n")
    (out / "MANIFEST.sha256").write_text("".join(lines + inside))


def check(out: Path) -> bool:
    """Each archive against the manifest, and each file inside against its line; every line accounted for."""
    pairs = (line.split(maxsplit=1) for line in (out / "MANIFEST.sha256").read_text().splitlines())
    listed = {name: digest for digest, name in pairs}
    ok = True
    seen: set[str] = set()
    for name in [n for n in listed if "/" not in n]:
        seen.add(name)
        if sha256(blocks(out / name)) != listed[name]:
            print(f"{name}: differs")
            ok = False
        if not name.endswith(".tar.gz"):
            continue
        with tarfile.open(out / name) as archive:
            for member in archive:
                f = archive.extractfile(member)
                if f is None:
                    continue
                seen.add(member.name)
                if sha256(stream(f)) != listed.get(member.name):
                    print(f"{member.name}: differs from the manifest")
                    ok = False
    for name in sorted(listed.keys() - seen):
        print(f"{name}: in the manifest, not in any archive")
        ok = False
    print(f"{len(seen)} files: " + ("every hash matches" if ok else "DIFFERENCES"))
    return ok


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path())
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("build")
    made.add_argument("out", type=Path)
    made.add_argument("--readme", type=Path)
    checked = commands.add_parser("check")
    checked.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.root, args.out, args.readme)
        return 0
    return 0 if check(args.out) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
