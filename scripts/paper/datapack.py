#!/usr/bin/env python3
"""
Paper 1's datapack (the paper workplan's D37): one record holding only what paper 1 cites, in two
packs. The full pack is every file paper 1's manifests list (MANIFESTS), one archive per part; the
verify pack is what `paper_tables.py check` reads to regenerate the manuscript's numbers and tables,
plus every long-tested (certified) network, so anyone can check every number and rerun every
certificate without the full download.

`datapack.py reads TEX` lists, relative to `--data`, the files the verify pack holds: those
`paper_tables.py check TEX` opens (recorded by an audit hook) and the networks the long tests name
by hash. It fails if any of them is attested by no manifest in MANIFESTS.

`datapack.py build TEX --out DIR` writes into DIR: `verify.tar.gz` and `verify.sha256`, the verify
pack and its manifest, made of the cited manifests' own lines (paths relative to `--data`, hashes
unchanged); `full-<part>.tar.gz` for each part, holding its manifests and every file they list, at
the paths they list, symlinks followed (`--parts` builds some parts only); `README.md`, the record's
how-to-verify page (`--readme`, by default indrajala-research's docs/datapack/README.md in the
sibling clone); and `MANIFEST.sha256`, the hash of every archive
written, of the README and of every manifest in MANIFESTS. Each manifest verifies with
`sha256sum -c` from its base, the directory its paths are relative to (`manifest_base`).

`datapack.py check DIR` re-hashes every archive DIR's `MANIFEST.sha256` lists and every file inside
each, against the manifests the archive carries (the verify pack against `verify.sha256`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables

PAPER = [
    *(f"paper/{test}.sha256" for test in ["t1", "t2-t5", "t3", "t4", "t4-fresh", "t4prime"]),
    *(f"paper/{test}.sha256" for test in ["t6", "t6-proof", "t6-strict", "t7-reruns", "t7-reruns-jebel"]),
    *(f"paper/{test}.sha256" for test in ["t10", "t10r", "t11", "t11-runs", "t12", "t12-runs", "p2-check"]),
]
MANIFESTS: dict[str, list[str]] = {  # each part of the full pack and its manifests, relative to --data
    **{Path(m).stem: [m] for m in PAPER},
    "approximation-claim": ["approximation-claim.sha256", "approximation-claim-long-tests.sha256"],
    "addition": [
        "matrix/results.sha256",
        "control/results.sha256",
        "control/models.sha256",
        "certificates.sha256",
        "certificates-strict.sha256",
        "long-tests.sha256",
        "e3-e4.sha256",
        "e3-e4/certificates.sha256",
        "e3-e4/certificates-strict.sha256",
        "e3-e4/long-tests.sha256",
        "stable-replay.sha256",
        "replay-guard.sha256",
        "machine-diff.sha256",
        "e2.sha256",
        "e5.sha256",
    ],
}


def sha256(path: Path) -> str:
    """A file's SHA-256, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entries(manifest: Path) -> list[tuple[str, str]]:
    """A manifest's lines as (hash, path as written)."""
    lines = [line.rstrip("\n") for line in manifest.read_text().splitlines() if line.strip()]
    return [(digest, path) for digest, path in (line.split("  ", 1) for line in lines)]


# the record's how-to-verify page, kept with the paper in the sibling indrajala-research clone
README = Path(__file__).resolve().parents[3] / "indrajala-research" / "docs" / "datapack" / "README.md"
BASES = {"paper/t4-fresh.sha256": "paper/t2-t5/approximations"}  # run from where its files were read


def manifest_base(data: Path, manifest: str) -> Path:
    """The directory a manifest's paths are relative to, relative to `data`: BASES's, else the first of
    its own directory, its stem's directory and the data root where every listed file exists."""
    own = Path(manifest).parent
    for base in [Path(BASES[manifest])] if manifest in BASES else [own, own / Path(manifest).stem, Path(".")]:
        if all((data / base / path).is_file() for _, path in entries(data / manifest)):
            return base
    raise ValueError(f"{manifest}: no base holds every file it lists")


def attested(data: Path) -> dict[str, tuple[str, str]]:
    """Every file MANIFESTS list, by its path relative to `data`: its hash and its manifest."""
    found: dict[str, tuple[str, str]] = {}
    for manifests in MANIFESTS.values():
        for manifest in manifests:
            base = manifest_base(data, manifest)
            for digest, path in entries(data / manifest):
                found[os.path.normpath(base / path)] = (digest, manifest)
    return found


def check_reads(data: Path, tex: Path) -> set[str]:
    """The files under `data` that `paper_tables.py check TEX` opens, relative to `data`."""
    root = data.resolve()
    opened: set[str] = set()

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event == "open" and isinstance(args[0], (str, os.PathLike)):
            path = Path(os.path.abspath(os.fspath(args[0])))  # abspath, not resolve: keep symlinked paths
            if path.is_relative_to(root) and path.is_file():
                opened.add(str(path.relative_to(root)))

    sys.addaudithook(hook)  # an audit hook can't be removed; `reads` runs once per process
    if paper_tables.main(["--data", str(data), "check", str(tex)]) != 0:
        raise SystemExit("paper_tables.py check differs: the verify pack would not reproduce the manuscript")
    return opened


def certified_networks(data: Path, listed: dict[str, tuple[str, str]]) -> set[str]:
    """The networks the long tests name by hash, at a path a manifest lists."""
    wanted = {
        json.loads(path.read_text())["sha256"]
        for part in ("paper", "approximation-claim")
        for path in (data / part).rglob("*.long-test.json")
    }
    by_hash = {digest: path for path, (digest, _) in sorted(listed.items())}
    missing = wanted - by_hash.keys()
    if missing:
        raise SystemExit(f"{len(missing)} long-tested networks are in no manifest")
    return {by_hash[digest] for digest in wanted}


def verify_files(data: Path, tex: Path) -> dict[str, str]:
    """The verify pack's files, at the paths the check reads them by, and each one's hash, attested by
    a manifest in MANIFESTS (at that path, or the same content at another: a copy, or a symlink)."""
    listed = attested(data)
    cited = {digest for digest, _ in listed.values()}
    files: dict[str, str] = {}
    unattested: list[str] = []
    for path in sorted(check_reads(data, tex)) + sorted(certified_networks(data, listed)):
        digest = listed[path][0] if path in listed else sha256(data / path)
        if digest in cited:
            files[path] = digest
        else:
            unattested.append(path)
    if unattested:
        print("\n".join(unattested), file=sys.stderr)
        raise SystemExit(f"{len(unattested)} files read by the check are attested by no manifest in MANIFESTS")
    return files


def add(archive: tarfile.TarFile, data: Path, paths: Iterable[str]) -> None:
    """Each path, relative to `data`, into the archive at that path, its symlinks followed."""
    for path in paths:
        info = archive.gettarinfo(str(data / path), arcname=path)
        if info.issym() or info.islnk():
            info = archive.gettarinfo(str((data / path).resolve()), arcname=path)
        with (data / path).open("rb") as f:
            archive.addfile(info, f)


def build(data: Path, tex: Path, out: Path, parts: list[str], readme: Path = README) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = verify_files(data, tex)
    (out / "verify.sha256").write_text("".join(f"{digest}  {path}\n" for path, digest in files.items()))
    with tarfile.open(out / "verify.tar.gz", "w:gz") as archive:
        add(archive, data, files)
    print(f"verify: {len(files)} files")
    for part in parts:
        with tarfile.open(out / f"full-{part}.tar.gz", "w:gz") as archive:
            for manifest in MANIFESTS[part]:
                base = manifest_base(data, manifest)
                add(archive, data, [manifest])
                add(archive, data, sorted({os.path.normpath(base / path) for _, path in entries(data / manifest)}))
        print(f"full-{part}: {sum(len(entries(data / m)) for m in MANIFESTS[part])} files")
    shutil.copyfile(readme, out / "README.md")
    split(out)
    write_manifest(data, out)


SPLIT = 10**9  # bytes: an archive above this is uploaded in parts, each on its own connection (Zenodo's are slow)


def split(out: Path) -> None:
    """Each archive above SPLIT into `<name>.part-00`, `-01`, ..., the whole moved to `out/whole/`."""
    (out / "whole").mkdir(exist_ok=True)
    for path in sorted(out.glob("*.tar.gz")):
        if path.stat().st_size <= SPLIT:
            continue
        count = 0
        with path.open("rb") as f:
            for block in iter(lambda: f.read(SPLIT), b""):
                (out / f"{path.name}.part-{count:02d}").write_bytes(block)
                count += 1
        path.rename(out / "whole" / path.name)
        print(f"{path.name}: {count} parts")


def archive_sources(out: Path, name: str) -> list[Path]:
    """An archive's bytes: the whole file, or its parts in order."""
    whole = [p for p in (out / name, out / "whole" / name) if p.exists()]
    return whole[:1] or sorted(out.glob(f"{name}.part-*"))


def write_manifest(data: Path, out: Path) -> None:
    """MANIFEST.sha256: every archive whole (rejoined from its parts if split), every part, the verify
    manifest, the README and every manifest in MANIFESTS."""
    names = sorted({p.name.split(".part-")[0] for p in out.glob("*.tar.gz*")} | {p.name for p in out.glob("whole/*")})
    lines = [f"{sha256_of(archive_sources(out, name))}  {name}" for name in names]
    lines += [f"{sha256(p)}  {p.name}" for p in sorted(out.glob("*.tar.gz.part-*"))]
    lines += [f"{sha256(out / n)}  {n}" for n in ("verify.sha256", "README.md")]
    lines += [f"{sha256(data / m)}  manifests/{m}" for ms in MANIFESTS.values() for m in ms]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def sha256_of(paths: list[Path]) -> str:
    """The SHA-256 of files' bytes one after another."""
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


class Joined(io.RawIOBase):
    """Files read as one stream, in order: an archive's parts rejoined without writing it out."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths, self.f = list(paths), None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while True:
            if self.f is None:
                if not self.paths:
                    return 0
                self.f = self.paths.pop(0).open("rb")
            n = self.f.readinto(buffer)
            if n:
                return n
            self.f.close()
            self.f = None


def check(out: Path) -> bool:
    """Every archive's hash, and every file inside it against the manifests it carries, each archive
    read once as a stream (a gzip stream can't seek back cheaply, so members are hashed as they pass)."""
    good = True
    for digest, name in entries(out / "MANIFEST.sha256"):
        if name.startswith("manifests/"):
            continue
        sources = archive_sources(out, name) if name.endswith(".tar.gz") else [out / name]
        if not sources or not all(p.exists() for p in sources) or sha256_of(sources) != digest:
            print(f"{name}: missing or its hash differs")
            good = False
            continue
        if not name.endswith(".tar.gz"):
            continue
        inside: dict[str, str] = {}  # member -> sha256
        carried: dict[str, bytes] = {}  # the archive's manifests, by their path
        part = name.removeprefix("full-").removesuffix(".tar.gz")
        manifests = [] if name == "verify.tar.gz" else MANIFESTS[part]
        with tarfile.open(fileobj=io.BufferedReader(Joined(sources), 1 << 20), mode="r|gz") as archive:
            for member in archive:
                f = archive.extractfile(member) if member.isfile() else None
                if f is None:
                    continue
                content = f.read()
                inside[member.name] = hashlib.sha256(content).hexdigest()
                if member.name in manifests:
                    carried[member.name] = content
        if name == "verify.tar.gz":
            expected = [(Path("."), entries(out / "verify.sha256"))]
        else:
            expected = [(_base_inside(inside, m, _lines(carried[m])), _lines(carried[m])) for m in manifests]
        for base, lines in expected:
            for digest_expected, path in lines:
                if inside.get(os.path.normpath(base / path)) != digest_expected:
                    print(f"{name}: {path} missing or differs")
                    good = False
        print(f"{name}: checked, {len(inside)} files")
    return good


def _lines(text: bytes) -> list[tuple[str, str]]:
    lines = [line for line in text.decode().splitlines() if line.strip()]
    return [(digest, path) for digest, path in (line.split("  ", 1) for line in lines)]


def _base_inside(inside: dict[str, str], manifest: str, lines: list[tuple[str, str]]) -> Path:
    """Where a carried manifest's paths sit inside its archive: BASES's, else the first of its own
    directory, its stem's and the root that holds every listed file."""
    own = Path(manifest).parent
    for base in [Path(BASES[manifest])] if manifest in BASES else [own, own / Path(manifest).stem, Path(".")]:
        if all(os.path.normpath(base / path) in inside for _, path in lines):
            return base
    return own


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/window-recurrent"))
    commands = parser.add_subparsers(dest="command", required=True)
    reads = commands.add_parser("reads", help="the verify pack's files, each attested")
    reads.add_argument("tex", type=Path)
    make = commands.add_parser("build", help="the verify pack, the full pack's parts and MANIFEST.sha256")
    make.add_argument("tex", type=Path)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--parts", nargs="*", choices=sorted(MANIFESTS), default=sorted(MANIFESTS))
    make.add_argument("--readme", type=Path, default=README, help="the record's how-to-verify page")
    parted = commands.add_parser("split", help="archives above 1 GB into parts, and MANIFEST.sha256 rewritten")
    parted.add_argument("out", type=Path)
    checked = commands.add_parser("check", help="every archive and every file inside, re-hashed")
    checked.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "reads":
        files = verify_files(args.data, args.tex)
        print("\n".join(files))
        size = sum((args.data / f).stat().st_size for f in files)
        print(f"{len(files)} files, {size / 1e9:.2f} GB, every one attested", file=sys.stderr)
        return 0
    if args.command == "build":
        build(args.data, args.tex, args.out, args.parts, args.readme)
        return 0
    if args.command == "split":
        split(args.out)
        write_manifest(args.data, args.out)
        return 0
    return 0 if check(args.out) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
