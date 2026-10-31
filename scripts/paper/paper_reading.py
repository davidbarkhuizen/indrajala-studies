"""
A runner batch's reading tools (the paper workplan's D15 and D16 rules, as T1's `read.sh` did by
hand for one task): every certified network of a batch's attempts, its task, arm and vocabulary from
its backdrop (`indrajala_studies.batch.BACKDROPS`, by the attempt's name `<backdrop>-<seed>`).

- `plan DATA`: each certified network and the arguments it is read with.
- `links DATA`: DATA/runs/<backdrop>/seed<s>, each a link to the run's log directory, as `readout`
  reads them.
- `certify DATA [--jobs J]`: each network recertified in both verdicts into DATA/certify/.
- `approximations DATA`: T4's reading (D10) of every certified network, a backdrop's at once, into
  DATA/approximations/<group>.txt.
- `proof DATA`: T6's reading (D11) of every coded run's log, the answers-only stop at the exact bound
  and 1.1 times it, then Chow's test set, a backdrop's at once, into DATA/proof/<group>/ and
  DATA/proof/<group>.txt.
- `long DATA --share I/N`: the long tests (U1, S1 at 128, 256, 1,024 columns) of the I-th of N
  shares (every N-th network from the I-th, in `plan`'s order) into DATA/long-tests/<network>/, one
  at a time; a network already long-tested is skipped, so a worker restarts where it stopped. A fixed
  share a worker needs no claims (T3's `mkdir` claims once let two workers take one network).
- `machines DATA --test T --arms A ... --pairs A:B ...`: a test whose unit is the machine (D19, D24:
  T10, T10r), its backdrops `<T>-<arm>-<machine>`, read from DATA/runs (after `links`) with the
  recertifications and long tests: each arm's machines and runs succeeding, and each pair's exact
  one-sided rank-sum test over the machines' counts, Holm-adjusted (mf.machine_unit_readout).
"""

import argparse
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from indrajala_studies import modular_family as mf
from indrajala_studies.batch import BACKDROPS, CELLS, Backdrop

# the batch runner's probe, which stays in ml (the studies-repo workplan)
PROBE = Path(__file__).resolve().parents[2] / "ml" / "scripts" / "studies" / "modular_family_probe.py"


@dataclass(frozen=True)
class Network:
    """A certified network of a batch: its attempt, backdrop, seed and file."""

    attempt: str
    backdrop: str
    seed: int
    path: Path

    @property
    def group(self) -> str:
        return group(self.attempt)

    @property
    def name(self) -> str:
        return f"{self.attempt}-{self.path.name.removesuffix('.network.json')}"


def split(attempt: str) -> tuple[str, int]:
    """
    An attempt's backdrop and seed: `d11-family-300` -> (`d11-family`, 300); a batch of several cells
    names its attempts `<cell>.<backdrop>-<seed>` (T3's `approximation.d5-family-wide-300`).
    """
    backdrop, _, seed = group(attempt).rpartition(".")[2], None, attempt.rpartition("-")[2]
    return backdrop, int(seed)


def group(attempt: str) -> str:
    """An attempt's cell (if named) and backdrop, its seed left off: the unit a reading is run over."""
    return attempt.rpartition("-")[0]


def coded(attempt: str) -> bool:
    """Whether the attempt's cell has codes (T3's continuous baseline has none, so no machine to read)."""
    cell = CELLS.get(group(attempt).partition(".")[0]) if "." in attempt else None
    return cell is None or "continuous" not in cell.args


def run_directory(data: Path, attempt: str) -> Path:
    """Where an attempt's log is: the attempt's directory for division, its seed<s>/ for a machine's."""
    backdrop, seed = split(attempt)
    return BACKDROPS[backdrop].log(data / "attempts" / attempt, seed).parent


def networks(data: Path) -> Iterator[Network]:
    for directory in sorted((data / "attempts").iterdir()):
        backdrop, seed = split(directory.name)
        for path in sorted(run_directory(data, directory.name).glob("certified-step-*.network.json")):
            yield Network(directory.name, backdrop, seed, path)


def task_args(backdrop: Backdrop) -> list[str]:
    """The reading's arguments for a backdrop's networks: its task, and division's arm and vocabulary."""
    if not backdrop.division:
        return ["--task", backdrop.task, *([] if backdrop.arm == "family" else ["--arm", backdrop.arm])]
    vocabulary = ["--vocabulary", *backdrop.vocabulary] if backdrop.vocabulary else []
    return ["--task", backdrop.task, "--arm", backdrop.arm, *vocabulary]


def share(items: list[Network], which: str) -> list[Network]:
    i, n = (int(part) for part in which.split("/"))
    assert 0 <= i < n, f"share {which}: I in 0..N-1"
    return items[i::n]


def say(message: str) -> None:
    print(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def probe(*args: str | Path, log: Path | None = None) -> int:
    command = [sys.executable, str(PROBE), *map(str, args)]
    if log is None:
        return subprocess.run(command, check=False).returncode
    with log.open("w") as out:
        return subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "links", "certify", "approximations", "proof", "long", "machines"):
        command = sub.add_parser(name)
        command.add_argument("data", type=Path)
        if name == "certify":
            command.add_argument("--jobs", type=int, default=1)
        if name == "long":
            command.add_argument("--share", default="0/1", metavar="I/N")
        if name == "machines":
            command.add_argument("--test", required=True, help="the backdrops' prefix: t10, t10r")
            command.add_argument("--arms", nargs="+", required=True)
            command.add_argument("--pairs", nargs="+", required=True, metavar="A:B", help="A tends larger than B")
    args = parser.parse_args()
    data: Path = args.data
    if args.command == "machines":
        runs = mf.machine_unit_runs(
            data / "runs",
            args.test,
            args.arms,
            mf.recertifications(data / "certify"),
            mf.long_tested(data / "long-tests"),
        )
        pairs = [tuple(pair.split(":")) for pair in args.pairs]
        for line in mf.machine_unit_readout(runs, [(a, b) for a, b in pairs]):
            print(line)
        return
    found = list(networks(data))
    if args.command == "plan":
        for network in found:
            print(network.name, network.path, *task_args(BACKDROPS[network.backdrop]))
    elif args.command == "links":
        for attempt in sorted(p.name for p in (data / "attempts").iterdir()):
            backdrop, seed = split(attempt)
            link = data / "runs" / backdrop / f"seed{seed}"
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(Path("../..") / run_directory(data, attempt).relative_to(data))
    elif args.command in ("approximations", "proof"):
        out = data / args.command
        out.mkdir(exist_ok=True)
        if args.command == "approximations":
            groups: dict[str, list[Path]] = {}
            for n in found:
                groups.setdefault(n.group, []).append(n.path)
            extra: list[str] = []
        else:
            groups = {}
            for directory in sorted((data / "attempts").iterdir()):
                if coded(directory.name):
                    groups.setdefault(group(directory.name), []).append(
                        run_directory(data, directory.name) / "log.jsonl"
                    )
            extra = ["--slack", "1", "1.1", "--limit", "20000000"]
        for name, paths in groups.items():
            report = out / f"{name}.txt"
            if report.exists():
                continue
            backdrop, _ = split(f"{name}-0")
            where = ["--out", str(out / name)] if args.command == "proof" else []
            say(f"{args.command} {name}: {len(paths)}")
            rc = probe(
                args.command if args.command == "approximations" else "answers-proof",
                *paths,
                *task_args(BACKDROPS[backdrop]),
                *extra,
                *where,
                log=report,
            )
            say(f"{args.command} {name} exit {rc}")
    elif args.command == "certify":
        (data / "certify").mkdir(exist_ok=True)
        commands = [
            [
                *map(str, (sys.executable, PROBE, "certify", n.path)),
                *task_args(BACKDROPS[n.backdrop]),
                *verdict,
                "--out",
                out,
            ]
            for n in found
            for verdict, out in (
                ([], str(data / "certify" / f"{n.name}.certificate.json")),
                (["--snap-ratio", "1.0"], str(data / "certify" / f"{n.name}.strict.certificate.json")),
            )
            if not Path(out).exists()
        ]
        running: list[subprocess.Popen[bytes]] = []
        for command in commands:
            while len(running) >= args.jobs:
                running.pop(0).wait()
            running.append(subprocess.Popen(command, stdout=subprocess.DEVNULL))
        for process in running:
            process.wait()
        say(f"recertified {len(found)} networks ({len(commands)} certificates written)")
    else:
        mine = share(found, args.share)
        say(f"share {args.share}: {len(mine)} of {len(found)} networks")
        for network in mine:
            out = data / "long-tests" / network.name
            if any(out.glob("*.long-test.json")):
                continue
            out.mkdir(parents=True, exist_ok=True)
            say(f"start {network.name}")
            rc = probe(
                "long-test",
                network.path,
                *task_args(BACKDROPS[network.backdrop]),
                "--out",
                out,
                log=out.with_suffix(".log"),
            )
            say(f"long-tested {network.name} exit {rc}")
        say(f"share {args.share} finished")


if __name__ == "__main__":
    main()
