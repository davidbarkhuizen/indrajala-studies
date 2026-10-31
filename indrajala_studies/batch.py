"""
The batch runner (its workplan's stage 3, retired: git show 53d7fef:docs/workplans/batch-runner-workplan.md): a controller that owns a batch's attempts
and its ledger, and a worker daemon on each machine that runs them, over newline-delimited JSON on a
TCP socket (D4). The standard library only, so the worker costs almost nothing at rest.

A batch is a directory holding `ledger.jsonl`: its first line the batch's registration (`batch`:
the mode, the attempts with their cells, tasks, seeds and cutoffs), then one line per event
(`assigned`, `started`, `ended`, `lost`, `done`). The controller rebuilds its state from it on start
(D5), so it resumes after its own restart. Each attempt names a registered cell (CELLS), never a
command line, so a worker runs nothing else; a batch may hold several cells, its attempts interleaved
onto one fleet (an older ledger names one cell, the batch's: read as each attempt's). An attempt
message carries its cell as it did before, so a worker at an older commit runs a newer controller's.

Messages (each a JSON object on one line, with `type` and the protocol's version `v`):
worker to controller `hello` (host, slots, the attempts it is running and the ended ones not yet
acknowledged), `ready` (free slots), `started` (an attempt's pid), `progress` (its last step), `ended`
(its summary); controller to worker `attempt` (an attempt to run), `drain` (no more work), `acked` (an
`ended` received: a report lost on a dropped connection is sent again on the next).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import hashlib
import itertools
import json
import os
import shlex
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VERSION = 1
CENSORED = ("spent", "deadline")  # modular_family.CENSORED: ends at a budget (the standard library only)
LEDGER = "ledger.jsonl"
ACKED = "acked"  # a worker's ROOT/acked/<attempt>: its `ended` acknowledged by the controller
POLL = 2.0  # seconds between a worker's looks at its attempts

# The continuous crystallisation's cell (D54's, with the margin stop under its deadline: D59, D60),
# as probe-d64.sh runs it but for its wall-clock budget (epochs bind: the owner, 2026-10-23); "-reheat"
# adds D63 (a).
CONTINUOUS_CELL = [
    "--factors",
    "2",
    "--codes-ratio",
    "32/11",
    "--reads",
    "output",
    "--window-width",
    "32",
    "--budget-epochs",
    "1600",
    "--anneal",
    "--anneal-tail",
    "40",
    "--anneal-never-commit",
    "--warm-up",
    "--melt-hot",
    "2",
    "--melt-kappa",
    "1e-2",
    "--melt-hold",
    "20",
    "--melt-halving",
    "30",
    "--melt-shape",
    "exponential",
    "--crucible",
    "--warm-up-screen",
    "1.01",
    "--continuous",
    "--workers",
    "1",
    "--margin-wait",
    "200",
]
# The paper's cell (paper-workplan.md, D3 and D12): the division and round-1 cell as W1 ran it at 169
# states (two factors 32 wide, ceil(32/11 sqrt(q)) codes, reading the output, the fork, the window
# 32 wide), stopped at its first certified step, 600 epochs (W1's and round 1's); no state bound.
PAPER_CELL = [
    "--factors",
    "2",
    "--codes-ratio",
    "32/11",
    "--factor-width",
    "32",
    "--reads",
    "output",
    "--fork",
    "--window-width",
    "32",
    "--budget-epochs",
    "600",
    "--workers",
    "1",
]

# The approximation claim's cell (`git show c201622:docs/approximation-claim-workplan.md`: two factors
# of 16, reading the output, the default window, 400 epochs, stop at certified, no fork): T1 repeats its
# sum arm beside the two controls, and T3's baselines compare with T1 and A1 (paper-workplan.md, D12).
APPROXIMATION_CELL = ["--factors", "2", "--reads", "output", "--budget-epochs", "400", "--workers", "1"]


@dataclass(frozen=True)
class Cell:
    """A registered method: its run arguments, and whether it stops on the answers under the backdrop's state bound."""

    args: tuple[str, ...]
    bounded: bool = False


CELLS: dict[str, Cell] = {
    "continuous": Cell(tuple(CONTINUOUS_CELL), bounded=True),
    "continuous-reheat": Cell((*CONTINUOUS_CELL, "--reheat"), bounded=True),
    "continuous-ireheat": Cell(
        (*CONTINUOUS_CELL, "--incremental-reheat"), bounded=True
    ),  # incremental-reheat-workplan.md
    "paper": Cell(tuple(PAPER_CELL)),
    "approximation": Cell(tuple(APPROXIMATION_CELL)),
    # T3's baselines: the approximation claim's cell with only the window (E6's) or the state (E2's) changed
    "approximation-linear": Cell((*APPROXIMATION_CELL, "--baseline", "linear")),
    "approximation-continuous": Cell((*APPROXIMATION_CELL, "--baseline", "continuous")),
}

DIVISION = "modular-division-"
T1_VOCABULARY = ("sum", "suffix", "token")  # T1's arms ask the same tokens (paper-workplan.md, T1)
WIDE = ("sum", "difference", "product")  # the approximation claim's vocabulary: T3's cover, as A1's
T12_VOCABULARY = ("sum", "difference", "state")  # T12's arms ask the same tokens: the cover, the state (D31)
# T10's draws (D19): P0 the first 15 random-q16 seeds (all primitive); P1 and P2 the first 15 planted
# q16-b4 seeds whose only nontrivial principal congruence is the planted one (1-3, 12, 14, 17 and 21
# gain others; 15 the existence probe's)
T10_P0 = tuple(range(15))
T10_PLANTED = (0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 18, 19, 20, 22)
T10_PROBE = 15
# T10r's draws (D24), the next seeds after T10's: P0 random-q16 15-39 (all primitive); P2 the first 25
# planted seeds from 23 whose only nontrivial principal congruence is the planted one (25, 26, 42, 45
# and 47 gain others)
T10R_P0 = tuple(range(15, 40))
T10R_PLANTED = (23, 24, *range(27, 42), 43, 44, 46, *range(48, 53))


@dataclass(frozen=True)
class Backdrop:
    """
    A task to run a cell on: a machine's (`run --runs TASK:SEED`, with `--arm` for a machine's queries,
    its log out/seed<s>/log.jsonl), or division at its p with an arm's queries and a vocabulary (`run
    --p P --seed S --arm A`, its log out/log.jsonl); `bound` the target's states, a bounded cell's state
    bound.
    """

    task: str
    bound: int
    arm: str = "family"
    vocabulary: tuple[str, ...] = ()

    @property
    def division(self) -> bool:
        return self.task.startswith(DIVISION)

    def args(self, seed: int) -> list[str]:
        if not self.division:
            arm = [] if self.arm == "family" else ["--arm", self.arm]  # a machine's queries (T10's P2: planted)
            return ["--runs", f"{self.task}:{seed}", *arm]
        vocabulary = ["--vocabulary", *self.vocabulary] if self.vocabulary else []
        return ["--p", self.task.removeprefix(DIVISION), "--seed", str(seed), "--arm", self.arm, *vocabulary]

    def log(self, directory: Path, seed: int) -> Path:
        return directory / "log.jsonl" if self.division else directory / f"seed{seed}" / "log.jsonl"


def division(p: int, arm: str = "family", vocabulary: tuple[str, ...] = ()) -> Backdrop:
    return Backdrop(f"{DIVISION}{p}", p * p, arm, vocabulary)


BACKDROPS: dict[str, Backdrop] = {  # a backdrop's short name, as attempts name it
    "s3": Backdrop("word-problem-s3", 6),
    "s4": Backdrop("word-problem-s4", 24),
    "mtf4": Backdrop("move-to-front-4", 24),
    "rq12": Backdrop("random-q12-0", 12),
    "rq16": Backdrop("random-q16-0", 16),
    # T1: division at p 5, the quotient and one more output, the three arms on one vocabulary
    **{f"d5-{arm}": division(5, arm, T1_VOCABULARY) for arm in ("sum", "suffix", "token")},
    # T2: the cover at p 7, 11 and 13, the quotient alone at 11
    **{f"d{p}-family": division(p) for p in (7, 11, 13)},
    "d11-quotient": division(11, "quotient"),
    # T3: the cover at p 5 on the approximation claim's vocabulary (A1's arm), under each window and state
    "d5-family-wide": division(5, "family", WIDE),
    # T12 (paper-workplan.md, D31): at p 5 the cover against the state as the one extra output, on one
    # vocabulary (so one output layer of p^2 classes in both arms)
    "d5-cover": division(5, "family", T12_VOCABULARY),
    "d5-state": division(5, "state", T12_VOCABULARY),
    # T10 (paper-workplan.md, D19, at q 16): P0 primitive random machines, P1 planted ones (the target
    # alone), P2 the same machines with the planted quotient asked (`--arm planted`); the probe's machine
    **{f"t10-p0-{m}": Backdrop(f"random-q16-{m}", 16) for m in T10_P0},
    **{f"t10-p1-{m}": Backdrop(f"random-planted-q16-b4-{m}", 16) for m in T10_PLANTED},
    **{f"t10-p2-{m}": Backdrop(f"random-planted-q16-b4-{m}", 16, "planted") for m in T10_PLANTED},
    "t10-probe-p1": Backdrop(f"random-planted-q16-b4-{T10_PROBE}", 16),
    "t10-probe-p2": Backdrop(f"random-planted-q16-b4-{T10_PROBE}", 16, "planted"),
    # T10r (D24): T10b on fresh machines, P0 against P2 alone
    **{f"t10r-p0-{m}": Backdrop(f"random-q16-{m}", 16) for m in T10R_P0},
    **{f"t10r-p2-{m}": Backdrop(f"random-planted-q16-b4-{m}", 16, "planted") for m in T10R_PLANTED},
    # T11 (paper-workplan.md, D20): T5's algebras with their coarsest cover asked as two outputs
    "t11-s3": Backdrop("word-problem-s3", 6, "cover"),
    "t11-s4": Backdrop("word-problem-s4", 24, "cover"),
    "t11-mtf4": Backdrop("move-to-front-4", 24, "cover"),
}


@dataclass(frozen=True)
class Window:
    """Hours a host may run: `days` (0 Monday .. 6 Sunday) on which a window opens at `start`, for `hours`."""

    days: tuple[int, ...]
    start: int  # the hour it opens
    hours: int  # how long it stays open (past midnight if start + hours > 24)


@dataclass(frozen=True)
class Host:
    """
    A machine in the fleet (D7): its run slots (its cores less the worker's one, bounded by memory), the
    worktree pinned at the batch's commit, its python and any PYTHONPATH before the worktree's, a
    shell line to run first (stone's env.sh), a wrapper for each attempt (klip's thermal guard, run from the worktree), its
    attempts' root, and its hours (none: always open). `remote` hosts are reached by ssh; every host's
    ended attempts are pulled into the batch, the controller's own machine's by a local copy.
    """

    name: str
    slots: int
    code: str
    python: str
    root: str
    pythonpath: str = ""
    setup: str = ""
    wrapper: tuple[str, ...] = ()
    windows: tuple[Window, ...] = ()
    remote: bool = True


WEEKDAYS, ALL_DAYS = (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5, 6)
# One crate build on every host (D6): the same .so, unpacked into a site directory named by its crate
# commit and put before the worktree on PYTHONPATH, so every host's crate hash is the same and a venv's
# own build (rebuilt in development) never reaches a batch. rust/ at 647f23c, built on pyramidon.
CRATE = "crate-647f23c"
# The owner's rules as data (2026-10-22; jebel on the grid around the clock since 2026-10-23): stone weekday nights and all weekend,
# the laptops always; slots are cores less the worker's, bounded by memory (pyramidon and klip 5 GB):
# klip all its cores (the owner: "klip always on, all cores", <= 85 C under the thermal guard; about
# 220 MB a paper-cell run, 7 runs about 1.5 GB), pyramidon 3 beside development.
# The worktree is wt-studies: each host's wt-batch is a worktree of its ml clone from before the separation.
HOSTS: dict[str, Host] = {
    "pyramidon": Host(
        "pyramidon",
        3,
        "/home/david/code/wt-studies",
        "/home/david/code/indrajala-ml/.venv/bin/python",
        "/home/david/code/batch-attempts",
        pythonpath=f"/home/david/code/{CRATE}",
        remote=False,
    ),
    "klip": Host(
        "klip",
        7,
        "/home/david/code/wt-studies",
        "/home/david/code/indrajala-ml/.venv/bin/python",
        "/home/david/code/batch-attempts",
        pythonpath=f"/home/david/code/{CRATE}",
        wrapper=("scripts/runner/thermal_guard.sh", "/home/david/code/batch-attempts/temp", "85", "--"),
    ),
    "jebel": Host(
        "jebel",
        7,
        "/home/david/code/wt-studies",
        "/home/david/code/indrajala-ml/.venv/bin/python",
        "/home/david/code/batch-attempts",
        pythonpath=f"/home/david/code/{CRATE}",
        # on grid power, around the clock, from 2026-10-23's evening (the owner); before, 07:00-18:00 on solar
    ),
    "stone": Host(
        "stone",
        11,
        "/home/david/labour/code/wt-studies",
        "/home/david/labour/code/wt-branches/.venv/bin/python",
        "/home/david/labour/batch-attempts",
        pythonpath=f"/home/david/labour/{CRATE}",
        setup="source ~/labour/env.sh",
        windows=(Window(WEEKDAYS, 18, 13), Window((5,), 0, 24), Window((6,), 0, 31)),
    ),
}


def opened(hosts: dict[str, Host], names: Sequence[str]) -> dict[str, Host]:
    """The hosts with `names` open at any hour (no windows): the owner's grant outside their hours."""
    return {n: dataclasses.replace(h, windows=()) if n in names else h for n, h in hosts.items()}


def tagged(hosts: dict[str, Host], tag: str) -> dict[str, Host]:
    """
    Each host's worktree and attempts' root set apart for one batch (`<path>-<tag>`, the wrapper's paths
    under the root with them), so a second batch's workers run beside another's at another commit.
    """

    def apart(h: Host) -> Host:
        root = f"{h.root}-{tag}"
        wrapper = tuple(root + w[len(h.root) :] if w.startswith(f"{h.root}/") else w for w in h.wrapper)
        return dataclasses.replace(h, code=f"{h.code}-{tag}", root=root, wrapper=wrapper)

    return {n: apart(h) for n, h in hosts.items()} if tag else hosts


def local_now() -> datetime.datetime:
    """The machine's local time, with its zone: hosts' hours are local wall-clock hours."""
    return datetime.datetime.now().astimezone()


def window_left(host: Host, now: datetime.datetime) -> float | None:
    """Seconds until the host's open window closes at `now`; None if no window is open; inf if always open."""
    if not host.windows:
        return float("inf")
    left: float | None = None
    for window in host.windows:
        for back in (0, 1, 2):  # a window opened today or up to two days before may still be open
            day = (now - datetime.timedelta(days=back)).replace(hour=0, minute=0, second=0, microsecond=0)
            if day.weekday() not in window.days:
                continue
            opens = day + datetime.timedelta(hours=window.start)
            closes = opens + datetime.timedelta(hours=window.hours)
            if opens <= now < closes:
                left = max(left or 0.0, (closes - now).total_seconds())
    return left


@dataclass(frozen=True)
class Attempt:
    """One attempt: a cell on a backdrop's seed, cut at `cutoff` epochs trained (None: run to its budget)."""

    id: str
    cell: str
    backdrop: str
    seed: int
    cutoff: int | None = None


def attempt_command(attempt: Attempt, python: str, script: str, out: Path) -> list[str]:
    """The command that runs an attempt: its backdrop's and cell's arguments, its bound and cutoff, into `out`."""
    backdrop, chosen = BACKDROPS[attempt.backdrop], CELLS[attempt.cell]
    cutoff = ["--cutoff-epochs", str(attempt.cutoff)] if attempt.cutoff is not None else []
    bound = ["--state-bound", str(backdrop.bound)] if chosen.bounded else []
    return [python, script, "run", *backdrop.args(attempt.seed), *chosen.args, *bound, *cutoff, "--out", str(out)]


@dataclass(frozen=True)
class Batch:
    """
    A batch's registration (D3, D8): `measure` runs every attempt; `find` runs each cell's backdrop's
    attempts in order until `k` of them succeed (the rest never start).
    """

    name: str
    mode: str
    attempts: tuple[Attempt, ...]
    k: int | None = None
    commit: str | None = None  # the code every worker must run (D6)
    crate: str | None = None  # the crate extension's SHA-256 every worker must import (D6)

    def __post_init__(self) -> None:
        assert self.mode in ("measure", "find"), f"a mode: measure or find; got {self.mode}"
        assert self.mode != "find" or (self.k is not None and self.k >= 1), "find needs k >= 1"
        assert all(a.cell in CELLS for a in self.attempts), f"registered cells: {sorted(CELLS)}"
        assert all(a.backdrop in BACKDROPS for a in self.attempts), f"backdrops: {sorted(BACKDROPS)}"
        assert len({a.id for a in self.attempts}) == len(self.attempts), "attempts' ids unique"

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a.cell for a in self.attempts))

    def to_json(self) -> dict[str, Any]:
        return {"type": "batch", "v": VERSION, **asdict(self)}

    @staticmethod
    def from_json(row: dict[str, Any]) -> Batch:
        cell = {"cell": row["cell"]} if "cell" in row else {}  # version 1: the batch's one cell
        attempts = tuple(Attempt(**{**cell, **a}) for a in row["attempts"])
        return Batch(row["name"], row["mode"], attempts, row.get("k"), row.get("commit"), row.get("crate"))


def attempts(spec: str, cells: Sequence[str], cutoffs: dict[str, int] | None = None) -> tuple[Attempt, ...]:
    """
    Attempts from `s3:40-47,mtf4:32-35` under each cell in turn (list the longest first: the fleet takes
    them in order), each backdrop cut at its `cutoffs` entry if given. With several cells an attempt's id
    starts with its cell's (`approximation-linear.s3-40`), so the cells' runs never share a directory.
    """
    made: list[Attempt] = []
    for cell in cells:
        for part in spec.split(","):
            backdrop, seeds = part.split(":")
            first, _, last = seeds.partition("-")
            for seed in range(int(first), int(last or first) + 1):
                name = f"{backdrop}-{seed}" if len(cells) == 1 else f"{cell}.{backdrop}-{seed}"
                made.append(Attempt(name, cell, backdrop, seed, (cutoffs or {}).get(backdrop)))
    return tuple(made)


def register(directory: Path, batch: Batch) -> None:
    """A new batch's directory and its ledger's first line (refused if it exists: never two registrations)."""
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / LEDGER
    assert not ledger.exists(), f"{ledger} exists: a batch is registered once"
    ledger.write_text(json.dumps(batch.to_json()) + "\n")


@dataclass
class State:
    """A batch's state as its ledger has it."""

    batch: Batch
    assigned: dict[str, str] = field(default_factory=dict[str, str])  # attempt -> host, not yet ended
    hosts_of: dict[str, str] = field(default_factory=dict[str, str])  # attempt -> the host that ran it
    ended: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])  # attempt -> summary
    done: bool = False

    @staticmethod
    def read(directory: Path) -> State:
        """The state from the ledger (D5): every event in order."""
        lines = (directory / LEDGER).read_text().splitlines()
        state = State(Batch.from_json(json.loads(lines[0])))
        for line in lines[1:]:
            state.apply(json.loads(line))
        return state

    def apply(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "assigned":
            self.assigned[event["id"]] = self.hosts_of[event["id"]] = event["host"]
        elif kind == "ended":
            self.assigned.pop(event["id"], None)
            self.ended[event["id"]] = event["summary"]
        elif kind == "lost":
            self.assigned.pop(event["id"], None)
        elif kind == "done":
            self.done = True

    def succeeded(self, cell: str, backdrop: str) -> int:
        return sum(
            1
            for a in self.batch.attempts
            if (a.cell, a.backdrop) == (cell, backdrop) and self.ended.get(a.id, {}).get("success") is not None
        )

    def next_attempt(self) -> Attempt | None:
        """The first attempt neither assigned nor ended, its cell's backdrop not yet found (find mode)."""
        for a in self.batch.attempts:
            if a.id in self.assigned or a.id in self.ended:
                continue
            if self.batch.mode == "find" and self.succeeded(a.cell, a.backdrop) >= (self.batch.k or 0):
                continue
            return a
        return None

    def finished(self) -> bool:
        return not self.assigned and self.next_attempt() is None


def last_run[R: Mapping[str, Any]](rows: Sequence[R]) -> list[R]:
    """
    A log's last run: a step end that doesn't follow its predecessor's step starts a run again (a lost
    attempt re-run in its old directory, before the worker set one aside, appended to its log).
    """
    ends = [i for i, r in enumerate(rows) if "end" in r and "certified" in r]
    restarts = [i for i, j in itertools.pairwise(ends) if int(rows[j]["step"]) <= int(rows[i]["step"])]
    return list(rows[restarts[-1] + 1 :] if restarts else rows)


def summary(log: Path) -> dict[str, Any]:
    """
    An attempt's outcome from its run's log.jsonl: its last step, its end, epochs trained and seconds,
    and its success in epochs trained (D41's: the first step's end certified at both verdicts; None if
    never). A run whose rows don't log the strict verdict (the paper's cells: it is checked after, by
    recertifying the saved network) succeeds at its first certified step; a continuous state (T3) never
    does here (long tests judge it). Epochs as modular_family.restart_course counts them (the time
    axis): each step end's, less the one a budget's stop logs untrained. Only the log's last run counts
    (last_run). The worker adds `peak_pss_mb`.
    """
    rows = last_run([json.loads(line) for line in log.read_text().splitlines()] if log.exists() else [])
    trained: dict[int, int] = {}
    total = 0
    for i, r in enumerate(rows):
        if "end" in r:
            total += int(r.get("epochs", 0)) - (r["end"] in CENSORED)
            trained[i] = total
    steps = [i for i, r in enumerate(rows) if "end" in r and "certified" in r]  # a step's end, not a screen's
    if not steps:
        return {"success": None, "step": None, "epochs": None, "end": "error", "seconds": None}
    success = next(
        (
            trained[i]
            for i in steps
            if rows[i].get("certified")
            and ("certificates" not in rows[i] or rows[i]["certificates"].get("strict", {}).get("certified"))
        ),
        None,
    )
    last = rows[steps[-1]]
    return {
        "success": success,
        "step": int(last["step"]),
        "epochs": trained[steps[-1]],
        "end": last["end"],
        "seconds": float(last["seconds"]),
    }


def last_step(log: Path) -> int | None:
    """The run's last logged step, read from the log's end (cheap on a long log)."""
    if not log.exists():
        return None
    with log.open("rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(f.tell() - 65536, 0))
        tail = f.read().decode(errors="replace").splitlines()[1:]
    for line in reversed(tail):
        with contextlib.suppress(ValueError):
            return int(json.loads(line)["step"])
    return None


async def send(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write((json.dumps({**message, "v": VERSION}) + "\n").encode())
    await writer.drain()


class Controller:
    """
    A batch's controller (D4): the protocol's server on 127.0.0.1, its queue and its ledger. Each
    attempt is assigned once, on a worker's `ready`, never more than its free slots less the attempts
    already sent it and not yet started. A worker on another commit or crate than the batch's is refused
    (D6); each host's crate is recorded. With `hosts` (D7), a host is given work only inside its hours and only
    an attempt whose expected time (its cutoff times the host's seconds an epoch so far) ends inside them;
    at a window's end its attempts are stopped and queued again. A host's ended attempt is pulled into
    `attempts/` and checked against its manifest.
    """

    def __init__(
        self,
        directory: Path,
        hosts: dict[str, Host] | None = None,
        now: Callable[[], datetime.datetime] | None = None,
        pull: Callable[[Host, str, Path], bool] | None = None,
        check: float = 30.0,
    ) -> None:
        self.directory = directory
        self.state = State.read(directory)
        self.hosts = hosts or {}
        self.now, self.check = now or local_now, check
        self.pull = pull or pull_attempt
        self.writers: dict[str, asyncio.StreamWriter] = {}
        self.pending: dict[str, set[str]] = {}  # host -> attempts sent, not yet started
        self.pulling: set[asyncio.Task[None]] = set()
        self.finished = asyncio.Event()
        if self.state.finished():
            self.finished.set()

    def log(self, event: dict[str, Any]) -> None:
        with (self.directory / LEDGER).open("a") as f:
            f.write(json.dumps({**event, "t": round(time.time(), 3)}) + "\n")
        self.state.apply(event)

    def seconds_an_epoch(self, host: str) -> float | None:
        """The host's mean seconds an epoch over its ended attempts (None before any)."""
        rates = [
            s["seconds"] / s["epochs"]
            for i, s in self.state.ended.items()
            if s.get("epochs") and s.get("seconds") and self.state.hosts_of.get(i) == host
        ]
        return sum(rates) / len(rates) if rates else None

    def fits(self, host: str, attempt: Attempt) -> bool:
        """Whether the host is open now and the attempt's expected time ends before it closes (D7)."""
        if host not in self.hosts:
            return True
        left = window_left(self.hosts[host], self.now())
        if left is None:
            return False
        rate = self.seconds_an_epoch(host)
        return attempt.cutoff is None or rate is None or attempt.cutoff * rate < left

    async def assign(self, host: str, writer: asyncio.StreamWriter, free: int) -> None:
        free -= len(self.pending.setdefault(host, set()))
        while free > 0:
            attempt = self.state.next_attempt()
            if attempt is None or not self.fits(host, attempt):
                break
            self.log({"type": "assigned", "id": attempt.id, "host": host})
            self.pending[host].add(attempt.id)
            await send(writer, {"type": "attempt", **asdict(attempt)})
            free -= 1
        if self.state.next_attempt() is None:
            await send(writer, {"type": "drain"})

    async def ended(self, host: str, message: dict[str, Any]) -> None:
        attempt = message["id"]
        self.log({"type": "ended", "id": attempt, "host": host, "summary": message["summary"]})
        files: dict[str, str] = message.get("files", {})
        if host in self.hosts and files:
            destination = self.directory / "attempts" / attempt

            async def pulling() -> None:
                ok = await asyncio.to_thread(self.pull, self.hosts[host], attempt, destination)
                verified = ok and verify(destination, files)
                self.log({"type": "pulled", "id": attempt, "host": host, "ok": verified})
                self.pulling.discard(task)  # this pull done before the batch's end is judged
                self.settle()

            task = asyncio.create_task(pulling())
            self.pulling.add(task)
            task.add_done_callback(self.pulling.discard)

    def settle(self) -> None:
        if self.state.finished() and not self.pulling and not self.state.done:
            self.log({"type": "done"})
            self.finished.set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        host = "?"
        try:
            while line := await reader.readline():
                message = json.loads(line)
                if message.get("v") != VERSION:
                    break
                kind = message["type"]
                if kind == "hello":
                    host = message["host"]
                    batch = self.state.batch
                    if (batch.commit is not None and message.get("commit") != batch.commit) or (
                        batch.crate is not None and message.get("crate") != batch.crate
                    ):
                        self.log(
                            {
                                "type": "refused",
                                "host": host,
                                "commit": message.get("commit"),
                                "crate": message.get("crate"),
                            }
                        )
                        await send(writer, {"type": "refused", "commit": batch.commit, "crate": batch.crate})
                        break
                    self.log(
                        {"type": "hello", "host": host, "commit": message.get("commit"), "crate": message.get("crate")}
                    )
                    self.writers[host] = writer
                    running = set(message.get("running", [])) | set(message.get("ended", []))
                    self.pending[host] = set()
                    for attempt, owner in list(self.state.assigned.items()):  # a host's attempts it no longer has
                        if owner == host and attempt not in running:
                            self.log({"type": "lost", "id": attempt, "host": host})
                elif kind == "ready":
                    await self.assign(host, writer, int(message["free"]))
                elif kind == "started":
                    self.pending.get(host, set()).discard(message["id"])
                    self.log({"type": "started", "id": message["id"], "host": host, "pid": message["pid"]})
                elif kind == "ended":
                    attempt = message["id"]
                    if self.state.hosts_of.get(attempt) == host and attempt not in self.state.ended:
                        await self.ended(host, message)  # else sent again, or another batch's: acked only
                    await send(writer, {"type": "acked", "id": attempt})
                elif kind == "stopped":
                    self.log({"type": "lost", "id": message["id"], "host": host, "why": "window"})
                self.settle()
        finally:
            if self.writers.get(host) is writer:
                del self.writers[host]
            writer.close()

    async def watch(self) -> None:
        """At each host's window's end, its attempts stopped (D7): they are queued again."""
        while not self.finished.is_set():
            for host, writer in list(self.writers.items()):
                if host in self.hosts and window_left(self.hosts[host], self.now()) is None:
                    with contextlib.suppress(ConnectionError):
                        await send(writer, {"type": "stop"})
            await asyncio.sleep(self.check)

    async def serve(self, port: int) -> None:
        """Serve on 127.0.0.1:port until the batch is done (every attempt ended or never to start, every pull done)."""
        server = await asyncio.start_server(self.handle, "127.0.0.1", port)
        watching = asyncio.create_task(self.watch())
        async with server:
            await self.finished.wait()
        watching.cancel()


def pull_attempt(host: Host, attempt: str, destination: Path) -> bool:
    """An ended attempt's directory copied from its host (rsync, over ssh for a remote host; retried by the caller's check)."""
    destination.mkdir(parents=True, exist_ok=True)
    source = f"{host.name}:{host.root}/{attempt}/" if host.remote else f"{host.root}/{attempt}/"
    for _ in range(3):
        done = subprocess.run(["rsync", "-a", "--exclude", "ab-runs", source, f"{destination}/"], check=False)
        if done.returncode == 0:
            return True
        time.sleep(10)
    return False


def manifest(directory: Path) -> dict[str, str]:
    """An attempt's files and their SHA-256 (its machine lock and pid file left out)."""
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and relative.parts[0] not in ("ab-runs", "pid"):
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def verify(directory: Path, files: dict[str, str]) -> bool:
    """Whether every manifest file is there with its hash."""
    return all(
        (directory / name).is_file() and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for name, digest in files.items()
    )


@dataclass
class Running:
    """An attempt a worker runs: its directory (its pid file, its run's log) and its process."""

    attempt: Attempt
    directory: Path
    pid: int
    step: int | None = None
    peak: int | None = None  # its session's largest sampled PSS, kB


def attempt_pss(session: int, proc: Path = Path("/proc")) -> int | None:
    """
    The summed PSS (kB) of an attempt: every process in its session and every process descended from
    one (its pool's workers, a forkserver's children outside the attempt's own rusage, and a run a
    wrapper moved into a session of its own, as klip's thermal guard's `setsid`), shared pages counted
    once. None without /proc.
    """
    if not proc.is_dir():
        return None
    parents: dict[int, int] = {}
    members: list[int] = []
    for stat in proc.glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            fields = stat.read_text().rpartition(")")[2].split()
            pid = int(stat.parent.name)
            parents[pid] = int(fields[1])
            if int(fields[3]) == session:
                members.append(pid)
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    tree: set[int] = set()
    while members:
        pid = members.pop()
        if pid not in tree:
            tree.add(pid)
            members.extend(children.get(pid, ()))
    total = 0
    for pid in tree:
        with contextlib.suppress(OSError, ValueError, IndexError):
            for line in (proc / str(pid) / "smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    total += int(line.split()[1])
    return total


class Worker:
    """
    A machine's worker daemon (D4): connects to the controller with its commit and crate (D6), asks for
    attempts while it has free slots, runs each as its own detached session (pid file, output beside it;
    `wrapper` before the command, `pythonpath` before the worktree on PYTHONPATH), reports its start,
    progress and end (with the attempt's manifest, for the pull), stops its attempts at the controller's
    `stop` (a window's end) and re-adopts its attempts from their pid files after its own restart. Its
    runs survive a dropped connection: it reconnects and reports, an end the controller hasn't acknowledged
    (ROOT/acked/) sent again; a connection to itself (Linux's TCP self-connect, the tunnel down) is
    refused and tried again. An attempt run again (lost, or stopped
    at a window's end) gets a fresh directory, the old one set aside under ROOT/lost/. Each look samples
    an attempt's memory (attempt_pss); its peak, kept in its directory, goes into its summary.
    """

    def __init__(
        self,
        root: Path,
        slots: int,
        python: str,
        code: Path,
        script: str = "scripts/studies/modular_family_probe.py",
        host: str | None = None,
        env: dict[str, str] | None = None,
        poll: float = POLL,
        pythonpath: str = "",
        wrapper: Sequence[str] = (),
    ) -> None:
        self.root, self.slots, self.python, self.code, self.script = root, slots, python, code, script
        self.poll, self.wrapper = poll, tuple(wrapper)
        self.host = host or socket.gethostname()
        self.env = {"PYTHONPATH": ":".join(p for p in (pythonpath, *code_paths(str(code))) if p), **(env or {})}
        self.running: dict[str, Running] = {}
        self.draining = False
        root.mkdir(parents=True, exist_ok=True)
        self.commit, self.crate = identity(code, python, self.env)
        self.adopt()

    def adopt(self) -> None:
        """Its attempts still running from before a restart, from their pid files."""
        for pid_file in self.root.glob("*/pid"):
            record = json.loads(pid_file.read_text())
            if alive(record["pid"]) and not (pid_file.parent / "ended.json").exists():
                attempt = Attempt(**record["attempt"])
                peak = pid_file.parent / "peak_pss"
                kb = int(peak.read_text()) if peak.exists() else None
                self.running[attempt.id] = Running(attempt, pid_file.parent, record["pid"], peak=kb)

    def command(self, attempt: Attempt) -> list[str]:
        return [*self.wrapper, *attempt_command(attempt, self.python, self.script, self.root / attempt.id)]

    def launch(self, attempt: Attempt) -> Running:
        directory = self.root / attempt.id
        if directory.exists():  # an earlier run's: set aside, never appended to
            lost = self.root / "lost"
            lost.mkdir(exist_ok=True)
            directory.rename(lost / f"{attempt.id}-{sum(1 for _ in lost.glob(f'{attempt.id}-*')) + 1}")
        directory.mkdir(parents=True)
        env = {
            **os.environ,
            **self.env,
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "AB_RUNS_ROOT": str(directory / "ab-runs"),  # each run its own machine lock
        }
        with (directory / "run.out").open("ab") as out:  # its own session: it outlives the worker
            child = subprocess.Popen(
                self.command(attempt),
                cwd=self.code,
                env=env,
                stdout=out,
                stderr=out,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        (directory / "pid").write_text(json.dumps({"pid": child.pid, "attempt": asdict(attempt)}))
        running = Running(attempt, directory, child.pid)
        self.running[attempt.id] = running
        return running

    def log_of(self, running: Running) -> Path:
        return BACKDROPS[running.attempt.backdrop].log(running.directory, running.attempt.seed)

    async def stop(self, writer: asyncio.StreamWriter) -> None:
        """Every running attempt stopped (its process group), reported, its directory marked."""
        for attempt_id, running in list(self.running.items()):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(running.pid, 15)
            (running.directory / "ended.json").write_text(json.dumps({"stopped": True}))
            await send(writer, {"type": "stopped", "id": attempt_id})
            del self.running[attempt_id]

    def unacked(self) -> list[Path]:
        """The ended attempts' directories whose `ended` the controller hasn't acknowledged (stopped ones aside)."""
        return [
            ended.parent
            for ended in sorted(self.root.glob("*/ended.json"))
            if not (self.root / ACKED / ended.parent.name).exists() and "stopped" not in json.loads(ended.read_text())
        ]

    async def session(self, port: int) -> bool:
        """One connection: hello, then work until drained with nothing running; whether to stop."""
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            if writer.get_extra_info("sockname") == writer.get_extra_info("peername"):
                raise ConnectionError("connected to itself: nothing listens on the port")
            unacked = self.unacked()
            hello = {"type": "hello", "host": self.host, "slots": self.slots, "running": list(self.running)}
            hello["ended"] = [d.name for d in unacked]
            await send(writer, {**hello, "commit": self.commit, "crate": self.crate})
            for directory in unacked:
                result = json.loads((directory / "ended.json").read_text())
                await send(
                    writer, {"type": "ended", "id": directory.name, "summary": result, "files": manifest(directory)}
                )
            while True:
                free = self.slots - len(self.running)
                if free > 0 and not self.draining:
                    await send(writer, {"type": "ready", "free": free})
                with contextlib.suppress(TimeoutError):
                    line = await asyncio.wait_for(reader.readline(), timeout=self.poll)
                    if not line:
                        return False  # the controller went away: reconnect
                    message = json.loads(line)
                    if message["type"] == "attempt":
                        fields = {k: message[k] for k in ("id", "cell", "backdrop", "seed", "cutoff")}
                        running = self.launch(Attempt(**fields))
                        await send(writer, {"type": "started", "id": running.attempt.id, "pid": running.pid})
                    elif message["type"] == "drain":
                        self.draining = True
                    elif message["type"] == "acked":
                        (self.root / ACKED).mkdir(exist_ok=True)
                        (self.root / ACKED / message["id"]).touch()
                    elif message["type"] == "stop":
                        await self.stop(writer)
                    elif message["type"] == "refused":
                        return True  # on another commit or crate than the batch's: nothing to run here
                await self.look(writer)
                if self.draining and not self.running and not self.unacked():  # every end acknowledged
                    return True
        finally:
            writer.close()

    async def look(self, writer: asyncio.StreamWriter) -> None:
        """Each running attempt's progress, and the ended ones' summaries and manifests."""
        for attempt_id, running in list(self.running.items()):
            step = last_step(self.log_of(running))
            kb = attempt_pss(running.pid)
            if kb is not None and (running.peak is None or kb > running.peak):
                running.peak = kb
                (running.directory / "peak_pss").write_text(str(kb))
            if step != running.step:
                running.step = step
                await send(writer, {"type": "progress", "id": attempt_id, "step": step})
            if not alive(running.pid):
                result = summary(self.log_of(running))
                result["peak_pss_mb"] = None if running.peak is None else round(running.peak / 1024, 1)
                (running.directory / "ended.json").write_text(json.dumps(result))
                files = manifest(running.directory)
                await send(writer, {"type": "ended", "id": attempt_id, "summary": result, "files": files})
                del self.running[attempt_id]

    async def serve(self, port: int, retry: float = 5.0) -> None:
        """Connect, work, and reconnect after a dropped connection, until drained."""
        while True:
            with contextlib.suppress(OSError):
                if await self.session(port):
                    return
            await asyncio.sleep(retry)


def code_paths(code: str) -> tuple[str, str]:
    """A worktree's import roots: the studies' worktree and its ml submodule (the research-separation plan, D6)."""
    return code, f"{code}/ml"


# each host's clones (the research-separation plan, D6): a batch's worktree is a studies worktree at the
# batch's commit, its ml submodule checked out from the host's own ml clone (no crate build: the batch's
# crate comes from CRATE on PYTHONPATH)
STUDIES_URL = "https://github.com/davidbarkhuizen/indrajala-studies.git"  # public: no key on the hosts
CLONES = {"stone": ("~/labour/code/indrajala-studies", "~/labour/code/indrajala-ml")}
DEFAULT_CLONES = ("~/code/indrajala-studies", "~/code/indrajala-ml")


def _worktree(checkout: str, commit: str, ml: str) -> str:
    """The shell line, run in the studies clone, that puts the worktree at commit and its ml submodule at its pin."""
    return (
        f"(git worktree add -q --detach {checkout} {commit} 2>/dev/null || git -C {checkout} checkout -q --detach {commit})"
        f" && git -C {checkout} submodule init -q ml && git -C {checkout} config submodule.ml.url {ml}"
        f" && git -C {checkout} -c protocol.file.allow=always submodule update -q ml"
    )


def prepare(host: Host, commit: str, repo: Path, scratch: Path) -> list[list[str]]:
    """
    The commands that put the host's worktree at `commit` (D6), a studies commit, with ml at the commit
    it pins: from GitHub on klip and jebel (each clone fetched, the studies clone made if missing), from
    bundles on stone (its sandbox: no authenticated service from it; the studies and ml commits each
    bundled), a local worktree on the controller's machine (its ml submodule from the sibling ml clone).
    """
    checkout = shlex.quote(host.code)
    if not host.remote:
        ml = shlex.quote(str(repo.parent / "indrajala-ml"))
        return [["bash", "-c", f"cd {shlex.quote(str(repo))} && {_worktree(checkout, commit, ml)}"]]
    studies, ml = CLONES.get(host.name, DEFAULT_CLONES)
    if host.name == "stone":
        bundles = scratch / "batch.bundle", scratch / "batch-ml.bundle"
        here, sub = shlex.quote(str(repo)), shlex.quote(str(repo / "ml"))
        return [
            ["git", "-C", str(repo), "branch", "-f", "batch-pin", commit],
            ["git", "-C", str(repo), "bundle", "create", "-q", str(bundles[0]), "batch-pin"],
            # the ml commit the studies commit pins, from the controller's ml submodule
            [
                "bash",
                "-c",
                f"git -C {sub} fetch -q origin && git -C {sub} branch -f batch-pin $(git -C {here} rev-parse {commit}:ml)",
            ],
            ["git", "-C", str(repo / "ml"), "bundle", "create", "-q", str(bundles[1]), "batch-pin"],
            ["ssh", host.name, "mkdir -p ~/labour/bundles"],
            ["scp", "-q", str(bundles[0]), f"{host.name}:labour/bundles/batch.bundle"],
            ["scp", "-q", str(bundles[1]), f"{host.name}:labour/bundles/batch-ml.bundle"],
            [
                "ssh",
                host.name,
                (
                    f"cd {ml} && git fetch -q --no-recurse-submodules ~/labour/bundles/batch-ml.bundle batch-pin:batch-pin"
                    f" && (test -d {studies} || git init -q {studies}) && cd {studies}"
                    " && git fetch -q --no-recurse-submodules ~/labour/bundles/batch.bundle batch-pin:batch-pin"
                    f" && {_worktree(checkout, commit, ml)}"
                ),
            ],
        ]
    return [
        [
            "ssh",
            host.name,
            (
                f"cd {ml} && git fetch -q origin && (test -d {studies} || git clone -q {STUDIES_URL} {studies})"
                f" && cd {studies} && git fetch -q origin && {_worktree(checkout, commit, ml)}"
            ),
        ]
    ]


def worker_line(host: Host, port: int) -> str:
    """The shell line that starts the host's worker, detached (its output in its root's worker.out)."""
    root, code = shlex.quote(host.root), shlex.quote(host.code)
    path = ":".join(p for p in (host.pythonpath, *code_paths(host.code)) if p)
    args = [
        host.python,
        "scripts/runner/batch.py",
        "worker",
        host.root,
        "--slots",
        str(host.slots),
        "--code",
        host.code,
        "--python",
        host.python,
        "--host",
        host.name,
        "--port",
        str(port),
        *(["--pythonpath", host.pythonpath] if host.pythonpath else []),
        *(f"--wrapper={w}" for w in host.wrapper),
    ]
    setup = f"{host.setup}; " if host.setup else ""
    return (
        f"{setup}mkdir -p {root} && cd {code} && PYTHONPATH={shlex.quote(path)} setsid nohup "
        f"{shlex.join(args)} >> {root}/worker.out 2>&1 < /dev/null &"
    )


async def tunnel(host: Host, port: int, retry: float = 10.0) -> None:
    """The host's reverse tunnel to the controller's 127.0.0.1:port (D4), opened again whenever it drops."""
    while True:
        process = await asyncio.create_subprocess_exec(
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            "-R",
            f"127.0.0.1:{port}:127.0.0.1:{port}",  # IPv4 or fail and retry: never bound on ::1 alone
            host.name,
            stdin=subprocess.DEVNULL,
        )
        await process.wait()
        await asyncio.sleep(retry)


async def fleet(hosts: Sequence[Host], port: int) -> None:
    """Each remote host's tunnel kept open and its worker started; the local host's worker started here."""
    tunnels = [asyncio.create_task(tunnel(h, port)) for h in hosts if h.remote]
    await asyncio.sleep(3)  # the tunnels up before the workers connect (a worker retries anyway)
    for h in hosts:
        line = worker_line(h, port)
        command = ["ssh", "-f", h.name, line] if h.remote else ["bash", "-c", line]
        started = await asyncio.create_subprocess_exec(*command, stdin=subprocess.DEVNULL)
        await started.wait()
    await asyncio.gather(*tunnels)


def identity(code: Path, python: str, env: dict[str, str]) -> tuple[str | None, str | None]:
    """The worktree's commit and the crate extension's SHA-256 as `python` imports it (D6)."""
    commit = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    found = subprocess.run(
        [python, "-c", "import indrajala_math_rust, pathlib; print(pathlib.Path(indrajala_math_rust.__file__).parent)"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )
    crate = None
    if found.returncode == 0:
        built = sorted(Path(found.stdout.strip()).glob("*.so"))
        crate = hashlib.sha256(built[0].read_bytes()).hexdigest() if built else None
    return (commit.stdout.strip() or None), crate


def alive(pid: int) -> bool:
    """Whether a process lives (a finished child reaped, so it doesn't linger as a zombie)."""
    with contextlib.suppress(ChildProcessError):
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def groups(batch: Batch) -> dict[tuple[str, str], list[Attempt]]:
    """A batch's attempts by cell and backdrop, the cells in their order, each one's backdrops sorted."""
    made: dict[tuple[str, str], list[Attempt]] = {}
    for cell in batch.cells:
        for backdrop in sorted({a.backdrop for a in batch.attempts if a.cell == cell}):
            made[cell, backdrop] = [a for a in batch.attempts if (a.cell, a.backdrop) == (cell, backdrop)]
    return made


def report(directory: Path) -> list[str]:
    """A batch's reading: each cell's backdrops' attempts, certificates and Wilson interval."""
    state = State.read(directory)
    cells = state.batch.cells
    lines = [f"{state.batch.name}: {state.batch.mode}, cell {', '.join(cells)}, {'done' if state.done else 'running'}"]
    for (cell, backdrop), mine in groups(state.batch).items():
        ended = [state.ended[a.id] for a in mine if a.id in state.ended]
        wins = sum(1 for s in ended if s["success"] is not None)
        low, high = wilson(wins, len(ended))
        lines.append(
            f"  {f'{cell} ' if len(cells) > 1 else ''}{backdrop}: {len(ended)} of {len(mine)} ended, {wins} certified"
            + (f" ({wins / len(ended):.2f}, {low:.2f}-{high:.2f})" if ended else "")
        )
    return lines


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def runs_of(summaries: Sequence[dict[str, Any]]) -> list[tuple[float | None, float, bool]]:
    """Ended attempts as restart_cost's runs (success and last step's end in epochs, censored at its budget)."""
    return [
        (None if s["success"] is None else float(s["success"]), float(s["epochs"]), s["end"] in CENSORED)
        for s in summaries
        if s.get("epochs") is not None
    ]
