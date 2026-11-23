"""
When a run forms its states, read on one clock across machines (the population method's diagnosis, 2026-11-22).

A run's state share (the share of a step's inputs whose state sits on a code) is 1.0 at every certificate
found in the studies' logs (395 curriculum runs across s3, s4, s5, move-to-front, random and division), and
most failed runs never reach it. Read on the step over the target's depth plus Chow's `k` (the coverage
rule's clock), the chance that a run not formed by then certifies later falls alike on every machine. A
stopping rule stops a run at the first step end at or past `ratio` times depth plus `k`, before a
certificate, whose share is below `below`; a stopped run is replaced by a fresh seed, so its measure is
certificates per epoch, not certificates kept.

Formed is not separated (the stages reading, 2026-11-23): a step end's machine summary (machine_reading.summary:
the codes reached, the (code, state) pairs and those right) says whether each code stands for one target state.
A run passes three stages, formed (share 1.0), separated (pairs equal codes, formed) and certified; a separated
run nearly always certifies. The network's architecture beside it (`Architecture`): the window's width W, its
blocks, each factor's width and codes, and the joint codes C over W squared, the load.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indrajala_studies import batch as runner
from indrajala_studies import window_tasks as wt

FULL = 0.999  # a step end's share at or above this has formed every state


@dataclass(frozen=True)
class StepEnd:
    """
    One step's end: its step, the epochs trained in all by then, its state share, whether it certified, and its
    machine summary when logged (codes reached, (code, state) pairs, pairs right; None in older logs).
    """

    step: int
    epochs: int
    share: float
    certified: bool
    codes: int | None = None
    pairs: int | None = None
    right: int | None = None

    @property
    def formed(self) -> bool:
        return self.share >= FULL

    @property
    def separated(self) -> bool:
        """Formed, and each code reached stands for one target state."""
        return self.formed and self.codes is not None and self.pairs == self.codes

    @property
    def states_per_code(self) -> float | None:
        return self.pairs / self.codes if self.codes and self.pairs is not None else None


@dataclass(frozen=True)
class Run:
    """A run's step ends and its target's depth plus Chow's k."""

    name: str
    ends: tuple[StepEnd, ...]
    depth_k: int

    @property
    def certified(self) -> bool:
        return any(e.certified for e in self.ends)

    @property
    def epochs(self) -> int:
        return self.ends[-1].epochs if self.ends else 0

    @property
    def traced(self) -> bool:
        """Every step end carries a machine summary."""
        return bool(self.ends) and all(e.codes is not None for e in self.ends)


@dataclass(frozen=True)
class Rule:
    """Stop at the first step end at or past `ratio` times depth plus k, before a certificate, with a share below `below`."""

    ratio: float
    below: float = FULL

    def __str__(self) -> str:
        return f"{self.ratio:g}(d+k), share < {self.below:g}"


def depth_plus_k(machine: wt.Mealy) -> int:
    """The coverage rule's measure: the deepest state's distance from the start plus Chow's k."""
    k = machine.distinguishing_length()
    assert k is not None, "a minimal target tells every pair of states apart"
    return machine.depth() + k


def step_ends(rows: Sequence[Mapping[str, Any]]) -> tuple[StepEnd, ...]:
    """A log's step ends that carry a share (its last run's: runner.last_run), the epochs counted as they accrue."""
    total, ends = 0, list[StepEnd]()
    for r in runner.last_run(list(rows)):
        if "end" in r and "state_share" in r:
            total += int(r.get("epochs", 0))
            m: dict[str, int] = dict(r["machine"]) if isinstance(r.get("machine"), Mapping) else {}
            ends.append(
                StepEnd(
                    int(r["step"]),
                    total,
                    float(r["state_share"]),
                    bool(r.get("certified")),
                    m.get("codes"),
                    m.get("pairs"),
                    m.get("right"),
                )
            )
    return tuple(ends)


def read_run(log: Path, depth_k: int, name: str | None = None) -> Run:
    """A run from its log.jsonl; a line cut short (a run stopped while writing it) is skipped."""
    rows: list[dict[str, Any]] = []
    for line in log.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return Run(name or str(log.parent), step_ends(rows), depth_k)


def formed_step(run: Run) -> int | None:
    """The first step whose end has every state formed (share at least FULL), or None."""
    return next((e.step for e in run.ends if e.formed), None)


def separated_step(run: Run) -> int | None:
    """The first step whose end is separated (formed, one target state a code), or None."""
    return next((e.step for e in run.ends if e.separated), None)


def certified_step(run: Run) -> int | None:
    return next((e.step for e in run.ends if e.certified), None)


@dataclass(frozen=True)
class Stages:
    """Runs counted at each stage: formed, separated, certified."""

    runs: int
    formed: int
    separated: int
    certified: int


def stages(runs: Sequence[Run]) -> Stages:
    return Stages(
        len(runs),
        sum(formed_step(r) is not None for r in runs),
        sum(separated_step(r) is not None for r in runs),
        sum(r.certified for r in runs),
    )


def stop(run: Run, rule: Rule) -> StepEnd | None:
    """Where `rule` stops `run`: the first step end at or past its point with the share below, before any certificate."""
    for e in run.ends:
        if e.certified:
            return None
        if e.step >= rule.ratio * run.depth_k and e.share < rule.below:
            return e
    return None


def flagged_later(runs: Iterable[Run], ratio: float, flag: Callable[[StepEnd], bool]) -> tuple[int, int]:
    """
    Of the runs alive (no certificate before it) at their first step end at or past `ratio` times depth plus k,
    those whose end there is flagged: (certified later, flagged). A signal for a stopping rule loses the first.
    """
    later = flagged = 0
    for r in runs:
        at = next((e for e in r.ends if e.step >= ratio * r.depth_k), None)
        if at is None or any(e.certified for e in r.ends if e.step < at.step) or not flag(at):
            continue
        flagged += 1
        later += r.certified
    return later, flagged


def unformed_later(runs: Iterable[Run], ratio: float) -> tuple[int, int]:
    """Of the runs reaching `ratio` times depth plus k with no formed step end and no certificate before it: (certified later, all)."""
    alive = [
        r
        for r in runs
        if r.ends
        and r.ends[-1].step >= ratio * r.depth_k
        and not any(e.share >= FULL or e.certified for e in r.ends if e.step < ratio * r.depth_k)
    ]
    return sum(r.certified for r in alive), len(alive)


@dataclass(frozen=True)
class RuleEffect:
    """A rule over a set of runs: runs stopped, certificates lost of those found, the epochs saved, certificates per epoch against none."""

    stopped: int
    lost: int
    certified: int
    saved: float  # share of all epochs spent
    gain: float  # certificates per epoch with the rule over without (stopped runs not replaced: the rate's ratio)


def effect(runs: Sequence[Run], rule: Rule) -> RuleEffect:
    spent = sum(r.epochs for r in runs)
    certified = sum(r.certified for r in runs)
    stopped = lost = saved = 0
    for r in runs:
        at = stop(r, rule)
        if at is not None:
            stopped, lost, saved = stopped + 1, lost + r.certified, saved + r.epochs - at.epochs
    kept = certified - lost
    gain = (kept / (spent - saved)) / (certified / spent) if certified and spent > saved else 0.0
    return RuleEffect(stopped, lost, certified, saved / spent if spent else 0.0, gain)


def effect_summary(rule: Rule, e: RuleEffect) -> str:
    return (
        f"{rule}: stops {e.stopped}, loses {e.lost} of {e.certified} certificates, saves {e.saved:.0%} of the epochs; "
        f"{e.gain:.2f}x certificates per epoch"
    )


@dataclass(frozen=True)
class Architecture:
    """A window-recurrent network's shape: the window's width W, its blocks, each factor's width and codes."""

    width: int
    blocks: int
    factor_widths: tuple[int, ...]
    factor_codes: tuple[int, ...]

    @property
    def joint_codes(self) -> int:
        """C: the factors' codes multiplied, the joint states the codebooks can name."""
        return math.prod(self.factor_codes)

    @property
    def load(self) -> float:
        """C over W squared."""
        return self.joint_codes / self.width**2


def architecture(network: Mapping[str, Any]) -> Architecture | None:
    """A saved network's architecture (its JSON), or None for one without codes (a continuous cell)."""
    layers: list[dict[str, Any]] = list(network.get("layers", []))
    width: int | None = next((layer["size"] for layer in layers if layer.get("kind") == "embedding"), None)
    recurrence: dict[str, Any] = dict(network.get("recurrence") or {})
    factors: list[dict[str, Any]] = list(recurrence.get("factors") or [])
    codes: list[int | None] = [dict(f.get("codes") or {}).get("count") for f in factors]
    if width is None or not factors or any(c is None for c in codes):
        return None
    blocks = sum(layer.get("kind") == "state_input" for layer in layers)
    widths = tuple(int(f["state_size"]) for f in factors)
    return Architecture(int(width), blocks, widths, tuple(int(c or 0) for c in codes))


def network_file(run_directory: Path) -> Path | None:
    """A run's saved network: network.json, else its last step's, else any."""
    if (run_directory / "network.json").exists():
        return run_directory / "network.json"
    steps = sorted(run_directory.glob("step-*.network.json"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    others = sorted(run_directory.glob("*.network.json"))
    return steps[-1] if steps else (others[-1] if others else None)
