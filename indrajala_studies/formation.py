"""
When a run forms its states, read on one clock across machines (the population method's diagnosis, 2026-11-22).

A run's state share (the share of a step's inputs whose state sits on a code) is 1.0 at every certificate
found in the studies' logs (395 curriculum runs across s3, s4, s5, move-to-front, random and division), and
most failed runs never reach it. Read on the step over the target's depth plus Chow's `k` (the coverage
rule's clock), the chance that a run not formed by then certifies later falls alike on every machine. A
stopping rule stops a run at the first step end at or past `ratio` times depth plus `k`, before a
certificate, whose share is below `below`; a stopped run is replaced by a fresh seed, so its measure is
certificates per epoch, not certificates kept.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indrajala_studies import batch as runner
from indrajala_studies import window_tasks as wt

FULL = 0.999  # a step end's share at or above this has formed every state


@dataclass(frozen=True)
class StepEnd:
    """One step's end: its step, the epochs trained in all by then, its state share, whether it certified."""

    step: int
    epochs: int
    share: float
    certified: bool


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
            ends.append(StepEnd(int(r["step"]), total, float(r["state_share"]), bool(r.get("certified"))))
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
    return next((e.step for e in run.ends if e.share >= FULL), None)


def stop(run: Run, rule: Rule) -> StepEnd | None:
    """Where `rule` stops `run`: the first step end at or past its point with the share below, before any certificate."""
    for e in run.ends:
        if e.certified:
            return None
        if e.step >= rule.ratio * run.depth_k and e.share < rule.below:
            return e
    return None


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
