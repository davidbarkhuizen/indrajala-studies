"""
A collapse trigger, generic over tasks (the p11-trigger workplan, D5; retired:
`git show 28894cf:docs/p11-trigger-workplan.md`): when a step's screen has fallen far enough from
the step's peak to call the fall a collapse. Three choices, none of them about a task:

- **the signal**: the screen's share of every answer right ("pooled"), the mean over answer kinds
  of each kind's share ("mean"), or the worst kind's share ("worst"); the kinds are whatever the
  task's accuracy tallies (division's quotient classes and queries), and a task with one kind has
  all three equal;
- **the rule**: below a fraction of the step's peak ("ratio": value < threshold * peak) or more
  than an amount below it ("fall": value < peak - threshold);
- **the floor**: fire only once the step's peak is at least this (0: always).

The default, pooled below 0.6 of the peak with no floor, is the collapse-fork workplan's trigger
(retired: `git show 89cf3a3:docs/collapse-fork-workplan.md`) exactly; a task tunes the settings,
never the code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

SIGNALS = ("pooled", "mean", "worst")
RULES = ("ratio", "fall")


@dataclass(frozen=True)
class Trigger:
    signal: str = "pooled"
    rule: str = "ratio"
    threshold: float = 0.6  # the collapse-fork workplan's D1: below 0.6 of the peak
    floor: float = 0.0

    def __post_init__(self) -> None:
        assert self.signal in SIGNALS, f"a trigger's signal is one of {SIGNALS}; got {self.signal!r}"
        assert self.rule in RULES, f"a trigger's rule is one of {RULES}; got {self.rule!r}"

    def value(self, pooled: float, kinds: Mapping[str, float]) -> float:
        """The signal, from the screen's pooled share and each answer kind's share."""
        if self.signal == "pooled" or not kinds:
            return pooled
        shares = list(kinds.values())
        return sum(shares) / len(shares) if self.signal == "mean" else min(shares)

    def fires(self, value: float, peak: float) -> bool:
        """Whether a signal's value against the step's peak of it is a collapse."""
        if peak < self.floor:
            return False
        return value < self.threshold * peak if self.rule == "ratio" else value < peak - self.threshold

    def describe(self) -> str:
        """The trigger as a line: its signal, rule, threshold and floor."""
        rule = f"below {self.threshold:g} of the peak" if self.rule == "ratio" else f"{self.threshold:g} below the peak"
        return f"{self.signal} {rule}" + (f", once the peak is {self.floor:g}" if self.floor else "")
