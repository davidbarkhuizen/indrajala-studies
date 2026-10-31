"""
The certificate guard (the guard workplan, retired: `git show
ed0e378:docs/guard-training-workplan.md`; the snap-wall workplan's stage 2d): the network's machine
read off after a training update, and an update that makes a right machine (no wrong (state, window)
pair against the target) wrong undone, the network's weights, its optimizer's state and its
generator restored, the batch skipped. A remedy for the collapse acting where it starts: a snapped
network changes its machine one transition at a time, silently to the batch's loss, and the collapse
is the optimizer's response to the first wrong one (stage 2c).

The guard acts on a network once armed for it (`arm`), so a caller reads only where a right
machine is possible: the family's run arms it for an epoch after a perfect quotient screen (the
guard workplan's D5). Armed, it reads the machine before the first update and after every one.
`right` is the read: breadth first by levels, a level's states in one forward pass, stopping at
the first wrong pair (D6). Whether a machine is right doesn't depend on the order its states are
reached in: where it is right, each state's target state is the same along every path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from indrajala_ml.pcg64 import generator_state, set_generator_state

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_tasks as wt


def right(machine: wc.WindowMachine, task: wt.Task) -> bool:
    """
    Whether the machine reachable from machine's start over every window is the task's target: no
    (state, window) pair whose labels or next state is wrong, each state's target state fixed where
    it is first reached (as the certificate fixes it).
    """
    target = task.machine
    windows = wc.all_windows(machine.columns, target.tokens)
    want_labels, want_state = target.windows(windows)
    n = len(windows)
    start, _ = machine.start()
    phi = {start: target.start}
    level = [start]
    while level:
        reached: list[wc.State] = []
        for begin in range(0, len(level), max(1, wc.BATCH // n)):
            states = level[begin : begin + max(1, wc.BATCH // n)]
            steps = wc.step_states(machine, states, windows)
            for i, q in enumerate(states):
                digits = steps.digits[i * n : (i + 1) * n]
                if (digits != want_labels[phi[q]]).any():
                    return False
                for j, row in enumerate(steps.states[i * n : (i + 1) * n]):
                    r = tuple(int(c) for c in row)
                    want = int(want_state[phi[q]][j])
                    if r not in phi:
                        phi[r] = want
                        reached.append(r)
                    elif phi[r] != want:
                        return False
        level = reached
    return True


def training_state(network: Any) -> tuple[Any, Any, Any]:
    """What a learn step changes: the weights, the optimizer's state and the generator's."""
    return network.snapshot(), network.optimizer.state(), generator_state(network.rng)


def restore_training_state(network: Any, state: tuple[Any, Any, Any]) -> None:
    weights, optimizer, rng = state
    network.restore(weights)
    network.optimizer.load_state(optimizer)
    set_generator_state(network.rng, rng)


class Guard:
    """
    The guard over one network at a time: `machine` reads a network as a WindowMachine (the
    family's quotient slot, say), `task` its target. Unarmed, or for another network, learn_batch
    is the network's own; `updates` counts the armed network's updates, `rollbacks` those undone.
    """

    def __init__(self, machine: Callable[[Any], wc.WindowMachine], task: wt.Task) -> None:
        self.machine, self.task = machine, task
        self.updates = self.rollbacks = 0
        self._network: Any = None
        self._right = False

    def arm(self, network: Any) -> None:
        """Guard network from here on, its machine read now."""
        self._network = network
        self._right = right(self.machine(network), self.task)

    def disarm(self) -> None:
        self._network = None

    @property
    def armed(self) -> bool:
        return self._network is not None

    def learn_batch(self, network: Any, rate: float, batch: Sequence[Any]) -> None:
        """network's update on batch, undone if it makes the armed network's right machine wrong."""
        if network is not self._network:
            network.learn_batch(rate, batch)
            return
        saved = training_state(network) if self._right else None
        network.learn_batch(rate, batch)
        self.updates += 1
        now = right(self.machine(network), self.task)
        if saved is not None and not now:
            restore_training_state(network, saved)
            self.rollbacks += 1
            return
        self._right = now
