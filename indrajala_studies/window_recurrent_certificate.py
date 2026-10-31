"""
The window-recurrent network's certificate against its task's machine
(indrajala-research/docs/research-notes/window-recurrent-report.md, E1, stage 2; over any target machine, D17, stage 5;
indrajala-studies' scripts/studies/window_recurrent_study.py's certify).

A window's work is a function of the state it receives and its k column tokens, and a snapped state
is one of the network's m codes, exactly (the codes' forward pass gives the code's own vector), so
the network is a finite-state transducer: its states the codes, its input symbols the T^k windows
(T the task's tokens: 27 for addition), its outputs k labels. The task's target is a minimal Mealy
machine over the same tokens (window_tasks: for addition the carry automaton, its states the
carries 0-2 and its labels the sum's digits). Starting from the snapped initial state (W8), breadth
first over every window, certify collects the reachable codes and, for each reachable code q and
window w, the labels D(q, w) and the next code G(q, w). The network is certified when a map phi
from its reachable codes to the target's states, with phi(start) the target's start, has for every
reachable q and every w:

- D(q, w) the target's k labels for w from state phi(q), and
- phi(G(q, w)) the target's state after w from phi(q),

and every label leads by its margin (D3): its log-probability (its logit) at least LOGIT_LEAD ahead
of the next label's. phi is fixed as the search goes, a code's state by the first window that
reaches it, then checked by every other. A case is a start state, a word of windows (the zero
columns of the alignment, W6-i, and the final window, W9, are windows like any other) and the
transducer's outputs, so a certified network is right at every length and alignment. Conversely, a
code reached with two target states fails, for one of them, on the shortest word that tells the two
apart (the target is minimal; for addition, one zero column, whose digit is the carry): the
certificate is exact, up to its margins.

The snap's margin (D3): a snap holds when every other code is at least SNAP_RATIO times as far as
the code it snaps to. Rounding elsewhere (another batch, another BLAS order) could snap a near tie
to another code, so a near tie branches: every code within the ratio is a next code too, explored
and checked as G's is. A certificate with near ties is still a proof, for either snap; they are
counted (ties) apart from wrong outputs, and those that split (split_ties: a branch into another
state of the minimised machine than the snap's, the machine's own nondeterminism, read without the
target). A label's near tie can't branch (one of its labels is
wrong): it fails the certificate. (D3 as written failed every near tie; branching proves the same
for both snaps instead, settled in stage 2.)

The search runs every window from every reachable code, failing or not: the counts of wrong labels
and states, the margins and the minimised machine (the coarsest partition of the reachable codes
that keeps D and G, the target's size for a certified network: 3 for addition) describe a near miss
too. The first failure in breadth-first order, the fewest windows, is the counterexample, as the
windows from the start and as the task's case at alignment 0, through one window more: for a wrong
state, the shortest word telling the two states apart, zero-padded to a window (for addition, a
zero window); for wrong labels, a zero window (W9).

The certificate's JSON names what stage 2 named for addition in the target's terms: wrong_outputs
(stage 2's wrong_digits), wrong_states (wrong_carries), the counterexample's kinds "outputs" and
"state" ("digits", "carry"), its state_in (carry_in) and its case (addition), with the task's name.

A factored state (the factored-state workplan, stage 6) is a machine of tuples: a state is a code
per factor (State, D4), m_1 ... m_G of them at most; "code" above reads "state", and the search,
the map phi and the minimised machine are over tuples. A snap's margin is per factor: a near tie in
any factor branches, to every combination of the factors' snapped and near-tied codes (next_states).
Each factor's readout (D9, FactorReadout) says what it holds over the reachable states: whether its
code is a function of the target's state, the target's states grouped by its codes, and that
grouping's name, a linear form of the target's coordinates (window_tasks.Coordinates: "A", "B",
"A + B", ...). The JSON writes a state as a list of codes, and phi as [state, target] pairs.

A WindowMachine is what certify reads: WindowRecurrentShape's networks through NetworkMachine, or a
stand-in (the tests').
"""

from __future__ import annotations

import hashlib
import itertools
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
from indrajala_ml.model.persistence.load_network import load_network

from indrajala_studies import window_tasks as wt

SNAP_RATIO = 1.05  # D3: every other code's distance over the snapped code's, at least
LOGIT_LEAD = 1e-6  # D3: a label's logit over the next's, at least
BATCH = 8_192  # windows a forward pass

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


# a machine's state: a code per factor of the network's state (the factored-state workplan, D4)
State = tuple[int, ...]


class Steps(NamedTuple):
    """
    A batch of windows from one state: each window's k labels and their logit leads, the state it
    snaps to, a code per factor, and each factor's distance to every one of its codes.
    """

    digits: IntArray  # (N, k)
    leads: FloatArray  # (N, k)
    states: IntArray  # (N, G)
    distances: list[FloatArray]  # per factor, (N, m_g)


class WindowMachine(Protocol):
    """A network as a transducer over its states: its window size, its codes, its start and its windows from a state."""

    @property
    def columns(self) -> int: ...

    @property
    def codes(self) -> list[int]:
        """Each factor's code count."""
        ...

    def start(self) -> tuple[State, list[FloatArray]]:
        """The initial state, and each factor's distance to every one of its codes (m_g,)."""
        ...

    def step(self, state: State, windows: IntArray) -> Steps:
        """windows (N, k) of column tokens, each run from the state."""
        ...


def distances_to(H: FloatArray, E: FloatArray) -> FloatArray:
    """Each row of H's Euclidean distance to each code (N, m)."""
    return np.sqrt(((H[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2).sum(axis=2))


def snap_margins(codes: IntArray, distances: FloatArray, tolerance: float) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """
    Each snap's ratio, the nearest other code's distance over the snapped code's (inf when it is
    alone, 1 when both are 0), and the other codes within tolerance of it (N, m), its near ties.
    """
    rows = np.arange(len(codes))
    chosen = distances[rows, codes]
    others = distances.copy()
    others[rows, codes] = np.inf
    nearest_other = others.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(chosen > 0, nearest_other / chosen, np.where(nearest_other > 0, np.inf, 1.0))
        near = others < tolerance * chosen[:, np.newaxis]
    near |= (others == 0) & (chosen == 0)[:, np.newaxis]
    return ratios, near


def logit_leads(probabilities: FloatArray, classes: int) -> tuple[IntArray, FloatArray]:
    """Each column's most probable label, and its log-probability's lead over the next's."""
    P = probabilities.reshape(probabilities.shape[0], -1, classes)
    top = np.sort(P, axis=2)[:, :, -2:]
    with np.errstate(divide="ignore"):
        leads = np.log(top[:, :, 1]) - np.log(top[:, :, 0])
    return np.argmax(P, axis=2), np.where(top[:, :, 0] == top[:, :, 1], 0.0, leads)


class NetworkMachine:
    """
    A window-recurrent array network (WindowRecurrentShape, either backend) as a WindowMachine: its
    states a code per factor, every factor snapped.
    """

    def __init__(self, network: Any) -> None:
        assert network.recurrence.snapped, "a continuous factor (codes=None) isn't a finite machine"
        self.network = network
        self.E = [np.array(f.codes.E.tolist(), dtype=np.float64) for f in network.recurrence_layer.factors]
        sizes = [f.state_size for f in network.recurrence.factors]
        self._blocks = [(sum(sizes[:g]), sum(sizes[: g + 1])) for g in range(len(sizes))]

    @property
    def columns(self) -> int:
        return int(self.network.columns)

    @property
    def codes(self) -> list[int]:
        return [len(E) for E in self.E]

    def _distances(self, H: Any) -> list[FloatArray]:
        # each factor's columns of H against its codes
        values = np.array(H.tolist(), dtype=np.float64)
        return [distances_to(values[:, a:b], E) for (a, b), E in zip(self._blocks, self.E, strict=True)]

    def start(self) -> tuple[State, list[FloatArray]]:
        H, states = self.network.first_step()
        return states[0], [d[0] for d in self._distances(H)]

    def step(self, state: State, windows: IntArray) -> Steps:
        return self.step_states([state], windows)

    def step_states(self, states: Sequence[State], windows: IntArray) -> Steps:
        """The windows from each of states in one forward pass, state by state (step_states)."""
        backend = self.network.backend
        X = backend.matrix(np.tile(windows, (len(states), 1)).astype(np.float64).tolist())
        codes = [np.concatenate([E[i] for E, i in zip(self.E, state, strict=True)]) for state in states]
        C = backend.matrix(np.repeat(np.array(codes), len(windows), axis=0).tolist())
        output, H, after = self.network.window_step(X, C)
        digits, leads = logit_leads(np.array(output.tolist(), dtype=np.float64), self.network.class_count)
        return Steps(digits, leads, np.array(after, dtype=np.int64).reshape(len(after), -1), self._distances(H))


def step_states(machine: WindowMachine, states: Sequence[State], windows: IntArray) -> Steps:
    """
    The windows from each of states, stacked state by state (row i * len(windows) + j is window j
    from states[i]): in one forward pass where the machine has its own step_states, else a step a
    state.
    """
    batched = getattr(machine, "step_states", None)
    if batched is not None:
        return batched(states, windows)
    parts = [machine.step(state, windows) for state in states]
    return Steps(
        np.concatenate([p.digits for p in parts]),
        np.concatenate([p.leads for p in parts]),
        np.concatenate([p.states for p in parts]),
        [np.concatenate([p.distances[g] for p in parts]) for g in range(len(parts[0].distances))],
    )


def all_windows(k: int, tokens: int) -> IntArray:
    """Every window of k column tokens, (tokens^k, k), in lexicographic order of its columns."""
    return np.array(list(itertools.product(range(tokens), repeat=k)), dtype=np.int64).reshape(-1, k)


@dataclass(frozen=True)
class Counterexample:
    """
    The first failure in breadth-first order: kind "outputs" (D(q, w) wrong for phi(q)) or "state"
    (a next state of (q, w) whose target state isn't the target's after w); the windows from the
    start through w, the states before each, the target's state in to w, the labels expected and
    given, or the states, and the window after (the word telling the states apart, padded; or zeros).
    """

    kind: str
    windows: list[list[int]]
    codes: list[State]
    state_in: int
    expected: list[int]
    got: list[int]
    after: list[int]

    def columns(self) -> list[int]:
        """The counterexample's columns at alignment 0: its windows, then the window after."""
        return [t for window in [*self.windows, self.after] for t in window]


@dataclass(frozen=True)
class Margin:
    """The smallest margin of a kind, where it is (the state and window), and how many fall short of it."""

    smallest: float
    code: State | None
    window: list[int] | None
    short: int


@dataclass(frozen=True)
class FactorReadout:
    """
    What a factor holds (the factored-state workplan, D9): whether its code is a function of the
    target's state over the reachable states; the reached target states grouped by its codes (two
    states joined when one code holds both); and that partition's name from the target's
    coordinates (window_tasks.Coordinates: "A", "B", "A + 4B", ...), None when none fits.
    """

    function: bool
    partition: list[list[int]]
    name: str | None


@dataclass(frozen=True)
class Certificate:
    """certify's result for one network."""

    columns: int
    codes: list[int]  # each factor's code count
    start: State
    reachable: list[State]  # in breadth-first order, near ties' states among them
    phi: dict[State, int]  # each reachable state's target state, as first fixed
    wrong_outputs: int  # (state, window) pairs whose labels are wrong for the state's target state
    wrong_states: int  # (state, window) pairs with a next state whose target state is wrong
    wrong_ties: int  # of those, the pairs whose snapped next state is right: wrong only on a near tie
    counterexample: Counterexample | None
    snap: Margin  # short: the near ties, each factor's, branched
    lead: Margin  # short: the labels' near ties, each a failure
    minimised: list[list[State]]  # the minimised machine's states (over G), each the reachable states it merges
    factors: list[FactorReadout]  # what each factor holds (D9)
    evaluations: int
    seconds: float
    task: str
    case: dict[str, Any] | None  # the counterexample as the task's case (window_tasks' case_json)
    split_ties: int = 0  # (state, window) pairs a near tie branches into another minimised state than the snap's

    @property
    def correct(self) -> bool:
        return self.wrong_outputs == 0 and self.wrong_states == 0

    @property
    def certified(self) -> bool:
        return self.correct and self.lead.short == 0

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        # a state is a list of codes, so phi is a list of [state, target state] pairs
        value["phi"] = [[list(state), target] for state, target in self.phi.items()]
        case = value.pop("case")
        if self.counterexample is not None:
            value["counterexample"]["case"] = case
        value.update(correct=self.correct, certified=self.certified)
        return value


class _Smallest:
    """The running minimum of a margin, its place, and the count under the tolerance."""

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.value = np.inf
        self.code: State | None = None
        self.window: list[int] | None = None
        self.short = 0

    def add(self, values: FloatArray, code: State, windows: IntArray | None) -> None:
        self.short += int((values < self.tolerance).sum())
        i = int(np.argmin(values))
        if values[i] < self.value:
            self.value = float(values[i])
            self.code = code
            self.window = None if windows is None else [int(t) for t in windows[i]]

    def margin(self) -> Margin:
        return Margin(self.value, self.code, self.window, self.short)


def _state(row: npt.NDArray[np.integer[Any]]) -> State:
    return tuple(int(v) for v in row)


def _lookup(table: Mapping[State, int], rows: IntArray) -> IntArray:
    """Each row's value in table, -1 where it has none, through the rows' distinct states."""
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    values = np.array([table.get(_state(row), -1) for row in unique], dtype=np.int64)
    return values[inverse.reshape(-1)]


def next_states(states: IntArray, near: Sequence[npt.NDArray[np.bool_]]) -> tuple[IntArray, IntArray]:
    """
    Every window's next states (windows (M,), states (M, G)): its snapped state, then, where a
    factor's snap is a near tie, every other state of its factors' snapped and near-tied codes.
    """
    windows, rows = [np.arange(len(states))], [states]
    tied = np.flatnonzero(np.any([n.any(axis=1) for n in near], axis=0))
    for w in tied:
        options = [[int(states[w, g]), *(int(c) for c in np.flatnonzero(n[w]))] for g, n in enumerate(near)]
        others = list(itertools.product(*options))[1:]  # the first is the snapped state
        windows.append(np.full(len(others), w, dtype=np.int64))
        rows.append(np.array(others, dtype=np.int64).reshape(-1, states.shape[1]))
    return np.concatenate(windows), np.concatenate(rows)


def minimise(
    reachable: Sequence[State], D: Mapping[State, npt.NDArray[np.int8]], G: Mapping[State, npt.NDArray[np.int16]]
) -> list[list[State]]:
    """
    The coarsest partition of the reachable states in which equivalent states give the same digits
    and move to equivalent states on every window (Moore's refinement).
    """
    following = {q: np.unique(G[q], axis=0, return_inverse=True) for q in reachable}
    block = {q: D[q].tobytes() for q in reachable}
    while True:
        ids = {key: i for i, key in enumerate(dict.fromkeys(block.values()))}
        number = {q: ids[block[q]] for q in reachable}
        refined: dict[State, bytes] = {}
        for q in reachable:
            unique, inverse = following[q]
            numbers = np.array([number[_state(row)] for row in unique], dtype=np.int64)
            refined[q] = block[q] + numbers[inverse.reshape(-1)].tobytes()
        if len(set(refined.values())) == len(ids):
            break
        block = refined
    classes: dict[bytes, list[State]] = {}
    for q in sorted(reachable):
        classes.setdefault(block[q], []).append(q)
    return sorted(classes.values())


def _joined(states: Sequence[int], groups: Iterable[set[int]]) -> list[list[int]]:
    """The states, those in one group joined, as connected components: sorted blocks."""
    parent = {s: s for s in states}

    def find(s: int) -> int:
        while parent[s] != s:
            s = parent[s]
        return s

    for group in groups:
        first, *rest = sorted(group)
        for s in rest:
            parent[find(s)] = find(first)
    blocks: dict[int, list[int]] = {}
    for s in states:
        blocks.setdefault(find(s), []).append(s)
    return sorted(blocks.values())


def readout(reachable: Sequence[State], phi: Mapping[State, int], target: wt.Mealy) -> list[FactorReadout]:
    """Each factor's readout (D9) over the reachable states."""
    readouts: list[FactorReadout] = []
    for g in range(len(reachable[0])):
        by_state: dict[int, set[int]] = {}
        by_code: dict[int, set[int]] = {}
        for q in reachable:
            by_state.setdefault(phi[q], set()).add(q[g])
            by_code.setdefault(q[g], set()).add(phi[q])
        partition = _joined(sorted(by_state), by_code.values())
        function = all(len(codes) == 1 for codes in by_state.values())
        readouts.append(FactorReadout(function, partition, name_partition(partition, target.coordinates)))
    return readouts


def name_partition(partition: Sequence[Sequence[int]], coordinates: wt.Coordinates | None) -> str | None:
    """
    The linear form over the target's coordinates (mod their modulus) whose values group the
    partition's states as it does, its first coefficient 1, the forms in order (A, then B, then
    A + B, A + 2B, ...); None without coordinates, when no form fits, and when a fit says nothing:
    one block (some form is constant on every state: a permutation's positions sum to the same
    value), or a partition of fewer than half the target's states (where many forms fit by chance:
    a network on 1-5 of 24 states).
    """
    if coordinates is None or len(partition) < 2:
        return None
    states = sorted(s for block in partition for s in block)
    if 2 * len(states) < len(coordinates.values):
        return None
    want = sorted(sorted(block) for block in partition)
    m = coordinates.modulus
    forms = itertools.product(range(m), repeat=len(coordinates.names))
    # the forms of fewer terms first, then by their coefficients: A, B, A + B, A + 2B, ...
    for coefficients in sorted(forms, key=lambda c: (sum(1 for x in c if x), [not x for x in c], c)):
        nonzero = [x for x in coefficients if x]
        if not nonzero or nonzero[0] != 1:
            continue
        groups: dict[int, list[int]] = {}
        for s in states:
            value = sum(int(c) * int(v) for c, v in zip(coefficients, coordinates.values[s], strict=True)) % m
            groups.setdefault(value, []).append(s)
        if sorted(groups.values()) == want:
            terms = [
                name if c == 1 else f"{c}{name}" for c, name in zip(coefficients, coordinates.names, strict=True) if c
            ]
            return " + ".join(terms)
    return None


def certify(
    machine: WindowMachine,
    task: wt.Task = wt.ADDITION,
    snap_ratio: float = SNAP_RATIO,
    logit_lead: float = LOGIT_LEAD,
) -> Certificate:
    """The certificate of a machine against its task's (the module's docstring)."""
    begun = time.perf_counter()
    k = machine.columns
    target = task.machine
    windows = all_windows(k, target.tokens)
    want_labels, want_state = target.windows(windows)
    snap, lead = _Smallest(snap_ratio), _Smallest(logit_lead)
    start, start_distances = machine.start()
    near_start: list[npt.NDArray[np.bool_]] = []
    for g, distances in enumerate(start_distances):
        ratios, near = snap_margins(np.array([start[g]]), distances[np.newaxis, :], snap_ratio)
        snap.add(ratios, start, None)
        near_start.append(near)
    _, start_rows = next_states(np.array([start], dtype=np.int64), near_start)
    starts = [_state(row) for row in start_rows]
    phi = dict.fromkeys(starts, target.start)
    parent: dict[State, tuple[State, int] | None] = dict.fromkeys(starts)
    queue = list(starts)
    D: dict[State, npt.NDArray[np.int8]] = {}
    G: dict[State, npt.NDArray[np.int16]] = {}
    wrong_outputs = wrong_states = wrong_ties = 0
    branches: list[tuple[State, IntArray, IntArray]] = []  # each state's near ties' windows and next states
    first: tuple[str, State, int, State | None] | None = None  # kind, state, window, the next state that failed
    for q in queue:  # breadth first: the queue grows as states are reached
        parts = [machine.step(q, windows[i : i + BATCH]) for i in range(0, len(windows), BATCH)]
        # narrow: at k 4, 531,441 windows a state
        D[q] = np.concatenate([p.digits for p in parts]).astype(np.int8)
        G[q] = np.concatenate([p.states for p in parts]).astype(np.int16)
        nears: list[list[npt.NDArray[np.bool_]]] = []
        for i, p in enumerate(parts):
            batch = windows[i * BATCH : (i + 1) * BATCH]
            factor_near: list[npt.NDArray[np.bool_]] = []
            for g, distances in enumerate(p.distances):
                ratios, near = snap_margins(p.states[:, g], distances, snap_ratio)
                snap.add(ratios, q, batch)
                factor_near.append(near)
            lead.add(p.leads.min(axis=1), q, batch)
            nears.append(factor_near)
        near_by_factor = [np.concatenate([n[g] for n in nears]) for g in range(G[q].shape[1])]
        # every next state, (window, state), the snapped ones then the near ties', in window order
        next_windows, next_rows = next_states(G[q].astype(np.int64), near_by_factor)
        state = phi[q]
        order = np.lexsort((*(next_rows[:, g] for g in reversed(range(next_rows.shape[1]))), next_windows))
        _, firsts = np.unique(next_rows[order], axis=0, return_index=True)  # each next state's first window
        for j in order[np.sort(firsts)]:
            w, r = int(next_windows[j]), _state(next_rows[j])
            if r not in phi:
                phi[r], parent[r] = int(want_state[state][w]), (q, w)
                queue.append(r)
        bad_outputs = np.flatnonzero((D[q] != want_labels[state]).any(axis=1))
        wrong_next = _lookup(phi, next_rows) != want_state[state][next_windows]
        bad_states = np.unique(next_windows[wrong_next])
        wrong_outputs += len(bad_outputs)
        wrong_states += len(bad_states)
        wrong_ties += len(bad_states) - int(wrong_next[: len(windows)].sum())
        if len(next_windows) > len(windows):
            branches.append((q, next_windows[len(windows) :], next_rows[len(windows) :]))
        if first is None and (len(bad_outputs) or len(bad_states)):
            d = int(bad_outputs[0]) if len(bad_outputs) else len(windows)
            c = int(bad_states[0]) if len(bad_states) else len(windows)
            if d <= c:
                first = ("outputs", q, d, None)
            else:
                failed = next_rows[wrong_next & (next_windows == c)]
                first = ("state", q, c, _state(failed[0]))
    counterexample = None
    if first is not None:
        kind, q, w, r = first
        path: list[tuple[State, int]] = [(q, w)]
        while (step := parent[path[-1][0]]) is not None:
            path.append(step)
        path.reverse()
        state = phi[q]
        after = [0] * k
        if r is not None:
            word = target.distinguishing(int(want_state[state][w]), phi[r])
            after = word + [0] * (-len(word) % k)
        counterexample = Counterexample(
            kind,
            [[int(t) for t in windows[i]] for _, i in path],
            [code for code, _ in path],
            state,
            [int(x) for x in want_labels[state][w]] if kind == "outputs" else [int(want_state[state][w])],
            [int(x) for x in D[q][w]] if r is None else [phi[r]],
            after,
        )
    minimised = minimise(queue, D, G)
    return Certificate(
        k,
        list(machine.codes),
        start,
        list(queue),
        dict(phi),
        wrong_outputs,
        wrong_states,
        wrong_ties,
        counterexample,
        snap.margin(),
        lead.margin(),
        minimised,
        readout(queue, phi, target),
        len(queue) * len(windows),
        time.perf_counter() - begun,
        task.name,
        None if counterexample is None else task.case_json(counterexample.columns()),
        split_ties(minimised, G, branches),
    )


def split_ties(
    minimised: Sequence[Sequence[State]],
    G: Mapping[State, npt.NDArray[np.int16]],
    branches: Sequence[tuple[State, IntArray, IntArray]],
) -> int:
    """
    The (state, window) pairs whose near ties branch into another minimised state than the snapped next
    state's: the machine's own nondeterminism, read without the target (the state formation workplan's
    D59). A tie between equivalent codes splits nothing; one that splits is a wrong state for some
    snap, whatever the target.
    """
    block = {q: i for i, states in enumerate(minimised) for q in states}
    split: set[tuple[State, int]] = set()
    for q, windows, rows in branches:
        for w, row in zip(windows, rows, strict=True):
            if block[_state(row)] != block[_state(G[q][w])]:
                split.add((q, int(w)))
    return len(split)


def certify_file(path: Path, task_name: str = wt.ADDITION.name, snap_ratio: float = SNAP_RATIO) -> dict[str, Any]:
    """
    A saved network's certificate against its task's machine as JSON, with the file's path, sha256
    and snap ratio (1.0: strict, the network's own snaps only, no near tie branched).
    """
    network = load_network(str(path))
    certificate = certify(NetworkMachine(network), wt.task_named(task_name), snap_ratio)
    return {
        "network": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "snap_ratio": snap_ratio,
        **certificate.to_json(),
    }


def summary(result: dict[str, Any]) -> str:
    """One line of a certificate's JSON: the verdict, the codes, the margins, the counterexample."""
    verdict = "certified" if result["certified"] else "correct, a label's near tie" if result["correct"] else "fails"
    line = (
        f"{verdict}: {len(result['reachable'])} codes reached, minimised to {len(result['minimised'])}, "
        f"{result['wrong_outputs']} wrong outputs, {result['wrong_states']} wrong states "
        f"({result['wrong_ties']} only on a near tie); "
        f"snap ratio >= {result['snap']['smallest']:.3f} ({result['snap']['short']} near ties), "
        f"logit lead >= {result['lead']['smallest']:.3g}; {result['seconds']:.0f} s"
    )
    counterexample = result["counterexample"]
    if counterexample is not None:
        case: dict[str, Any] = counterexample["case"]
        line += (
            f"; e.g. {counterexample['kind']} after {len(counterexample['windows'])} windows "
            f"(state in {counterexample['state_in']}: expected {counterexample['expected']}, got "
            f"{counterexample['got']}), the case {', '.join(f'{key} {value}' for key, value in case.items())}"
        )
    return line
