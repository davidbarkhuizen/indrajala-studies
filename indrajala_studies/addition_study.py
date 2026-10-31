"""
The addition study's candidates and harness (the addition study workplan's Stage 3, retired:
git show 8990f74:docs/addition-study-workplan.md; scripts/studies/addition_study.py).

A candidate is an encoder, a network spec with its size knobs, and a decoder: build() gives a Model
that trains an epoch on a batch of triples and adds (an Adder, as addition_data's catalogue
verifies). A candidate's sizes are shapes (the knobs held fixed in one search: layers and heads, or
depth) times a ladder of rungs (the knob searched: d, or width).

A run is one candidate, n, rate, shape, rung and seed. It trains on a fresh training mixture each
epoch (D7), screens the catalogue each epoch on a spread of its cases, and stops at the first epoch
whose screen and then whole catalogue pass, at a plateau (no more screen cases passed for `patience` epochs) or at the
budget; then the whole catalogue runs once. Each run writes its result file atomically and saves
its trained network(s) in format 2 with a manifest, so any run's model reloads as an adder
(load_model) and can be checked again.

A study (one entry of a config) is a candidate at one n over rates, shapes and rungs: "grid" runs
every rung, "bisect" (D6) bisects each shape's ladder on success (4 of 5 seeds passing every
property, D9) and then runs the rungs either side of the threshold. run() schedules every study's
runs over worker processes, choosing the next rungs from the results so far, and skips every run
whose result is on disk: a sweep that stops resumes where it stopped. A run that raises is recorded
with its traceback and run once more.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import math
import multiprocessing
import os
import random
import socket
import subprocess
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from indrajala_ml.data import addition_data as ad
from indrajala_ml.data.addition_data import BASE, Triple
from indrajala_ml.model.layers.array.array_backend import RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerNorm, LayerSpec, Position, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies.common import crate_hash, table

PASSING_SEEDS = 4  # D9: a size succeeds when this many of its seeds pass every property
RETRIES = 1

Size = dict[str, int]


class Model(Protocol):
    def train_epoch(self, triples: Sequence[Triple], rate: float, batch_size: int, rng: random.Random) -> None: ...

    def add(self, triples: Sequence[Triple]) -> list[int]: ...

    def parameter_count(self) -> int: ...

    def save(self, directory: Path) -> None: ...


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    shapes: tuple[Mapping[str, int], ...]
    rung: str  # the knob a search moves along
    rungs: tuple[int, ...]
    build: Callable[[int, Size, int], Model]  # (n, size, seed)
    load: Callable[[int, Size, Path], Model]  # (n, size, directory)

    def sizes(self, shape: Mapping[str, int]) -> list[Size]:
        return [{**shape, self.rung: r} for r in self.rungs if _fits(shape, self.rung, r)]


def _fits(shape: Mapping[str, int], rung: str, value: int) -> bool:
    # a token width splits evenly over the heads
    return rung != "d" or value % shape.get("heads", 1) == 0


# encodings


MULTISETS = sorted(itertools.combinations_with_replacement(range(BASE), 3))  # 10
_MULTISET_IDS: dict[tuple[int, ...], int] = {m: i for i, m in enumerate(MULTISETS)}
PLUS, EQUALS = BASE, BASE + 1  # the string format's two symbols beside the digits


def column_tokens(triple: Triple, n: int, ordered: bool = True) -> tuple[int, ...]:
    """One token a column, least significant first, n + 1 of them (the last column's digits 0)."""
    a, b, c = (ad.digits(x, n) + (0,) for x in triple)
    if ordered:
        return tuple(BASE * BASE * a[t] + BASE * b[t] + c[t] for t in range(n + 1))
    return tuple(_MULTISET_IDS[tuple(sorted((a[t], b[t], c[t])))] for t in range(n + 1))


def string_tokens(triple: Triple, n: int) -> tuple[int, ...]:
    """a+b+c= then the sum, every number least significant digit first: 4n + 4 tokens."""
    a, b, c = (ad.digits(x, n) for x in triple)
    return (*a, PLUS, *b, PLUS, *c, EQUALS, *ad.digits(sum(triple), n + 1))


def answer_start(n: int) -> int:
    """The label index of the sum's first digit in the string format (label j is token j + 1)."""
    return 3 * n + 2


def scaled_digits(triple: Triple, n: int) -> tuple[float, ...]:
    return tuple(d / (BASE - 1) for x in triple for d in ad.digits(x, n))


# models


NetworkShape = Literal["multiclass", "single_output", "sequence"]


def _network(input_shape: tuple[int], specs: list[LayerSpec], shape: NetworkShape, seed: int) -> Any:
    network = SequentialArrayNetwork(input_shape, specs, Adam(), shape, RUST)
    network.rng = RUST.default_rng(seed)
    network.randomize()
    return network


def _parameters(state: Any) -> int:
    if hasattr(state, "shape"):
        return math.prod(state.shape)
    if state is None or isinstance(state, int | float):
        return 0
    return sum(_parameters(part) for part in state)


class SequenceModel:
    """A sequence network over column tokens (candidates 1-4) or the string format (5)."""

    def __init__(self, network: Any, n: int, encoding: str) -> None:
        self.network, self.n, self.encoding = network, n, encoding

    def _example(self, triple: Triple) -> tuple[tuple[float, ...], tuple[int, ...]]:
        n = self.n
        if self.encoding == "string":
            tokens = string_tokens(triple, n)
            return tuple(float(t) for t in tokens[:-1]), tokens[1:]
        state = column_tokens(triple, n, ordered=self.encoding == "ordered")
        return tuple(float(t) for t in state), ad.digits(sum(triple), n + 1)

    def train_epoch(self, triples: Sequence[Triple], rate: float, batch_size: int, rng: random.Random) -> None:
        bss.train_epoch(self.network, [self._example(t) for t in triples], batch_size, rate, 0, rng)

    def add(self, triples: Sequence[Triple]) -> list[int]:
        if not triples:
            return []
        examples = [self._example(t) for t in triples]
        if self.encoding != "string":
            predictions = self.network.classify_rows(self.network.prepare_dataset(examples))
            return [ad.from_digits(p) for p in predictions]
        # greedy decoding: each digit predicted from the digits predicted before it
        states = [list(state) for state, _labels in examples]
        start = answer_start(self.n)
        answers: list[list[int]] = [[] for _ in triples]
        for k in range(self.n + 1):
            rows = [(tuple(state), labels) for state, (_s, labels) in zip(states, examples, strict=True)]
            predictions = self.network.classify_rows(self.network.prepare_dataset(rows))
            for state, answer, prediction in zip(states, answers, predictions, strict=True):
                answer.append(prediction[start + k])
                if start + k + 1 < len(state):
                    state[start + k + 1] = float(prediction[start + k])
        return [ad.from_digits(answer) for answer in answers]

    def parameter_count(self) -> int:
        return _parameters(self.network.snapshot())

    def save(self, directory: Path) -> None:
        self.network.save(str(directory / "network.json"))


class PerDigitModel:
    """Candidate 6: one multiclass network per sum digit, over the operands' scaled digits."""

    def __init__(self, networks: list[Any], n: int) -> None:
        self.networks, self.n = networks, n

    def train_epoch(self, triples: Sequence[Triple], rate: float, batch_size: int, rng: random.Random) -> None:
        states = [scaled_digits(t, self.n) for t in triples]
        sums = [ad.digits(sum(t), self.n + 1) for t in triples]
        for i, network in enumerate(self.networks):
            examples = [(state, s[i]) for state, s in zip(states, sums, strict=True)]
            bss.train_epoch(network, examples, batch_size, rate, 0, random.Random(rng.random()))

    def add(self, triples: Sequence[Triple]) -> list[int]:
        if not triples:
            return []
        rows = [(scaled_digits(t, self.n), 0) for t in triples]
        columns = [network.classify_rows(network.prepare_dataset(rows)) for network in self.networks]
        return [ad.from_digits(ds) for ds in zip(*columns, strict=True)]

    def parameter_count(self) -> int:
        return sum(_parameters(network.snapshot()) for network in self.networks)

    def save(self, directory: Path) -> None:
        for i, network in enumerate(self.networks):
            network.save(str(directory / f"digit-{i}.json"))


class RegressionModel:
    """Candidates 7 and 8: one sigmoid output, the sum scaled to [0, 1], decoded by rounding."""

    def __init__(self, network: Any, n: int, inputs: str) -> None:
        self.network, self.n, self.inputs = network, n, inputs
        self.top = BASE ** (n + 1) - 1

    def _state(self, triple: Triple) -> tuple[float, ...]:
        if self.inputs == "digits":
            return scaled_digits(triple, self.n)
        return tuple(x / (BASE**self.n - 1) for x in triple)

    def train_epoch(self, triples: Sequence[Triple], rate: float, batch_size: int, rng: random.Random) -> None:
        examples = [(self._state(t), sum(t) / self.top) for t in triples]
        bss.train_epoch(self.network, examples, batch_size, rate, 0, rng)

    def add(self, triples: Sequence[Triple]) -> list[int]:
        if not triples:
            return []
        prepared = self.network.prepare_dataset([(self._state(t), 0.0) for t in triples])
        outputs = [row[0] for batch in self.network.forward_rows(prepared) for row in batch.tolist()]
        return [min(self.top, max(0, round(y * self.top))) for y in outputs]

    def parameter_count(self) -> int:
        return _parameters(self.network.snapshot())

    def save(self, directory: Path) -> None:
        self.network.save(str(directory / "network.json"))


# the candidates' specs


def _transformer_specs(vocabulary: int, size: Size, causal: bool | None) -> list[LayerSpec]:
    """causal None: FFN blocks only (candidate 4)."""
    d = size["d"]
    ffn = Residual((LayerNorm(), Dense(2 * d, activation="relu"), Dense(d, activation="linear", bias=True)))
    blocks: list[LayerSpec] = []
    for _ in range(size["layers"]):
        if causal is not None:
            blocks.append(Residual((LayerNorm(), Attention(heads=size["heads"], causal=causal))))
        blocks.append(ffn)
    return [
        Embedding(vocabulary, d),
        Position(),
        *blocks,
        LayerNorm(),
        Dense(BASE, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _dense_specs(size: Size, output: Dense) -> list[LayerSpec]:
    return [*(Dense(size["width"], activation="relu") for _ in range(size["depth"])), output]


def _sequence_candidate(name: str, description: str, encoding: str, causal: bool | None) -> Candidate:
    vocabulary = {"ordered": BASE**3, "unordered": len(MULTISETS), "string": BASE + 2}[encoding]

    def tokens(n: int) -> int:
        return 4 * n + 3 if encoding == "string" else n + 1

    def build(n: int, size: Size, seed: int) -> Model:
        specs = _transformer_specs(vocabulary, size, causal)
        return SequenceModel(_network((tokens(n),), specs, "sequence", seed), n, encoding)

    def load(n: int, size: Size, directory: Path) -> Model:
        return SequenceModel(load_network(str(directory / "network.json")), n, encoding)

    if causal is None:
        shapes: tuple[Mapping[str, int], ...] = ({"layers": 1, "heads": 1}, {"layers": 2, "heads": 1})
        rungs = (8, 16, 32, 64)
    else:
        shapes = tuple({"layers": layers, "heads": heads} for layers in (1, 2, 3) for heads in (1, 2))
        rungs = (4, 6, 8, 12, 16, 24, 32)
    return Candidate(name, description, shapes, "d", rungs, build, load)


def _per_digit_build(n: int, size: Size, seed: int) -> Model:
    output = Dense(BASE, output=True, activation="softmax", loss="cross_entropy")
    networks = [_network((3 * n,), _dense_specs(size, output), "multiclass", seed * 1000 + i) for i in range(n + 1)]
    return PerDigitModel(networks, n)


def _per_digit_load(n: int, size: Size, directory: Path) -> Model:
    return PerDigitModel([load_network(str(directory / f"digit-{i}.json")) for i in range(n + 1)], n)


def _regression_candidate(name: str, description: str, inputs: str) -> Candidate:
    def build(n: int, size: Size, seed: int) -> Model:
        width = 3 * n if inputs == "digits" else 3
        return RegressionModel(
            _network((width,), _dense_specs(size, Dense(1, output=True)), "single_output", seed), n, inputs
        )

    def load(n: int, size: Size, directory: Path) -> Model:
        return RegressionModel(load_network(str(directory / "network.json")), n, inputs)

    shapes = ({"depth": 1}, {"depth": 2})
    return Candidate(name, description, shapes, "width", (8, 16, 32, 64, 128), build, load)


CANDIDATES: dict[str, Candidate] = {
    c.name: c
    for c in (
        _sequence_candidate("column-causal", "1: column-aligned causal transformer", "ordered", True),
        _sequence_candidate("column-unordered", "2: 1 with unordered column tokens", "unordered", True),
        _sequence_candidate("column-noncausal", "3: 1 without the causal mask", "ordered", False),
        _sequence_candidate("column-ffn", "4: FFN blocks only (control)", "ordered", None),
        _sequence_candidate("string-causal", "5: string-format causal transformer", "string", True),
        Candidate(
            "dense-per-digit",
            "6: one dense network per sum digit",
            ({"depth": 1}, {"depth": 2}),
            "width",
            (2, 4, 8, 16, 32, 64),
            _per_digit_build,
            _per_digit_load,
        ),
        _regression_candidate("regression-digits", "7: dense regression, digits in", "digits"),
        _regression_candidate("regression-scalars", "8: dense regression, scalars in (control)", "scalars"),
    )
}


# runs


@dataclass(frozen=True)
class Settings:
    """A sweep's training and screening settings (D7, its budget and plateau rule from Stage 4's
    calibration), the same for every study in it."""

    epoch_examples: int = 2**16
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 30
    screen: int = 500
    seeds: int = 5


@dataclass(frozen=True)
class RunSpec:
    candidate: str
    n: int
    rate: float
    size: tuple[tuple[str, int], ...]  # sorted (knob, value) pairs, hashable
    seed: int

    @property
    def key(self) -> str:
        knobs = "-".join(f"{k}{v}" for k, v in self.size)
        return f"{self.candidate}-n{self.n}-r{self.rate:g}-{knobs}-s{self.seed}"


def run_spec(candidate: str, n: int, rate: float, size: Mapping[str, int], seed: int) -> RunSpec:
    return RunSpec(candidate, n, rate, tuple(sorted(size.items())), seed)


def _provenance() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {"commit": commit, "crate": crate_hash(), "machine": socket.gethostname()}


def _result_json(result: ad.PropertyResult) -> dict[str, Any]:
    return {
        "cases": result.cases,
        "passed": result.passed,
        "strata": {str(k): list(v) for k, v in sorted(result.strata.items())},
        "failures": [{"triple": list(c.triple), "column": c.column} for c in result.failures],
    }


def train_and_evaluate(
    spec: RunSpec, settings: Settings, candidates: Mapping[str, Candidate], models: Path | None
) -> dict[str, Any]:
    """One run, as the module docstring says; its model saved under models/<key>/ when models."""
    candidate = candidates[spec.candidate]
    size = dict(spec.size)
    start = time.perf_counter()
    model = candidate.build(spec.n, size, spec.seed)
    screens: list[float] = []
    best, since_best, stop = -1, 0, "budget"
    epochs = 0
    for epoch in range(settings.max_epochs):
        rng = random.Random(f"addition-train-{spec.seed}-{epoch}")
        model.train_epoch(
            ad.training_triples(spec.n, settings.epoch_examples, rng), spec.rate, settings.batch_size, rng
        )
        epochs = epoch + 1
        screen = ad.evaluate(model.add, spec.n, limit=settings.screen, shrink=False)
        passed = sum(r.passed for r in screen)
        screens.append(passed / sum(r.cases for r in screen))
        # a passing screen triggers the whole catalogue: a spread of cases can miss a rare failure
        if all(r.holds for r in screen) and all(r.holds for r in ad.evaluate(model.add, spec.n, shrink=False)):
            stop = "passed"
            break
        if passed > best:
            best, since_best = passed, 0
        else:
            since_best += 1
            if since_best >= settings.patience:
                stop = "plateau"
                break
    train_seconds = time.perf_counter() - start
    results = ad.evaluate(model.add, spec.n)
    by_id = {r.id: r for r in results}
    if models is not None:
        directory = models / spec.key
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory)
        _write_json(directory / "manifest.json", {"candidate": spec.candidate, "n": spec.n, "size": size})
    return {
        "key": spec.key,
        "candidate": spec.candidate,
        "n": spec.n,
        "rate": spec.rate,
        "size": size,
        "seed": spec.seed,
        "parameters": model.parameter_count(),
        "epochs": epochs,
        "stop": stop,
        "screens": screens,
        "passes": all(r.holds for r in results),
        "reach": by_id["M3"].reach() if "M3" in by_id else None,
        "properties": {r.id: _result_json(r) for r in results},
        "train_seconds": train_seconds,
        "seconds": time.perf_counter() - start,
        **_provenance(),
    }


def load_model(directory: Path, candidates: Mapping[str, Candidate] | None = None) -> tuple[Model, int]:
    """A saved run's model and its n, from the manifest beside its network files."""
    manifest = json.loads((directory / "manifest.json").read_text())
    candidate = (candidates or CANDIDATES)[manifest["candidate"]]
    return candidate.load(manifest["n"], manifest["size"], directory), manifest["n"]


def _write_json(path: Path, value: Any) -> None:
    # atomic: a reader sees the old file or the whole new one
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    os.replace(temporary, path)


# the search


@dataclass
class Study:
    """One config entry: a candidate at one n, over rates, shapes and rungs."""

    candidate: Candidate
    n: int
    rates: list[float]
    search: str  # "bisect" or "grid"
    shapes: list[Mapping[str, int]]
    rungs: list[int] | None = None  # None: the candidate's ladder

    def ladders(self) -> Iterator[tuple[float, list[Size]]]:
        for rate in self.rates:
            for shape in self.shapes:
                sizes = self.candidate.sizes(shape)
                if self.rungs is not None:
                    sizes = [s for s in sizes if s[self.candidate.rung] in self.rungs]
                yield rate, sizes


@dataclass
class Rung:
    """A size's runs so far: None while undecided."""

    passed: int = 0
    finished: int = 0

    def succeeds(self, seeds: int) -> bool | None:
        if self.passed >= min(PASSING_SEEDS, seeds):
            return True
        if self.finished - self.passed > seeds - min(PASSING_SEEDS, seeds):
            return False
        return None if self.finished < seeds else self.passed >= min(PASSING_SEEDS, seeds)


def wanted(study: Study, settings: Settings, outcomes: Mapping[RunSpec, bool]) -> tuple[list[RunSpec], bool]:
    """
    The runs a study wants next given the outcomes so far (a run's key to whether it passed), and
    whether the study is finished. Grid: every run of every rung. Bisect: each ladder's middle
    undecided rung between the highest failing and lowest succeeding, then the rungs either side
    of the threshold.
    """
    runs: list[RunSpec] = []
    done = True
    for rate, sizes in study.ladders():

        def specs(size: Size, rate: float = rate) -> list[RunSpec]:
            return [run_spec(study.candidate.name, study.n, rate, size, seed) for seed in range(settings.seeds)]

        def verdict(size: Size) -> bool | None:
            rung = Rung()
            for spec in specs(size):
                if spec in outcomes:
                    rung.finished += 1
                    rung.passed += outcomes[spec]
            return rung.succeeds(settings.seeds)

        if study.search == "grid":
            needed = list(range(len(sizes)))
        else:
            low, high = -1, len(sizes)  # the highest known failure, the lowest known success
            needed = []
            while high - low > 1:
                middle = (low + high) // 2
                result = verdict(sizes[middle])
                if result is None:
                    needed = [middle]
                    break
                low, high = (low, middle) if result else (middle, high)
            if not needed:
                needed = [i for i in (high - 1, high, high + 1) if 0 <= i < len(sizes)]
        for i in needed:
            if verdict(sizes[i]) is None:
                done = False
                runs.extend(spec for spec in specs(sizes[i]) if spec not in outcomes)
    return runs, done


# the sweep


@dataclass
class Sweep:
    directory: Path
    settings: Settings
    studies: list[Study]
    workers: int
    candidates: Mapping[str, Candidate] = field(default_factory=lambda: CANDIDATES)

    @property
    def results(self) -> Path:
        return self.directory / "results"

    @property
    def models(self) -> Path:
        return self.directory / "models"


def load_config(path: Path, candidates: Mapping[str, Candidate] = CANDIDATES) -> Sweep:
    """
    A sweep's JSON config: "directory" (results/ and models/ under it), "workers", the Settings'
    fields, and "studies", each {"candidate", "n", "rates", "search" ("bisect" or "grid"),
    optionally "shapes" (else the candidate's) and "rungs" (else its ladder)}.
    """
    config = json.loads(path.read_text())
    settings = Settings(**{k: config[k] for k in Settings.__dataclass_fields__ if k in config})
    studies = [
        Study(
            candidates[entry["candidate"]],
            entry["n"],
            [float(r) for r in entry["rates"]],
            entry.get("search", "bisect"),
            entry.get("shapes") or list(candidates[entry["candidate"]].shapes),
            entry.get("rungs"),
        )
        for entry in config["studies"]
    ]
    return Sweep(Path(config["directory"]), settings, studies, config.get("workers", 1), candidates)


def read_results(results: Path) -> dict[str, dict[str, Any]]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(results.glob("*.json"))}


def _run_job(
    spec: RunSpec, directory: Path, settings: Settings, candidates: Mapping[str, Candidate] | None = None
) -> dict[str, Any]:
    """
    One run, its result written; a raise recorded as an error result. In a worker process
    candidates is None, CANDIDATES (a candidate's functions don't pickle).
    """
    results, models = directory / "results", directory / "models"
    marker = results / f"{spec.key}.running"
    marker.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    previous = results / f"{spec.key}.json"
    attempts = json.loads(previous.read_text()).get("attempts", 0) if previous.exists() else 0
    result: dict[str, Any]
    try:
        result = train_and_evaluate(spec, settings, candidates or CANDIDATES, models)
    except Exception:  # noqa: BLE001 - any run that raises is recorded, and run once more
        result = {"key": spec.key, "error": traceback.format_exc()}
    result["attempts"] = attempts + 1
    _write_json(previous, result)
    marker.unlink(missing_ok=True)
    return result


def run(sweep: Sweep, log: Callable[[str], None] = print) -> dict[str, dict[str, Any]]:
    """
    Every study's runs, the next chosen from the results so far, over sweep.workers processes (0:
    in this process); the results on disk are kept and their runs skipped.
    """
    sweep.results.mkdir(parents=True, exist_ok=True)
    sweep.models.mkdir(parents=True, exist_ok=True)
    for stale in sweep.results.glob("*.running"):
        stale.unlink()
    results = read_results(sweep.results)

    def outcomes() -> dict[RunSpec, bool]:
        known: dict[RunSpec, bool] = {}
        for study in sweep.studies:
            for rate, sizes in study.ladders():
                for size in sizes:
                    for seed in range(sweep.settings.seeds):
                        spec = run_spec(study.candidate.name, study.n, rate, size, seed)
                        result = results.get(spec.key)
                        if result is None:
                            continue
                        if "error" in result:
                            if result.get("attempts", 1) > RETRIES:
                                known[spec] = False  # failed twice: counts as a failing seed
                            continue
                        known[spec] = bool(result["passes"])
        return known

    def pending(running: set[RunSpec]) -> list[RunSpec]:
        known = outcomes()
        return [
            spec for study in sweep.studies for spec in wanted(study, sweep.settings, known)[0] if spec not in running
        ]

    def record(result: dict[str, Any]) -> None:
        results[result["key"]] = result
        if "error" in result:
            log(f"{result['key']}: error (attempt {result['attempts']})")
        else:
            verdict = "passes" if result["passes"] else "fails"
            log(
                f"{result['key']}: {verdict}, {result['parameters']} parameters, {result['epochs']} epochs, {result['stop']}"
            )

    if sweep.workers == 0:
        while todo := pending(set()):
            record(_run_job(todo[0], sweep.directory, sweep.settings, sweep.candidates))
        return results
    assert sweep.candidates is CANDIDATES, "worker processes run the module's candidates; workers=0 for others"
    context = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(sweep.workers, mp_context=context) as pool:
        running: dict[concurrent.futures.Future[dict[str, Any]], RunSpec] = {}
        while True:
            for spec in pending(set(running.values()))[: sweep.workers - len(running)]:
                running[pool.submit(_run_job, spec, sweep.directory, sweep.settings)] = spec
            if not running:
                return results
            finished, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                spec = running.pop(future)
                try:
                    record(future.result())
                except Exception:  # noqa: BLE001 - a worker that died: recorded, so the retry rule applies
                    error = {"key": spec.key, "error": traceback.format_exc(), "attempts": 1}
                    _write_json(sweep.results / f"{spec.key}.json", error)
                    record(error)


# status and report


def status(sweep: Sweep) -> str:
    results = read_results(sweep.results) if sweep.results.exists() else {}
    errors = [k for k, r in results.items() if "error" in r]
    running = sorted(p.stem for p in sweep.results.glob("*.running")) if sweep.results.exists() else []
    passed = sum(1 for r in results.values() if r.get("passes"))
    lines = [
        f"{len(results) - len(errors)} runs finished ({passed} passing), {len(errors)} errors, {len(running)} running",
        *(f"running: {key}" for key in running),
        *(f"error: {key}" for key in errors),
    ]
    return "\n".join(lines)


def report(sweep: Sweep) -> str:
    """Per study and ladder, each rung's seeds passing, parameters, epochs, L* and the failing
    properties; then each study's smallest succeeding size."""
    results = read_results(sweep.results)
    sections: list[str] = []
    thresholds: list[list[str]] = []
    seeds = sweep.settings.seeds
    for study in sweep.studies:
        name = study.candidate.name
        for rate, sizes in study.ladders():
            rows: list[list[str]] = []
            smallest: tuple[int, Size] | None = None
            for size in sizes:
                runs = [results.get(run_spec(name, study.n, rate, size, seed).key) for seed in range(seeds)]
                done = [r for r in runs if r is not None and "error" not in r]
                if not done:
                    continue
                passing = sum(r["passes"] for r in done)
                failing = sorted({p for r in done for p, v in r["properties"].items() if v["passed"] < v["cases"]})
                succeeds = passing >= min(PASSING_SEEDS, seeds)
                parameters = done[0]["parameters"]
                if succeeds and (smallest is None or parameters < smallest[0]):
                    smallest = (parameters, size)
                rows.append(
                    [
                        str(size[study.candidate.rung]),
                        str(parameters),
                        f"{passing}/{len(done)}" + ("" if len(done) == seeds else f" of {seeds}"),
                        "yes" if succeeds else "no",
                        f"{sum(r['epochs'] for r in done) / len(done):.1f}",
                        "/".join(str(r["reach"]) for r in done),
                        ", ".join(failing),
                    ]
                )
            shape = {k: v for k, v in sizes[0].items() if k != study.candidate.rung} if sizes else {}
            title = f"{name}, n = {study.n}, rate {rate:g}, {shape}"
            header = [study.candidate.rung, "parameters", "seeds passing", "succeeds", "epochs", "L*", "failing"]
            sections.append(f"{title}\n\n{table(header, rows)}" if rows else f"{title}\n\nno runs yet")
            thresholds.append(
                [name, str(study.n), f"{rate:g}", str(shape)]
                + (["none"] * 2 if smallest is None else [str(smallest[1][study.candidate.rung]), str(smallest[0])])
            )
    summary = table(["candidate", "n", "rate", "shape", "smallest rung", "parameters"], thresholds)
    return "\n\n".join([summary, *sections])
