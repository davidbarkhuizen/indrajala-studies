"""
Staggered factors (the factored-state workplan, D14; indrajala_studies/window_recurrent_study.py):
the turns, the optimizer's rate per factor on a real factored network, and a step's rotation.
"""

import contextlib
import dataclasses
import json
import random
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt
from tests.helpers import TINY

DIVISION = wt.ModularDivision(5)
THREE = ((4, 8), (4, 8), (4, 8))


def spec(stagger: float | None = 0.1, seed: int = 0) -> wr.RunSpec:
    return wr.RunSpec(
        "curriculum", 1, 12, None, seed, DIVISION.name, factors=THREE, reads="before_state", stagger=stagger
    )


def network() -> Any:
    s = spec()
    return wr.build(DIVISION, 1, s.s, s.m, 0, recurrence=s.recurrence)


def weights(net: Any, indices: Sequence[int]) -> list[np.ndarray]:
    stepped = [*net.layers, *net.parts]
    trained = [stepped[i] for i in indices if hasattr(stepped[i], "parameters")]  # a fork or a norm-less layer has none
    return [np.array(p.tolist(), dtype=float) for layer in trained for p in layer.parameters()]


def batch() -> list[Any]:
    rng = random.Random(0)
    return wr.training_examples(DIVISION, [3], 1, 16, rng)  # one window count: one width


def test_turns_rotate_and_rate_each_factor():
    turns = wr.Turns()
    assert turns.rates(3, 0.1) == [1.0, 0.0, 0.0]
    turns = turns.next(3)
    assert turns == wr.Turns(1, (0, 1)) and turns.rates(3, 0.1) == [0.1, 1.0, 0.0]
    turns = turns.next(3).next(3)
    assert turns == wr.Turns(0, (0, 1, 2)) and turns.rates(3, 0.1) == [1.0, 0.1, 0.1]


def test_a_staggered_key_and_label_name_the_fraction():
    assert spec().key.endswith("-before-state-stagger0.1-seed0")
    assert spec(None).key.endswith("-before-state-seed0")
    assert spec().label == "curriculum (before state, stagger 0.1)"


def test_each_factors_parts_are_its_own_indices():
    net = network()
    indices = wr.factor_indices(net)
    assert len(indices) == 3 and all(len(i) == 4 for i in indices)  # pooling, cell, codes, initial state
    assert indices[0][0] == len(net.layers) and indices[2][-1] == len(net.layers) + len(net.parts) - 1


def test_a_frozen_factor_keeps_its_weights_and_the_others_learn():
    net = network()
    indices = wr.factor_indices(net)
    before = [weights(net, i) for i in indices]
    window = weights(net, range(len(net.layers)))
    with wr.factor_rates(net, [1.0, 0.1, 0.0]):
        net.learn_batch(0.01, batch())
    after = [weights(net, i) for i in indices]
    assert all(np.array_equal(a, b) for a, b in zip(before[2], after[2], strict=True))  # frozen: bit for bit
    assert any(not np.array_equal(a, b) for a, b in zip(before[0], after[0], strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(before[1], after[1], strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(window, weights(net, range(len(net.layers))), strict=True))
    assert not isinstance(net.optimizer, wr.FactorRates)  # restored after the block


def test_every_factor_at_the_rate_trains_as_the_network_alone():
    plain, wrapped = network(), network()
    plain.learn_batch(0.01, batch())
    with wr.factor_rates(wrapped, [1.0, 1.0, 1.0]):
        wrapped.learn_batch(0.01, batch())
    everything = range(len(plain.layers) + len(plain.parts))
    assert all(
        np.array_equal(a, b) for a, b in zip(weights(plain, everything), weights(wrapped, everything), strict=True)
    )


class NeverLearns:
    """A stand-in for run_step that answers 0 everywhere, so every turn plateaus."""

    class _Backend:
        def matrix(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
            return [list(row) for row in rows]

    backend = _Backend()

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> list[tuple[float, ...]]:
        return [tokens for tokens, _labels in rows]

    def classify_rows(self, prepared: Sequence[tuple[float, ...]]) -> list[tuple[int, ...]]:
        return [tuple(0 for _ in tokens) for tokens in prepared]

    def state_codes(self, X: Sequence[Sequence[float]]) -> list[list[tuple[int, ...]]]:
        return [[(0, 0, 0)] * (len(row) + 1) for row in X]

    def learn_batch(self, rate: float, batch: list[Any]) -> None:
        pass


def test_a_staggered_step_turns_to_each_factor_at_its_plateau_then_ends(monkeypatch: pytest.MonkeyPatch):
    rates: list[list[float]] = []

    @contextlib.contextmanager
    def record(net: Any, fractions: Sequence[float]) -> Generator[None]:
        rates.append(list(fractions))
        yield

    monkeypatch.setattr(wr, "factor_rates", record)
    settings = wr.Settings(
        epoch_examples=16,
        max_epochs=50,
        patience=2,
        screen=10,
        test_lengths=(3,),
        sampled=20,
        relation_sampled=5,
        per_stratum=2,
        codes_cases=5,
    )
    result = wr.run_step(NeverLearns(), spec(), 1, [1], settings)
    assert result["stop"] == "plateau" and result["epochs"] == 9
    assert result["turns"] == [[0, 0], [3, 1], [6, 2]]
    assert result["active"] == 0 and result["started"] == [0, 1, 2]  # the next step starts with factor 0 again
    assert rates == [[1.0, 0.0, 0.0]] * 3 + [[0.1, 1.0, 0.0]] * 3 + [[0.1, 0.1, 1.0]] * 3
    carried = wr.run_step(NeverLearns(), spec(), 2, [1, 2], settings, turns=wr.Turns(1, (0, 1)))
    assert carried["turns"][0] == [0, 1] and carried["active"] == 1


def test_a_staggered_run_carries_its_turns_and_resumes_them(tmp_path: Path):
    settings = dataclasses.replace(TINY, max_epochs=2)
    study = wr.Study(tmp_path, settings, [spec()], 0)
    result = wr.run(study, log=lambda line: None)[spec().key]
    assert result["stagger"] == 0.1 and all("turns" in step for step in result["steps"])
    first, second = result["steps"]
    assert second["turns"][0] == [0, first["active"]]
    # resumed after step 1: step 2 starts from step 1's saved turns
    (tmp_path / "results" / f"{spec().key}.json").unlink()
    (tmp_path / "models" / spec().key / "step-2.json").unlink()
    rerun = wr.run(study, log=lambda line: None)[spec().key]
    assert rerun["steps"][1]["turns"] == second["turns"]
    assert json.loads((tmp_path / "models" / spec().key / "step-1.json").read_text())["started"] == first["started"]


def test_a_config_staggers_a_factored_cell_and_numbers_its_seeds_from_first_seed(tmp_path: Path):
    config = tmp_path / "study.json"
    cell = {"k": [1], "factors": [[16, 16]] * 3, "reads": "before_state", "stagger": 0.1}
    config.write_text(
        json.dumps({"directory": "x", "task": DIVISION.name, "seeds": 3, "first_seed": 6, "matrix": [cell]})
    )
    study = wr.load_config(config)
    assert [r.seed for r in study.runs] == [6, 7, 8] and all(r.stagger == 0.1 for r in study.runs)
    config.write_text(json.dumps({"directory": "x", "matrix": [{"k": [1], "s": [4], "m": [3], "stagger": 0.1}]}))
    with pytest.raises(AssertionError, match="staggered cell is factored"):
        wr.load_config(config)
