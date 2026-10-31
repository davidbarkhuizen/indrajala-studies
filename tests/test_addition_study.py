"""The addition study's candidates and harness (indrajala_studies/addition_study.py)."""

import json
import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from indrajala_studies import addition_data as ad
from indrajala_studies import addition_study as st
from indrajala_studies.addition_data import Triple
from tests.addition_adders import correct, no_carry

# fast settings: few cases per epoch and per screen; n = 2 keeps every catalogue run small
SETTINGS = st.Settings(epoch_examples=64, max_epochs=2, patience=1, screen=50, seeds=5)
LADDER = (1, 2, 4, 8, 16, 32)
THRESHOLD = 8  # a fake model adds from this width up


class FakeModel:
    """A stand-in for a trained network: adds correctly from THRESHOLD up, carry-free below."""

    def __init__(self, n: int, width: int, seed: int, log: list[tuple[int, int]], raises: set[tuple[int, int]]) -> None:
        if (width, seed) in raises:
            raises.discard((width, seed))
            raise RuntimeError("a run that dies")
        log.append((width, seed))
        self.n, self.width = n, width
        self.adder = correct if width >= THRESHOLD else no_carry(n)

    def train_epoch(self, triples: Sequence[Triple], rate: float, batch_size: int, rng: random.Random) -> None:
        pass

    def add(self, triples: Sequence[Triple]) -> list[int]:
        return self.adder(triples)

    def parameter_count(self) -> int:
        return 10 * self.width

    def save(self, directory: Path) -> None:
        (directory / "fake.json").write_text(json.dumps({"width": self.width}))


def fake_candidates(log: list[tuple[int, int]], raises: set[tuple[int, int]] | None = None) -> dict[str, st.Candidate]:
    pending_raises: set[tuple[int, int]] = set() if raises is None else raises

    def build(n: int, size: st.Size, seed: int) -> st.Model:
        return FakeModel(n, size["width"], seed, log, pending_raises)

    def load(n: int, size: st.Size, directory: Path) -> st.Model:
        return FakeModel(n, json.loads((directory / "fake.json").read_text())["width"], 0, [], set())

    candidate = st.Candidate("fake", "a stand-in", ({"depth": 1},), "width", LADDER, build, load)
    return {"fake": candidate}


def sweep_of(tmp_path: Path, candidates: dict[str, st.Candidate], search: str = "bisect") -> st.Sweep:
    study = st.Study(candidates["fake"], 2, [0.01], search, [{"depth": 1}])
    return st.Sweep(tmp_path, SETTINGS, [study], 0, candidates)


# encodings


def test_column_tokens_are_least_significant_first_with_a_last_zero_column():
    # 5 = 12 (digits 2, 1), 1 = 01 (1, 0), 0
    assert st.column_tokens((5, 1, 0), 2) == (9 * 2 + 3 * 1 + 0, 9 * 1 + 0 + 0, 0)


def test_unordered_tokens_ignore_the_order_and_ordered_ones_dont():
    n = 3
    triple = (5, 17, 22)
    tokens = {st.column_tokens((x, y, z), n, ordered=False) for x, y, z in __import__("itertools").permutations(triple)}
    assert len(tokens) == 1
    assert len({st.column_tokens((x, y, z), n) for x, y, z in __import__("itertools").permutations(triple)}) == 6
    assert len(st.MULTISETS) == 10


def test_the_string_formats_answer_labels_are_the_sums_digits():
    n = 2
    triple = (5, 7, 8)
    tokens = st.string_tokens(triple, n)
    assert len(tokens) == 4 * n + 4
    labels = tokens[1:]
    start = st.answer_start(n)
    assert labels[start : start + n + 1] == ad.digits(sum(triple), n + 1)
    assert tokens[start] == st.EQUALS  # the input at the first answer label


# the real candidates


@pytest.mark.parametrize("name", list(st.CANDIDATES))
def test_every_candidate_trains_adds_and_reloads_its_answers(name: str, tmp_path: Path):
    n = 2
    candidate = st.CANDIDATES[name]
    size = candidate.sizes(candidate.shapes[0])[0]
    model = candidate.build(n, size, 3)
    rng = random.Random(0)
    model.train_epoch(ad.training_triples(n, 128, rng), 0.003, 32, rng)
    triples = ad.training_triples(n, 200, random.Random(1))
    answers = model.add(triples)
    assert all(0 <= a < ad.BASE ** (n + 1) for a in answers)
    assert model.parameter_count() > 0
    model.save(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"candidate": name, "n": n, "size": size}))
    reloaded, reloaded_n = st.load_model(tmp_path)
    assert reloaded_n == n
    assert reloaded.add(triples) == answers


def test_a_wide_column_transformer_learns_three_digit_addition(tmp_path: Path):
    # the pipeline end to end: a size well above the threshold passes the whole catalogue. Not at
    # n = 2: its held-out tenth removes whole contexts (every order of a column-0 carry with one
    # column-1 multiset), which a network that fits the rest gets wrong
    settings = st.Settings(epoch_examples=2**14, max_epochs=10, patience=5, screen=200, seeds=1)
    spec = st.run_spec("column-causal", 3, 0.003, {"layers": 2, "heads": 1, "d": 16}, 0)
    result = st.train_and_evaluate(spec, settings, st.CANDIDATES, tmp_path)
    assert result["passes"], {k: v["passed"] for k, v in result["properties"].items()}
    assert result["stop"] == "passed"
    assert result["reach"] == 3


# the search


def test_a_rung_is_decided_as_soon_as_its_seeds_settle_it():
    assert st.Rung(passed=4, finished=4).succeeds(5) is True
    assert st.Rung(passed=0, finished=2).succeeds(5) is False  # 4 of 5 can't pass any more
    assert st.Rung(passed=3, finished=4).succeeds(5) is None
    assert st.Rung(passed=1, finished=1).succeeds(1) is True


def test_bisection_finds_the_threshold_and_measures_either_side(tmp_path: Path):
    log: list[tuple[int, int]] = []
    sweep = sweep_of(tmp_path, fake_candidates(log))
    st.run(sweep, log=lambda line: None)
    widths = sorted({w for w, _seed in log})
    # the middle (4) fails, then 16 passes, then 8 passes: the threshold, with 4 and 16 beside it
    assert widths == [4, 8, 16]
    report = st.report(sweep)
    assert "| fake | 2 | 0.01 | {'depth': 1} | 8 | 80 |" in report


def test_a_grid_runs_every_rung(tmp_path: Path):
    log: list[tuple[int, int]] = []
    st.run(sweep_of(tmp_path, fake_candidates(log), "grid"), log=lambda line: None)
    assert sorted({w for w, _seed in log}) == list(LADDER)


def test_a_failing_rung_stops_after_two_failing_seeds(tmp_path: Path):
    log: list[tuple[int, int]] = []
    st.run(sweep_of(tmp_path, fake_candidates(log)), log=lambda line: None)
    assert sorted(seed for w, seed in log if w == 4) == [0, 1]  # in-process runs go in order


# the sweep


def test_a_rerun_resumes_from_the_results_on_disk(tmp_path: Path):
    log: list[tuple[int, int]] = []
    sweep = sweep_of(tmp_path, fake_candidates(log))
    st.run(sweep, log=lambda line: None)
    (tmp_path / "results" / f"{st.run_spec('fake', 2, 0.01, {'depth': 1, 'width': 8}, 2).key}.json").unlink()
    log.clear()
    st.run(sweep, log=lambda line: None)
    assert log == [(8, 2)]


def test_a_run_that_raises_is_recorded_and_run_once_more(tmp_path: Path):
    log: list[tuple[int, int]] = []
    sweep = sweep_of(tmp_path, fake_candidates(log, raises={(8, 0)}))
    lines: list[str] = []
    st.run(sweep, log=lines.append)
    key = st.run_spec("fake", 2, 0.01, {"depth": 1, "width": 8}, 0).key
    assert any(line.startswith(f"{key}: error") for line in lines)
    result = json.loads((tmp_path / "results" / f"{key}.json").read_text())
    assert result["passes"] and result["attempts"] == 2


def test_each_run_saves_its_model_and_manifest(tmp_path: Path):
    candidates = fake_candidates([])
    st.run(sweep_of(tmp_path, candidates), log=lambda line: None)
    key = st.run_spec("fake", 2, 0.01, {"depth": 1, "width": 8}, 0).key
    model, n = st.load_model(tmp_path / "models" / key, candidates)
    assert n == 2 and model.add([(1, 2, 2)]) == [5]


def test_status_counts_runs(tmp_path: Path):
    sweep = sweep_of(tmp_path, fake_candidates([]))
    st.run(sweep, log=lambda line: None)
    assert st.status(sweep).startswith("10 runs finished (8 passing), 0 errors, 0 running")


def test_a_config_names_the_sweep(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "directory": str(tmp_path / "sweep"),
                "workers": 6,
                "max_epochs": 30,
                "studies": [{"candidate": "column-causal", "n": 4, "rates": [0.003]}],
            }
        )
    )
    sweep = st.load_config(path)
    assert sweep.workers == 6 and sweep.settings.max_epochs == 30 and sweep.settings.seeds == 5
    study = sweep.studies[0]
    assert study.search == "bisect" and len(study.shapes) == 6 and study.rates == [0.003]
