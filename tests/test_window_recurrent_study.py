"""
The window-recurrent adder's harness (indrajala_studies/window_recurrent_study.py): the encoding
at an alignment through the final window, the curriculum's examples and batches, the checks and the
break length against stand-in networks that add (or carry only inside a window), and a real
network's run, its steps saved and resumed.
"""

import dataclasses
import hashlib
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs import window_presets as wp
from indrajala_ml.model.specs.layer_specs import Attention, Residual, StateInput
from indrajala_ml.model.specs.window_recurrence import BEFORE_STATE

from indrajala_studies import addition_data as ad
from indrajala_studies import inference
from indrajala_studies import modular_data as md
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt
from tests.helpers import TINY

# small catalogue sizes, and test lengths past A3's and M1's exhaustive spaces, so a check is quick
SMALL = wr.Settings(
    epoch_examples=64,
    max_epochs=3,
    patience=1,
    screen=20,
    steps=2,
    test_lengths=(8, 14, 16),
    sampled=200,
    relation_sampled=20,
    per_stratum=10,
    codes_cases=20,
    seeds=1,
)


class _Backend:
    def matrix(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [list(row) for row in rows]


class AddingNetwork:
    """
    A stand-in that reads the column tokens and adds them, the carry passed along every column, or,
    with window set, reset at each window's start (a network whose state carries nothing).
    """

    backend = _Backend()

    def __init__(self, window: int | None = None) -> None:
        self.window = window
        self.batches: list[list[Any]] = []

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> list[tuple[float, ...]]:
        return [tokens for tokens, _labels in rows]

    def classify_rows(self, prepared: Sequence[tuple[float, ...]]) -> list[tuple[int, ...]]:
        predictions: list[tuple[int, ...]] = []
        for tokens in prepared:
            carry, digits = 0, list[int]()
            for t, token in enumerate(int(x) for x in tokens):
                if self.window is not None and t % self.window == 0:
                    carry = 0
                total = token // 9 + token // 3 % 3 + token % 3 + carry
                digits.append(total % 3)
                carry = total // 3
            predictions.append(tuple(digits))
        return predictions

    def state_codes(self, X: Sequence[Sequence[float]]) -> list[list[tuple[int, ...]]]:
        return [[(0,), (len(row) % 5,)] for row in X]

    def learn_batch(self, rate: float, batch: list[Any]) -> None:
        self.batches.append(batch)


class TaskNetwork(AddingNetwork):
    """A stand-in that runs a task's target machine over the column tokens, the state kept across windows."""

    def __init__(self, task: wt.Task) -> None:
        super().__init__()
        self.task = task

    def classify_rows(self, prepared: Sequence[tuple[float, ...]]) -> list[tuple[int, ...]]:
        return [self.task.machine.run([int(x) for x in tokens]) for tokens in prepared]


DIVISION = wt.ModularDivision(5)


# the encoding


def test_an_example_is_its_columns_at_the_alignment_through_one_more_window():
    triple = (5, 1, 0)  # 12 + 01 + 00 in base 3, least significant first: (2, 1), (1, 0)
    tokens, labels = wr.example(wt.ADDITION, triple, 2, 3, 2)
    # 2 zero columns, the 2 columns, 2 more to fill 2 windows, then the final window
    assert wr.data_windows(2, 2, 3) == 2
    assert tokens == (0.0, 0.0, 9 * 2 + 3 * 1 + 0, 9 * 1 + 0 + 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert labels == (0, 0, *ad.digits(6, 7))


@pytest.mark.parametrize("k", [1, 2, 4])
def test_lengths_fill_exactly_their_window_count(k: int):
    for windows in (1, 2, 3):
        for a in range(k):
            lengths = wr.lengths_for(windows, a, k)
            assert len(lengths) > 0
            assert all(wr.data_windows(n, a, k) == windows for n in lengths)
            assert wr.data_windows(lengths[0] - 1, a, k) < windows or lengths[0] == 1


def test_a_cases_alignment_is_its_own_whatever_the_order():
    rng = random.Random(3)
    triples = [ad.uniform_triple(8, rng) for _ in range(400)]
    assert {wr.alignment(wt.ADDITION, t, 8, 1) for t in triples} == {0}
    assert {wr.alignment(wt.ADDITION, t, 8, 4) for t in triples} == {0, 1, 2, 3}
    for x, y, z in triples[:20]:
        assert wr.alignment(wt.ADDITION, (x, y, z), 8, 4) == wr.alignment(wt.ADDITION, (z, x, y), 8, 4)
    # independent of D8's tenth: the held-out triples take every alignment too
    held = [t for t in (ad.uniform_triple(8, rng) for _ in range(4000)) if ad.held_out(t, 8)]
    assert {wr.alignment(wt.ADDITION, t, 8, 4) for t in held} == {0, 1, 2, 3}


def test_an_epoch_shares_its_examples_equally_among_the_window_counts():
    examples = wr.training_examples(wt.ADDITION, [1, 2, 3], 2, 300, random.Random(5))
    widths = [len(tokens) for tokens, _labels in examples]
    # data windows w, then the final window: (w + 1) * k columns
    assert {width: widths.count(width) for width in set(widths)} == {4: 100, 6: 100, 8: 100}
    adder = AddingNetwork()
    for (tokens, labels), digits in zip(examples, adder.classify_rows([t for t, _ in examples]), strict=True):
        assert labels == digits
        assert len(labels) == len(tokens)


def test_an_epochs_batches_each_hold_one_window_count():
    network = AddingNetwork()
    examples = wr.training_examples(wt.ADDITION, [1, 2, 3], 2, 300, random.Random(5))
    wr.train_epoch(network, examples, 16, 0.01, random.Random(6))
    assert all(len({len(tokens) for tokens, _ in batch}) == 1 for batch in network.batches)
    assert sorted(map(str, (e for b in network.batches for e in b))) == sorted(map(str, examples))


def test_short_lengths_train_on_every_triple():
    # below n = 3 nothing is held out, so B1 there can pass whole (W24)
    examples = wr.training_examples(wt.ADDITION, [1], 1, 3000, random.Random(1))
    seen = {tokens for tokens, _labels in examples}
    assert len(seen) == 27


# the checks


@pytest.mark.parametrize("k", [1, 3])
def test_an_adding_network_passes_the_catalogue_at_any_alignment(k: int):
    model = wr.WindowModel(AddingNetwork(), k, wt.ADDITION)
    for n in (2, 8):
        results = model.evaluate(n, wr.check_properties(wt.ADDITION, n, SMALL.catalogue(wt.ADDITION)))
        assert all(r.holds for r in results), [r.id for r in results if not r.holds]


def test_a_network_carrying_nothing_across_windows_fails_the_carry_at_distance():
    model = wr.WindowModel(AddingNetwork(window=2), 2, wt.ADDITION)
    results = {r.id: r for r in model.evaluate(8, wr.test_properties(wt.ADDITION, SMALL.catalogue(wt.ADDITION)))}
    assert not results["M3"].holds and not results["B1"].holds
    assert results["M3"].reach() < 8


def _sums(model: wr.WindowModel, n: int, triples: Sequence[ad.Triple]) -> list[int]:
    return [ad.from_digits(labels) for labels in model.predict(n)(triples)]


def test_the_adder_runs_in_chunks_of_bounded_tokens_with_the_same_sums(monkeypatch: pytest.MonkeyPatch):
    k, n = 3, 8
    rng = random.Random(0)
    triples = [(rng.randrange(3**n), rng.randrange(3**n), rng.randrange(3**n)) for _ in range(50)]
    whole = wr.WindowModel(AddingNetwork(), k, wt.ADDITION)
    sums, codes = _sums(whole, n, triples), whole.codes_used(triples, n)
    monkeypatch.setattr(wr, "CHUNK_TOKENS", 40)  # 3 or 4 windows of 3 columns: 3 rows a pass
    chunked = wr.WindowModel(AddingNetwork(), k, wt.ADDITION)
    widths = [len(rows[0][0]) * len(rows) for _, rows in chunked._chunks(triples, n)]
    assert max(widths) <= 40 and len(widths) > len(triples) // 4
    assert _sums(chunked, n, triples) == sums == [sum(t) for t in triples]
    assert chunked.codes_used(triples, n) == codes


def test_b3_is_checked_from_three_columns_and_never_at_a_test_length():
    catalogue = SMALL.catalogue(wt.ADDITION)
    assert "B3" not in {p.id for p in wr.check_properties(wt.ADDITION, 2, catalogue)}
    assert "B3" in {p.id for p in wr.check_properties(wt.ADDITION, 3, catalogue)}
    assert "B3" not in {p.id for p in wr.test_properties(wt.ADDITION, catalogue)}


def test_a_division_example_reads_most_significant_first_its_labels_the_running_quotients():
    pair = (21, 5)  # 210 and 012 in base 3
    tokens, labels = wr.example(DIVISION, pair, 3, 2, 1)
    # 1 zero column (B 0: undefined), the 3 columns, then the final window
    assert tokens == (0.0, 6.0, 4.0, 2.0, 0.0, 0.0)
    answers = md.answers(pair, 3, 5)  # b = 5: undefined at the end
    assert answers == (5, 2, 5) and labels == (5, *answers, 5, 5)


@pytest.mark.parametrize("k", [1, 3])
def test_a_dividing_network_passes_the_division_catalogue_at_any_alignment(k: int):
    model = wr.WindowModel(TaskNetwork(DIVISION), k, DIVISION)
    for n in (2, 8):
        results = model.evaluate(n, wr.check_properties(DIVISION, n, SMALL.catalogue(DIVISION)))
        assert {r.id for r in results} == ({"U1", "S1", "S2"} if n < 3 else {"U1", "U2", "S1", "S2"})
        assert all(r.holds for r in results), [r.id for r in results if not r.holds]


def test_a_division_step_passes_at_once_and_records_the_undefined_runs_reach():
    spec = wr.RunSpec("curriculum", 1, 4, 32, 0, DIVISION.name)
    assert spec.key == "modular-division-5-curriculum-k1-s4-m32-seed0"
    result = wr.run_step(TaskNetwork(DIVISION), spec, 2, [1, 2], SMALL)
    assert result["stop"] == "passed" and result["passes"] and result["break"] == 16
    assert all(result["tests"][str(n)]["reach"] == n for n in SMALL.test_lengths)
    assert set(result["tests"]["16"]["properties"]) == {"U1", "S1", "S2"}


def test_the_break_length_is_the_last_of_the_passing_prefix():
    assert wr.break_length({8: True, 10: True, 12: False, 16: True}) == 10
    assert wr.break_length({8: False, 10: True}) == 0
    assert wr.break_length({8: True, 10: True}) == 10


def test_a_step_with_an_adding_network_passes_at_once_and_breaks_at_the_longest_test():
    spec = wr.RunSpec("curriculum", 1, 4, 8, 0)
    result = wr.run_step(AddingNetwork(), spec, 2, [1, 2], SMALL)
    assert result["stop"] == "passed" and result["epochs"] == 1 and result["passes"]
    assert result["lengths"] == [1, 2]
    assert result["break"] == 16 and result["window_ratio"] == 16 / 2
    assert result["examples"] == SMALL.epoch_examples
    assert all(result["tests"][str(n)]["reach"] == n for n in SMALL.test_lengths)
    # a state a code per factor (the factored-state workplan, D4), of one factor here
    assert (0,) in result["codes_used"] and set(result["codes_used"]) <= {(c,) for c in range(5)}


def test_a_step_with_a_network_that_never_learns_stops_on_the_plateau():
    spec = wr.RunSpec("curriculum", 1, 4, 8, 0)
    result = wr.run_step(AddingNetwork(window=1), spec, 2, [1, 2], SMALL)
    assert result["stop"] == "plateau" and result["epochs"] == 2 and not result["passes"]
    assert result["break"] == 0 and result["window_ratio"] == 0


def test_without_tests_past_the_break_a_step_tests_up_to_its_first_failing_length():
    spec = wr.RunSpec("curriculum", 1, 4, 8, 0)
    settings = dataclasses.replace(SMALL, tests_past_break=False)
    failing = wr.run_step(AddingNetwork(window=1), spec, 2, [1, 2], settings)
    assert list(failing["tests"]) == ["8"] and not failing["tests"]["8"]["passes"]
    assert failing["break"] == 0 and failing["window_ratio"] == 0
    assert [wr._test_cell(failing, n, "reach", "{}") for n in SMALL.test_lengths][1:] == ["-", "-"]
    passing = wr.run_step(AddingNetwork(), spec, 2, [1, 2], settings)
    assert list(passing["tests"]) == [str(n) for n in SMALL.test_lengths] and passing["break"] == 16


def test_the_arms_steps():
    curriculum, control = wr.RunSpec("curriculum", 2, 4, 8, 0), wr.RunSpec("all-at-once", 2, 4, 8, 0)
    assert [list(c) for c in curriculum.step_counts(SMALL)] == [[1], [1, 2]]
    assert [list(c) for c in control.step_counts(SMALL)] == [[1, 2]]


# a real network's run


@pytest.mark.parametrize("task", ["addition", "modular-division-5", "modular-sum-4"])
def test_a_run_trains_each_step_saves_it_and_resumes_after_the_last_saved(tmp_path: Path, task: str):
    settings = TINY
    study = wr.Study(tmp_path, settings, [wr.RunSpec("curriculum", 1, 4, 3, 0, task)], 0)
    lines: list[str] = []
    results = wr.run(study, log=lines.append)
    key = study.runs[0].key
    result = results[key]
    assert [s["counts"] for s in result["steps"]] == [[1], [1, 2]] and result["task"] == task
    assert result["parameters"] > 0 and len(result["crate"]) == 12
    assert set(result["steps"][1]["codes_used"]) <= {(c,) for c in range(3)}
    assert len(lines) == 1 and lines[0].startswith(f"{key}: ") and "step 2 " in lines[0]
    step_one = (tmp_path / "models" / key / "step-1.json").read_text()
    # a stopped run: its result and last step gone; the rerun keeps step 1 and trains step 2 again
    (tmp_path / "results" / f"{key}.json").unlink()
    (tmp_path / "models" / key / "step-2.json").unlink()
    rerun = wr.run(study, log=lambda line: None)[key]
    assert (tmp_path / "models" / key / "step-1.json").read_text() == step_one
    assert rerun["steps"][1] == {**result["steps"][1], **{k: rerun["steps"][1][k] for k in _TIMES}}
    assert "s/epoch train" in wr.report(study)
    assert wr.status(study).startswith("1 of 1 runs finished")


_TIMES = ("train_seconds", "screen_seconds", "check_seconds", "tests")


def test_a_rerun_reproduces_its_run_and_a_change_shows(tmp_path: Path):
    """C0: the same run in two directories matches but for timings and provenance; a changed weight or result shows."""
    settings = TINY
    spec = wr.RunSpec("curriculum", 1, 4, 3, 0)
    original, rerun = tmp_path / "original", tmp_path / "rerun"
    for directory in (original, rerun):
        wr.run(wr.Study(directory, settings, [spec], 0), log=lambda line: None)
    assert wr.compare_runs(original, rerun) == {spec.key: []}
    # E1's results have no task field: addition's
    result_file = original / "results" / f"{spec.key}.json"
    result = json.loads(result_file.read_text())
    del result["task"]
    result["machine"], result["seconds"] = "elsewhere", 1.0e9
    result_file.write_text(json.dumps(result))
    assert wr.compare_runs(original, rerun) == {spec.key: []}
    result["steps"][1]["break"] += 1
    result_file.write_text(json.dumps(result))
    network_file = rerun / "models" / spec.key / "step-2.network.json"
    network = json.loads(network_file.read_text())
    network["weights"][0][0][0][0] += 2.0**-40
    network_file.write_text(json.dumps(network))
    assert wr.compare_runs(original, rerun) == {
        spec.key: ["result.steps[1].break", "step-2.network.weights[0][0][0][0]"]
    }
    (rerun / "models" / spec.key / "step-1.network.json").unlink()
    assert "step-1.network.json (missing)" in wr.compare_runs(original, rerun)[spec.key]


def test_differences_name_each_path():
    assert wr.differences({"a": [1, 2.0], "b": 1}, {"a": [1, 2], "c": 1}) == [".a[1]", ".b", ".c"]
    assert wr.differences([1], [1, 2], "x") == ["x (length 1 against 2)"]
    assert wr.comparable({"steps": [{"check_seconds": 1, "tests": {"8": {"seconds": 2, "passes": True}}}]}) == {
        "steps": [{"tests": {"8": {"passes": True}}}],
        "task": "addition",
        "window": "transformer",
    }


def test_a_continuous_run_trains_without_codes(tmp_path: Path):
    """E2: m None builds a continuous state; its run counts no codes and the report says so."""
    settings = TINY
    spec = wr.RunSpec("curriculum", 1, 4, None, 0)
    assert spec.key == "curriculum-k1-s4-continuous-seed0"
    study = wr.Study(tmp_path, settings, [spec], 0)
    result = wr.run(study, log=lambda line: None)[spec.key]
    assert result["m"] is None and all(step["codes_used"] is None for step in result["steps"])
    assert result["steps"][0]["tests"]["3"]["codes"] is None
    network = load_network(str(tmp_path / "models" / spec.key / "step-2.network.json"))
    assert not network.recurrence.snapped
    report = wr.report(study)
    assert "| curriculum | 1 | 4 | continuous |" in report and f"| {spec.key} | 2 |" in report


def test_explore_counts_the_codes_past_the_check_lengths(tmp_path: Path):
    """E5: a run's last network's codes at its check lengths and past them, the new ones apart."""
    settings = TINY
    spec, unrun = wr.RunSpec("curriculum", 1, 4, 3, 0), wr.RunSpec("curriculum", 1, 4, 3, 1)
    study = wr.Study(tmp_path, settings, [spec], 0)
    wr.run(study, log=lambda line: None)
    explored = wr.explore_run(study, spec, (4, 8))
    assert explored is not None and wr.explore_run(study, unrun) is None
    assert set(explored["check_codes"]) == {"1", "2"} and set(explored["codes"]) == {"4", "8"}
    assert explored["past"] == 4  # the first of the lengths past the check's longest, 2
    network = study.models / spec.key / "step-2.network.json"
    assert explored["network"] == hashlib.sha256(network.read_bytes()).hexdigest()
    seen = {code for used in explored["check_codes"].values() for code in used}
    for n in ("4", "8"):
        assert set(explored["codes"][n]) <= {(c,) for c in range(3)}
        assert explored["new"][n] == [code for code in explored["codes"][n] if code not in seen]


def test_explore_summary_compares_failing_with_working_runs():
    def run(works: bool, passes: bool, new: list[int], past: int = 8) -> dict[str, Any]:
        return {"works": works, "passes": passes, "new": {"8": new, "16": []}, "past": past}

    explored = [
        *(run(False, True, [5]) for _ in range(4)),
        run(False, True, []),
        run(False, True, [5], past=16),  # trained past 8 (`k` 4): read at 16, where it uses no new code
        run(False, False, [5]),  # a failing check: not E5's
        *(run(True, True, []) for _ in range(5)),
    ]
    summary = wr.explore_summary(explored, (8, 16))
    assert f"| 8 | 5/6 | 0/5 | {inference.fisher_exact(5, 1, 0, 5):.3g} |" in summary
    assert f"| past | 4/6 | 0/5 | {inference.fisher_exact(4, 2, 0, 5):.3g} |" in summary


def test_verdicts_count_each_cell_by_its_results_certificates_and_long_tests(tmp_path: Path):
    """The findings' tables: works from the results, the class from both certificates, failures from the long tests."""
    settings = dataclasses.replace(SMALL, seeds=3, steps=2)
    specs = [wr.RunSpec("curriculum", 1, 4, 3, seed) for seed in range(3)]
    study = wr.Study(tmp_path / "study", settings, specs, 0)
    branched, strict, long_tests = tmp_path / "branched", tmp_path / "strict", tmp_path / "long"
    for directory in (study.results, branched, strict, long_tests):
        directory.mkdir(parents=True)
    # seed 0 robustly certified, seed 1 strictly only (working, a long-test failure), seed 2 neither (failing)
    for spec, works, b, s in zip(specs, (True, True, False), (True, False, False), (True, True, False), strict=True):
        (study.results / f"{spec.key}.json").write_text(
            json.dumps({"key": spec.key, "works": works, "steps": [{"step": 2}]})
        )
        name = f"{spec.key}-step-2"
        (branched / f"{name}.json").write_text(json.dumps({"certified": b}))
        (strict / f"{name}.json").write_text(json.dumps({"certified": s}))
        passed = 9 if spec.seed == 1 else 10
        lengths = {"128": {"properties": {"B1": {"cases": 10, "passed": passed}}}}
        (long_tests / f"{name}.json").write_text(json.dumps({"lengths": lengths}))
    assert wr.last_networks(study) == {spec.key: study.models / spec.key / "step-2.network.json" for spec in specs}
    assert wr.network_name(study.models / specs[0].key / "step-2.network.json") == f"{specs[0].key}-step-2"
    text = wr.verdicts(study, branched, strict, long_tests)
    two, one = wr.rate(2, 3), wr.rate(1, 3)
    assert f"| curriculum | 1 | 4 | 3 | 3 | {two} | {two} | {one} |" in text
    assert "| robustly certified | 1 | 0 |" in text and "| not certified | 0 | 1 |" in text
    assert f"| certified only strictly | 1 | 1 | 10 | {specs[1].key}-step-2: B1@128 1 |" in text
    assert "| robustly certified | 1 | 0 | 10 | - |" in text
    low, high = inference.wilson(2, 3)
    assert two == f"2/3 ({low:.2f}-{high:.2f})" and wr.rate(0, 0) == "-"
    # without certificates the cell's certified columns are empty; no long tests, no long-test table
    for path in branched.iterdir():
        path.unlink()
    text = wr.verdicts(study, branched, strict)
    assert f"| curriculum | 1 | 4 | 3 | 3 | {two} | - | - |" in text and "long-tested" not in text


def test_a_long_test_writes_each_length_and_resumes(tmp_path: Path):
    """long-test: a network's properties at each length, as check --json; a network done before is skipped."""
    settings = dataclasses.replace(SMALL, max_epochs=1, test_lengths=(3,), sampled=50, per_stratum=5)
    spec = wr.RunSpec("curriculum", 1, 4, 3, 0)
    study = wr.Study(tmp_path / "study", settings, [spec], 0)
    wr.run(study, log=lambda line: None)
    path = wr.last_networks(study)[spec.key]
    lines: list[str] = []
    long = wr.long_test(path, wt.ADDITION, (8, 12), ["M3", "B1"], sampled=30, per_stratum=2, log=lines.append)
    assert long["network"] == str(path) and set(long["lengths"]) == {"8", "12"}
    m3 = long["lengths"]["8"]["properties"]["M3"]
    assert set(long["lengths"]["8"]["properties"]) == {"B1", "M3"}
    assert m3["cases"] == 2 * len(ad.doubling_distances(8)) and len(lines) == 4
    out = tmp_path / "long"
    out.mkdir()
    line = wr.long_test_file(path, out, wt.ADDITION.name, (4,), ["M3"])
    written = json.loads((out / f"{spec.key}-step-2.json").read_text())
    cases, failures = wr.long_test_failures(written)
    assert line.startswith(f"{spec.key}-step-2: {cases:,} cases, ") and cases == 1_000 * len(ad.doubling_distances(4))
    assert ("every case passed" in line) == (not failures)
    assert wr.long_test_file(path, out, wt.ADDITION.name, (4,), ["M3"]) == f"{spec.key}-step-2: done before"


def test_e6s_windows_drop_the_attention_or_every_block():
    def inner(specs: list[Any]) -> list[Any]:
        return [part for spec in specs for part in (spec.body if isinstance(spec, Residual) else (spec,))]

    for window in wr.WINDOWS:
        network = wr.build(wt.ADDITION, 1, 16, 8, 0, window)
        assert network.recurrence.snapped
    transformer, linear, ffn = (inner(wr.WINDOWS[w](wt.ADDITION)) for w in ("transformer", "linear", "ffn"))
    assert sum(isinstance(spec, Attention) for spec in transformer) == wp.BLOCKS
    assert not any(isinstance(spec, Attention) for spec in linear + ffn)
    assert sum(isinstance(spec, StateInput) for spec in ffn) == wp.BLOCKS
    assert sum(isinstance(spec, StateInput) for spec in linear) == 1
    assert all(getattr(spec, "activation", "linear") in ("linear", "softmax") for spec in linear)
    assert wr.RunSpec("curriculum", 1, 16, 8, 0, window="linear").key == "curriculum-linear-k1-s16-m8-seed0"
    assert wr.RunSpec("curriculum", 1, 16, 8, 0).key == "curriculum-k1-s16-m8-seed0"


def test_a_baseline_window_run_names_its_window(tmp_path: Path):
    settings = dataclasses.replace(TINY, steps=1)
    spec = wr.RunSpec("curriculum", 1, 4, 3, 0, window="ffn")
    study = wr.Study(tmp_path, settings, [spec], 0)
    result = wr.run(study, log=lambda line: None)[spec.key]
    assert result["window"] == "ffn" and "| curriculum (ffn window) | 1 | 4 | 3 |" in wr.report(study)
    # E1's results have no window field: a transformer's
    assert wr.comparable({"key": "x"}) == {"key": "x", "task": "addition", "window": "transformer"}


def test_a_run_that_raises_is_recorded_and_run_once_more(tmp_path: Path):
    calls: list[str] = []

    def train(spec: wr.RunSpec, settings: wr.Settings, models: Path) -> dict[str, Any]:
        calls.append(spec.key)
        if len(calls) == 1:
            raise RuntimeError("a run that dies")
        return {"key": spec.key, "steps": [], "works": False}

    study = wr.Study(tmp_path, SMALL, [wr.RunSpec("curriculum", 1, 4, 3, 0)], 0, train)
    lines: list[str] = []
    wr.run(study, log=lines.append)
    assert len(calls) == 2 and lines[0].endswith("error (attempt 1)")
    assert json.loads((tmp_path / "results" / f"{calls[0]}.json").read_text())["attempts"] == 2


def test_a_config_names_the_matrix(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "directory": str(tmp_path / "study"),
                "workers": 2,
                "steps": 2,
                "seeds": 2,
                "test_lengths": [8, 16],
                "matrix": [{"k": [1, 2], "s": [4], "m": [4, 8]}, {"arm": "all-at-once", "k": [2], "s": [4], "m": [8]}],
            }
        )
    )
    study = wr.load_config(path)
    assert study.workers == 2 and study.settings.steps == 2 and study.settings.test_lengths == (8, 16)
    assert len(study.runs) == 2 * 2 * 2 + 2
    assert study.runs[-1].key == "all-at-once-k2-s4-m8-seed1"
    assert {spec.task for spec in study.runs} == {"addition"}


def test_a_config_names_its_task(tmp_path: Path):
    path = tmp_path / "config.json"
    config = {
        "directory": str(tmp_path),
        "task": "modular-division-7",
        "seeds": 1,
        "matrix": [{"k": [4], "s": [4], "m": [64]}],
    }
    path.write_text(json.dumps(config))
    (spec,) = wr.load_config(path).runs
    assert spec.task == "modular-division-7" and spec.key == "modular-division-7-curriculum-k4-s4-m64-seed0"


def test_e3_e4_reruns_the_matrix_and_control_at_their_cells():
    """C0: E3 and E4's seeds 0-4 at the matrix's and the control's cells are those runs again, same keys and settings."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    e3_e4, matrix, control = (
        wr.load_config(scripts / f"window_recurrent_{name}.json") for name in ("e3_e4", "matrix", "control")
    )
    keys = {spec.key for spec in e3_e4.runs}
    assert len(keys) == len(e3_e4.runs) == (6 + 4) * 20 + 2 * 20
    cells = {("curriculum", 4, 4), ("curriculum", 1, 16), ("all-at-once", 4, 4), ("all-at-once", 1, 16)}
    reruns = [spec for spec in matrix.runs + control.runs if (spec.arm, spec.k, spec.s) in cells]
    assert len(reruns) == 6 * 5 + 2 * 5 and {spec.key for spec in reruns} <= keys
    assert (
        e3_e4.settings
        == dataclasses.replace(matrix.settings, seeds=20)
        == dataclasses.replace(control.settings, seeds=20)
    )


def test_e3_e4_trimmed_is_e3_e4s_k4_cells_without_m32():
    """D34: the trimmed stage 7 is a subset of E3 and E4's runs, in their directory, and keeps C0's reruns at k 4."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    trimmed, e3_e4, matrix, control = (
        wr.load_config(scripts / f"window_recurrent_{name}.json")
        for name in ("e3_e4_trimmed", "e3_e4", "matrix", "control")
    )
    keys = {spec.key for spec in trimmed.runs}
    assert len(keys) == len(trimmed.runs) == 6 * 20 and keys <= {spec.key for spec in e3_e4.runs}
    assert (trimmed.directory, trimmed.settings, trimmed.workers) == (e3_e4.directory, e3_e4.settings, e3_e4.workers)
    assert {(spec.k, spec.s) for spec in trimmed.runs} == {(4, 4)} and 32 not in {spec.m for spec in trimmed.runs}
    reruns = {spec.key for spec in matrix.runs + control.runs if (spec.k, spec.s) == (4, 4)}
    assert len(reruns) == 3 * 5 + 5 and reruns <= keys


def test_e2_is_e3s_curriculum_m16_cell_with_a_continuous_state():
    """The claims workplan's D36: E2 at k 4 s 4 only, E3's settings and seeds, so its runs compare
    with E3's curriculum runs at m 16 seed by seed."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    e3 = wr.load_config(scripts / "window_recurrent_e3_e4_trimmed.json")
    e2 = wr.load_config(scripts / "window_recurrent_e2.json")
    assert e2.runs == [
        dataclasses.replace(spec, m=None) for spec in e3.runs if spec.arm == "curriculum" and spec.m == 16
    ]
    assert len(e2.runs) == 20 and e2.settings == e3.settings and e2.directory != e3.directory


def test_e2b_is_e6s_linear_cell_snapped_against_continuous():
    """The claims workplan's D38: E6's linear window at k 1 s 16, 20 seeds an arm, m 16 against a
    continuous state, the curriculum capped at the exclusion bound, every snapped step certified and
    no run stopped early; one study an arm, the same cell and settings but for m."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    snapped = wr.load_config(scripts / "window_recurrent_e2b_snapped.json")
    continuous = wr.load_config(scripts / "window_recurrent_e2b_continuous.json")
    e6 = wr.load_config(scripts / "window_recurrent_e6_linear.json")
    assert snapped.runs == [wr.RunSpec("curriculum", 1, 16, 16, seed, window="linear") for seed in range(20)]
    assert continuous.runs == [dataclasses.replace(spec, m=None) for spec in snapped.runs]
    assert snapped.settings == continuous.settings
    assert snapped.settings == dataclasses.replace(e6.settings, seeds=20, stop_at_certified=False)
    assert snapped.settings.steps == 16 + wt.ADDITION.machine.states - 1
    assert snapped.directory != continuous.directory


def test_e2b_prime_is_e2b_snapped_with_no_run_stopped_early():
    """The claims workplan's D39: E2b's snapped arm again, every run trained to the cap past its first
    certified step, so that it differs from E2b's continuous arm in the snap alone."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    prime = wr.load_config(scripts / "window_recurrent_e2b_prime.json")
    snapped = wr.load_config(scripts / "window_recurrent_e2b_snapped.json")
    assert prime.runs == snapped.runs and prime.workers == snapped.workers
    assert prime.settings == dataclasses.replace(snapped.settings, end_at_certified_step=False)
    assert prime.directory not in (snapped.directory, Path("data/window-recurrent/e2b-continuous"))


@pytest.mark.parametrize("window", ["linear", "ffn"])
def test_e6_is_an_existence_test_past_the_exclusion_bound(window: str):
    """The claims workplan's D37: E6's window (D21) at k 1 s 16 m 16 on addition, trained to the cap
    m + states - 1, E8c's protocol (6 seeds, each step certified, stopped at the first certified run)."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    e6 = wr.load_config(scripts / f"window_recurrent_e6_{window}.json")
    states = wt.ADDITION.machine.states
    assert e6.runs == [wr.RunSpec("curriculum", 1, 16, 16, seed, window=window) for seed in range(6)]
    assert 16 >= states and e6.settings.steps == 16 + states - 1
    e8c = wr.load_config(scripts / "window_recurrent_e8c_q4.json")
    assert e6.settings == dataclasses.replace(e8c.settings, steps=e6.settings.steps)
    assert e6.workers == 6 and e6.directory == Path(f"data/window-recurrent/e6-{window}")


def test_e7_prime_is_e7s_k1_cells_with_five_and_ten_times_the_states_codes():
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    for p, codes in ((5, {128, 256}), (7, {256, 512})):
        e7_prime, e7 = (wr.load_config(scripts / f"window_recurrent_{name}_p{p}.json") for name in ("e7_prime", "e7"))
        states = wt.task_named(f"modular-division-{p}").machine.states
        assert {spec.m for spec in e7_prime.runs} == codes and all(5 * states <= m <= 11 * states for m in codes)
        assert {(spec.arm, spec.k, spec.s) for spec in e7_prime.runs} == {("curriculum", 1, 16), ("all-at-once", 1, 16)}
        assert len(e7_prime.runs) == 2 * 2 * 10 and e7_prime.settings == dataclasses.replace(e7.settings, seeds=10)


def test_e8_is_the_modular_sum_at_e7_primes_settings_with_32_codes():
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    e7_prime = wr.load_config(scripts / "window_recurrent_e7_prime_p5.json")
    for q in (4, 5):
        e8 = wr.load_config(scripts / f"window_recurrent_e8_q{q}.json")
        assert {spec.task for spec in e8.runs} == {f"modular-sum-{q}"} and e8.settings == e7_prime.settings
        assert {(spec.arm, spec.k, spec.s, spec.m) for spec in e8.runs} == {
            ("curriculum", 1, 16, 32),
            ("all-at-once", 1, 16, 32),
        }
        assert len(e8.runs) == 2 * 10 and wt.task_named(f"modular-sum-{q}").machine.states == q


# D23: a cell kept below its bound on purpose, and why
BELOW_BOUND_ON_PURPOSE = {
    ("window_recurrent_e7_p7.json", 32): "E7's design flaw, run as registered: the negative control",
}


def network_states(spec: wr.RunSpec) -> int | None:
    """A snapped run's machine states, the product of its factors' codes (the factored-state
    workplan's D7); None for a continuous state."""
    return math.prod(m for _, m in spec.state if m is not None) if spec.snapped else None


def test_every_configs_codes_can_hold_its_tasks_machine():
    """D23: every snapped cell has at least as many codes as its task's minimal machine has states
    (each machine is minimal and every state reachable: test_window_tasks), or a recorded reason."""
    for path in sorted((Path(__file__).parents[1] / "scripts" / "studies" / "configs").glob("window_recurrent_*.json")):
        study = wr.load_config(path)
        for spec in study.runs:
            states, m = wt.task_named(spec.task).machine.states, network_states(spec)
            if m is not None and m < states:
                assert (path.name, m) in BELOW_BOUND_ON_PURPOSE, (
                    f"{path.name}: {spec.key} has {m} codes for {states} states"
                )


def test_e8b_is_e8_trained_long_enough_that_no_smaller_wrong_machine_fits():
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    e8 = wr.load_config(scripts / "window_recurrent_e8_q4.json")
    for q in (4, 5):
        e8b = wr.load_config(scripts / f"window_recurrent_e8b_q{q}.json")
        assert {(spec.arm, spec.k, spec.s, spec.m) for spec in e8b.runs} == {
            ("curriculum", 1, 16, 16),
            ("all-at-once", 1, 16, 16),
        }
        assert len(e8b.runs) == 2 * 10 and e8b.settings == dataclasses.replace(e8.settings, steps=20)


# D23's exclusion bound: configs registered before it, which train too short to rule out a wrong
# machine of at most m states (two machines of n and n' states that agree on every input up to
# length n + n' - 1 are equivalent)
BEFORE_THE_EXCLUSION_BOUND = {
    "window_recurrent_calibration.json",
    "window_recurrent_control.json",
    "window_recurrent_e2.json",
    "window_recurrent_e3_e4.json",
    "window_recurrent_e3_e4_trimmed.json",  # a subset of e3_e4's cells, the same runs (D34)
    "window_recurrent_e7_p5.json",
    "window_recurrent_e7_p7.json",
    "window_recurrent_e7_prime_p5.json",
    "window_recurrent_e7_prime_p7.json",
    "window_recurrent_e8_q4.json",
    "window_recurrent_e8_q5.json",
    "window_recurrent_matrix.json",
}


def test_e8c_is_e8bs_curriculum_cell_as_an_existence_test():
    """The claims workplan's D35: E8b's curriculum cell and settings, 6 seeds, each step certified and
    each rung stopped at its first certified run."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    for q in (4, 5):
        e8b = wr.load_config(scripts / f"window_recurrent_e8b_q{q}.json")
        e8c = wr.load_config(scripts / f"window_recurrent_e8c_q{q}.json")
        assert e8c.runs == [spec for spec in e8b.runs if spec.arm == "curriculum" and spec.seed < 6]
        assert e8c.settings == dataclasses.replace(e8b.settings, seeds=6, certify_steps=True, stop_at_certified=True)
        assert e8c.workers == 6 and e8c.directory != e8b.directory


def test_every_new_config_trains_long_enough_to_rule_out_a_wrong_machine():
    """D23: a snapped cell's longest training length (steps x k columns) is at least m + states - 1,
    so a network exactly right on its training lengths is its task's machine."""
    for path in sorted((Path(__file__).parents[1] / "scripts" / "studies" / "configs").glob("window_recurrent_*.json")):
        if path.name in BEFORE_THE_EXCLUSION_BOUND:
            continue
        study = wr.load_config(path)
        for spec in study.runs:
            states, m = wt.task_named(spec.task).machine.states, network_states(spec)
            if m is not None:
                longest = study.settings.steps * spec.k
                assert longest >= m + states - 1, f"{path.name}: {spec.key} trains to {longest} columns"


@pytest.mark.parametrize(
    ("name", "codes", "cap", "budget"),
    [("factored_p5", 8, 88, None), ("factored_p5_fallback", 6, 60, None), ("factored_p5_m16", 16, 280, 4)],
)
def test_c9_is_one_factored_division_cell_capped_at_the_product_bound(
    name: str, codes: int, cap: int, budget: float | None
):
    """The factored-state workplan's C9 (D7, D8, D10, D11) and C9' (D13): division at p 5, k 1, the
    curriculum, two independent factors of width 16, each with codes for its residue (A or B, 5
    states), 6 seeds, trained to the cap m_1 m_2 + q - 1, each step certified and the study stopped
    at its first certified run; C9' with E8c's counter (16 codes) and a wall-clock budget."""
    study = wr.load_config(
        Path(__file__).parents[1] / "scripts" / "studies" / "configs" / f"window_recurrent_{name}.json"
    )
    task = wt.task_named("modular-division-5")
    assert {(s.arm, s.k, s.state, s.cross, s.reads, s.task) for s in study.runs} == {
        ("curriculum", 1, ((16, codes), (16, codes)), False, BEFORE_STATE, task.name)
    }
    assert sorted(spec.seed for spec in study.runs) == list(range(6)) and study.workers == 6
    assert codes >= 5 and task.machine.states == 25
    assert study.settings.steps == cap == codes * codes + task.machine.states - 1
    assert study.settings.certify_steps and study.settings.stop_at_certified
    assert study.settings == dataclasses.replace(
        wr.load_config(
            Path(__file__).parents[1] / "scripts" / "studies" / "configs" / "window_recurrent_e8b_q5.json"
        ).settings,
        seeds=6,
        steps=cap,
        certify_steps=True,
        stop_at_certified=True,
        budget_hours=budget,
    )


@pytest.mark.parametrize(
    ("name", "seeds", "workers"),
    [("factored_p5_stagger", range(6), 6), ("factored_p5_stagger_pyramidon", range(6, 9), 3)],
)
def test_c9_double_prime_is_c9_prime_with_a_spare_factor_staggered(name: str, seeds: range, workers: int):
    """The factored-state workplan's C9'' (D14-D17): C9''s cell with a third factor of 16 codes,
    the factors staggered at 0.1 of the rate, the cap the product bound 16^3 + 25 - 1, a 4 h budget;
    seeds 0-5 on jebel and 6-8 on pyramidon, one directory."""
    scripts = Path(__file__).parents[1] / "scripts" / "studies" / "configs"
    study = wr.load_config(scripts / f"window_recurrent_{name}.json")
    c9_prime = wr.load_config(scripts / "window_recurrent_factored_p5_m16.json")
    task = wt.task_named("modular-division-5")
    assert {(s.arm, s.k, s.state, s.cross, s.reads, s.task, s.stagger) for s in study.runs} == {
        ("curriculum", 1, ((16, 16),) * 3, False, BEFORE_STATE, task.name, 0.1)
    }
    assert [spec.seed for spec in study.runs] == list(seeds) and study.workers == workers
    assert study.settings.steps == 16**3 + task.machine.states - 1 == 4120
    assert study.settings == dataclasses.replace(c9_prime.settings, steps=4120, seeds=len(seeds), first_seed=seeds[0])
    assert study.directory.name == "factored-p5-stagger"


def test_status_counts_each_running_runs_own_steps(tmp_path: Path):
    runs = [wr.RunSpec("curriculum", 1, 4, 3, 0), wr.RunSpec("all-at-once", 1, 4, 3, 0)]
    study = wr.Study(tmp_path, SMALL, runs, 0)
    study.results.mkdir(parents=True)
    for spec in runs:
        (study.results / f"{spec.key}.running").write_text("")
    lines = wr.status(study).splitlines()
    assert f"running: {runs[0].key}, 0 of 2 steps saved" in lines
    assert f"running: {runs[1].key}, 0 of 1 steps saved" in lines


# factored states and existence tests (the factored-state workplan, stage 7)


def _certified(network: Any, spec: wr.RunSpec, path: Path) -> dict[str, Any]:
    # certify_step's summary for a step that certifies
    return {"certified": True, "correct": True, "states": 25, "minimised": 25, "factors": ["A", "B"], "seconds": 0.0}


FACTORED_TASK = wt.ModularDivision(5).name
FACTORED = wr.RunSpec("curriculum", 1, 8, None, 0, FACTORED_TASK, factors=((4, 8), (4, 8)), reads=BEFORE_STATE)


def test_a_factored_runs_key_names_each_factor_and_a_single_factors_is_unchanged():
    assert FACTORED.key == "modular-division-5-curriculum-k1-s4-m8-s4-m8-before-state-seed0"
    assert FACTORED.label == "curriculum (before state)" and FACTORED.cell[2:] == ("4+4", "8+8")
    crossed = dataclasses.replace(FACTORED, cross=True, reads=wr.OUTPUT)
    assert crossed.key == "modular-division-5-curriculum-k1-s4-m8-s4-m8-cross-seed0"
    assert wr.RunSpec("curriculum", 4, 4, 16, 3).key == "curriculum-k4-s4-m16-seed3"
    assert wr.RunSpec("curriculum", 4, 4, None, 3).cell == ("curriculum", 4, "4", "continuous")
    assert FACTORED.snapped and not dataclasses.replace(FACTORED, factors=((4, 8), (4, None))).snapped


def test_a_factored_cell_in_a_config_builds_its_factored_network(tmp_path: Path):
    config = {
        "directory": str(tmp_path),
        "task": FACTORED_TASK,
        "seeds": 2,
        "matrix": [
            {"k": [1], "factors": [[4, 8], [4, 8]], "reads": "before_state"},
            {"k": [1], "s": [4], "m": [16]},
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    study = wr.load_config(path)
    assert [spec.key for spec in study.runs] == [
        "modular-division-5-curriculum-k1-s4-m8-s4-m8-before-state-seed0",
        "modular-division-5-curriculum-k1-s4-m8-s4-m8-before-state-seed1",
        "modular-division-5-curriculum-k1-s4-m16-seed0",
        "modular-division-5-curriculum-k1-s4-m16-seed1",
    ]
    network = wr.build(wt.task_named(FACTORED_TASK), 1, 0, None, 0, recurrence=study.runs[0].recurrence)
    assert network.recurrence == study.runs[0].recurrence and len(network.recurrence.factors) == 2


def test_a_certified_step_ends_the_run_and_its_certificate_is_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = dataclasses.replace(TINY, certify_steps=True)
    study = wr.Study(tmp_path, settings, [FACTORED], 0)
    results = wr.run(study, log=lambda line: None)
    result = results[FACTORED.key]
    # untrained: neither step certifies, so both run, each certified as it ends
    assert [s["certificate"]["certified"] for s in result["steps"]] == [False, False]
    assert all(len(s["certificate"]["factors"]) == 2 for s in result["steps"])
    assert (study.models / FACTORED.key / "step-1.certificate.json").exists()
    assert result["factors"] == [[4, 8], [4, 8]] and result["reads"] == BEFORE_STATE and not result["certified"]
    assert all(len(state) == 2 for state in result["steps"][0]["codes_used"])

    # a step that certifies ends its run there
    monkeypatch.setattr(wr, "certify_step", _certified)
    other = dataclasses.replace(FACTORED, seed=1)
    result = wr.run(wr.Study(tmp_path / "again", settings, [other], 0), log=lambda line: None)[other.key]
    assert len(result["steps"]) == 1 and result["certified"] and result["certified_step"] == 1


def test_without_end_at_certified_step_a_run_trains_every_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The claims workplan's D39: every step trained and certified, past the first certified one."""
    settings = dataclasses.replace(TINY, certify_steps=True, end_at_certified_step=False)
    monkeypatch.setattr(wr, "certify_step", _certified)
    result = wr.run(wr.Study(tmp_path, settings, [FACTORED], 0), log=lambda line: None)[FACTORED.key]
    assert len(result["steps"]) == settings.steps and all(s["certificate"]["certified"] for s in result["steps"])
    assert result["certified"] and result["certified_step"] == 1 and result["last_certified"]


def test_the_first_certified_run_stops_the_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = dataclasses.replace(TINY, certify_steps=True, stop_at_certified=True)
    specs = [dataclasses.replace(FACTORED, seed=seed) for seed in range(3)]
    monkeypatch.setattr(wr, "certify_step", _certified)
    lines: list[str] = []
    study = wr.Study(tmp_path, settings, specs, 0)
    results = wr.run(study, log=lines.append)
    assert list(results) == [specs[0].key] and results[specs[0].key]["certified"]
    assert wr.stop_file(study.models).exists() and "the first certified run" in lines[-1]
    # run again: the existence is found, nothing more runs
    assert list(wr.run(study, log=lines.append)) == [specs[0].key]


def test_a_spent_budget_stops_the_study_and_a_resumed_study_keeps_its_deadline(tmp_path: Path):
    """The factored-state workplan's D13: the first launch sets the deadline; past it nothing runs."""
    settings = dataclasses.replace(TINY, certify_steps=True, stop_at_certified=True, budget_hours=1)
    specs = [dataclasses.replace(FACTORED, seed=seed) for seed in range(2)]
    study = wr.Study(tmp_path, settings, specs, 0)
    study.models.mkdir(parents=True)
    spent = time.time() - 1
    wr.deadline_file(study.models).write_text(repr(spent))
    assert wr.run(study, log=lambda line: None) == {}
    assert float(wr.deadline_file(study.models).read_text()) == spent
    assert "stopped: the budget is spent" in wr.status(study)

    fresh = wr.Study(tmp_path / "fresh", settings, specs[:1], 0)
    before = time.time()
    assert list(wr.run(fresh, log=lambda line: None)) == [specs[0].key]
    deadline = float(wr.deadline_file(fresh.models).read_text())
    assert before + 3600 <= deadline <= time.time() + 3600
    assert "budget: until" in wr.status(fresh)


def test_a_spent_budget_stops_a_run_at_its_next_epoch(tmp_path: Path):
    settings = dataclasses.replace(TINY, certify_steps=True, budget_hours=1)
    tmp_path.mkdir(exist_ok=True)
    wr.deadline_file(tmp_path).write_text(repr(time.time() - 1))
    result = wr.train_run(FACTORED, settings, tmp_path)
    assert result["stopped"] and [s["stop"] for s in result["steps"]] == ["stopped"]
    assert "certificate" not in result["steps"][0]


def test_status_reads_the_stop_files_reason(tmp_path: Path):
    study = wr.Study(tmp_path, TINY, [FACTORED], 0)
    study.models.mkdir(parents=True)
    wr.stop_file(study.models).write_text("stopped by the owner (D13)")
    assert "stopped: stopped by the owner (D13)" in wr.status(study).splitlines()
    wr.stop_file(study.models).write_text("")
    assert f"stopped: {wr.FIRST_CERTIFIED}" in wr.status(study).splitlines()


def test_the_report_reads_runs_stopped_at_different_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Workers running in parallel end their runs at different steps when the study stops (D8)."""
    settings = dataclasses.replace(TINY, certify_steps=True, stop_at_certified=True)
    specs = [dataclasses.replace(FACTORED, seed=seed) for seed in range(2)]
    monkeypatch.setattr(wr, "certify_step", _certified)
    study = wr.Study(tmp_path, settings, specs, 0)
    certified = wr.run(study)[specs[0].key]
    # the first run in the report stopped a step later than the certified one (E8c's crash)
    stopped = {**certified, "certified": False, "stopped": True}
    stopped["steps"] = [*certified["steps"], {**certified["steps"][0], "step": 2, "stop": "stopped", "passes": False}]
    (study.results / f"{specs[0].key}.json").write_text(json.dumps(stopped))
    (study.results / f"{specs[1].key}.json").write_text(json.dumps({**certified, "key": specs[1].key, "seed": 1}))
    report = wr.report(study)
    assert "| 1/2 | step 1: 0 pass, breaks 0/0; step 2: 0 pass of 1, breaks 0 |" in report


def test_the_report_reads_a_run_stopped_before_its_first_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A run the study stops before its first epoch has no epoch timings (E6's crash)."""
    settings = dataclasses.replace(TINY, certify_steps=True, stop_at_certified=True)
    specs = [dataclasses.replace(FACTORED, seed=seed) for seed in range(2)]
    monkeypatch.setattr(wr, "certify_step", _certified)
    study = wr.Study(tmp_path, settings, specs, 0)
    certified = wr.run(study)[specs[0].key]
    first = certified["steps"][0]
    unstarted: dict[str, Any] = {
        **first,
        "stop": "stopped",
        "epochs": 0,
        "passes": False,
        "train_seconds": [],
        "screen_seconds": [],
    }
    stopped: dict[str, Any] = {
        **certified,
        "key": specs[1].key,
        "seed": 1,
        "certified": False,
        "stopped": True,
        "steps": [unstarted],
    }
    (study.results / f"{specs[1].key}.json").write_text(json.dumps(stopped))
    cost = [line for line in wr.report(study).splitlines() if line.startswith(f"| {specs[1].key} | 1 |")]
    assert len(cost) == 2 and "| - | - |" in cost[1]


def test_a_probe_times_each_steps_epochs_the_check_and_a_certificate():
    figures = wr.probe(FACTORED, TINY, [1, 2], epochs=2)
    assert [(row["step"], row["epoch"]) for row in figures["epochs"]] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert all(row["train_seconds"] > 0 and 0 <= row["screen"] <= 1 and row["peak_mb"] > 0 for row in figures["epochs"])
    assert figures["check"]["n"] == 2 and figures["check"]["seconds"] > 0
    assert figures["certificate"]["states"] >= 1 and figures["machine"]


def test_a_results_peak_memory_is_recorded_and_not_compared(tmp_path: Path):
    result = wr.run(wr.Study(tmp_path, TINY, [FACTORED], 0), log=lambda line: None)[FACTORED.key]
    assert result["peak_mb"] > 0
    assert wr.comparable(result) == wr.comparable({**result, "peak_mb": result["peak_mb"] * 2})


def test_a_step_stopped_by_the_study_ends_unchecked():
    spec = wr.RunSpec("curriculum", 1, 4, 8, 0)
    result = wr.run_step(AddingNetwork(), spec, 1, [1], SMALL, stopped=lambda: True)
    assert result["stop"] == "stopped" and result["epochs"] == 0 and result["tests"] == {}
