"""The modular family's encoding and probe (the modular-family workplan, stage 2)."""

import dataclasses
import hashlib
import itertools
import json
import math
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from indrajala_ml.model.specs import window_presets as wp
from indrajala_ml.model.specs.layer_specs import Attention, Branches, Dense, Residual

from indrajala_studies import inference
from indrajala_studies import modular_data as md
from indrajala_studies import modular_family as mf
from indrajala_studies import window_tasks as wt


def test_the_column_answers_division_and_the_queries_their_operations():
    rng = random.Random(0)
    for p in (5, 7):
        encoding = mf.Encoding(p, mf.FAMILY)
        for _ in range(50):
            n = rng.randrange(1, 9)
            pair = (rng.randrange(3**n), rng.randrange(3**n))
            columns = md.column_tokens(pair, n)
            tokens, labels = encoding.example(columns)
            assert len(tokens) == len(labels) == 3 * n
            assert labels[0::3] == md.answers(pair, n, p)
            assert labels[1::3] == md.sums(pair, n, p)
            a, b = md.digits(pair[0], n), md.digits(pair[1], n)
            x = y = 0  # the prefixes mod p
            for t in reversed(range(n)):
                x, y = (3 * x + a[t]) % p, (3 * y + b[t]) % p
            assert labels[-1] == (x - y) % p
            assert tokens[0::3] == tuple(float(c) for c in columns)
            assert set(tokens[1::3]) == {9.0} and set(tokens[2::3]) == {10.0}


def test_the_quotient_alone_is_division_one_token_a_window():
    encoding = mf.Encoding(5, mf.QUOTIENT)
    pair, n = (17, 40), 4
    tokens, labels = encoding.example(md.column_tokens(pair, n))
    assert encoding.width == 1 and len(tokens) == n
    assert labels == md.answers(pair, n, 5)
    assert encoding.tokens == 11 and encoding.classes == 6


def test_the_state_arm_asks_the_state_itself_and_t12s_vocabulary_widens_both_arms_layer():
    # the paper workplan's T12 (D31, design (a)): the state A p + B as the one extra output, the cover's
    # arm on the same vocabulary, so both share one output layer of p^2 classes and differ in queries alone
    p, vocabulary = 5, ("sum", "difference", "state")
    cols = md.column_tokens((17, 40), 6)
    state, cover = mf.Encoding(p, mf.STATE, vocabulary), mf.Encoding(p, mf.FAMILY, vocabulary)
    (tokens, labels), (_, covered) = state.example(cols), cover.example(cols)
    assert labels[0::2] == covered[0::3] == md.answers((17, 40), 6, p)  # the certified answer the same
    assert labels[1::2] == tuple(A * p + B for A, B in state.states(cols))
    assert set(tokens[1::2]) == {float(mf.COLUMNS + 2)}  # the state's token, the vocabulary's third
    assert state.classes == cover.classes == p * p and max(labels) < state.classes
    assert mf.Encoding(p, mf.FAMILY).classes == mf.Encoding(p, mf.FAMILY, mf.WIDE).classes == p + 1  # as before


def test_the_sum_arm_asks_the_sum_alone_with_the_familys_token():
    encoding = mf.Encoding(5, mf.SUM)
    pair, n = (17, 40), 4
    tokens, labels = encoding.example(md.column_tokens(pair, n))
    assert encoding.width == 2 and len(tokens) == 2 * n
    assert labels[0::2] == md.answers(pair, n, 5) and labels[1::2] == md.sums(pair, n, 5)
    assert set(tokens[1::2]) == {9.0} and encoding.tokens == 11


def test_the_cap_is_the_cells_exclusion_bound():
    assert mf.exclusion_bound(25, [(16, 16), (16, 16)]) == 280  # the division workplan's table
    assert mf.exclusion_bound(25, [(32, 128)]) == 152
    assert mf.exclusion_bound(49, [(16, 16), (16, 16)]) == 304


def test_a_probe_run_trains_diagnoses_and_saves(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4), (4, 4)], 0, 1e9, tmp_path, settings, cap=2)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    ends = [r for r in rows if "end" in r]
    assert [r["step"] for r in ends] == [1, 2]
    assert set(rows[0]["accuracy"]) >= {"sum", "difference"}
    assert len(rows[0]["factors"]) == 2 and set(rows[0]["factors"][0]["shares"]) == {"A", "B", "S", "D", "A=0", "B=0"}
    assert set(ends[-1]["long"]) == {"4"}
    assert all(isinstance(r["certified"], bool) and r["minimised"] >= 1 for r in ends)  # each step's certificate
    # the walk's counts (R2): every pair has a code, and no more right pairs than pairs
    assert all(
        1 <= r["machine"]["codes"] <= r["machine"]["pairs"] and 0 <= r["machine"]["right"] <= r["machine"]["pairs"]
        for r in ends
    )
    assert (tmp_path / "network.json").exists()
    assert (tmp_path / "step-1.network.json").exists() and (tmp_path / "step-2.network.json").exists()
    starts = [r for r in rows if r.get("epoch") == 0 and r["step"] > 0]  # the step before's network, scored
    assert [r["step"] for r in starts] == [1, 2] and all(
        set(r) == {"step", "epoch", "screen", "kinds", "seconds"} for r in starts
    )


def test_every_epoch_logs_its_screen_score(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=3, patience=10, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.QUOTIENT), [(4, 4)], 0, 1e9, tmp_path, settings, cap=1)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    epochs = [(r["epoch"], "factors" in r) for r in rows if r["step"] == 1 and "epoch" in r and "screen" in r]
    assert epochs[:4] == [(0, False), (1, False), (2, True), (3, False)]  # a diagnosis every 2 carries it too
    assert mf.trace(rows)[1].startswith("step   1 epoch   2")  # trace shows the diagnoses alone


def test_a_probe_run_stops_when_its_epochs_are_spent(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.QUOTIENT), [(4, 4)], 0, 1e9, tmp_path, settings, budget_epochs=3)
    ends = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines() if '"end"' in line]
    assert ends[-1]["end"] == "spent" and len(ends) <= 3


def test_a_fork_restores_the_steps_best_network_cools_the_rate_and_stops_at_its_limit(tmp_path: Path):
    """A threshold of 2 calls every epoch that doesn't raise the peak a collapse (below twice it), so
    steps fork, each fork halving the step's rate, at most `limit` times a step."""
    settings = mf.Settings(
        epoch_examples=64, max_epochs=8, patience=10, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    fork = mf.Fork(trigger=mf.Trigger(threshold=2.0), cool=0.5, limit=2)
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=2, fork=fork)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    forks = [r for r in rows if "fork" in r]
    ends = {r["step"]: r["forks"] for r in rows if "end" in r}
    assert forks and set(ends) == {1, 2}
    for step, count in ends.items():
        mine = [r for r in forks if r["step"] == step]
        assert len(mine) == count <= fork.limit
        assert [(r["fork"], r["rate"]) for r in mine] == [(f, 0.003 * 0.5**f) for f in range(1, count + 1)]
    assert all(r["screen"] < 2 * r["peak"] and r["restored"] == r["peak"] for r in forks)  # the peak's network back
    assert (tmp_path / "snapshot.network.json").exists()


def test_without_a_fork_a_run_logs_no_forks(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=1)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert not any("fork" in r or "forks" in r for r in rows)
    assert not (tmp_path / "snapshot.network.json").exists()


def test_the_query_machine_runs_each_column_with_its_queries_and_reads_one_slot():
    from indrajala_studies import window_recurrent_certificate as wc

    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, reads="output")
    state, _ = wc.NetworkMachine(net).start()
    columns = np.arange(9, dtype=np.int64)[:, np.newaxis]
    full = np.concatenate([columns, np.full((9, 1), 9), np.full((9, 1), 10)], axis=1)
    whole = wc.NetworkMachine(net).step(state, full)
    for slot in range(3):
        steps = mf.QueryMachine(net, encoding, slot).step(state, columns)
        assert steps.digits.shape == (9, 1)
        assert (steps.digits[:, 0] == whole.digits[:, slot]).all()
        assert (steps.states == whole.states).all()


class Oracle:
    """A stand-in network answering every window right: its rows' own labels."""

    def __init__(self, columns: int, shift: int = 0) -> None:
        self.columns, self.shift = columns, shift

    def prepare_dataset(self, rows: list[mf.Example]) -> list[mf.Example]:
        return rows

    def classify_rows(self, rows: list[mf.Example]) -> list[tuple[int, ...]]:
        return [labels[self.shift :] + labels[: self.shift] for _, labels in rows]


def test_the_query_model_reads_its_slot_so_a_right_network_passes_the_long_tests():
    from indrajala_studies import window_tasks as wt

    encoding = mf.Encoding(5, mf.FAMILY)
    for name, slot in (("modular-division-5", 0), ("modular-sum-5", 1)):
        task = wt.task_named(name)
        tested = [p for p in task.catalogue(50, 50, 4) if p.id in mf.LONG_PROPERTIES]
        model = mf.QueryModel(Oracle(3), encoding, task, slot)
        for n in (2, 9):
            results = task.evaluate(model.predict(n), n, tested)
            assert [r.id for r in results] == ["U1", "S1"] and all(r.holds for r in results)
        # the wrong slot's labels fail
        wrong = mf.QueryModel(Oracle(3, shift=1), encoding, task, slot)
        assert not all(r.holds for r in task.evaluate(wrong.predict(9), 9, tested))


def test_a_saved_networks_long_test_is_the_harness_json(tmp_path: Path):
    from indrajala_studies import window_recurrent_study as wr

    encoding = mf.Encoding(5, mf.FAMILY)
    path = tmp_path / "network.json"
    mf.network(encoding, [(4, 4), (4, 4)], 0, reads="output").save(str(path))
    long = mf.long_test(path, encoding, "modular-division-5", lengths=(2, 3))
    assert set(long["lengths"]) == {"2", "3"} and long["slot"] == 0 and len(long["sha256"]) == 64
    assert set(long["lengths"]["3"]["properties"]) == {"U1", "S1"}
    cases, _ = wr.long_test_failures(long)
    assert cases == 3**4 + 3**6 + 2 * 25 * 1000  # U1 every pair at 2 columns and at 3; S1 1,000 a state


def test_trace_marks_a_collapse_where_the_codes_in_use_halve():
    def row(step: int, codes: list[int], **extra: object) -> dict[str, object]:
        return {"step": step, "epoch": 5, "accuracy": {"sum": 0.5}, "factors": [{"codes": c} for c in codes], **extra}

    lines = mf.trace([row(1, [16, 15]), row(1, [14, 14]), row(2, [3, 4]), row(2, [4, 4], end="plateau", minimised=7)])
    assert ["collapse" in line for line in lines] == [False, False, True, False]
    assert lines[0].startswith("step   1 epoch   5 codes [16, 15] share 0.000 sum 0.500")
    assert lines[-1].endswith("END plateau min 7")
    walked = row(2, [4, 4], end="plateau", minimised=7, machine={"codes": 8, "pairs": 20, "right": 3})
    assert mf.trace([walked])[0].endswith("END plateau min 7 pairs/code 2.50 right 3")
    fork = {"step": 2, "epoch": 3, "fork": 1, "screen": 0.3, "peak": 0.9, "restored": 0.9, "rate": 0.0015}
    assert (
        mf.trace([row(1, [16, 15]), fork])[1]
        == "step   2 epoch   3 FORK 1: screen 0.300 peak 0.900 restored 0.900 rate 0.0015"
    )


def test_large_drops_and_whether_each_recovered():
    def row(step: int, worst: float) -> dict[str, object]:
        return {"step": step, "accuracy": {"sum": 1.0, "difference": worst}}

    rows = [row(1, 0.95), row(1, 0.5), row(2, 0.92), row(2, 0.7), row(3, 0.91), row(3, 0.2), row(4, 0.4)]
    assert mf.large_drops(rows) == [(1, True), (3, False)]  # 0.92 to 0.7 isn't large; 0.91 to 0.2 never back


def test_outcome_stops_at_the_first_certified_step_and_counts_its_epochs():
    def row(step: int, codes: int, worst: float, **extra: object) -> dict[str, object]:
        accuracy = {"sum": 1.0, "quotient-0": worst}
        return {"step": step, "accuracy": accuracy, "factors": [{"codes": codes}], **extra}

    rows = [
        row(0, 1, 0.1, epoch=0, seconds=0.2),
        row(1, 8, 0.6, end="passed", epochs=3, certified=False, seconds=30.0),
        row(2, 16, 0.997, epoch=5, seconds=200.0),  # kept to three decimals, not 1.0
        row(2, 7, 0.3, end="plateau", epochs=40, certified=False, seconds=300.0),  # a collapse
        row(3, 9, 1.0, end="passed", epochs=2, certified=True, seconds=330.0),
    ]
    assert mf.outcome(rows) == mf.Outcome(3, "certified", 45, 5.5, 1.0, 3, 1, 1, 0)
    assert mf.outcome(rows[:4]) == mf.Outcome(2, "plateau", 43, 5.0, 0.997, 2, 1, 1, 1)
    assert mf.outcome(rows[:1]).end == "running"
    spent = row(3, 7, 0.4, end="spent", epochs=6, certified=False, seconds=330.0)  # 5 trained
    assert mf.outcome([*rows[:4], spent]).epochs == 48
    assert mf.progress(rows[:4]) == "step 2 plateau, 43 epochs, 5 min, peak 0.997 at step 2"
    annealing = {"step": 2, "epoch": 6, "seconds": 250.0, "anneal": {"phase": "cool"}}
    assert mf.progress([*rows[:3], annealing]) == (
        "step 2 running at epoch 6, 3 epochs before it, 4.2 min, peak 0.997 at step 2, anneal cool"
    )


def test_readout_counts_a_run_only_when_recertified_in_both_verdicts(tmp_path: Path):
    def run(arm: str, seed: int, certified: bool, network: bytes = b"") -> None:
        out = tmp_path / arm / f"seed{seed}"
        out.mkdir(parents=True)
        end = {
            "step": 4,
            "end": "passed" if certified else "spent",
            "epochs": 7,
            "certified": certified,
            "seconds": 60.0,
            "accuracy": {"sum": 1.0 if certified or seed == 10 else 0.5},  # seed 10 came near, then failed
            "factors": [{"codes": 9}],
        }
        (out / "log.jsonl").write_text(json.dumps(end) + "\n")
        if certified:
            (out / "certified-step-4.network.json").write_bytes(network)

    def certificate(name: str, network: bytes, ratio: float) -> None:
        sha = hashlib.sha256(network).hexdigest()
        result = {"sha256": sha, "snap_ratio": ratio, "certified": True}
        (tmp_path / "certify" / f"{name}.certificate.json").write_text(json.dumps(result))

    (tmp_path / "certify").mkdir()
    run("family", 1, True, b"one")
    run("family", 2, True, b"two")  # only one verdict: not counted
    run("family", 10, False)
    run("quotient", 1, False)
    run("quotient", 2, False)
    run("quotient", 3, False)
    for ratio in mf.VERDICTS:
        certificate(f"one-{ratio}", b"one", ratio)
    certificate("two", b"two", 1.0)
    lines = mf.readout(tmp_path, ["family", "quotient"], tmp_path / "certify")
    assert lines[1] == (
        "seed1: certified at step 4, 7 epochs, 1.0 min; worst answer at best 1.000 (step 4), collapses 0, "
        "large drops 0 (0 unrecovered)"
    )
    assert "NOT RECERTIFIED" in lines[2] and lines[3].startswith("seed10: spent")  # seeds in numeric order
    assert lines[4].startswith("family: 1 of 3 certified (Wilson 95%: 0.06-0.79)")
    assert lines[5] == "family's 2 failures: 1 came near a solution (worst answer >= 0.95), 0 collapsed"
    assert (
        lines[-4].startswith("quotient: 0 of 3 certified")
        and lines[-2] == "quotient: 0 large drops in 0 runs, 0 recovered"
    )
    assert lines[-1] == "Fisher's exact, one-sided (family more often than quotient): p 0.5"
    pairs = [("family", "quotient"), ("quotient", "family")]
    lines = mf.readout(tmp_path, ["family", "quotient"], tmp_path / "certify", pairs)
    assert lines[-2:] == [
        "Fisher's exact, one-sided (family more often than quotient): p 0.5, Holm-adjusted over 2: 1",
        "Fisher's exact, one-sided (quotient more often than family): p 1, Holm-adjusted over 2: 1",
    ]


def test_readout_with_long_tests_counts_a_certified_run_only_long_tested_and_a_codeless_run_by_its_last_network(
    tmp_path: Path,
):
    def run(arm: str, seed: int, network: bytes, certified: bool) -> None:
        out = tmp_path / arm / f"seed{seed}"
        out.mkdir(parents=True)
        end: dict[str, Any] = {"step": 4, "end": "passed" if certified else "spent", "epochs": 7, "seconds": 60.0}
        end |= {"certified": certified, "accuracy": {"sum": 1.0 if certified else 0.3}}
        if arm == "family":
            end["factors"] = [{"codes": 9}]
        rows = [{**end, "end": "passed", "step": 3, "certified": False}, end]  # a codeless log: no collapse read
        (out / "log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (out / ("certified-step-4.network.json" if certified else "network.json")).write_bytes(network)
        sha = hashlib.sha256(network).hexdigest()
        for ratio in mf.VERDICTS if certified else ():
            result = {"sha256": sha, "snap_ratio": ratio, "certified": True}
            (tmp_path / "certify" / f"{arm}-{seed}-{ratio}.certificate.json").write_text(json.dumps(result))

    def long_test(name: str, network: bytes, wrong: int) -> None:
        result = {"cases": 10, "passed": 10 - wrong}
        long = {"sha256": hashlib.sha256(network).hexdigest(), "lengths": {"128": {"properties": {"U1": result}}}}
        (tmp_path / "long" / name).mkdir(parents=True)
        (tmp_path / "long" / name / "net.long-test.json").write_text(json.dumps(long))

    (tmp_path / "certify").mkdir()
    for seed, network in enumerate([b"pass", b"fail", b"untested"]):
        run("family", seed, network, certified=True)
    run("continuous", 0, b"last-pass", certified=False)
    run("continuous", 1, b"last-fail", certified=False)
    for name, network, wrong in [("a", b"pass", 0), ("b", b"fail", 2), ("c", b"last-pass", 0), ("d", b"last-fail", 1)]:
        long_test(name, network, wrong)
    lines = mf.readout(tmp_path, ["family", "continuous"], tmp_path / "certify", long_tests=tmp_path / "long")
    assert lines[1].endswith("; long tests passed") and lines[2].endswith("; LONG TEST FAILED")
    assert lines[3].endswith("; NOT LONG-TESTED")
    assert lines[4].startswith("family: 1 of 3 succeeded")
    assert lines[8].startswith("seed0: spent") and lines[8].endswith("; long tests passed")
    assert lines[10].startswith("continuous: 1 of 2 succeeded")
    assert mf.readout(tmp_path, ["family", "continuous"], tmp_path / "certify")[10].startswith(
        "continuous: 0 of 2 certified"
    )


def test_a_machine_unit_test_counts_each_machines_successes_and_ranks_the_arms_machines(tmp_path: Path):
    # the paper workplan's D19 and D24 (T10, T10r): backdrops <test>-<arm>-<machine>, two seeds a
    # machine, a run counted when certified in both verdicts and long-tested
    (tmp_path / "certify").mkdir()
    (tmp_path / "long").mkdir()

    def run(backdrop: str, seed: int, succeeded: bool) -> None:
        out = tmp_path / "runs" / backdrop / f"seed{seed}"
        out.mkdir(parents=True)
        end = {"step": 4, "end": "passed" if succeeded else "spent", "epochs": 7, "seconds": 60.0}
        end |= {"certified": succeeded, "accuracy": {"answers": 1.0 if succeeded else 0.3}, "factors": [{"codes": 9}]}
        (out / "log.jsonl").write_text(json.dumps(end) + "\n")
        if not succeeded:
            return
        network = f"{backdrop}-{seed}".encode()
        (out / "certified-step-4.network.json").write_bytes(network)
        sha = hashlib.sha256(network).hexdigest()
        for ratio in mf.VERDICTS:
            result = {"sha256": sha, "snap_ratio": ratio, "certified": True}
            (tmp_path / "certify" / f"{backdrop}-{seed}-{ratio}.certificate.json").write_text(json.dumps(result))
        long = {"sha256": sha, "lengths": {"128": {"properties": {"U1": {"cases": 4, "passed": 4}}}}}
        (tmp_path / "long" / f"{backdrop}-{seed}").mkdir()
        (tmp_path / "long" / f"{backdrop}-{seed}" / "net.long-test.json").write_text(json.dumps(long))

    wins = {"p0": {0: 0, 1: 0, 2: 1}, "p2": {3: 2, 4: 1, 5: 0}}  # each machine's successes, of two seeds
    for arm, machines in wins.items():
        for machine, won in machines.items():
            for seed in (400, 401):
                run(f"t10-{arm}-{machine}", seed, seed - 400 < won)
    run("t10r-p2-9", 400, True)  # another test's backdrop: not read
    runs = mf.machine_unit_runs(
        tmp_path / "runs",
        "t10",
        ["p0", "p2"],
        mf.recertifications(tmp_path / "certify"),
        mf.long_tested(tmp_path / "long"),
    )
    assert sorted(runs["p0"]) == [0, 1, 2] and sorted(runs["p2"]) == [3, 4, 5]
    assert mf.machine_successes(runs["p2"]) == [2, 1, 0] and mf.machine_successes(runs["p2"], 1) == [1, 1, 0]
    lines = mf.machine_unit_readout(runs, [("p2", "p0")])
    assert lines[0] == "p0: 1 of 3 machines, 1 of 6 runs (2: 1)"
    assert lines[1] == "p2: 2 of 3 machines, 3 of 6 runs (3: 2, 4: 1)"
    p = inference.rank_sum_greater([2, 1, 0], [0, 0, 1])
    once = inference.rank_sum_greater([1, 1, 0], [0, 0, 1])
    assert lines[2] == f"p2 > p0: rank-sum p {p:.4f}, Holm {p:.4f}; each machine at most once: p {once:.4f}"


def test_each_screen_row_logs_every_answer_kinds_share_beside_the_pooled_one(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=1)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    screened = [r for r in rows if "screen" in r]
    assert screened and all(set(r["kinds"]) >= {"quotient-nonzero"} for r in screened)
    assert all(0.0 <= share <= 1.0 for r in screened for share in r["kinds"].values())


def _point(step: int, epoch: int, screen: float, **kinds: float) -> dict[str, object]:
    return {"step": step, "epoch": epoch, "screen": screen, "kinds": kinds}


def test_the_trigger_scorer_counts_drops_catches_and_false_alarms_per_epoch():
    rows = [
        _point(1, 1, 0.5, a=0.4, b=0.6),  # still learning
        _point(1, 2, 0.95, a=0.92, b=0.98),
        _point(1, 3, 0.55, a=0.2, b=0.9),  # a large drop (0.92 to 0.2), pooled below 0.6 of 0.95
        _point(1, 4, 0.97, a=0.95, b=0.99),  # recovered
        _point(2, 1, 0.96, a=0.94, b=0.98),
        _point(2, 2, 0.7, a=0.85, b=0.6),  # a large drop (the worst 0.94 to 0.6) the pooled screen misses; never back
    ]
    pooled = mf.score_trigger(rows, mf.Trigger())
    assert (pooled.points, pooled.drops, pooled.caught, pooled.unrecovered) == (6, 2, 1, 1)
    assert pooled.false_alarms == 0 and pooled.early == 0
    worst = mf.score_trigger(rows, mf.Trigger(signal="worst", rule="fall", threshold=0.3))
    assert (worst.drops, worst.caught, worst.unrecovered_caught) == (2, 2, 1)


def test_the_vocabulary_keeps_every_earlier_studys_tokens_and_gives_the_product_its_own():
    assert mf.Encoding(5, mf.FAMILY).tokens == mf.COLUMNS + 2  # the studies before the product's
    assert mf.Encoding(5, mf.QUOTIENT).tokens == mf.Encoding(5, mf.FAMILY).tokens
    wide = [mf.Encoding(5, arm, mf.WIDE) for arm in (mf.QUOTIENT, mf.SUM, mf.FAMILY, mf.PRODUCT)]
    assert {e.tokens for e in wide} == {mf.COLUMNS + 3}
    assert wide[0].query_token("sum") == mf.Encoding(5, mf.SUM).query_token("sum")
    tokens, labels = mf.Encoding(5, mf.PRODUCT, mf.WIDE).example([7, 5])  # columns (2, 1) then (1, 2)
    states = mf.Encoding(5, mf.PRODUCT, mf.WIDE).states([7, 5])
    assert tokens[1::2] == (float(mf.COLUMNS + 2),) * 2
    assert labels[1::2] == tuple((a * b) % 5 for a, b in states)
    with pytest.raises(AssertionError):
        mf.Encoding(5, mf.PRODUCT)  # the product has no token in the earlier vocabulary


CONTROLS = ("sum", "suffix", "token")  # T1's vocabulary: its three arms ask the same tokens


@pytest.mark.parametrize("p", [5, 7, 11, 13])
def test_t1s_controls_answer_the_columns_digits_not_the_state(p: int):
    rng = random.Random(p)
    cols = [rng.randrange(mf.COLUMNS) for _ in range(40)]
    digits = [t // mf.BASE + t % mf.BASE for t in cols]  # a column's a + b
    _, token = mf.Encoding(p, mf.TOKEN, CONTROLS).example(cols)
    _, suffix = mf.Encoding(p, mf.SUFFIX, CONTROLS).example(cols)
    _, total = mf.Encoding(p, mf.SUM, CONTROLS).example(cols)
    states = mf.Encoding(p, mf.SUM, CONTROLS).states(cols)
    quotients = tuple(md.quotient(A, B, p) for A, B in states)
    assert token[0::2] == suffix[0::2] == total[0::2] == quotients  # the certified answer the same
    assert token[1::2] == tuple(d % p for d in digits)
    assert suffix[1::2] == tuple((d + e) % p for d, e in zip(digits, [0, *digits[:-1]], strict=True))
    assert total[1::2] == tuple((A + B) % p for A, B in states)
    # neither control is a function of division's state: two prefixes reach one state, their answers differ
    seen: dict[tuple[int, int], set[int]] = {}
    for n in range(1, 4):
        for word in itertools.product(range(mf.COLUMNS), repeat=n):
            (A, B) = mf.Encoding(p, mf.TOKEN, CONTROLS).states(list(word))[-1]
            seen.setdefault((A, B), set()).add(mf.Encoding(p, mf.SUFFIX, CONTROLS).example(list(word))[1][-1])
    assert any(len(answers) > 1 for answers in seen.values())
    tokens = mf.Encoding(p, mf.SUFFIX, CONTROLS).example(cols)[0]
    assert tokens[1::2] == (float(mf.COLUMNS + 1),) * len(cols)  # suffix's query token, the vocabulary's second


def _example_by_the_states(encoding: mf.Encoding, columns: list[int]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Encoding.example as first written (before the window-program workplan's stage 1): the states and the operations."""
    tokens: list[float] = []
    labels: list[int] = []
    for token, (A, B) in zip(columns, encoding.states(columns), strict=True):
        tokens.append(float(token))
        labels.append(md.quotient(A, B, encoding.p))
        for operation in encoding.queries:
            tokens.append(float(encoding.query_token(operation)))
            labels.append(mf.OPERATIONS[operation](A, B, encoding.p))
    return tuple(tokens), tuple(labels)


@pytest.mark.parametrize("p", md.PRIMES)
def test_an_example_is_the_states_answers_for_every_encoding_and_length(p: int):
    rng = random.Random(p)
    encodings = [mf.Encoding(p, queries) for queries in (mf.QUOTIENT, mf.SUM, mf.FAMILY)] + [
        mf.Encoding(p, queries, mf.WIDE) for queries in (mf.QUOTIENT, mf.SUM, mf.PRODUCT, mf.FAMILY)
    ]
    for encoding in encodings:
        for n in [*range(65), 1024]:
            columns = mf.columns(n, rng)
            assert encoding.example(columns) == _example_by_the_states(encoding, columns)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (path / "log.jsonl").read_text().splitlines()]


def test_a_guarded_run_logs_each_epochs_arming_and_rollbacks_and_each_steps_sum(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=2, guard=True)
    rows = _rows(tmp_path)
    epochs = [r for r in rows if r.get("epoch", 0) > 0 and "end" not in r]
    assert epochs and all(isinstance(r["armed"], bool) and r["rollbacks"] == 0 for r in epochs)
    assert all(r["updates"] == (1 if r["armed"] else 0) for r in epochs)  # one batch an epoch, counted when armed
    assert all(r["rollbacks"] == 0 for r in rows if "end" in r)


def _always_right(encoding: object, kinds: object) -> bool:
    return True


def test_the_guards_reads_leave_training_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Armed every epoch, the machine read after every batch (never right, so nothing undone): the
    networks and logs are the unguarded run's, but for timings and the guard's fields."""
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    encoding = mf.Encoding(5, mf.FAMILY)
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path / "off", settings, cap=2, fork=mf.Fork())
    monkeypatch.setattr(mf, "certified_right", _always_right)
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path / "on", settings, cap=2, fork=mf.Fork(), guard=True)
    on = _rows(tmp_path / "on")
    assert all(r["armed"] for r in on if r.get("epoch", 0) > 0 and "end" not in r)
    guard_fields = {"seconds", "armed", "updates", "rollbacks"}
    strip = [
        [{k: v for k, v in r.items() if k not in guard_fields} for r in rows] for rows in (_rows(tmp_path / "off"), on)
    ]
    assert strip[0] == strip[1]
    for name in ("step-1.network.json", "step-2.network.json"):
        assert (tmp_path / "off" / name).read_bytes() == (tmp_path / "on" / name).read_bytes()


def test_a_guard_that_undoes_nearly_every_update_for_its_stall_epochs_ends_the_run_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Armed every epoch and every update undone (right before, wrong after): the run stops
    "stalled" at its STALL_EPOCHS-th epoch, the step's network certified as any step's end."""
    from indrajala_studies import certificate_guard as cg

    settings = mf.Settings(
        epoch_examples=64, max_epochs=20, patience=30, diagnose_every=30, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    reads = iter([True, False] * 100)  # each epoch: right when armed, wrong after its one update
    monkeypatch.setattr(mf, "certified_right", _always_right)

    def stub(machine: object, task: object) -> bool:
        return next(reads)

    monkeypatch.setattr(cg, "right", stub)
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=3, guard=True)
    rows = _rows(tmp_path)
    ends = [r for r in rows if "end" in r]
    assert [(r["step"], r["end"], r["epochs"]) for r in ends] == [(1, "stalled", mf.STALL_EPOCHS)]
    assert ends[0]["rollbacks"] == mf.STALL_EPOCHS and isinstance(ends[0]["certified"], bool)
    assert mf.outcome(rows).epochs == mf.STALL_EPOCHS


# the division ladder's feasibility table (git show f9101d8:docs/division-ladder-workplan.md): p, codes a factor, the cap
LADDER = {
    13: (38, 1612),
    17: (50, 2788),
    19: (56, 3496),
    23: (67, 5017),
    29: (85, 8065),
    31: (91, 9241),
    37: (108, 13032),
    41: (120, 16080),
}


@pytest.mark.parametrize("p", sorted(LADDER))
def test_the_ladders_cells_are_feasible_with_the_surplus_held(p: int):
    from fractions import Fraction

    codes, cap = LADDER[p]
    factors = mf.scaled_factors(p * p, Fraction(32, 11), 2)
    assert p in md.PRIMES and factors == [(16, codes), (16, codes)]
    assert codes >= p and codes * codes >= 8 * p * p  # a counter fits each factor; joint codes 8.5x the states
    assert mf.exclusion_bound(p * p, factors) == cap
    assert mf.scaled_factors(121, Fraction(32, 11), 2) == [(16, 32), (16, 32)]  # p 11's cell, exactly


# run --resume (the code-consolidation workplan's stage 3)

RESUMED = mf.Settings(
    epoch_examples=64, max_epochs=4, patience=10, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
)
# random-3-0 with A under it: step 2 passes, step 3 ends on a plateau
PLATEAUS = mf.Settings(
    epoch_examples=64, max_epochs=6, patience=2, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
)
RESUMED_FORK = mf.Fork(trigger=mf.Trigger(threshold=2.0), cool=0.5, limit=1)
# five runs: one forking, ending on its epochs; one guarded, ending steps at a plateau; one consolidated;
# one consolidated restoring its plateaued step's best; one forking without the commitment at step 1
RESUMABLE: dict[str, dict[str, Any]] = {
    "forked": {
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 2,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
    },
    "guarded": {
        "encoding": mf.Encoding(5, mf.QUOTIENT),
        "settings": mf.Settings(
            epoch_examples=64, max_epochs=6, patience=2, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
        ),
        "cap": 3,
        "budget_epochs": 10,
        "guard": True,
    },
    # the consolidation workplan's arms together: s0 2, the long tail, A's groups regrouped at a fork
    "consolidated": {
        "encoding": mf.target_named("random-3-0"),
        "settings": RESUMED,
        "cap": 3,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
        "long_tail": 4,
        "consolidate": True,
    },
    "restored": {
        "encoding": mf.target_named("random-3-0"),
        "settings": PLATEAUS,
        "cap": 3,
        "budget_epochs": 10,
        "consolidate": True,
        "restore_best": True,
    },
    "uncommitted": {
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 2,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
        "commitment_from": 2,
    },
    # R3 and R4 (the one-code-one-state workplan): random-3-0's step 3 ends on a plateau, restoring the kept machine
    "kept": {
        "encoding": mf.target_named("random-3-0"),
        "settings": PLATEAUS,
        "cap": 3,
        "budget_epochs": 10,
        "keep_machine": True,
        "counterexamples": 0.25,
    },
    # the usage thermostat (the commitment-control workplan's stage 2), its levels low enough to move on 4 codes
    "thermostat": {
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 2,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
        "thermostat": mf.Thermostat(floor=1.5, restore=0.5, ramp=3),
    },
    # annealing (the commitment-control workplan's stage 3), cooling fast enough to harden within the run
    "annealed": {
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 2,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
        "anneal": mf.Anneal(start=0.5, halving=1, ramp=2),
    },
    "crucible": {  # D26: the warm-up's end, the crucible's blocks, then growth, within the budget
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,  # the warm-up's 2 steps of up to 4 epochs, the crucible's block, growth's
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1, handover=True),
        "crucible": True,
    },
    "gated": {  # D29: a gate no screen meets, so the cool's clock holds from its second epoch
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1, handover=True, gate=2.0, window=3.0),
        "crucible": True,
    },
    "adaptive": {  # D33: the cool's temperatures and measurements carried over every stop
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, handover=True, shape="adaptive", settle=2),
        "crucible": True,
    },
    "capped": {  # D38's coupling cap and D35's answers-only stop: the cap and the stop's readings resumed
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1, handover=True),
        "crucible": True,
        "coupling_cap": 0.0,
        "state_bound": 25,
    },
    "continuous": {  # D48-D53: the clock's shares, the sharpness's start, the floors and the length resumed
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1),
        "crucible": True,
        "continuous": mf.Continuous(low=0.0, high=2.0, growth=0.5, curve_every=3, save_every=2),
    },
    "paced": {  # D61 (b): each coupling's allowance and last move carried over every stop
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1),
        "crucible": True,
        "continuous": mf.Continuous(low=0.0, high=2.0, growth=0.5, curve_every=3, save_every=2, pace=True),
    },
    "smoothed": {  # D61 (c): the window's readings carried over every stop
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1),
        "crucible": True,
        "continuous": mf.Continuous(low=0.0, high=2.0, growth=0.5, curve_every=3, save_every=2, smooth=True),
    },
    "reheated": {  # D63 (a): a clock run back (the screen below the level), its held share carried over every stop
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1),
        "crucible": True,
        "continuous": mf.Continuous(low=1.2, high=2.0, growth=0.5, curve_every=3, save_every=2, reheat=True),
    },
    "slowed": {  # D66: the reheats counted and the clock's back carried over every stop
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1),
        "crucible": True,
        "continuous": mf.Continuous(
            low=1.2, high=2.0, growth=0.5, curve_every=3, save_every=2, reheat=True, slowing=True
        ),
    },
    "backtracked": {  # the incremental reheat: its stages, mode and wait carried over every stop
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 6,
        "budget_epochs": 14,
        "anneal": mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False),
        "warm_up": mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=3),
        "crucible": True,
        "continuous": mf.Continuous(low=1.2, high=2.0, growth=0.5, curve_every=3, save_every=2, incremental=True),
    },
}


def test_the_coupling_cap_holds_every_coupling_and_the_answers_stop_judges_every_steps_end(tmp_path: Path):
    config = dict(RESUMABLE["capped"])
    encoding, settings = config.pop("encoding"), config.pop("settings")
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path, settings, **config)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    couplings = [c for r in rows if "coupling" in r.get("anneal", {}) for c in r["anneal"]["coupling"]]
    assert couplings and all(c == 0.0 for c in couplings)  # every phase, hard factors' too
    ends = [r for r in rows if "end" in r]
    assert ends and all("answers_stop" in r for r in ends)
    judged = [r["answers_stop"] for r in ends if not r["answers_stop"].get("warm")]
    assert judged and all(set(j["answers"]) == {"8", "16", "24", "48"} for j in judged)
    passed = [r for r in ends if r["answers_stop"]["passed"]]
    assert all(isinstance(r["answers_stop"]["strict"], bool) for r in passed)
    assert len(list(tmp_path.glob("answers-stop-step-*.network.json"))) == len(passed) <= 1  # the stop ends it
    assert not passed or passed[0] is ends[-1]


CONTINUOUS = mf.Continuous(low=0.0, high=2.0, growth=0.5, curve_every=3, save_every=2)


def test_the_continuous_rate_coupling_and_growth_are_continuous_and_bounded():
    c = mf.Continuous()
    assert c.rate(0.99, 1.0) == pytest.approx(1.0) and c.rate(1.0, 1.0) == 1.0
    assert c.rate(0.95, 1.0) == pytest.approx(0.0) and c.rate(0.5, 1.0) == 0.0
    assert c.rate(0.97, 1.0) == pytest.approx(0.5)  # linear between (D49)
    assert mf.Continuous.coupling(0.4, 0.4) == 0.0 and mf.Continuous.coupling(1.0, 0.4) == 1.0
    assert mf.Continuous.coupling(0.7, 0.4) == pytest.approx(0.5)  # half of what was left (D50)
    assert mf.Continuous.coupling(0.3, 0.4) == 0.0  # less sharp than the cool's start: uncoupled
    assert c.grow(8.0, 0.0, 1.0, 69) == 8.0 and c.grow(8.0, 1.0, 0.0, 69) == 8.0  # no rate or no coupling
    assert c.grow(8.0, 1.0, 1.0, 69) == pytest.approx(8.0 * math.exp(0.02))
    assert c.grow(68.9, 1.0, 1.0, 69) == 69.0  # to the cap, no further (D52)


def test_growth_lengths_mix_the_floors_and_the_next_lengths_by_the_rest():
    assert mf.growth_lengths(4.0, 8) == [1, 2, 3, 4, 1, 2, 3, 4]  # whole: epoch_lengths's
    assert mf.growth_lengths(4.0, 8) == mf.epoch_lengths(4, 8)
    mixed = mf.growth_lengths(4.5, 10)  # half the examples over 1..5, half over 1..4
    assert mixed == [1, 2, 3, 4, 5, 1, 2, 3, 4, 1]
    assert max(mf.growth_lengths(4.05, 100)) == 5 and max(mf.growth_lengths(4.04, 100)) == 4  # 5 from 1..5 first
    assert mf.growth_screen(8, 8.0) == list(range(1, 9))
    assert mf.growth_screen(8, 20.7) == sorted(set(range(1, 9)) | set(mf.whole_lengths(20)))


def test_sharpness_reads_hardnesss_share_beside_the_mean_largest_weight():
    net = mf.network(mf.Encoding(5, mf.FAMILY), [(4, 4), (4, 4)], 0)
    rows = [mf._windows(mf.Encoding(5, mf.FAMILY), n, 6, f"sharp-{n}")[0] for n in (1, 3)]
    for temperatures in ([0.5, None], [1e-3, 2.0]):
        sharp = mf.sharpness(net, rows, temperatures, 0.9)
        assert [None if v is None else v[1] for v in sharp] == mf.hardness(net, rows, temperatures, 0.9)
        assert all(v is None or 1 / 4 <= v[0] <= 1.0 for v in sharp)
    cold, hot = mf.sharpness(net, rows, [1e-6, 1e6], 0.9)
    assert cold is not None and hot is not None and cold[0] > 0.999 and hot[0] < 0.26


def test_a_continuous_run_moves_every_switch_smoothly_and_reads_every_epoch(tmp_path: Path):
    config = dict(RESUMABLE["continuous"])
    encoding, settings = config.pop("encoding"), config.pop("settings")
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path, settings, **config)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    (end,) = [r for r in rows if "warm_up_end" in r]
    after = [r for r in rows if r["step"] > end["step"]]
    epochs = [r for r in after if "anneal" in r]
    assert epochs and all(r["anneal"]["temperature"][0] is not None for r in epochs)  # never hard (D51)
    assert all(0.0 <= r["anneal"]["rate"] <= 1.0 for r in epochs)
    assert all(c == 0.0 for r in epochs if r["anneal"]["phase"] != "cool" for c in r["anneal"]["coupling"])
    cools = [r for r in epochs if r["anneal"]["phase"] == "cool"]
    assert cools and cools[0]["anneal"]["coupling"] == [0.0]  # the cool starts uncoupled (D50)
    lengths = [r["anneal"]["length"] for r in epochs]
    assert lengths[0] == end["crucible"] and lengths == sorted(lengths) and lengths[-1] > lengths[0]  # D52
    assert all(len(r["anneal"]["sharpness"]) == len(r["anneal"]["hard"]) == 1 for r in epochs)
    ends = [r for r in after if "end" in r]
    assert ends and all(r["epochs"] == 1 for r in ends)  # a step an epoch (D53)
    assert all(set(r["certificates"]) == {"default", "strict"} for r in ends)
    assert all(set(r["snapped_answers"]) == {"8", "16", "24", "48"} for r in ends)
    assert [r["step"] for r in ends if "melting" in r] == [r["step"] for r in ends if r["step"] % 3 == 0]
    saved = {int(p.name.split("-")[1].split(".")[0]) for p in tmp_path.glob("step-*.network.json")}
    assert {r["step"] for r in ends} & saved == {r["step"] for r in ends if r["step"] % 2 == 0}
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["continuous"] == repr(CONTINUOUS)
    with pytest.raises(AssertionError, match="continuous crystallisation"):
        mf.run(
            encoding,
            [(4, 4)],
            0,
            1e9,
            tmp_path / "x",
            settings,
            cap=6,
            anneal=config["anneal"],
            warm_up=mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1, handover=True),
            crucible=True,
            continuous=CONTINUOUS,
        )


def test_a_cutoff_ends_an_uncertified_attempt_at_its_epochs_as_the_uncut_run_was_there(tmp_path: Path):
    # the batch runner workplan's D1, in epochs (its stage 5): rows identical to the uncut run's up to the
    # first step end with the cutoff's epochs trained, then "cut"
    config = dict(RESUMABLE["continuous"])
    encoding, settings = config.pop("encoding"), config.pop("settings")

    def rows(out: Path, **more: Any) -> list[dict[str, Any]]:
        mf.run(encoding, [(4, 4)], 0, 1e9, out, settings, **config, **more)
        return [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]

    def timeless(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k != "seconds"}

    whole = rows(tmp_path / "whole")
    trained: dict[int, int] = {}  # each step end's epochs trained in all
    total = 0
    for r in whole:
        if "end" in r:
            total += r["epochs"] - (r["end"] in mf.CENSORED)
            trained[r["step"]] = total
    cutoff = trained[4] + 1  # past step 4's epochs: step 5's end is the first at it
    cut = rows(tmp_path / "cut", cutoff_epochs=cutoff)
    assert [timeless(r) for r in cut[:-1]] == [timeless(r) for r in whole[: len(cut) - 1]]
    ends = [r for r in cut if "end" in r and "state_share" in r]
    assert ends[-1]["step"] == 5 and ends[-1]["end"] == "cut" and cut[-1]["step"] == 5
    assert max(r["step"] for r in whole if "end" in r) > 5
    assert {k: v for k, v in ends[-1].items() if k not in ("seconds", "end")} == {
        k: v
        for k, v in next(r for r in whole if r.get("step") == 5 and "end" in r).items()
        if k not in ("seconds", "end")
    }
    identity = json.loads((tmp_path / "cut" / "checkpoint.json").read_text())["run"]
    assert (
        identity["cutoff_epochs"] == cutoff
        and "cutoff_epochs" not in json.loads((tmp_path / "whole" / "checkpoint.json").read_text())["run"]
    )
    assert mf.restart_course(cut) == (None, float(trained[5]), False)  # a finished failure, not censored
    with pytest.raises(AssertionError, match="cutoff"):
        mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path / "x", settings, cap=2, cutoff_epochs=3)


def test_a_cutoff_leaves_a_run_certified_at_both_verdicts_to_its_own_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # certified at both verdicts from the start, but D35's stop never passes: the run goes past the cutoff
    config = dict(RESUMABLE["continuous"], state_bound=25)
    encoding, settings = config.pop("encoding"), config.pop("settings")
    stopping, certify = mf.answers_stopping, mf.wc.certify

    def never(*args: Any) -> tuple[bool, dict[str, Any]]:  # D35's stop never passes
        return False, stopping(*args)[1]

    monkeypatch.setattr(mf, "answers_stopping", never)

    def clean(*args: Any, **kwargs: Any) -> Any:  # every verdict certified: no wrong outputs or states, no label tie
        c = certify(*args, **kwargs)
        lead = dataclasses.replace(c.lead, short=0)
        return dataclasses.replace(c, wrong_outputs=0, wrong_states=0, wrong_ties=0, counterexample=None, lead=lead)

    monkeypatch.setattr(mf.wc, "certify", clean)
    # certified at both verdicts at step 3's end, epoch 9 (the warm-up's steps train 4 epochs each, the
    # strict verdict judged from step 3), so a cutoff at 10 epochs (step 4's end) leaves it running
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path, settings, **config, cutoff_epochs=10)
    ends = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    ends = [r for r in ends if "end" in r and "state_share" in r]
    assert "cut" not in {r["end"] for r in ends} and ends[-1]["step"] > 4


def _margin_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_until: int, passes: int = 10**6
) -> list[dict[str, Any]]:
    # D35's stop passes at the first `passes` steps after the warm-up (and on a restored network); the
    # certificate's ties split until a step
    config = dict(RESUMABLE["continuous"], budget_epochs=40, state_bound=25, margin_wait=3)
    encoding, settings = config.pop("encoding"), config.pop("settings")
    stopping, passed = mf.answers_stopping, [0]

    def passing(*args: Any) -> tuple[bool, dict[str, Any]]:
        if args[-1]:  # warm
            return False, stopping(*args)[1]
        passed[0] += 1
        verdict = passed[0] <= passes or passed[0] > 4  # the first pass's step and W 3 after it, then the restore's
        return verdict, {"answers_stop": {"passed": verdict}}

    certify, steps = mf.wc.certify, [0]

    def splitting(*args: Any, **kwargs: Any) -> Any:
        c = certify(*args, **kwargs)
        if "snap_ratio" in kwargs:  # the strict certificate (continuous_readings)
            return c
        steps[0] += 1
        return dataclasses.replace(c, split_ties=1 if steps[0] <= split_until else 0)

    monkeypatch.setattr(mf, "answers_stopping", passing)
    monkeypatch.setattr(mf.wc, "certify", splitting)
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path, settings, **config)
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["margin_wait"] == 3
    return [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]


def _cool(tmp_path: Path, kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = dict(RESUMABLE[kind])
    encoding, settings = config.pop("encoding"), config.pop("settings")
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path, settings, **config)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    return [r["anneal"] for r in rows if r.get("anneal", {}).get("phase") == "cool"], config


def test_the_paced_coupling_moves_at_most_its_allowance_which_the_rate_doubles_or_halves(tmp_path: Path):
    cool, _ = _cool(tmp_path, "paced")
    assert cool[0]["coupling"] == [0.0] and cool[0]["allowance"] == [cool[0]["target"][0]]  # D61 (b)
    for before, after in itertools.pairwise(cool):
        (a,), (b,), (rate,) = before["allowance"], after["allowance"], [after["rate"]]
        first = max(after["target"][0] - before["coupling"][0], 0.0)  # its first rise, taken whole
        assert b == pytest.approx(a * 2 ** (2 * rate - 1) if a > 0 else first, abs=2e-4)
        assert abs(after["coupling"][0] - before["coupling"][0]) <= b + 1e-4
        assert (
            after["coupling"][0] == after["target"][0] or abs(after["coupling"][0] - before["coupling"][0]) > b - 1e-4
        )
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["coupling_pace"] is True and "sharpness_smoothing" not in identity


def test_the_savitzky_golay_line_is_exact_on_a_line_and_the_mean_at_one_time():
    assert mf.savitzky_golay([[0.0, 0.1, 1.0], [1.0, 0.3, 1.0], [3.0, 0.7, 1.0]], 3.0) == pytest.approx([0.7, 1.0])
    assert mf.savitzky_golay([[2.0, 0.2], [2.0, 0.6]], 2.0) == pytest.approx([0.4])
    assert mf.savitzky_golay([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]], 2.0) == pytest.approx([5 / 6])  # a jump, spread


def test_the_smoothed_coupling_reads_the_sharpness_line_over_the_cools_e_folding_time(tmp_path: Path):
    cool, config = _cool(tmp_path, "smoothed")
    span = config["warm_up"].halving / math.log(2)
    assert cool[0]["smoothed"] == pytest.approx(cool[0]["sharpness"], abs=1e-6)  # the cool's first (D61 (c))
    for i, now in enumerate(cool):
        window = [[a["clock"], a["sharpness"][0]] for a in cool[: i + 1] if a["clock"] > now["clock"] - span]
        assert now["smoothed"][0] == pytest.approx(mf.savitzky_golay(window, now["clock"])[0], abs=1e-3)
    start = cool[0]["sharpness"][0]
    assert all(a["target"][0] == pytest.approx(mf.Continuous.coupling(a["smoothed"][0], start), abs=1e-3) for a in cool)
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["sharpness_smoothing"] is True and identity["continuous"] == repr(CONTINUOUS)


def test_the_reheats_rate_runs_below_zero_to_minus_one_and_only_with_it():
    continuous, reheated = mf.Continuous(), mf.Continuous(reheat=True)
    assert continuous.rate(0.5, 1.0) == 0.0 and reheated.rate(0.5, 1.0) == -1.0  # D63 (a): 2 low - high and below
    assert reheated.rate(2 * 0.95 - 0.99 + 0.02, 1.0) == pytest.approx(-0.5)
    assert reheated.rate(0.97, 1.0) == continuous.rate(0.97, 1.0) == pytest.approx(0.5)


def test_the_reheat_slowing_halves_the_forward_rate_after_each_reheat_and_leaves_the_run_back():
    slowed = mf.Continuous.slowed
    assert slowed(0.8, 0, False) == (0.8, 0, False)  # never reheated: unchanged
    assert slowed(-0.5, 0, False) == (-0.5, 0, True)  # running back: unchanged, marked
    assert slowed(0.0, 0, True) == (0.0, 0, True)  # held: not yet forward
    assert slowed(0.8, 0, True) == (0.4, 1, False)  # forward again: one reheat, half the pace
    assert slowed(1.0, 1, False) == (0.5, 1, False)
    assert slowed(-1.0, 1, False) == (-1.0, 1, True) and slowed(1.0, 1, True) == (0.25, 2, False)


def test_a_reheated_run_runs_its_clock_back_to_the_cools_start_and_grows_only_forward(tmp_path: Path):
    config = dict(RESUMABLE["reheated"], continuous=mf.Continuous(low=1.5, high=2.0, growth=0.5, reheat=True))
    encoding, settings = config.pop("encoding"), config.pop("settings")
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path, settings, **config)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    cool = [r["anneal"] for r in rows if r.get("anneal", {}).get("phase") == "cool"]
    warm_up = config["warm_up"]
    assert cool and all(a["rate"] < 0 for a in cool[1:])  # the screen about its level, far below `low`
    assert all(a["clock"] == warm_up.heat + warm_up.hold for a in cool)  # held at the cool's start, never before it
    assert len({a["length"] for a in cool}) == 1  # no growth running back
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["reheat"] is True and "reheat" not in identity["continuous"]  # in the identity apart


def test_the_incremental_reheat_records_a_stage_at_each_new_most_groups_only_cooling_forward():
    b = mf.Backtrack().recorded([4, 3], 10.0, True)
    assert b.maxima == [4, 3] and b.stages == [] and b.clock == 10.0  # the first reading sets the maxima
    b = b.recorded([5, 3], 11.0, True)
    assert b.maxima == [5, 3] and b.stages == [[10.0, 4, 3]]  # D1: the epoch before the split, the maxima then
    b = b.recorded([4, 3], 12.0, True).recorded([5, 3], 13.0, True)
    assert len(b.stages) == 1  # a flicker back to the maximum is no stage
    b = b.recorded([5, None], 14.0, True).recorded([6, 4], 15.0, True)
    assert b.stages[-1] == [14.0, 5, 3] and b.maxima == [6, 4]  # a hard factor read as no groups
    assert b.recorded([9, 9], 16.0, False).stages == b.stages  # not in the cool
    assert mf.Backtrack(mode="wait").recorded([9, 9], 16.0, True).stages == []  # not cooling forward


def test_the_incremental_reheat_holds_then_runs_back_a_stage_at_a_time_and_ends_with_none_left():
    b = mf.Backtrack(maxima=[6, 4], stages=[[20.0, 3, 3], [30.0, 5, 4]])
    assert b.moved(0.7, 40.0, 3) == (0.7, b)  # cooling: the rate as D49's
    rate, b = b.moved(0.0, 40.0, 3)
    rate, b = b.moved(0.0, 40.0, 3)
    assert rate == 0.0 and b.waited == 2 and b.mode == "cool"  # D2: held, short of the wait
    rate, b = b.moved(0.0, 40.0, 3)
    assert rate == -1.0 and b.mode == "back" and b.ceiling == 30.0  # D2, D3: the wait over, back to the latest
    for clock in (39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0, 32.0):
        rate, b = b.moved(0.5, clock, 3)  # D4: an epoch of clock an epoch, whatever the screen
        assert rate == -1.0 and b.mode == "back"
    rate, b = b.moved(0.5, 31.0, 3)
    assert rate == -1.0 and b.mode == "wait" and b.stages == [[20.0, 3, 3]] and b.maxima == [5, 4]  # popped
    recovered = b.moved(0.4, 30.0, 3)
    assert recovered == (0.4, mf.Backtrack(maxima=[5, 4], stages=[[20.0, 3, 3]], ceiling=30.0))  # D5: resumed
    for _ in range(2):
        rate, b = b.moved(0.0, 30.0, 3)
    rate, b = b.moved(0.0, 30.0, 3)
    assert rate == -1.0 and b.mode == "back" and b.ceiling == 20.0  # D6: the stage before
    rate, b = b.moved(0.0, 20.5, 3)
    assert rate == -0.5 and b.mode == "wait" and b.stages == [] and b.maxima == [3, 3]  # no further than it
    for _ in range(3):
        rate, b = b.moved(0.0, 20.0, 3)
    assert rate == 0.0 and b.exhausted  # D7: none left


def test_the_transitions_reading_finds_the_stages_holds_and_reheats_in_a_log():
    def row(step: int, clock: float, groups: list[int], rate: float) -> dict[str, Any]:
        return {
            "step": step,
            "anneal": {"phase": "cool", "clock": clock, "rate": rate, "groups": groups, "temperature": [1.0, 1.0]},
        }

    rows: list[dict[str, Any]] = [
        row(1, 1.0, [2, 2], 1.0),
        {"step": 2, "accuracy": {}, "state_share": 0.5},
        row(2, 2.0, [3, 2], 1.0),
        row(3, 3.0, [3, 4], 1.0),
        row(4, 3.0, [3, 4], 0.0),
        row(5, 3.0, [3, 4], 0.0),
        row(6, 3.5, [3, 4], 0.5),
        row(7, 2.5, [3, 4], -1.0),
        row(8, 1.5, [3, 4], -1.0),
        row(9, 2.0, [3, 4], 0.5),
        row(10, 2.0, [3, 4], 0.0),
    ]
    course = mf.transitions_course(rows)
    assert course is not None
    assert [s["step"] for s in course["stages"]] == [2, 3] and course["stages"][0]["share"] == 0.5
    assert [(h["step"], h["epochs"], h["recovered"]) for h in course["holds"]] == [(4, 2, True), (10, 1, False)]
    assert course["reheats"] == [{"step": 7, "from": 2.5, "to": 1.5, "crossed": 1, "past_first": False}]
    assert "2 stages" in mf.transitions_summary(course) and mf.transitions_course([]) is None


def test_an_incremental_reheat_run_never_runs_back_past_its_first_stage_and_ends_when_exhausted(tmp_path: Path):
    config = dict(RESUMABLE["backtracked"], budget_epochs=60)
    encoding, settings = config.pop("encoding"), config.pop("settings")
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path, settings, **config)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    cool = [r["anneal"] for r in rows if r.get("anneal", {}).get("phase") == "cool"]
    assert cool and all("backtrack" in a for a in cool)
    assert all(a["rate"] >= -1.0 for a in cool)  # D4: quasi-static
    warm_up = config["warm_up"]
    assert all(a["clock"] >= warm_up.heat + warm_up.hold for a in cool)
    backs = [a for a in cool if a["backtrack"]["mode"] == "back"]
    assert all(a["clock"] >= a["backtrack"]["ceiling"] for a in backs)  # D3: never past the ceiling
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["incremental_reheat"] is True and "incremental" not in identity["continuous"]


def test_the_margin_stop_waits_past_a_split_tie_and_stops_once_none_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = _margin_run(tmp_path, monkeypatch, split_until=10**6)  # never clears: waits W steps, then stops
    judged = [r for r in rows if "end" in r and r["answers_stop"].get("passed") and r["end"] != "waited"]
    first = judged[0]["step"]
    assert judged[0]["answers_stop"]["first_pass"] == first and judged[0]["answers_stop"]["split_ties"] == 1
    assert [r["step"] for r in judged] == list(range(first, first + 4))  # D59: W steps past the first pass
    assert [r["answers_stop"]["waited"] for r in judged] == [False, False, False, True]
    last = rows[-1]  # D60: the wait's stop, its row the best pass's network's
    assert last["end"] == "waited" and last["step"] == first + 3 and last["restored"] == first
    assert last["answers_stop"]["best_pass"] == first and last["answers_stop"]["waited"]
    assert "certificates" in last  # judged as a step's end (D53's readings)
    assert (tmp_path / f"first-pass-step-{first}.network.json").exists()
    assert (tmp_path / f"answers-stop-step-{first + 3}.network.json").read_text() == (
        tmp_path / f"best-pass-step-{first}.network.json"
    ).read_text()


def test_the_margin_stop_counts_the_steps_past_its_pass_and_ends_on_the_best_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = _margin_run(tmp_path, monkeypatch, split_until=10**6, passes=2)  # passes twice, then never
    ends = [r for r in rows if "end" in r and not r["answers_stop"].get("warm")]
    first = ends[0]["step"]
    assert [r["answers_stop"]["passed"] for r in ends] == [True, True, False, False, True]
    assert [r["step"] for r in ends] == [first, first + 1, first + 2, first + 3, first + 3]  # D60: every step counts
    assert ends[-1]["end"] == "waited" and ends[-1]["answers_stop"]["best_pass"] == first
    assert (tmp_path / f"answers-stop-step-{first + 3}.network.json").read_text() == (
        tmp_path / f"best-pass-step-{first}.network.json"
    ).read_text()


def test_the_margin_stop_stops_at_the_first_pass_whose_ties_split_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = _margin_run(tmp_path, monkeypatch, split_until=0)
    ends = [r for r in rows if "end" in r]
    judged = [r for r in ends if r["answers_stop"].get("passed")]
    assert len(judged) == 1 and judged[0] is ends[-1]  # no split tie: D35's stop as it was
    assert judged[0]["answers_stop"]["split_ties"] == 0 and "waited" not in judged[0]["answers_stop"]
    assert judged[0]["answers_stop"]["first_pass"] == -1


class Stop(Exception):
    """A run stopped from outside, as a kill would stop it."""


def _resumable(out: Path, resume: bool = False, kind: str = "forked") -> None:
    config = dict(RESUMABLE[kind])
    encoding, settings = config.pop("encoding"), config.pop("settings")
    mf.run(encoding, [(4, 4)], 0, 1e9, out, settings, resume=resume, **config)


def _outcome(out: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    # every row but its seconds, and every network's file
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    for row in rows:
        row.pop("seconds")
    return rows, {path.name: path.read_text() for path in sorted(out.glob("*.network.json"))}


def _stopped_at(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, call: int, after: bool) -> None:
    # module.name raises Stop on its call-th call, before its work or after it
    original, calls = getattr(module, name), [0]

    def stopping(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == call and not after:
            raise Stop
        result = original(*args, **kwargs)
        if calls[0] == call:
            raise Stop
        return result

    monkeypatch.setattr(module, name, stopping)


@pytest.mark.parametrize("kind", RESUMABLE)
def test_a_stopped_run_resumes_as_if_it_had_not_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    # stopped just before and just after each checkpoint is written (the rows past the one before
    # cut back, or none), so within a step, at a fork and at a step's start and end; and part way
    # through an epoch
    # the guard armed by a looser rule than a perfect screen, so the kinds carried over a stop decide it
    def armed(_encoding: Any, kinds: dict[str, float]) -> bool:
        return max(kinds.values()) >= 0.6

    monkeypatch.setattr(mf, "certified_right", armed)
    writes = [0]
    write = mf._write_checkpoint

    def counted_write(*args: Any) -> None:
        writes[0] += 1
        write(*args)

    with monkeypatch.context() as counted:
        counted.setattr(mf, "_write_checkpoint", counted_write)
        _resumable(tmp_path / "whole", kind=kind)
    whole = _outcome(tmp_path / "whole")
    ends = [(r["step"], r["end"]) for r in whole[0] if "end" in r]
    if kind == "forked":
        assert any("fork" in r for r in whole[0]) and ends == [(1, "budget"), (2, "spent")]
    elif kind == "consolidated":
        assert [step for step, _ in ends] == [2, 3]  # from s0
        assert any("fork" in r and "consolidation" in r for r in whole[0])
    elif kind == "kept":
        assert any("machine" in r and "counterexamples" in r for r in whole[0])
        assert any("restore_machine" in r for r in whole[0])
    elif kind == "thermostat":
        assert all("thermostat" in r for r in whole[0] if r.get("epoch", 0) > 0 and "end" not in r)
    elif kind == "annealed":
        annealed = [r["anneal"] for r in whole[0] if r.get("epoch", 0) > 0 and "end" not in r and "fork" not in r]
        assert annealed and annealed[0]["temperature"] == [0.5]
        assert annealed[-1]["temperature"] == [None] and annealed[-1]["beta"][0] > 0  # hardened, ramping back
    elif kind == "crucible":
        marks = [r["crucible"] for r in whole[0] if r.get("epoch") == 0 and "crucible" in r]
        assert marks and marks[0] != "grown" and marks[-1] == "grown"
    elif kind == "gated":  # the clock held, the held count carried over every stop
        assert max(r["anneal"].get("held", 0) for r in whole[0] if "anneal" in r) >= 2
    elif kind == "adaptive":  # the cool stepped more than once, part way through a temperature at some stop
        cool = [r["anneal"] for r in whole[0] if r.get("anneal", {}).get("phase") == "cool"]
        assert len({a["temperature"][0] for a in cool}) >= 3 and any(a["at"] == [2] for a in cool)
    elif kind == "capped":  # every coupling capped, every step's end judged by the answers-only stop
        couplings = [c for r in whole[0] if "coupling" in r.get("anneal", {}) for c in r["anneal"]["coupling"]]
        assert couplings and set(couplings) == {0.0} and all("answers_stop" in r for r in whole[0] if "end" in r)
    elif kind == "continuous":  # the length grown and the clock's shares carried over every stop
        cool = [r["anneal"] for r in whole[0] if r.get("anneal", {}).get("phase") == "cool"]
        assert cool and cool[-1]["length"] > cool[0]["length"] and any(0 < a["rate"] < 1 for a in cool[1:])
    elif kind == "paced":  # an allowance set and moved by the rate at some stop
        cool = [r["anneal"] for r in whole[0] if r.get("anneal", {}).get("phase") == "cool"]
        assert cool and any(a["allowance"][0] > 0 for a in cool) and any(0 < a["rate"] < 1 for a in cool[1:])
    elif kind == "smoothed":
        cool = [r["anneal"] for r in whole[0] if r.get("anneal", {}).get("phase") == "cool"]
        assert cool and any(a["smoothed"] != a["sharpness"] for a in cool)
    elif kind == "reheated":  # the rate below 0 at some stop, the clock never back past the cool's start
        cool = [r["anneal"] for r in whole[0] if r.get("anneal", {}).get("phase") == "cool"]
        assert cool and any(a["rate"] < 0 for a in cool) and all(a["clock"] >= 2 for a in cool)
    elif kind == "slowed":  # the reheats logged every cool epoch, the rate below 0 at some stop
        cool = [r["anneal"] for r in whole[0] if r.get("anneal", {}).get("phase") == "cool"]
        assert cool and all("reheats" in a for a in cool) and any(a["rate"] < 0 for a in cool)
    elif (
        kind == "backtracked"
    ):  # the incremental reheat's state logged every cool epoch, held or reheating at some stop
        cool = [r["anneal"] for r in whole[0] if r.get("anneal", {}).get("phase") == "cool"]
        assert cool and all("backtrack" in a for a in cool)
        assert any(a["backtrack"]["waited"] > 0 or a["backtrack"]["mode"] != "cool" for a in cool)
    elif kind == "uncommitted":
        assert any("fork" in r for r in whole[0]) and [step for step, _ in ends] == [1, 2]
    elif kind == "restored":
        assert ends == [(2, "passed"), (3, "plateau")] and any("restore" in r for r in whole[0])
    else:
        assert any(r.get("armed") for r in whole[0]) and ("plateau" in dict(ends).values())
    stops = [("_write_checkpoint", n, after) for n in range(1, writes[0] + 1) for after in (False, True)]
    stops += [("screen", 3, False), ("diagnose", 4, False)]
    for name, call, after in stops[:-3] + stops[-2:]:  # stopped after the last write, the run had ended
        out = tmp_path / f"{name}-{call}-{after}"
        with monkeypatch.context() as patched:
            _stopped_at(patched, mf, name, call, after)
            with pytest.raises(Stop):
                _resumable(out, kind=kind)
        _resumable(out, resume=True, kind=kind)
        assert _outcome(out) == whole, (name, call, after)
        shutil.rmtree(out)  # /tmp may be memory: one stopped run's files at a time


def test_resuming_skips_an_ended_run_and_starts_one_not_begun(tmp_path: Path):
    _resumable(tmp_path / "new", resume=True)  # nothing to resume: it starts
    rows = (tmp_path / "new" / "log.jsonl").read_text()
    _resumable(tmp_path / "new", resume=True)  # it ended: nothing trains
    assert (tmp_path / "new" / "log.jsonl").read_text() == rows
    assert json.loads((tmp_path / "new" / mf.CHECKPOINT).read_text())["place"] == mf.ENDED


@pytest.mark.parametrize("kind", ["forked", "continuous", "margin"])
def test_a_deadlines_stop_resumes_with_a_larger_guard_as_the_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
):
    # the wall clock a counter (a second a reading), so each guard stops at the same place every time
    config = (
        dict(RESUMABLE["continuous"], budget_epochs=40, state_bound=25, margin_wait=3)
        if kind == "margin"
        else dict(RESUMABLE[kind])
    )
    encoding, settings = config.pop("encoding"), config.pop("settings")
    whole_out = tmp_path / "whole"
    mf.run(encoding, [(4, 4)], 0, 1e9, whole_out, settings, **config)
    whole = _outcome(whole_out)
    clock = [0.0]

    def ticking() -> float:
        clock[0] += 1.0
        return clock[0]

    with monkeypatch.context() as patched:
        patched.setattr(mf, "time", SimpleNamespace(perf_counter=ticking))
        mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path / "counted", settings, **config)
    readings = clock[0]
    for guard in (readings / 4, readings / 2, 3 * readings / 4):
        out = tmp_path / f"guard-{guard:g}"
        clock[0] = 0.0
        with monkeypatch.context() as patched:
            patched.setattr(mf, "time", SimpleNamespace(perf_counter=ticking))
            mf.run(encoding, [(4, 4)], 0, guard, out, settings, **config)
        rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        assert any(r.get("end") == "deadline" for r in rows), guard  # the guard stopped it
        assert json.loads((out / mf.CHECKPOINT).read_text())["place"] != mf.ENDED
        mf.run(encoding, [(4, 4)], 0, 1e9, out, settings, resume=True, **config)
        assert _outcome(out) == whole, guard
        mf.run(encoding, [(4, 4)], 0, 1e9, out, settings, resume=True, **config)  # ended now: nothing trains
        assert _outcome(out) == whole, guard


def test_resuming_refuses_another_runs_checkpoint_and_a_log_without_one(tmp_path: Path):
    _resumable(tmp_path / "run")
    with pytest.raises(AssertionError, match="another run"):
        mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 1, 1e9, tmp_path / "run", RESUMED, cap=2, resume=True)
    (tmp_path / "run" / mf.CHECKPOINT).unlink()
    with pytest.raises(AssertionError, match="no checkpoint"):
        _resumable(tmp_path / "run", resume=True)


# the code-consolidation workplan's stage 3: D3's s0, D4's long tail (B), A's groups and pull

# the general algebras' round-1 machines (E1), each s0 1 but random-3-0's
ROUND_1 = [f.format(n=n) for n in (3, 4, 5) for f in ("word-problem-s{n}", "move-to-front-{n}", "random-{n}-0")]


@pytest.mark.parametrize("name", [*ROUND_1, "modular-division-5", "modular-division-7"])
def test_s0_is_1_for_every_round_1_machine_but_random_3_0_s(name: str):
    from indrajala_studies import window_tasks as wt

    assert mf.first_length(wt.task_named(name).machine) == (2 if name == "random-3-0" else 1)


def test_random_3_0_s_one_token_words_answer_alike_so_a_stateless_network_passes_them():
    from indrajala_studies import window_tasks as wt

    machine = wt.task_named("random-3-0").machine
    assert {label for t in range(machine.tokens) for label in machine.run([t, 0])} == {2}
    two = {
        (t, label)
        for u in range(machine.tokens)
        for t in range(machine.tokens)
        for t_, label in zip((u, t, 0), machine.run([u, t, 0]), strict=True)
        if t_ == t
    }
    assert len(two) > len({t for t, _ in two})  # some token answers two ways: s0 is 2


# the plan's feasibility tables (docs/code-consolidation-workplan.md): codes a factor, joint codes,
# the cap (the exclusion bound), s0 (D3), d (D2); and stage 2b's probes
CONSOLIDATION_CELLS = {
    "random-3-0": (8, 64, 69, 2, 3),
    "word-problem-s4": (15, 225, 248, 1, 2),
    "random-4-0": (15, 225, 248, 1, 2),
}
B_PROBES = {
    "random-q8-1": (9, 81, 88),
    "random-q12-0": (11, 121, 132),
    "random-q16-0": (12, 144, 159),
    "random-perm-4-0": (15, 225, 248),
}


@pytest.mark.parametrize("name", sorted(CONSOLIDATION_CELLS))
def test_the_consolidation_cells_are_as_the_plan_s_table(name: str):
    from fractions import Fraction

    from indrajala_studies import machine_reading as mr
    from indrajala_studies import window_tasks as wt

    codes, joint, cap, s0, d = CONSOLIDATION_CELLS[name]
    machine = wt.task_named(name).machine
    factors = mf.scaled_factors(machine.states, Fraction(32, 11), 2)
    assert factors == [(16, codes)] * 2 and codes * codes == joint >= machine.states
    assert mf.exclusion_bound(machine.states, factors) == cap
    assert mf.first_length(machine) == s0 and mr.word_length(machine) == d


@pytest.mark.parametrize("name", sorted(B_PROBES))
def test_the_b_probes_are_as_the_plan_s_table(name: str):
    from fractions import Fraction

    from indrajala_studies import window_tasks as wt

    codes, joint, cap = B_PROBES[name]
    machine = wt.task_named(name).machine
    factors = mf.scaled_factors(machine.states, Fraction(32, 11), 2)
    assert factors == [(16, codes)] * 2 and codes * codes == joint >= machine.states
    assert mf.exclusion_bound(machine.states, factors) == cap


def test_without_a_long_tail_an_epoch_s_lengths_are_the_curriculum_s_in_turn():
    assert mf.epoch_lengths(3, 7) == [1, 2, 3, 1, 2, 3, 1]


def test_the_long_tail_takes_every_other_example_beyond_the_step_to_its_factor_capped():
    lengths = mf.epoch_lengths(3, 16, long_tail=4, cap=10)
    assert lengths[::2] == [1, 2, 3, 1, 2, 3, 1, 2]  # half as before
    assert lengths[1::2] == [4, 5, 6, 7, 8, 9, 10, 4]  # 4..12, capped at 10
    assert mf.epoch_lengths(3, 16, long_tail=4, cap=40)[1::2] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert mf.epoch_lengths(10, 4, long_tail=4, cap=10) == [1, 2, 3, 4]  # at the cap: the curriculum's alone


def test_a_networks_groups_count_its_reached_codes_and_its_table_is_what_it_consolidates():
    from indrajala_studies import machine_reading as mr

    encoding = mf.target_named("random-3-0")
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, "output")
    table, codes, groups = mf.consolidation(net, encoding, 1)
    assert codes >= groups >= 1 and len(table) == codes - groups
    assert all(duplicate != representative for duplicate, representative in table.items())
    net.consolidate(table)
    assert dict(net.consolidation) == table
    assert mf.consolidation(net, encoding, mr.word_length(encoding.task.machine))[1] == codes  # the same walk


def test_a_consolidated_run_starts_at_s0_and_logs_its_groups_at_each_diagnosis(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=4, patience=10, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.target_named("random-3-0"), [(4, 4)], 0, 1e9, tmp_path, settings, cap=3, long_tail=4, consolidate=True)
    rows = _rows(tmp_path)
    assert [r["step"] for r in rows if "end" in r] == [2, 3]
    diagnoses = [r for r in rows if "factors" in r and isinstance(r.get("epoch"), int) and r["epoch"] > 0]
    assert diagnoses and all(set(r["consolidation"]) == {"codes", "groups", "duplicates"} for r in diagnoses)
    assert all(
        r["consolidation"]["codes"] == r["consolidation"]["groups"] + r["consolidation"]["duplicates"]
        for r in diagnoses
    )
    assert not any("consolidation" in r for r in rows if r not in diagnoses)
    saved = json.loads((tmp_path / mf.CHECKPOINT).read_text())
    assert saved["run"]["first"] == 2 and saved["run"]["long_tail"] == 4 and saved["run"]["consolidate"]


def test_a_run_without_the_plan_s_arms_keeps_its_identity_as_before(tmp_path: Path):
    _resumable(tmp_path / "run")
    identity = json.loads((tmp_path / "run" / mf.CHECKPOINT).read_text())["run"]
    assert not {"first", "long_tail", "consolidate", "restore_best", "commitment_from"} & set(identity)


# run --restore-best (the code-consolidation workplan's stage 2c, P2)


def _restoring(out: Path, cap: int, restore_best: bool) -> None:
    mf.run(
        mf.target_named("random-3-0"),
        [(4, 4)],
        0,
        1e9,
        out,
        PLATEAUS,
        cap=cap,
        consolidate=True,
        restore_best=restore_best,
    )


def test_restoring_the_best_leaves_a_run_without_a_plateau_as_it_was(tmp_path: Path):
    _restoring(tmp_path / "off", 2, False)
    _restoring(tmp_path / "on", 2, True)
    off, on = _outcome(tmp_path / "off"), _outcome(tmp_path / "on")
    assert [(r["step"], r["end"]) for r in on[0] if "end" in r] == [(2, "passed")]
    assert on[0] == off[0]
    assert {name: text for name, text in on[1].items() if name != "best.network.json"} == off[1]


def test_a_plateaued_step_ends_on_its_best_network_regrouped(tmp_path: Path):
    _restoring(tmp_path / "off", 3, False)
    _restoring(tmp_path / "on", 3, True)
    rows = _rows(tmp_path / "on")
    assert [(r["step"], r["end"]) for r in rows if "end" in r] == [(2, "passed"), (3, "plateau")]
    (restore,) = [r for r in rows if "restore" in r]
    assert restore["step"] == 3 and restore["restored"] == restore["restore"]
    assert restore["restore"] == max(
        r["screen"] for r in rows if r["step"] == 3 and r.get("epoch", 0) > 0 and "screen" in r
    )
    assert set(restore["consolidation"]) == {"codes", "groups", "duplicates"}
    # the step's saved network is its best, which the run without the restore didn't end on
    on = tmp_path / "on"
    assert (on / "step-3.network.json").read_bytes() == (on / "best.network.json").read_bytes()
    assert (on / "step-3.network.json").read_bytes() != (tmp_path / "off" / "step-3.network.json").read_bytes()
    assert any("RESTORE" in line for line in mf.trace(rows))
    assert json.loads((on / mf.CHECKPOINT).read_text())["run"]["restore_best"]


# run --commitment-from (the commitment-control workplan's diagnostic)


def test_the_commitment_from_the_first_step_is_the_run_without_it(tmp_path: Path):
    configs: dict[str, dict[str, Any]] = {"off": {}, "on": {"commitment_from": 1}}
    for name, config in configs.items():
        mf.run(
            mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path / name, RESUMED, cap=2, fork=RESUMED_FORK, **config
        )
    assert _outcome(tmp_path / "on") == _outcome(tmp_path / "off")


def test_the_steps_before_commitment_from_train_without_the_commitment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # every epoch's training (a fork's too) reads the commitments its step should have
    seen: list[list[float]] = []
    train_epoch = mf.wr.train_epoch

    def recording(net: Any, *args: Any) -> None:
        seen.append(list(net.recurrence_layer.commitments))
        train_epoch(net, *args)

    monkeypatch.setattr(mf.wr, "train_epoch", recording)
    out = tmp_path / "run"
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, out, RESUMED, cap=2, fork=RESUMED_FORK, commitment_from=2)
    rows = _rows(out)
    trained = [r["step"] for r in rows if r.get("epoch", 0) > 0 and "screen" in r]  # an epoch's row or its fork's
    assert len(trained) == len(seen) and set(trained) == {1, 2}
    assert [c for step, c in zip(trained, seen, strict=True) if step == 1] == [[0.0]] * trained.count(1)
    assert [c for step, c in zip(trained, seen, strict=True) if step == 2] == [[0.25]] * trained.count(2)
    assert json.loads((out / mf.CHECKPOINT).read_text())["run"]["commitment_from"] == 2


def test_tuning_the_readout_trains_the_output_layer_alone(tmp_path: Path):
    encoding = mf.Encoding(5, mf.FAMILY)
    path = tmp_path / "network.json"
    mf.network(encoding, [(4, 4), (4, 4)], 0, reads="output").save(str(path))
    settings = mf.Settings(epoch_examples=64)
    rows = mf.tune_readout(path, encoding, "modular-division-5", 2, tmp_path / "tuned", 2, settings)
    assert [row["epoch"] for row in rows] == list(range(len(rows))) and 1 <= len(rows) <= 3
    assert [json.loads(line) for line in (tmp_path / "tuned" / "log.jsonl").read_text().splitlines()] == rows
    before, after = mf.load_network(str(path)), mf.load_network(str(tmp_path / "tuned" / "network.json"))

    def arrays(net: Any) -> list[list[np.ndarray]]:
        # each layer's, then each part's, parameters as the optimizer indexes them
        owners: list[Any] = [*net.layers, *net.parts]
        parameters: list[list[Any]] = [list(getattr(owner, "parameters", list)()) for owner in owners]
        return [[np.asarray(array.tolist()) for array in arrays] for arrays in parameters]

    old, new = arrays(before), arrays(after)
    output = len(before.layers) - 1
    assert old[output] and any(not np.array_equal(a, b) for a, b in zip(old[output], new[output], strict=True))
    for index, (a, b) in enumerate(zip(old, new, strict=True)):
        if index != output:
            assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True)), f"index {index} changed"


def test_the_window_width_widens_the_stream_and_its_default_is_w11s(tmp_path: Path):
    from indrajala_ml.model.specs.layer_specs import Dense, Embedding, LayerNorm, Residual

    assert wp.transformer_window(9, 14) == wp.transformer_window(9, 14, wp.D)
    wide = wp.transformer_window(9, 14, 32)
    assert wide[0] == Embedding(9, 32) and wide[4] == Residual(
        (LayerNorm(), Dense(64, activation="relu"), Dense(32, activation="linear", bias=True))
    )
    encoding = mf.Encoding(5, mf.FAMILY)
    settings = mf.Settings(epoch_examples=32, max_epochs=1, screen=4, cases=4, lengths=(2,), long=(3,))
    mf.run(encoding, [(4, 4), (4, 4)], 0, 60, tmp_path / "w16", settings, cap=2, reads="output", budget_epochs=1)
    mf.run(
        encoding, [(4, 4), (4, 4)], 0, 60, tmp_path / "w32", settings, cap=2, reads="output", budget_epochs=1, width=32
    )
    identity = {name: json.loads((tmp_path / name / "checkpoint.json").read_text())["run"] for name in ("w16", "w32")}
    assert "width" not in identity["w16"] and identity["w32"]["width"] == 32


def test_separability_reads_every_reachable_codes_features_and_fits_them(tmp_path: Path):
    encoding = mf.Encoding(5, mf.FAMILY)
    path = tmp_path / "network.json"
    mf.network(encoding, [(4, 4), (4, 4)], 0, reads="output").save(str(path))
    result = mf.separability(path, encoding, "modular-division-5", steps=5000, passes=3)
    assert result["pairs"] == result["codes"] * len(
        mf.wc.all_windows(1, mf.wt.task_named("modular-division-5").machine.tokens)
    )
    assert result["features"] == wp.D and len(result["fit_wrong"]) == 1 and result["perceptron"]["passes"] <= 3
    assert all(row["other"] >= 0 for row in result["wrong"])


def test_a_branched_run_records_its_branches_and_its_window_has_a_branch_a_factor(tmp_path: Path):
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, "output", branches=True)
    assert len(net._pooling_reads) == 2 and len(set(net._pooling_reads)) == 2
    settings = mf.Settings(epoch_examples=32, max_epochs=1, screen=4, cases=4, lengths=(2,), long=(3,))
    mf.run(
        encoding,
        [(4, 4), (4, 4)],
        0,
        60,
        tmp_path / "b",
        settings,
        cap=2,
        reads="output",
        budget_epochs=1,
        branches=True,
    )
    identity = json.loads((tmp_path / "b" / "checkpoint.json").read_text())["run"]
    assert identity["branches"] is True
    with pytest.raises(AssertionError, match="reads OUTPUT"):
        mf.network(encoding, [(4, 4)], 0, branches=True)


def test_an_answer_branch_run_records_it_and_every_factor_reads_the_state_branch(tmp_path: Path):
    # the one-code-one-state workplan's R1: W11's third block unchanged as the state branch, which both
    # factors read, and the answers' own single-head branch
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, "output", answer_branch=True)
    assert len(net._pooling_reads) == 2 and len(set(net._pooling_reads)) == 1
    specs = wp.answer_branch_window(encoding.tokens, encoding.classes, 2)
    plain = wp.transformer_window(encoding.tokens, encoding.classes)
    branches = specs[-2]
    assert isinstance(branches, Branches) and branches.reads == (0, 0)
    assert list(branches.branches[0]) == plain[-5:-1]  # the state branch: W11's last block and LayerNorm
    settings = mf.Settings(epoch_examples=32, max_epochs=1, screen=4, cases=4, lengths=(2,), long=(3,))
    mf.run(
        encoding,
        [(4, 4), (4, 4)],
        0,
        60,
        tmp_path / "r1",
        settings,
        cap=2,
        reads="output",
        budget_epochs=1,
        answer_branch=True,
    )
    identity = json.loads((tmp_path / "r1" / "checkpoint.json").read_text())["run"]
    assert identity["answer_branch"] is True and "branches" not in identity
    with pytest.raises(AssertionError, match="one at most"):
        mf.network(encoding, [(4, 4)], 0, "output", branches=True, answer_branch=True)


def _branch_heads(branch: Sequence[Any]) -> int:
    # a branch block's attention heads: its first residual block's Attention (wr's branch blocks)
    block = branch[1]
    assert isinstance(block, Residual) and isinstance(block.body[1], Attention)
    return block.body[1].heads


def test_a_recombined_run_at_width_32_records_it_and_each_factor_reads_its_own_branch(tmp_path: Path):
    # the post-paper workplan's recombining window (D7, D8 (a)) at the base width 32, factors 32 wide:
    # W11's first two blocks as the trunk, a two-head block a factor, the ends merged into W11's last
    # block; the factors' paths 3 blocks deep, as the base's, the answers' 4
    encoding = mf.Encoding(5, mf.FAMILY)
    factors = [(32, 8), (32, 8)]
    net = mf.network(encoding, factors, 0, "output", width=32, recombine=True)
    assert len(net._pooling_reads) == 2 and len(set(net._pooling_reads)) == 2
    specs = wp.recombined_window(encoding.tokens, encoding.classes, 2, 32)
    plain = wp.transformer_window(encoding.tokens, encoding.classes, 32)
    branches = specs[8]
    assert isinstance(branches, Branches) and branches.merge and len(branches.branches) == 2
    assert specs[:8] == plain[:8] and specs[9:] == plain[-5:]  # the trunk and the recombining block, W11's
    assert specs[:8] == wp.branched_window(encoding.tokens, encoding.classes, 2, 32)[:8]  # branches-32's trunk
    for branch in branches.branches:
        assert _branch_heads(branch) == wp.HEADS
        ffn = branch[2]
        assert isinstance(ffn, Residual) and isinstance(ffn.body[2], Dense) and ffn.body[2].size == 32
    settings = mf.Settings(epoch_examples=32, max_epochs=1, screen=4, cases=4, lengths=(2,), long=(3,))
    mf.run(
        encoding,
        factors,
        0,
        60,
        tmp_path / "x",
        settings,
        cap=2,
        reads="output",
        budget_epochs=1,
        width=32,
        recombine=True,
    )
    identity = json.loads((tmp_path / "x" / "checkpoint.json").read_text())["run"]
    assert identity["recombine"] is True and identity["width"] == 32 and "branches" not in identity
    with pytest.raises(AssertionError, match="reads OUTPUT"):
        mf.network(encoding, [(4, 4)], 0, recombine=True)
    with pytest.raises(AssertionError, match="one at most"):
        mf.network(encoding, [(4, 4)], 0, "output", branches=True, recombine=True)


def test_a_branched_run_with_two_heads_a_branch_at_width_32_records_them(tmp_path: Path):
    # the post-paper workplan's branches-32 (D8): the retired design at width 32, two heads a branch
    encoding = mf.Encoding(5, mf.FAMILY)
    specs = wp.branched_window(encoding.tokens, encoding.classes, 2, 32, heads=2)
    branches = specs[-2]
    assert isinstance(branches, Branches) and len(branches.branches) == 3
    assert all(_branch_heads(branch) == 2 for branch in branches.branches)
    retired = wp.branched_window(encoding.tokens, encoding.classes, 2, 32)[-2]
    assert isinstance(retired, Branches)
    assert all(_branch_heads(branch) == 1 for branch in retired.branches)  # one by default, the retired design's
    settings = mf.Settings(epoch_examples=32, max_epochs=1, screen=4, cases=4, lengths=(2,), long=(3,))
    mf.run(
        encoding,
        [(32, 8), (32, 8)],
        0,
        60,
        tmp_path / "b",
        settings,
        cap=2,
        reads="output",
        budget_epochs=1,
        width=32,
        branches=True,
        branch_heads=2,
    )
    identity = json.loads((tmp_path / "b" / "checkpoint.json").read_text())["run"]
    assert identity["branches"] is True and identity["branch_heads"] == 2
    with pytest.raises(AssertionError, match="branch heads"):
        mf.network(encoding, [(4, 4)], 0, "output", recombine=True, branch_heads=2)


@pytest.mark.parametrize("window", ["branches", "answer_branch", "recombine"])
def test_tuning_a_branched_readout_trains_the_answers_branch_alone(tmp_path: Path, window: str):
    # recombined, the answers' own path is the recombining block after the merge
    encoding = mf.Encoding(5, mf.FAMILY)
    path = tmp_path / "network.json"
    branched = mf.network(
        encoding,
        [(4, 4), (4, 4)],
        0,
        "output",
        branches=window == "branches",
        answer_branch=window == "answer_branch",
        recombine=window == "recombine",
    )
    branched.save(str(path))
    mf.tune_readout(path, encoding, "modular-division-5", 2, tmp_path / "tuned", 1, mf.Settings(epoch_examples=64))
    before, after = mf.load_network(str(path)), mf.load_network(str(tmp_path / "tuned" / "network.json"))

    def arrays(net: Any) -> list[list[np.ndarray]]:
        owners: list[Any] = [*net.layers, *net.parts]
        parameters: list[list[Any]] = [list(getattr(owner, "parameters", list)()) for owner in owners]
        return [[np.asarray(array.tolist()) for array in arrays] for arrays in parameters]

    old, new = arrays(before), arrays(after)
    answers = max(before._pooling_reads) + 1  # the answers' branch, then the output layer
    changed = [
        i
        for i, (a, b) in enumerate(zip(old, new, strict=True))
        if any(not np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    ]
    assert changed and all(i >= answers and i < len(before.layers) for i in changed)
    assert len(before.layers) - 1 in changed


def test_the_thermostat_holds_off_below_its_floor_ramps_back_from_half_the_codebook_and_drops_at_the_floor():
    thermostat = mf.Thermostat(floor=3.0, restore=0.5, ramp=4)
    counts = [8, 32]
    levels = [0, 0]
    trace: list[list[int]] = []
    # factor 0: 2 codes (off), 5 codes (half of 8 reached: ramp), on up, then 2.5 (below the floor: off)
    # factor 1: 10 codes stay off (between the floor and half of 32: the hysteresis), then 16 restores it
    for effective in [(2.0, 10.0), (5.0, 10.0), (3.5, 10.0), (3.5, 16.0), (3.5, 10.0), (3.5, 10.0), (2.5, 3.0)]:
        levels = thermostat.levels(levels, effective, counts)
        trace.append(levels)
    assert trace == [[0, 0], [1, 0], [2, 0], [3, 1], [4, 2], [4, 3], [0, 4]]
    assert thermostat.commitments([0, 2, 4], [0.25, 0.25, 0.5]) == [0.0, 0.125, 0.5]


def test_the_thermostat_with_a_screen_restores_only_once_the_answers_come_right():
    # D2'': from off, back once the step's best screen reaches 0.95, whatever the codes in use
    thermostat = mf.Thermostat(floor=3.0, restore=0.5, ramp=4, restore_screen=0.95)
    levels = [0]
    trace: list[int] = []
    for codes, screen in [(6.0, 0.4), (8.0, 0.9), (8.0, 0.96), (8.0, 0.96), (2.0, 0.99), (5.0, 0.99)]:
        levels = thermostat.levels(levels, [codes], [8], screen)
        trace.append(levels[0])
    # half the codebook (4 of 8) passed at once, but no restore until the screen; below the floor, off
    assert trace == [0, 0, 1, 2, 0, 1]


def test_effective_codes_are_two_to_the_entropy_of_each_factors_use():
    class Coded:
        # a network whose windows' next states are given (after the initial state): factor 0 on 4 codes
        # evenly, factor 1 on one; its backend's matrix the rows themselves
        @staticmethod
        def _rows(rows: list[Any]) -> list[Any]:
            return rows

        backend = SimpleNamespace(matrix=_rows)

        def state_codes(self, X: Any) -> list[list[tuple[int, ...]]]:
            return [[(0, 0), (0, 7), (1, 7), (2, 7), (3, 7)] for _ in X]

    rows = [[((0,), None), ((1,), None)]]
    assert mf.effective_codes(Coded(), rows) == pytest.approx([4.0, 1.0])


def test_annealing_halves_its_temperature_hardens_for_good_and_ramps_the_commitment_back():
    anneal = mf.Anneal(start=1.0, halving=10, ramp=4)
    assert [anneal.temperature(n) for n in (0, 9, 10, 25)] == [1.0, 1.0, 0.5, 0.25]
    levels = [0, 0, 4]
    trace: list[list[int]] = []
    for hard in ([0.5, 0.995, None], [0.995, 0.2, None], [0.1, 0.1, None], [0.1, 0.1, None], [0.1, 0.1, None]):
        levels = anneal.levels(levels, hard)
        trace.append(levels)
    # factor 1 hard at once, factor 0 a step later; neither softens again; a level at the ramp stays there
    assert trace == [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 4, 4], [4, 4, 4]]
    assert anneal.commitments([0, 2, 4], [0.25, 0.25, 0.5]) == [0.0, 0.125, 0.5]


def test_annealing_with_a_tail_stays_soft_past_hard_then_ramps_the_commitment():
    anneal = mf.Anneal(ramp=2, tail=3)
    assert anneal.top == 5
    levels = [0]
    trace: list[int] = []
    for hard in (0.5, 0.995, None, None, None, None, None):
        levels = anneal.levels(levels, [hard])
        trace.append(levels[0])
    assert trace == [0, 1, 2, 3, 4, 5, 5]
    assert [anneal.soft(level) for level in (0, 3, 4)] == [True, True, False]
    assert anneal.commitments([0, 3, 4, 5], [0.5] * 4) == [0.0, 0.0, 0.25, 0.5]


def test_the_melt_profile_heats_doubling_holds_then_cools_exponentially():
    warm_up = mf.WarmUp(heat=3, hold=2, halving=4)
    assert [warm_up.phase(n) for n in (0, 2, 3, 4, 5, 99)] == ["heat", "heat", "hold", "hold", "cool", "cool"]
    temperatures = [warm_up.temperature(n, 8.0) for n in range(10)]
    assert temperatures[:5] == [1.0, 2.0, 4.0, 8.0, 8.0]
    assert temperatures[5] == 8.0 and temperatures[9] == pytest.approx(4.0)
    assert all(a > b for a, b in itertools.pairwise(temperatures[5:]))  # smooth, no floor


def test_the_shaped_cools_fall_their_octaves_in_their_epochs_between_the_limits():
    shaped = {
        shape: mf.WarmUp(heat=0, hold=0, halving=10, shape=shape, cool=100, octaves=10)
        for shape in ("concave", "convex", "s")
    }
    for warm_up in shaped.values():
        assert warm_up.temperature(0, 1.0) == 1.0
        assert warm_up.temperature(100, 1.0) == pytest.approx(2.0**-10)
        assert warm_up.temperature(110, 1.0) == pytest.approx(2.0**-11)  # then halved as the exponential
    straight = 2.0**-5  # the exponential's at the halfway point, the same end points
    assert shaped["concave"].temperature(50, 1.0) > straight > shaped["convex"].temperature(50, 1.0)
    assert shaped["s"].temperature(50, 1.0) == pytest.approx(straight)
    assert shaped["s"].temperature(10, 1.0) > shaped["convex"].temperature(10, 1.0)  # slow near the melt


def test_the_handover_holds_the_state_liquid_through_the_melt_then_couples_it_as_the_codes_split():
    warm_up = mf.WarmUp(heat=2, hold=3, handover=True)
    assert warm_up.coupling(0, 0.01, 2.0, 0.02) == 0.0  # heating, though below the melting point
    assert warm_up.coupling(4, 8.0, 2.0, 0.02) == 0.0  # holding
    assert warm_up.coupling(5, 4.0, 2.0, 0.02) == 0.0  # cooling, above the melting point
    assert warm_up.coupling(9, 0.2, 2.0, 0.02) == pytest.approx(0.5)  # halfway in log T
    assert warm_up.coupling(9, 0.001, 2.0, 0.02) == 1.0
    assert mf.WarmUp(heat=2, hold=3).coupling(0, 4.0, 2.0, 0.02) == 1.0  # without the handover


def test_the_handover_window_reads_each_factors_melting_curve():
    made: list[dict[str, Any]] = [
        {"melting_point": 2.5, "critical_temperature": 3.5, "transitions": [[2.5, 1, 4], [0.3, 4, 6]]},
        {"melting_point": None, "critical_temperature": 3.0, "transitions": []},
        {},
    ]
    melting, final = mf.handover_window(made)
    assert melting == [2.5, 3.0, 1.0] and final == [0.3, pytest.approx(0.003), 1.0]
    one = [{"melting_point": 2.0, "critical_temperature": 3.0, "transitions": [[2.0, 1, 8]]}]  # all at once
    assert mf.handover_window(one) == ([2.0], [1.0])  # an octave
    assert mf.handover_window(one, 3.0) == ([2.0], [0.25])  # D30: at least three
    assert mf.handover_window(made, 3.0)[1] == [0.3, pytest.approx(0.003), 1.0]  # wider windows kept


def test_the_gate_holds_the_cools_clock_while_the_screen_is_below_the_holds_level():
    gated = mf.WarmUp(heat=2, hold=3, gate=0.99)
    assert not gated.holds(5, 0.5, 1.0)  # the cool's first epoch: the clock runs from the hold
    assert gated.holds(6, 0.98, 1.0) and not gated.holds(6, 0.99, 1.0)
    assert not gated.holds(4, 0.0, 1.0)  # the hold's own epochs never wait
    assert not mf.WarmUp(heat=2, hold=3).holds(6, 0.0, 1.0)  # without a gate


def test_a_held_clock_holds_the_temperature_and_the_coupling(tmp_path: Path):
    _resumable(tmp_path, kind="gated")
    cool = [row["anneal"] for row in _outcome(tmp_path)[0] if row.get("anneal", {}).get("phase") == "cool"]
    assert len(cool) >= 3 and [a["held"] for a in cool] == list(range(len(cool)))
    assert all(a["temperature"] == cool[0]["temperature"] and a["coupling"] == cool[0]["coupling"] for a in cool)


def test_the_adaptive_cool_steps_at_equilibrium_or_its_cap_by_its_heat_capacity():
    warm_up = mf.WarmUp(shape="adaptive", pace=0.1, settle=3, within=0.75)
    cooling = mf.Cooling.start([2.0, 1.0])
    cooling.advance(warm_up)  # nothing measured yet: the cool's start
    assert cooling.temperature == [2.0, 1.0]
    cooling.record([(5.0, 0.25), None])  # factor 1 hard: never measured, never stepped
    cooling.advance(warm_up)  # one epoch, no prediction to test it against
    assert cooling.temperature == [2.0, 1.0]
    cooling.record([(5.5, 0.25), None])  # sigma 2 sqrt(0.25) = 1: within 0.75 of the epoch before
    cooling.advance(warm_up)
    step = math.exp(-0.1 / 0.5)
    assert cooling.temperature == pytest.approx([2.0 * step, 1.0])
    assert cooling.predicted[0] == pytest.approx(5.5 - 0.25 * (2.0 - 2.0 * step))
    assert cooling.energies == [[], []] and cooling.capacities == [[], []]
    t = cooling.temperature[0]
    cooling.record([(9.0, 0.25), None])  # far from its prediction: holds
    cooling.advance(warm_up)
    cooling.record([(7.0, 0.25), None])  # moved more than 0.75 sigma since: holds
    cooling.advance(warm_up)
    assert cooling.temperature[0] == t
    cooling.record([(1.0, 1.0), None])  # the cap: steps whatever the test, by the capacity's mean at t
    cooling.advance(warm_up)
    assert cooling.temperature[0] == pytest.approx(t * math.exp(-0.1 / math.sqrt(0.5)))
    flat = mf.Cooling.start([1.0])  # a capacity near 0 (no transition): at most an octave
    flat.record([(1.0, 1e-8)])
    flat.record([(1.0, 1e-8)])
    flat.advance(warm_up)
    assert flat.temperature == [0.5]
    bounded = mf.WarmUp(shape="adaptive", pace=0.1, settle=1, handover=True, rise=0.02)
    melting, final = 2.0, 2.0 * 2.0**-4  # a 4-octave window: 0.02 of it is 0.08 of an octave
    for t, share in ((4.0, 0.0), (1.0, 0.25)):  # above the window: from its top; within it: from the coupling's
        edge = bounded.floor(t, melting, final)
        assert bounded.coupling(100, edge, melting, final) == pytest.approx(share + 0.02)  # in the cool
        steep = mf.Cooling.start([t])
        steep.record([(1.0, 1e-8)])  # a flat capacity: an octave unbounded, the coupling's bound here
        steep.advance(bounded, [melting], [final])
        assert steep.temperature == pytest.approx([max(edge, t / 2)])
    assert bounded.floor(final * 1.01, melting, final) == 0.0  # the window's end within the bound
    assert mf.WarmUp(handover=True).floor(1.0, melting, final) == 0.0  # unbounded without rise
    assert mf.WarmUp(shape="adaptive", heat=2, hold=3).temperature(9, 4.0) == 4.0  # Cooling's from the cool on
    assert mf.WarmUp(shape="adaptive", heat=2, hold=3).reads_hardness(5)


def test_the_hardness_test_waits_for_a_shaped_cool_to_run_its_curve():
    exponential = mf.WarmUp(heat=2, hold=3, shape="exponential")
    convex = mf.WarmUp(heat=2, hold=3, shape="convex", cool=100)
    assert [exponential.reads_hardness(n) for n in (4, 5, 6)] == [False, True, True]
    assert [convex.reads_hardness(n) for n in (5, 104, 105)] == [False, False, True]


def test_the_warm_up_ends_only_with_its_screen_and_enough_distinct_states_or_at_its_cap():
    warm_up = mf.WarmUp(screen=0.95, cap=8, distinct=4.0)
    assert not warm_up.ends(1, 1, 1.0, [7, 40], [8, 8])  # screened, too few states in one factor
    assert warm_up.ends(2, 1, 0.96, [32, 40], [8, 8])
    assert not warm_up.ends(2, 1, 0.9, [32, 40], [8, 8])
    assert warm_up.ends(8, 1, 0.5, [1, 1], [8, 8])  # the cap


def test_thermal_noise_scales_with_the_temperature_and_spares_hard_factors():
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, mf.BEFORE_STATE)
    before = [np.array(f.codes.E.tolist()) for f in net.recurrence_layer.factors]
    mf.jitter(net, [4.0, None], 1e-2, np.random.default_rng(0))
    after = [np.array(f.codes.E.tolist()) for f in net.recurrence_layer.factors]
    assert np.array_equal(before[1], after[1])
    moved = np.sum((after[0] - before[0]) ** 2, axis=1)  # each code's squared move, about kappa T = 0.04
    assert 0.0 < float(moved.mean()) < 0.2
    assert mf.code_groups(net, [4.0, None])[1] is None
    assert mf.code_groups(net, [1e9, None])[0] == 1  # every code within T / 100: one group, the melt


def test_whole_lengths_spread_from_one_to_the_cap():
    assert mf.whole_lengths(69) == [1, 2, 3, 4, 5, 8, 13, 19, 30, 45, 69]
    assert mf.whole_lengths(3) == [1, 2, 3]


def test_annealing_without_commit_keeps_the_commitment_at_zero():
    anneal = mf.Anneal(ramp=2, tail=1, commit=False)
    assert anneal.commitments([0, 2, 3], [0.25] * 3) == [0.0, 0.0, 0.0]


def test_code_groups_join_codes_within_the_thermal_noise():
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4)], 0, mf.BEFORE_STATE)
    codes = net.recurrence_layer.factors[0].codes
    codes.E = net.backend.matrix([[0.0] * 4, [0.1] + [0.0] * 3, [5.0] + [0.0] * 3, [-5.0] + [0.0] * 3])
    # squared gaps 0.01, 25, 25...: at T 0.5 apart under T / 100, together under 10 kappa T at kappa 0.01
    assert mf.code_groups(net, [0.5]) == [4]
    assert mf.code_groups(net, [0.5], 0.01) == [3]


def test_a_whole_run_trains_and_screens_every_length_to_the_cap_from_its_first_step(tmp_path: Path):
    settings = mf.Settings(epoch_examples=32, max_epochs=2, screen=4, cases=4, lengths=(2,), long=(3,))
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 60, tmp_path, settings, cap=6, budget_epochs=3, whole=True)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    starts = [r for r in rows if r.get("epoch") == 0 and "screen" in r]
    assert starts and all(r["kinds"] for r in starts)
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["whole"] is True
    assert mf.epoch_lengths(6, 12) == [1, 2, 3, 4, 5, 6] * 2  # the whole distribution, in turn
    # blocks go on past the cap until the budget: a cap of 2, blocks of at most 1 epoch, 5 epochs
    short = mf.Settings(epoch_examples=32, max_epochs=1, screen=4, cases=4, lengths=(2,), long=(3,))
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 60, tmp_path / "long", short, cap=2, budget_epochs=5, whole=True)
    blocks = [
        r["step"] for r in map(json.loads, (tmp_path / "long" / "log.jsonl").read_text().splitlines()) if "end" in r
    ]
    assert max(blocks) > 2
    with pytest.raises(AssertionError, match="no long tail"):
        mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 60, tmp_path / "x", settings, cap=6, whole=True, long_tail=2)


def test_the_melting_curve_splits_at_the_critical_temperature_into_the_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 3.0]])
    h = np.concatenate([c + rng.normal(0.0, 0.05, size=(40, 2)) for c in centres])
    melting = mf.melting_curve(h, 6, np.random.default_rng(1))
    critical = 2.0 * float(np.linalg.eigvalsh(np.cov(h, rowvar=False))[-1])
    assert melting["critical_temperature"] == pytest.approx(critical)
    assert melting["curve"][0] == [2.0 * critical, 1]  # the melt: every code at the mean
    # Rose: the first split at T_c, seen within an octave below it (critical slowing down); 3 groups by the end
    assert critical / 2 <= melting["melting_point"] <= critical
    groups = [g for _, g in melting["curve"]]
    assert groups == sorted(groups) and 3 in groups
    assert melting["transitions"][0][1] == 1


def test_target_states_follow_the_machine_window_by_window():
    machine = wt.task_named("word-problem-s3").machine
    words = [[1, 2, 0], [2, 2, 0]]
    rows = [[(tuple(float(t) for t in w), None) for w in words]]
    states = mf.target_states(machine, rows)
    expected: list[int] = []
    for j in range(3):  # factor_states' order: each window, then each word
        for w in words:
            state = machine.start
            for t in w[: j + 1]:
                state = int(machine.transition[state, t])
            expected.append(state)
    assert states.tolist() == expected


def test_state_separation_reads_the_target_from_the_cells_against_chance():
    encoding = mf.target_named("word-problem-s3")
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, mf.OUTPUT)
    rows = [mf.windows_drawn(encoding, n, 20, f"separation-test-{n}") for n in (2, 4)]
    net.set_continuous([True, True])
    separation = mf.state_separation(net, rows, encoding.task.machine, sample=100)
    assert separation["windows"] == 100 and 0.0 < separation["chance"] <= 1.0
    for name in ("joint", "factor-0", "factor-1"):
        assert 0.0 <= separation[name]["neighbour"] <= 1.0 and 0.0 <= separation[name]["centroid"] <= 1.0


def test_distinct_states_counts_each_snapped_factors_cell_outputs():
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4)], 0, mf.BEFORE_STATE)
    rows = [[encoding.example(encoding.draw(2, random.Random(i))) for i in range(6)]]
    (n,) = mf.distinct_states(net, rows)
    assert 1 <= n <= 6 * 2


def test_hardness_is_the_share_of_windows_whose_largest_weight_passes_the_test():
    net = mf.network(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, mf.BEFORE_STATE)
    encoding = mf.Encoding(5, mf.FAMILY)
    rows = [[encoding.example(encoding.draw(2, random.Random(i))) for i in range(6)]]
    assert mf.hardness(net, rows, [None], 0.99) == [None]
    cold, hot = mf.hardness(net, rows, [1e-9], 0.99)[0], mf.hardness(net, rows, [1e9], 0.99)[0]
    assert cold == 1.0 and hot == 0.0


def test_thermals_are_the_soft_assignments_energy_and_its_derivative_in_temperature():
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4)], 0, mf.BEFORE_STATE)
    rows = [[encoding.example(encoding.draw(2, random.Random(i))) for i in range(6)]]
    assert mf.thermals(net, rows, [None]) == [None]
    h = np.concatenate(
        [np.array(c.tolist())[:, :4] for c in net.window_states(net.backend.matrix([t for t, _ in rows[0]]))[2][1:]]
    )
    E = np.array(net.recurrence_layer.factors[0].codes.E.tolist())
    D = np.sum((h[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2)
    (cold,) = mf.thermals(net, rows, [1e-9])
    assert cold is not None and cold[0] == pytest.approx(D.min(axis=1).mean()) and cold[1] == pytest.approx(0.0)
    (hot,) = mf.thermals(net, rows, [1e9])
    assert hot is not None and hot[0] == pytest.approx(D.mean(), rel=1e-6)
    # the capacity is d<D>/dT: a central difference of the energy
    t, dt = 0.1, 1e-5
    (here,), (up,), (down,) = (mf.thermals(net, rows, [x]) for x in (t, t + dt, t - dt))
    assert here is not None and up is not None and down is not None
    assert here[1] == pytest.approx((up[0] - down[0]) / (2 * dt), rel=1e-3)


def _step_end(step: int, right: bool, minimised: int, certified: bool) -> dict[str, Any]:
    v = 1.0 if right else 0.9
    return {
        "step": step,
        "end": "passed",
        "accuracy": {"token-0": 1.0},
        "long": {"24": v, "48": 1.0},
        "minimised": minimised,
        "certified": certified,
    }


def test_answers_stop_judges_the_first_step_right_within_the_bound_by_the_certificate():
    def run(*ends: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"step": 0, "epoch": 0, "screen": 0.5}, *ends]  # a row with no end is skipped

    # all right but above the bound (a larger machine agreeing on the samples), then the certified step
    kept = run(_step_end(1, True, 9, False), _step_end(2, True, 6, True))
    assert mf.answers_stop(kept, 6) == {"verdict": "kept", "stop": 2, "certified": 2}
    assert mf.answers_stop(kept, 6 * 1.5)["verdict"] == "false-stop"  # a loose bound stops on the larger one
    assert mf.answers_stop(run(_step_end(1, True, 6, False), _step_end(2, True, 6, True)), 6)["stop"] == 1
    assert mf.answers_stop(run(_step_end(1, True, 5, False)), 6)["verdict"] == "false-discovery"
    assert mf.answers_stop(run(_step_end(1, False, 6, True)), 6)["verdict"] == "missed"
    assert mf.answers_stop(run(_step_end(1, False, 6, False)), 6) == {
        "verdict": "silent",
        "stop": None,
        "certified": None,
    }
    # a coupled run's answers re-read from the saved networks: the logged (soft) ones are ignored
    soft = run(_step_end(1, True, 6, False), _step_end(2, True, 6, True))
    reread = {1: {"24": 0.5}, 2: {"24": 1.0}}
    assert mf.answers_stop(soft, 6, lambda step: reread[step])["verdict"] == "kept"
    assert mf.answers_stop(soft, 6, lambda step: None)["verdict"] == "missed"  # no saved network: no stop
    # a stop exact at snap ratio 1.0 but not with near ties branched (the margin's case, D59)
    strict = {**_step_end(1, True, 6, False), "certificates": {"strict": {"certified": True}}}
    assert mf.answers_stop(run(strict), 6)["verdict"] == "strict-only"
    # a row logging no certificates (the `paper` cell's): the saved step network's verdict, when given
    bare = run(_step_end(1, True, 6, False), _step_end(2, True, 6, True))
    assert mf.answers_stop(bare, 6)["verdict"] == "false-stop"  # nothing to read the strict verdict from
    assert mf.answers_stop(bare, 6, strict=lambda step: step == 1)["verdict"] == "strict-only"
    assert mf.answers_stop(bare, 6, strict=lambda step: None)["verdict"] == "false-stop"  # not saved
    assert mf.answers_stop(run(strict), 6, strict=lambda step: False)["verdict"] == "strict-only"  # the log's first


def test_formation_course_reads_when_the_state_formed_was_kept_and_the_melting_curves_crowding():
    def step(n: int, share: float, seconds: float, **more: Any) -> dict[str, Any]:
        return {"step": n, "end": "epoch", "state_share": share, "seconds": seconds, **more}

    melting: list[dict[str, Any]] = [
        {"transitions": [[4.0, 1, 4], [2.0, 4, 6], [1.0, 6, 8]]},  # half at the first, 3/4 two octaves down
        {"transitions": []},
    ]
    rows: list[dict[str, Any]] = [
        {"step": 1, "melting": melting},
        step(1, 0.5, 10.0),
        step(2, 0.97, 20.0, certificates={"strict": {"certified": True}}),
        step(3, 0.4, 30.0, answers_stop={"passed": True}),
        step(4, 0.99, 40.0, certified=True),
    ]
    course = mf.formation_course(rows, by=3)
    assert course is not None
    assert (course["formed"], course["strict"], course["passed"], course["certified"], course["lost"]) == (
        2,
        2,
        3,
        4,
        3,
    )
    assert (course["max_share"], course["seconds_by"], course["seconds"]) == (0.99, 30.0, 40.0)
    assert course["cascades"] == [
        {"foot": 8, "transitions": 3, "first_jump": 0.5, "octaves_half": 0.0, "octaves_three_quarters": 1.0},
        None,
    ]
    assert "formed 2" in mf.formation_summary(course) and "3/4 in 1.0 oct; -" in mf.formation_summary(course)
    never = mf.formation_course([step(1, 0.5, 10.0)], by=200)
    assert never is not None and (never["formed"], never["lost"], never["seconds_by"], never["cascades"]) == (
        None,
        None,
        None,
        [],
    )
    assert mf.formation_course([{"step": 1, "melting": melting}]) is None
    assert mf.formation_summary(None) == "no step ends"


def _cool_row(step: int, coupling: float, screen: float, rate: float = 1.0, **anneal: Any) -> dict[str, Any]:
    return {
        "step": step,
        "epoch": 1,
        "screen": screen,
        "anneal": {
            "phase": "cool",
            "coupling": [coupling, 0.0],
            "rate": rate,
            "clock": float(step),
            "temperature": [1.0, 0.5],
            **anneal,
        },
        "seconds": float(step),
    }


def test_cool_course_orders_the_first_jump_against_the_screens_fall_and_finds_the_freeze_and_reheats():
    rows: list[dict[str, Any]] = [
        {"step": 1, "epoch": 1, "screen": 1.0, "anneal": {"phase": "heat", "coupling": [0.0, 0.0], "rate": 1.0}},
        _cool_row(2, 0.0, 0.85),  # below the fall's line before ever reaching it: not a fall
        _cool_row(3, 0.05, 0.95),
        _cool_row(4, 0.2, 0.97),  # the first jump (0.15)
        _cool_row(5, 0.25, 0.6, rate=0.0),  # the fall, an epoch after the jump
        _cool_row(6, 0.3, 0.6, rate=0.0),
        _cool_row(7, 0.3, 0.7, rate=-0.5),  # a reheat
        _cool_row(8, 0.3, 0.7, rate=0.0),
        _cool_row(9, 0.6, 0.9, rate=-1.0),  # a second reheat, and a jump
    ]
    course = mf.cool_course(rows)
    assert course is not None
    assert (course["epochs"], course["start"], course["jumps"], course["largest"], course["jump"]) == (8, 2, 2, 0.3, 4)
    assert (course["fall"], course["order"], course["gap"]) == (5, "jump first", 1)
    assert course["frozen"] == {"epochs": 2, "step": 5, "clock": 5.0, "temperature": 1.0, "coupling": 0.25}
    assert (course["reheats"], course["reheat"]) == (2, 7)
    assert "jump first (+1 epochs)" in mf.cool_summary(course)
    same = mf.cool_course([_cool_row(1, 0.0, 0.95), _cool_row(2, 0.2, 0.5)])
    assert same is not None and (same["order"], same["gap"]) == ("the same epoch", 0)
    drop = mf.cool_course([_cool_row(1, 0.0, 0.95), _cool_row(2, 0.0, 0.5), _cool_row(3, 0.5, 0.5)])
    assert drop is not None and (drop["order"], drop["gap"], drop["frozen"]) == ("drop first", -1, None)
    calm = mf.cool_course([_cool_row(1, 0.0, 0.95), _cool_row(2, 0.05, 0.99)])
    assert calm is not None and (calm["jumps"], calm["fall"], calm["order"]) == (0, None, None)
    assert "never frozen" in mf.cool_summary(calm) and ": -;" in mf.cool_summary(calm)
    assert mf.cool_course(rows[:1]) is None and mf.cool_summary(None) == "no cool"


def test_same_prefix_reads_a_paired_arms_rows_against_its_control_but_for_seconds_and_its_own_fields():
    control = [{"step": 1, "screen": 1.0, "seconds": 1.0}, _cool_row(2, 0.1, 0.99), _cool_row(3, 0.2, 0.99)]
    arm = [
        {"step": 1, "screen": 1.0, "seconds": 9.0},  # seconds ignored
        _cool_row(2, 0.1, 0.99, target=0.5),
        _cool_row(3, 0.3, 0.99, target=0.5),  # the arm acts in the cool
    ]
    assert mf.same_prefix(arm, control) == {"same": 1, "before_cool": 1, "differs": 2, "intact": True}
    assert mf.same_prefix(arm, control, ["target"]) == {"same": 2, "before_cool": 1, "differs": 3, "intact": True}
    early = [{"step": 1, "screen": 0.5, "seconds": 1.0}, *control[1:]]
    assert mf.same_prefix(early, control)["intact"] is False
    assert mf.same_prefix(control, control) == {"same": 3, "before_cool": 1, "differs": None, "intact": True}
    assert mf.same_prefix(control[:2], control)["differs"] == 3  # the longer log's next row


def test_restart_cost_finds_lubys_cutoff_and_reads_censored_runs_only_to_the_shortest():
    def step(
        n: int, seconds: float, end: str = "epoch", certified: bool = False, strict: bool = False
    ) -> dict[str, Any]:
        return {
            "step": n,
            "end": end,
            "state_share": 1.0,
            "seconds": seconds,
            "certified": certified,
            "certificates": {"strict": {"certified": strict}},
        }

    # D41: both verdicts; the default alone is not a success; "spent" censors, a stop that fired doesn't
    fast = [step(1, 5.0), step(2, 10.0, "passed", certified=True, strict=True)]
    spent = [step(1, 5.0, certified=True), step(100, 500.0, "spent")]
    false_stop = [step(1, 5.0), step(30, 150.0, "passed", certified=True)]
    assert mf.restart_course(fast, clock="steps") == (2.0, 2.0, False)
    assert mf.restart_course(fast, clock="seconds") == (10.0, 10.0, False)
    assert mf.restart_course(spent, clock="steps") == (None, 100.0, True)
    # the wall-clock guard's stop censors too (epochs are the binding budget, the owner 2026-10-23)
    assert mf.restart_course([step(1, 5.0), step(80, 690.0, "deadline")], clock="steps") == (None, 80.0, True)
    assert mf.restart_course(false_stop, clock="steps") == (None, 30.0, False)
    # the default clock, epochs trained in all (the time axis): each step end's epochs summed, a
    # budget's stop logging one epoch not trained (outcome's count); a step end without a reading
    # (no state share) still counts its epochs
    trained = [
        {**step(1, 5.0), "epochs": 3},
        {"step": 2, "end": "epoch", "epochs": 2},
        {**step(3, 15.0, "passed", certified=True, strict=True), "epochs": 4},
    ]
    assert mf.restart_course(trained) == (9.0, 9.0, False)
    assert mf.restart_course([{**step(1, 5.0), "epochs": 3}, {**step(2, 9.0, "spent"), "epochs": 5}]) == (
        None,
        7.0,
        True,
    )
    assert mf.restart_course([{"step": 1, "melting": []}]) is None
    # half the runs certify at 2, the rest are spent at 100: l(2) = (0.5*2 + 0.5*2) / 0.5 = 4, against
    # (2 + 100) / 1 = 102 per certificate running each out
    cost = mf.restart_cost([(2.0, 2.0, False), (None, 100.0, True)])
    assert cost["horizon"] == 100.0 and cost["best"] == (2.0, 0.5, 4.0) and cost["run_out"] == 102.0
    assert "t* 2" in mf.restart_summary(cost) and "(25.5x)" in mf.restart_summary(cost)
    # a success later than the shortest censored run is unread beyond it; a finished failure isn't censored
    late = mf.restart_cost([(2.0, 2.0, False), (None, 50.0, True), (80.0, 80.0, False), (None, 30.0, False)])
    assert late["horizon"] == 50.0 and [t for t, _, _ in late["curve"]] == [2.0, 50.0]
    assert late["curve"][0] == (2.0, 0.25, 8.0)  # every run costs min(T, 2) = 2; one of four certified
    none = mf.restart_cost([(None, 10.0, True)])
    assert none["best"] is None and "no cutoff pays" in mf.restart_summary(none)


def test_the_answer_profile_reads_how_fast_recent_tokens_remove_the_answers_uncertainty():
    # the answer is the last token: uncertain alone (2 bits over 4 tokens), certain given one token
    last = wt.Mealy(np.array([[t for t in range(4)] for _ in range(4)]), np.array([[t for t in range(4)]] * 4))
    profile = mf.answer_profile(last, random.Random(0), depth=2, words=200, length=10)
    assert abs(profile[0] - 2.0) < 0.05 and profile[1] == profile[2] == 0.0


def test_order_against_random_puts_a_random_machine_near_its_ensemble_and_a_group_far_from_it():
    random_q12 = mf.order_against_random(mf.target_named("random-q12-0").task.machine, 12, ensemble=8)
    group = mf.order_against_random(mf.target_named("word-problem-s4").task.machine, 24, ensemble=8)
    assert random_q12["rms_z"] < 1.5 < group["rms_z"]
    assert group["drop_z"] < 0  # a group conserves its past: recent tokens leave the answer uncertain


def test_heat_course_reads_each_factors_splits_and_capacity_peak_in_the_cool():
    def row(phase: str, groups: list[int], capacity: list[float], t: list[float | None]) -> dict[str, Any]:
        return {
            "step": 1,
            "epoch": 1,
            "anneal": {"phase": phase, "temperature": t, "groups": groups, "capacity": capacity},
        }

    rows: list[dict[str, Any]] = [
        row("hold", [2, 8], [9.0, 0.1], [4.0, None]),  # the hold's peak isn't the cool's
        row("cool", [2, 8], [0.1, 0.1], [3.0, None]),
        row("cool", [3, 8], [0.3, 0.1], [2.0, None]),  # nucleation at epoch 1, jump x3
        row("cool", [3, 8], [0.5, 0.1], [1.5, None]),
        row("cool", [6, 8], [0.4, 0.1], [1.0, None]),  # the largest split, epoch 3
        row("cool", [5, 8], [0.6, 0.1], [0.8, None]),  # a merge is no split
        row("cool", [8, 8], [0.2, 0.1], [0.5, None]),
        {"step": 1, "factors": []},
    ]
    course, frozen = mf.heat_course(rows)
    assert frozen is None and course is not None
    assert course["splits"] == [[1, 1, 2.0], [3, 3, 1.0], [5, 3, 0.5]]
    assert course["nucleation"] == 1 and course["largest"] == 3 and course["jump"] == 3.0
    assert course["peak"] == 4 and course["peak_capacity"] == 0.6 and course["peak_temperature"] == 0.8
    assert course["from_nucleation"] == 3 and course["from_largest"] == 1 and course["in_cascade"]
    assert course["maxima"] == 2 and course["groups"] == [2, 8] and course["epochs"] == 6
    assert "factor 1 -" in mf.heat_summary([course, frozen])
    assert mf.heat_course([{"step": 1, "anneal": {"phase": "cool", "temperature": [1.0]}}]) == []


def test_state_spread_reads_the_outputs_critical_temperature_and_the_codes_groups():
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4)], 0, mf.BEFORE_STATE)
    rows = [[encoding.example(encoding.draw(3, random.Random(i))) for i in range(8)]]
    (spread,) = mf.state_spread(net, rows)
    h = np.concatenate(
        [np.array(c.tolist())[:, :4] for c in net.window_states(net.backend.matrix([t for t, _ in rows[0]]))[2][1:]]
    )
    top = float(np.linalg.eigvalsh(np.cov(h, rowvar=False))[-1])
    assert spread["windows"] == len(h) > 0
    assert spread["critical_temperature"] == pytest.approx(2 * top)
    assert 1 <= spread["effective_dimension"] <= 4
    assert spread["codes"] == 4 and spread["groups"] == 4 and 1 <= spread["in_use"] <= 4
    # three codes made identical: two groups, the nearest pair at distance 0
    codes = net.recurrence_layer.factors[0].codes
    E = np.array(codes.E.tolist())
    E[1] = E[2] = E[0]
    codes.E = net.backend.matrix(E.tolist()) if not isinstance(codes.E, np.ndarray) else E
    (merged,) = mf.state_spread(net, rows)
    assert merged["groups"] == 2 and merged["nearest_squared"] == 0.0


def test_weight_norms_read_every_trained_parameter_the_readout_and_each_factors_cell_and_codes():
    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4)], 0, mf.BEFORE_STATE)
    norms = mf.weight_norms(net)
    out = net.output_layer
    readout = math.sqrt(sum(float(np.sum(np.square(np.array(a.tolist())))) for a in (out.W, out.b)))
    assert norms["readout"] == pytest.approx(readout)
    assert len(norms["cells"]) == len(norms["codes"]) == 1 and norms["codes"][0] is not None
    assert norms["total"] > max(norms["readout"], norms["cells"][0], norms["codes"][0]) > 0
    # the codes doubled: their norm doubles, the total grows, the readout doesn't move
    codes = net.recurrence_layer.factors[0].codes
    E = np.array(codes.E.tolist()) * 2
    codes.E = net.backend.matrix(E.tolist()) if not isinstance(codes.E, np.ndarray) else E
    doubled = mf.weight_norms(net)
    assert doubled["codes"][0] == pytest.approx(2 * norms["codes"][0])
    assert doubled["readout"] == pytest.approx(norms["readout"]) and doubled["total"] > norms["total"]


def test_an_annealed_run_logs_each_epochs_temperatures_hardness_and_commitments(tmp_path: Path):
    settings = mf.Settings(epoch_examples=32, max_epochs=3, screen=4, cases=4, lengths=(2,), long=(3,))
    anneal = mf.Anneal(start=1.0, halving=1, ramp=2)
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 60, tmp_path, settings, cap=2, budget_epochs=3, anneal=anneal)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    epochs = [r["anneal"] for r in rows if r.get("epoch", 0) > 0 and "end" not in r]
    assert epochs and all(len(r["temperature"]) == len(r["hard"]) == len(r["beta"]) == 1 for r in epochs)
    assert all((r["temperature"][0] is None) == (r["beta"][0] > 0) for r in epochs)  # soft, uncommitted
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["anneal"] == repr(anneal)
    with pytest.raises(AssertionError, match="annealing sets the commitment"):
        mf.run(
            mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 60, tmp_path / "x", settings, cap=2, anneal=anneal, consolidate=True
        )


def test_kmeans_finds_separated_clusters_and_perturbs_the_codes_it_cant_fill():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0]])
    points = np.concatenate([c + rng.normal(0.0, 0.1, size=(30, 2)) for c in centres])
    found, nearest = mf.kmeans(points, 3, np.random.default_rng(1), 25)
    rounded: list[list[float]] = np.round(found).tolist()
    assert sorted(rounded) == sorted(centres.tolist())
    assert len(set(nearest.tolist())) == 3
    # two distinct points, four codes: no two codes equal
    few, _ = mf.kmeans(np.array([[0.0, 0.0], [1.0, 1.0]] * 5), 4, np.random.default_rng(2), 5)
    gaps = np.sum((few[:, np.newaxis, :] - few[np.newaxis, :, :]) ** 2, axis=2)[np.triu_indices(4, 1)]
    assert gaps.min() > 0.0


def test_a_warm_up_run_trains_continuous_then_heats_from_its_codes_t_hot(tmp_path: Path):
    settings = mf.Settings(epoch_examples=32, max_epochs=2, screen=40, cases=4, lengths=(2,), long=(3,))
    anneal, warm_up = mf.Anneal(start=1.0, halving=1, ramp=2), mf.WarmUp(screen=2.0, cap=1, heat=2, hold=1)
    mf.run(
        mf.Encoding(5, mf.FAMILY),
        [(4, 4)],
        0,
        60,
        tmp_path,
        settings,
        cap=3,
        budget_epochs=6,
        anneal=anneal,
        warm_up=warm_up,
    )
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    (end,) = [r for r in rows if "warm_up_end" in r]
    (made,) = end["warm_up_end"]
    assert made["hot"] == pytest.approx(2.0 * made["critical_temperature"]) and made["within_temperature"] >= 0.0
    assert "melting_point" in made and isinstance(made["transitions"], list)
    assert any("distinct" in r for r in rows)
    at = rows.index(end)
    assert all(r.get("warm") for r in rows[:at] if r.get("epoch", 0) > 0 and "end" not in r)
    after = [r["anneal"] for r in rows[at:] if r.get("epoch", 0) > 0 and "end" not in r and "anneal" in r]
    # D15: heated from T_hot / 4, doubling, hardness unread until the cool
    assert [a["phase"] for a in after[:3]] == ["heat", "heat", "hold"]
    assert [a["temperature"][0] for a in after[:3]] == pytest.approx([made["hot"] / 4, made["hot"] / 2, made["hot"]])
    assert all(a["hard"][0] is None for a in after[:3]) and all(a["beta"][0] == 0.0 for a in after[:3])
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["warm_up"] == repr(warm_up)
    with pytest.raises(AssertionError, match="annealed snap"):
        mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 60, tmp_path / "x", settings, cap=2, warm_up=warm_up)
    # D25: with the handover, couplings 0 through the heat and the hold
    handed = mf.WarmUp(screen=2.0, cap=1, heat=2, hold=1, handover=True)
    mf.run(
        mf.Encoding(5, mf.FAMILY),
        [(4, 4)],
        0,
        60,
        tmp_path / "handed",
        settings,
        cap=3,
        budget_epochs=6,
        anneal=anneal,
        warm_up=handed,
    )
    handed_rows = [json.loads(line) for line in (tmp_path / "handed" / "log.jsonl").read_text().splitlines()]
    profiled = [r["anneal"] for r in handed_rows if "anneal" in r and "phase" in r["anneal"]]
    assert [a["coupling"] for a in profiled[:3]] == [[0.0], [0.0], [0.0]]
    with pytest.raises(AssertionError, match="never goes back"):
        mf.run(
            mf.Encoding(5, mf.FAMILY),
            [(4, 4)],
            0,
            60,
            tmp_path / "y",
            settings,
            cap=2,
            anneal=anneal,
            warm_up=warm_up,
            restore_best=True,
        )


def test_a_crucibles_reading_marks_a_loss_after_every_factor_is_hard():
    start = {"in_use": 8, "effective_dimension": 6, "melting_point": 2.5, "transitions": [1.0, 0.5]}
    soft = {"phase": "cool", "coupling": [0.5, 0.5], "temperature": [0.5, 0.3]}
    hard = {"phase": "cool", "coupling": [1.0, 1.0], "temperature": [None, None]}
    rows: list[dict[str, Any]] = [
        {"step": 8, "warm_up_end": [start, start], "crucible": 8},
        {"step": 9, "epoch": 1, "screen": 1.0, "anneal": soft},
        {"step": 9, "factors": [{"codes": 8}, {"codes": 8}], "accuracy": {"a": 0.9}, "state_share": 0.9},
        {"step": 10, "epoch": 1, "screen": 0.4, "anneal": hard},
        {"step": 10, "factors": [{"codes": 2}, {"codes": 2}], "accuracy": {"a": 0.3}, "state_share": 0.2},
        {"step": 10, "end": "spent", "epochs": 2, "seconds": 180.0},
    ]
    course = mf.crucible_course(rows)
    assert course["best"] == {"phase": "cool", "coupling": 0.5, "temperature": 0.4}
    assert (course["lost_step"], course["lost"]["temperature"]) == (10, None)
    assert "lost 10 (cool, coupling 1.00, hard)" in mf.crucible_summary(course)
    assert (course["held"], course["rise"], course["trained"]) == (0, 0.5, None)
    assert mf.crucible_course(rows, run={"cap": 69})["trained"] == 8
    whole = [{**rows[0], "crucible": None}, *rows[1:]]
    del whole[0]["crucible"]  # a --whole run's warm-up end logs no crucible (D22)
    assert mf.crucible_course(whole, run={"cap": 69, "whole": True})["trained"] == 69


def test_the_trained_length_is_the_cap_once_whole_or_grown_and_the_crucibles_before():
    rows: list[dict[str, Any]] = [
        {"step": 5, "epoch": 1, "screen": 1.0, "warm": True},
        {"step": 8, "warm_up_end": [], "crucible": 8},
        {"step": 30, "epoch": 1, "screen": 1.0, "crucible": "grown"},
    ]
    assert [mf.trained_length(rows, {"cap": 69}, s) for s in (5, 8, 9, 29, 30)] == [5, 8, 8, 8, 69]
    assert mf.trained_length(rows[:1], {"cap": 69, "whole": True}, 5) == 69


def test_a_crucible_holds_the_warm_ups_material_then_grows_on_the_whole_distribution(tmp_path: Path):
    settings = mf.Settings(epoch_examples=32, max_epochs=1, screen=40, cases=4, lengths=(2,), long=(3,))
    # hard at the first reading (every window's weight above 0), read from the cool's start: after 2 epochs
    anneal = mf.Anneal(start=1.0, halving=1, ramp=1, hard=0.0, commit=False)
    warm_up = mf.WarmUp(screen=2.0, cap=2, heat=1, hold=1, halving=1, handover=True)
    encoding = mf.Encoding(5, mf.FAMILY)
    mf.run(
        encoding,
        [(4, 4)],
        0,
        60,
        tmp_path,
        settings,
        cap=6,
        budget_epochs=8,
        anneal=anneal,
        warm_up=warm_up,
        crucible=True,
    )
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    (end,) = [r for r in rows if "warm_up_end" in r]
    first = mf.first_length(encoding.task.machine)
    assert end["crucible"] == end["step"] == first + 1  # the warm-up's cap: its second step
    starts = [r for r in rows if r.get("epoch") == 0 and "screen" in r and r["step"] > end["step"]]
    marks = [r["crucible"] for r in starts]
    assert marks[0] == end["step"] and "grown" in marks  # the crucible's blocks, then growth
    grown = marks.index("grown")
    assert all(m == end["step"] for m in marks[:grown]) and all(m == "grown" for m in marks[grown:])
    assert set(starts[grown]["kinds"]) >= set(starts[0]["kinds"])
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["crucible"] is True
    with pytest.raises(AssertionError, match="crucible"):
        mf.run(
            encoding,
            [(4, 4)],
            0,
            60,
            tmp_path / "x",
            settings,
            cap=6,
            anneal=anneal,
            warm_up=warm_up,
            crucible=True,
            whole=True,
        )


def test_a_thermostat_run_logs_each_epochs_effective_codes_and_commitments(tmp_path: Path):
    settings = mf.Settings(epoch_examples=32, max_epochs=3, screen=4, cases=4, lengths=(2,), long=(3,))
    thermostat = mf.Thermostat(floor=1.5, restore=0.5, ramp=2)
    mf.run(
        mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 60, tmp_path, settings, cap=2, budget_epochs=3, thermostat=thermostat
    )
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    epochs = [r for r in rows if r.get("epoch", 0) > 0 and "end" not in r]
    assert epochs and all(len(r["thermostat"]["effective"]) == len(r["thermostat"]["beta"]) == 1 for r in epochs)
    assert all(r["thermostat"]["beta"][0] in (0.0, 0.125, 0.25) for r in epochs)  # off, half, or the specs' 0.25
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["thermostat"] == repr(thermostat)
    with pytest.raises(AssertionError, match="the thermostat sets the commitment"):
        mf.run(
            mf.Encoding(5, mf.FAMILY),
            [(4, 4)],
            0,
            60,
            tmp_path / "x",
            settings,
            cap=2,
            thermostat=thermostat,
            commitment_from=2,
        )


def test_a_run_keeping_its_machine_and_training_on_counterexamples_logs_them(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=32, max_epochs=4, diagnose_every=2, screen=4, cases=4, lengths=(2,), long=(3,)
    )
    encoding = mf.target_named("random-3-0")
    mf.run(
        encoding, [(4, 4)], 0, 60, tmp_path, settings, cap=3, budget_epochs=4, keep_machine=True, counterexamples=0.5
    )
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    diagnoses = [r for r in rows if "machine" in r and "end" not in r]
    assert diagnoses and all(r["counterexamples"] >= 0 for r in diagnoses)
    assert any(r.get("kept") for r in diagnoses) and (tmp_path / "machine-best.network.json").exists()
    identity = json.loads((tmp_path / "checkpoint.json").read_text())["run"]
    assert identity["keep_machine"] is True and identity["counterexamples"] == 0.5
    with pytest.raises(AssertionError, match="not both"):
        mf.run(encoding, [(4, 4)], 0, 60, tmp_path / "x", settings, cap=3, keep_machine=True, restore_best=True)
    with pytest.raises(AssertionError, match="a share"):
        mf.run(encoding, [(4, 4)], 0, 60, tmp_path / "y", settings, cap=3, counterexamples=1.5)


def test_t3s_baselines_build_e6s_window_and_e2s_continuous_state():
    encoding = mf.Encoding(5, mf.FAMILY)
    linear = mf.network(encoding, [(4, 4), (4, 4)], 0, mf.OUTPUT, baseline=mf.LINEAR)
    names = [type(layer).__name__ for layer in linear.layers]
    assert "AttentionArrayLayer" not in " ".join(names) and linear.recurrence.snapped
    continuous = mf.network(encoding, [(4, 4), (4, 4)], 0, mf.OUTPUT, baseline=mf.CONTINUOUS_STATE)
    assert not continuous.recurrence.snapped
    assert all(f.codes is None for f in continuous.recurrence.factors)
    with pytest.raises(AssertionError):
        mf.network(encoding, [(4, 4)], 0, mf.OUTPUT, branches=True, baseline=mf.LINEAR)


def test_a_continuous_state_trains_unjudged_by_a_certificate_and_refuses_a_state_bound(tmp_path: Path):
    settings = mf.Settings(epoch_examples=32, max_epochs=1, screen=4, cases=4, lengths=(2,), long=(3,))
    encoding = mf.Encoding(5, mf.FAMILY)
    mf.run(
        encoding,
        [(4, 4), (4, 4)],
        0,
        1e9,
        tmp_path / "run",
        settings,
        cap=3,
        reads=mf.OUTPUT,
        budget_epochs=2,
        baseline=mf.CONTINUOUS_STATE,
    )
    rows = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    ends = [r for r in rows if "end" in r]
    assert ends and all(r["certified"] is False and "minimised" not in r and "long" in r for r in ends)
    assert not list((tmp_path / "run").glob("certified-step-*"))
    with pytest.raises(AssertionError):
        mf.run(
            encoding,
            [(4, 4)],
            0,
            1e9,
            tmp_path / "bounded",
            settings,
            budget_epochs=2,
            state_bound=25,
            baseline=mf.CONTINUOUS_STATE,
        )


def test_a_runs_outcome_counts_a_logs_last_run_only():
    run = [{"step": s, "end": "passed", "epochs": 10, "certified": False, "seconds": float(s)} for s in (1, 2)]
    spent = {"step": 3, "end": "spent", "epochs": 401, "certified": False, "seconds": 9.0}
    once = mf.outcome([*run, spent])
    assert mf.outcome([*run, *run, spent]) == once and once.epochs == 420  # a re-run appended to the log


def test_long_summary_counts_every_run_and_names_a_failing_one(tmp_path: Path):
    def run(name: str, wrong: int) -> None:
        lengths = {
            str(n): {
                "properties": {
                    "U1": {"cases": 10, "passed": 10 - wrong, "failures": [{"case": i} for i in range(wrong)]}
                }
            }
            for n in (128, 256)
        }
        (tmp_path / name).mkdir()
        long = {"sha256": f"{name:0<64}", "lengths": lengths}
        (tmp_path / name / "certified-step-4.network.long-test.json").write_text(json.dumps(long))

    run("sum-1", 0)
    run("token-2", 1)
    assert mf.long_summary(tmp_path) == [
        "sum-1: 20 of 20 passed, 0 failures listed, sha256 sum-10000000",
        "token-2: 18 of 20 passed (U1@128 1, U1@256 1), 2 failures listed, sha256 token-200000",
        "all: 38 of 40 passed; networks with a failure: token-2",
    ]


@dataclasses.dataclass(frozen=True)
class _Profile:
    """A continuous profile's flags, as the identity reads them."""

    pace: bool = True
    smooth: bool = False
    reheat: bool = True
    slowing: bool = False
    incremental: bool = True


_PLAIN: dict[str, Any] = {
    "encoding": mf.target_named("random-3-0"), "factors": ((3, 3),), "seed": 7, "budget_seconds": None, "out": Path("x"),
    "settings": mf.PROBE, "cap": 5, "reads": "all", "budget_epochs": 100, "fork": None, "guard": False, "long_tail": None,
    "consolidate": False, "restore_best": False, "commitment_from": None, "width": wp.D, "branches": False,
    "answer_branch": False, "thermostat": None, "keep_machine": False, "counterexamples": None, "anneal": None, "warm_up": None,
    "whole": False, "crucible": False, "state_bound": None, "coupling_cap": 1.0, "continuous": None, "margin_wait": None,
}  # fmt: skip
_COMMON = {
    "target_states": 6, "tokens": 4, "factors": [[3, 3]], "seed": 7, "settings": repr(mf.PROBE), "cap": 5,
    "reads": "all", "budget_epochs": 100,
}  # fmt: skip


def test_a_runs_identity_names_its_defaults_alone_and_every_other_setting():
    # the identity a checkpoint records, fixed: a resumed run compares it, so its keys and values never change
    plain = mf.RunConfig(**_PLAIN)
    assert plain.identity(1) == {**_COMMON, "fork": None, "guard": False}
    changes: dict[str, Any] = {
        "fork": "F", "guard": True, "long_tail": 3, "consolidate": True, "restore_best": True,
        "commitment_from": 2, "width": 32, "branches": True, "answer_branch": True, "thermostat": "T",
        "keep_machine": True, "counterexamples": 0.5, "anneal": "A", "warm_up": "W", "whole": True,
        "crucible": True, "state_bound": 4, "coupling_cap": 0.5, "continuous": _Profile(), "margin_wait": 3,
        "cutoff_epochs": 50, "baseline": "b", "recombine": True, "branch_heads": 2,
    }  # fmt: skip
    every = dataclasses.replace(plain, **changes)
    assert every.identity(2) == {
        **_COMMON, "fork": "'F'", "guard": True, "first": 2, "long_tail": 3, "consolidate": True,
        "restore_best": True, "commitment_from": 2, "width": 32, "branches": True, "answer_branch": True,
        "thermostat": "'T'", "keep_machine": True, "counterexamples": 0.5, "anneal": "'A'", "warm_up": "'W'",
        "whole": True, "crucible": True, "state_bound": 4, "coupling_cap": 0.5, "continuous": repr(_Profile()),
        "coupling_pace": True, "reheat": True, "incremental_reheat": True, "margin_wait": 3, "cutoff_epochs": 50,
        "baseline": "b", "recombine": True, "branch_heads": 2,
    }  # fmt: skip
