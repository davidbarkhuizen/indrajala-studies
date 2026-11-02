"""
scripts/certnote/certificate_faults.py (the certificate note's D8): a table read as a machine, faults
planted and caught, and the exact chance of a sampled hit against simulation and closed forms.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_tasks as wt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "certnote"))
import certificate_check as cc  # scripts/ isn't a package
import certificate_faults as cf


class Machine:
    """A stand-in for any task: code q the target's state q % S, copies by the last token's parity."""

    def __init__(self, task: wt.Task, k: int, copies: int = 2) -> None:
        self.task, self.columns, self.copies = task, k, copies
        self.states = task.machine.states
        self.count = self.states * copies

    @property
    def codes(self) -> list[int]:
        return [self.count]

    def _distances(self, codes: np.ndarray) -> np.ndarray:
        d = np.full((len(codes), self.count), 2.0)
        d[np.arange(len(codes)), codes] = 1.0
        return d

    def start(self) -> tuple[wc.State, list[np.ndarray]]:
        return (0,), [self._distances(np.zeros(1, dtype=np.int64))[0]]

    def step(self, state: wc.State, windows: np.ndarray) -> wc.Steps:
        labels, after = self.task.machine.windows(windows)
        target = state[0] % self.states
        codes = after[target] + self.states * (windows[:, -1] % self.copies)
        return wc.Steps(labels[target], np.full(labels[target].shape, 5.0), codes[:, None], [self._distances(codes)])


def table(tmp_path: Path, task: wt.Task, k: int = 1) -> Path:
    certificate, t = wc.certify_table(Machine(task, k), task, 1.0)
    assert certificate.certified
    path = tmp_path / f"{task.name}-{k}.npz"
    t.save(path, wc.table_meta(certificate, task, 1.0, tmp_path / "none", "0" * 64))
    return path


def test_a_tables_machine_certifies_as_its_network_did(tmp_path: Path):
    task = wt.task_named("move-to-front-3")
    meta, a = cc.load(table(tmp_path, task))
    certificate = wc.certify(cf.TableMachine(meta, a), task, snap_ratio=1.0)
    assert certificate.certified and len(certificate.reachable) == len(a["states"])


@pytest.mark.parametrize("kind", cf.KINDS)
def test_every_planted_fault_is_caught_with_a_counterexample(tmp_path: Path, kind: str):
    reports = cf.fault_report(table(tmp_path, wt.ModularSum(4)), (8, 16), 10, 2)
    picked = [r for r in reports if r["kind"] == kind]
    assert {r["rank"] for r in picked} == set(cf.RANKS)
    assert all(r["clean_certified"] and r["caught"] and r["counterexample_windows"] >= 1 for r in picked)


def simulate(meta: dict[str, Any], a: cc.Arrays, task: wt.Task, n: int, cases: int, seed: int) -> float:
    """The share of uniform words of n tokens, then a zero token, on which the table's labels aren't the target's."""
    machine = cf.TableMachine(meta, a)
    rng, found = random.Random(seed), 0
    for _ in range(cases):
        word = [rng.randrange(meta["tokens"]) for _ in range(n)] + [0]
        state, got = machine.start()[0], list[int]()
        for t in word:
            steps = machine.step(state, np.array([[t]]))
            got.append(int(steps.digits[0, 0]))
            state = (int(steps.states[0, 0]),)
        found += tuple(got) != task.machine.run(word)
    return found / cases


def test_the_chance_per_case_is_the_simulated_share(tmp_path: Path):
    task = wt.ModularSum(4)
    meta, a = cc.load(table(tmp_path, task))
    nxt = cf.next_rows(a, meta["codes"])
    for kind in cf.KINDS:
        faulty = cf.plant(a, nxt, kind, 3, 2, task.machine.classes)
        assert faulty is not None
        pairs, targets, probabilities, final, _ = cf.pair_chain(meta, faulty, task.machine)
        start = np.zeros(len(pairs))
        start[0] = 1.0
        p = cf.found_chance(start, targets, probabilities, final, 6)
        share = simulate(meta, faulty, task, 6, 20_000, 0)
        assert abs(p - share) < 4 * np.sqrt(p * (1 - p) / 20_000) + 1e-3, (kind, p, share)


def test_a_label_fault_at_the_start_is_found_by_the_first_token_alone(tmp_path: Path):
    # from the start, window w's fault is hit by a word's first token w: chance 1/T at least at n 1
    task = wt.ModularSum(5)
    meta, a = cc.load(table(tmp_path, task))
    faulty = cf.plant(a, cf.next_rows(a, meta["codes"]), "label", 0, 1, task.machine.classes)
    assert faulty is not None
    pairs, targets, probabilities, final, _ = cf.pair_chain(meta, faulty, task.machine)
    start = np.zeros(len(pairs))
    start[0] = 1.0
    assert cf.found_chance(start, targets, probabilities, final, 1) == pytest.approx(1 / meta["tokens"])


def test_main_writes_a_line_per_fault(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "faults.jsonl"
    path = table(tmp_path, wt.ModularSum(4))
    assert cf.main([str(path), "--out", str(out), "--lengths", "8", "--sampled", "5", "--per-stratum", "1"]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 6 and all(0.0 <= r["catalogue_miss"] <= 1.0 for r in lines)
    assert "caught True" in capsys.readouterr().out
