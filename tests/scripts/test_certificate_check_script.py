"""
scripts/certnote/certificate_check.py, the certificate's independent checker (the certificate
note's D5): its structure checks against stand-in machines with every failure kind and against
tables cut or mislabelled, and its fidelity checks against real networks' tables, faithful and
corrupted, plain and through a family network's query adapter.
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import BEFORE_STATE, AttentionPooling, Codes, Factor, WindowRecurrence

from indrajala_studies import modular_family as mf
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "certnote"))
import certificate_check as cc  # scripts/ isn't a package

ALL_TWOS = 26  # addition's column token of digits (2, 2, 2)
NEAR = 1.01


class Adder:
    """
    A stand-in adder over the carry automaton, code q the carry q % 3 (q = carry + 3 * copy, the copy
    the window's last token's parity). Faults: "drops" hands a carry of 2 on as 1 (a wrong state);
    "digit" gives an all-twos window a wrong first digit (wrong labels); "digit-tie" gives an all-twos
    window's labels a lead of 1e-9; "snap-tie" puts the next carry's code at NEAR after an all-twos
    window (a near tie the branching must cover).
    """

    def __init__(self, k: int, copies: int = 1, fault: str | None = None) -> None:
        self.columns, self.copies, self.fault = k, copies, fault
        self.count = 3 * copies

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
        labels, after = wt.ADDITION.machine.windows(windows)
        digits, carry = labels[state[0] % 3].copy(), after[state[0] % 3].copy()
        twos = (windows == ALL_TWOS).all(axis=1)
        if self.fault == "drops":
            carry[carry == 2] = 1
        if self.fault == "digit":
            digits[twos, 0] = (digits[twos, 0] + 1) % 3
        codes = carry + 3 * (windows[:, -1] % self.copies)
        distances = self._distances(codes)
        if self.fault == "snap-tie":
            distances[twos, (carry[twos] + 1) % 3 + 3 * (windows[twos, -1] % self.copies)] = NEAR
        leads = np.full(digits.shape, 5.0)
        if self.fault == "digit-tie":
            leads[twos] = 1e-9
        return wc.Steps(digits, leads, codes[:, np.newaxis], [distances])


def stand_in_table(
    tmp_path: Path, machine: Any, snap_ratio: float = wc.SNAP_RATIO, name: str = "table.npz"
) -> tuple[wc.Certificate, Path]:
    certificate, table = wc.certify_table(machine, wt.ADDITION, snap_ratio)
    path = tmp_path / name
    meta = wc.table_meta(certificate, wt.ADDITION, snap_ratio, tmp_path / "none.network.json", "0" * 64)
    table.save(path, meta)
    return certificate, path


def structure(path: Path) -> dict[str, int]:
    return cc.check(path, structure_only=True)["structure"]


def rewrite(path: Path, **changes: Any) -> Path:
    """A copy of a table with some arrays (or the meta, as a dict) replaced."""
    meta, arrays = cc.load(path)
    meta = changes.pop("meta", meta)
    arrays.update(changes)
    out = path.with_name("changed-" + path.name)
    savez: Any = np.savez_compressed
    savez(out, meta=np.array(json.dumps(meta)), **arrays)
    return out


# the structure, on stand-ins


@pytest.mark.parametrize("k", [1, 2])
def test_a_certified_stand_ins_table_is_valid_and_its_verdict_agrees(tmp_path: Path, k: int):
    certificate, path = stand_in_table(tmp_path, Adder(k, copies=2))
    assert certificate.certified
    report = cc.check(path, structure_only=True)
    assert report["valid"] and report["agrees"] and not any(report["structure"].values())
    assert report["rows"] == 6 and report["windows"] == 27**k and report["entries"] == 6 * 27**k


def test_the_table_is_certifys_reachable_states_phi_and_minimised_machine(tmp_path: Path):
    certificate, table = wc.certify_table(Adder(2, copies=2), wt.ADDITION)
    assert [tuple(row) for row in table.states] == certificate.reachable
    assert table.phi.tolist() == [certificate.phi[q] for q in certificate.reachable]
    blocks = {q: b for b, states in enumerate(certificate.minimised) for q in states}
    assert table.blocks.tolist() == [blocks[q] for q in certificate.reachable]
    assert table.labels.shape == (6, 27**2, 2) and table.next.shape == (6, 27**2, 1)
    assert len(table.branch_rows) == 0 and table.starts.tolist() == [[0]]


@pytest.mark.parametrize(
    "fault,kind",
    [("drops", "next"), ("digit", "labels"), ("digit-tie", "lead")],
)
def test_each_wrong_machines_table_fails_its_check_and_agrees_with_certify(tmp_path: Path, fault: str, kind: str):
    certificate, path = stand_in_table(tmp_path, Adder(1, fault=fault))
    assert not certificate.certified
    report = cc.check(path, structure_only=True)
    assert not report["valid"] and report["agrees"]
    failed = {name for name, count in report["structure"].items() if count}
    assert kind in failed and failed <= {kind, "onto", "blocks"}


def test_a_near_tie_to_a_wrong_carry_branches_and_its_branch_fails_next(tmp_path: Path):
    certificate, path = stand_in_table(tmp_path, Adder(1, fault="snap-tie"))
    assert not certificate.certified and certificate.wrong_ties > 0
    meta, arrays = cc.load(path)
    assert len(arrays["branch_rows"]) > 0
    assert (arrays["ratios"][arrays["branch_rows"], arrays["branch_windows"]] < meta["snap_ratio"]).all()
    assert structure(path)["next"] == certificate.wrong_states
    # strict, the same machine certifies and has no branches
    certificate, path = stand_in_table(tmp_path, Adder(1, fault="snap-tie"), snap_ratio=1.0, name="strict.npz")
    assert certificate.certified and not any(structure(path).values())


def test_a_truncated_table_fails_closure(tmp_path: Path):
    _, path = stand_in_table(tmp_path, Adder(1))
    _, a = cc.load(path)
    cut = {name: a[name][:-1] for name in ("states", "labels", "next", "leads", "ratios", "phi", "blocks")}
    fails = structure(rewrite(path, **cut))
    assert fails["closure"] > 0 and fails["onto"] == 1


def test_a_nan_lead_fails_rather_than_passing_the_margin(tmp_path: Path):
    _, path = stand_in_table(tmp_path, Adder(1))
    _, a = cc.load(path)
    leads = a["leads"].copy()
    leads[0, 0] = np.nan
    assert structure(rewrite(path, leads=leads))["lead"] == 1


def test_onto_asks_only_for_the_states_whole_windows_reach():
    # every token toggles the state, so two tokens return to it: s1 is reachable by tokens, not by whole 2-windows
    toggle = np.array([[1, 1], [0, 0]])
    output = np.zeros((2, 2), dtype=np.int64)
    _, after = cc.target_windows(toggle, output, cc.windows(2, 2))
    assert cc.target_reachable(toggle, 0) == {0, 1} and cc.target_reachable(after, 0) == {0}


def test_a_wrong_phi_fails_labels_and_next(tmp_path: Path):
    _, path = stand_in_table(tmp_path, Adder(1))
    _, a = cc.load(path)
    fails = structure(rewrite(path, phi=a["phi"][[0, 2, 1]]))
    # a permuted phi keeps its kernel, so the blocks still match it
    assert fails["labels"] > 0 and fails["next"] > 0 and fails["blocks"] == 0
    # one that merges two states doesn't reach the third
    fails = structure(rewrite(path, phi=np.array([0, 1, 1])))
    assert fails["onto"] == 1 and fails["blocks"] > 0


def test_a_start_not_first_a_duplicate_row_and_a_wrong_minimised_machine_each_fail(tmp_path: Path):
    _, path = stand_in_table(tmp_path, Adder(1))
    _, a = cc.load(path)
    swapped = {name: a[name][[1, 0, 2]] for name in ("states", "labels", "next", "leads", "ratios", "phi", "blocks")}
    assert structure(rewrite(path, **swapped))["start"] == 1  # the start isn't the first row
    assert structure(rewrite(path, phi=np.array([1, 0, 2])))["start"] == 1  # phi(start) isn't the target's start
    states = a["states"].copy()
    states[2] = states[1]
    assert structure(rewrite(path, states=states))["duplicate"] == 1
    assert structure(rewrite(path, blocks=np.array([0, 0, 1])))["blocks"] > 0


def test_a_mismatched_or_out_of_range_table_fails_shape(tmp_path: Path):
    _, path = stand_in_table(tmp_path, Adder(1))
    meta, a = cc.load(path)
    assert structure(rewrite(path, meta={**meta, "columns": 2}))["shape"] == 1
    bad = a["next"].copy()
    bad[0, 0, 0] = 7
    assert structure(rewrite(path, next=bad))["shape"] == 1


# fidelity, on real networks


def saved_network(tmp_path: Path, network_class: Any, k: int = 2, factored: bool = False) -> Path:
    recurrence = (
        WindowRecurrence((Factor(2, Codes(3)), Factor(3, Codes(4))), pooling=AttentionPooling(BEFORE_STATE))
        if factored
        else WindowRecurrence((Factor(4, Codes(4)),))
    )
    network: Any = network_class.randomized((k,), wr.window_specs(wt.ADDITION), recurrence, Adam(), seed=3)
    path = tmp_path / f"{network_class.__name__}-{factored}.network.json"
    network.save(str(path))
    return path


@pytest.mark.parametrize("factored", [False, True], ids=["one factor", "two factors"])
@pytest.mark.parametrize("network_class", [WindowRecurrentArrayNetwork, WindowRecurrentRustArrayNetwork])
def test_a_networks_table_is_faithful_to_its_forward_pass(tmp_path: Path, network_class: Any, factored: bool):
    network = saved_network(tmp_path, network_class, factored=factored)
    # a wide ratio, so near ties are many and every branch is checked
    for ratio in (1.0, 3.0):
        emitted = tmp_path / f"table-{ratio}.npz"
        certificate = wc.certify_file(network, snap_ratio=ratio, emit=emitted)
        report = cc.check(emitted, batch=97, seed=1)
        assert report["sha256_matches"] and not any(report["fidelity"].values()), report["fidelity"]
        assert report["agrees"]  # an untrained network: invalid, and not certified
        assert not report["valid"] and not certificate["certified"]
        if ratio > 1.0:
            assert report["branches"] > 0


def test_a_corrupted_entry_fails_fidelity(tmp_path: Path):
    network = saved_network(tmp_path, WindowRecurrentRustArrayNetwork)
    emitted = tmp_path / "table.npz"
    wc.certify_file(network, snap_ratio=3.0, emit=emitted)
    _, a = cc.load(emitted)
    labels = a["labels"].copy()
    labels[0, 5, 0] = (labels[0, 5, 0] + 1) % 3
    assert cc.check(rewrite(emitted, labels=labels))["fidelity"]["labels"] == 1
    # every branch dropped: the forward pass's near ties are no longer covered
    none = {
        "branch_rows": a["branch_rows"][:0],
        "branch_windows": a["branch_windows"][:0],
        "branch_next": a["branch_next"][:0],
    }
    assert cc.check(rewrite(emitted, **none))["fidelity"]["ties"] > 0
    # a snapped next state moved to another code with no branch to it
    strict = tmp_path / "strict.npz"
    wc.certify_file(network, snap_ratio=1.0, emit=strict)
    _, s = cc.load(strict)
    following = s["next"].copy()
    following[0, 3, 0] = (following[0, 3, 0] + 1) % 4
    assert cc.check(rewrite(strict, next=following))["fidelity"]["next"] == 1


def test_another_networks_file_fails_its_sha256(tmp_path: Path):
    network = saved_network(tmp_path, WindowRecurrentArrayNetwork)
    other = saved_network(tmp_path, WindowRecurrentRustArrayNetwork)
    emitted = tmp_path / "table.npz"
    wc.certify_file(network, emit=emitted)
    report = cc.check(emitted, network_path=other)
    assert report["sha256_matches"] is False and not report["valid"]


def test_a_family_networks_table_is_read_through_its_query_adapter(tmp_path: Path):
    encoding = mf.Encoding(5, mf.FAMILY)
    network = mf.network(encoding, [(4, 4)], 0)
    path = tmp_path / "family.network.json"
    network.save(str(path))
    emitted = tmp_path / "family.npz"
    certificate = mf.certify_file(path, encoding, "modular-division-5", snap_ratio=3.0, emit=emitted)
    meta, _ = cc.load(emitted)
    assert meta["adapter"] == {"kind": "query", "query_tokens": list(encoding.query_tokens), "slot": 0}
    report = cc.check(emitted, batch=50)
    assert not any(report["fidelity"].values()) and report["agrees"] and not certificate["certified"]
    assert report["differences"]["lead_difference"] == 0.0
    # the adapter is part of the claim: other query tokens give another forward pass (slot 0's labels
    # are causal, blind to the queries after them, but the state the window hands on isn't)
    wrong = rewrite(emitted, meta={**meta, "adapter": {**meta["adapter"], "query_tokens": [9, 9]}})
    assert cc.check(wrong)["differences"]["ratio_difference"] > 0.0


def test_main_checks_tables_and_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _, good = stand_in_table(tmp_path, Adder(1), name="good.npz")
    _, bad = stand_in_table(tmp_path, Adder(1, fault="drops"), name="bad.npz")
    out = tmp_path / "reports.json"
    assert cc.main([str(good), "--structure-only", "--json", str(out)]) == 0
    assert cc.main([str(good), str(bad), "--structure-only"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert ": valid;" in lines[0] and "invalid: structure next" in lines[2]
    assert json.loads(out.read_text())[0]["valid"] is True


def test_a_table_is_not_a_dataclass_the_checker_needs():
    # the checker names its arrays itself: the same names as the Table's fields
    assert tuple(f.name for f in dataclasses.fields(wc.Table)) == cc.ARRAYS
