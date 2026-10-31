"""scripts/data/migrate_factored_state.py: old files rewritten as today's code writes them (the factored-state workplan, stage 5)."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from indrajala_ml.model.persistence.load_network import load_network

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt
from tests.test_window_recurrent_certificate import CarryMachine, TaskMachine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "data"))
import migrate_factored_state as migrate  # scripts/ isn't a package


def as_written(value: Any) -> Any:
    return json.loads(json.dumps(value))


def old_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    """A certificate (as written) in the layout before the factored state: one code a state, phi a map."""
    [codes] = certificate["codes"]
    old = {key: value for key, value in certificate.items() if key != "factors"}
    old.update(
        codes=codes,
        start=certificate["start"][0],
        reachable=[state[0] for state in certificate["reachable"]],
        phi={str(state[0]): target for state, target in certificate["phi"]},
        snap={
            **certificate["snap"],
            "code": None if certificate["snap"]["code"] is None else certificate["snap"]["code"][0],
        },
        lead={
            **certificate["lead"],
            "code": None if certificate["lead"]["code"] is None else certificate["lead"]["code"][0],
        },
        minimised=[[state[0] for state in block] for block in certificate["minimised"]],
    )
    if certificate["counterexample"] is not None:
        old["counterexample"] = {
            **certificate["counterexample"],
            "codes": [state[0] for state in certificate["counterexample"]["codes"]],
        }
    return old


CERTIFICATES = [
    pytest.param(lambda: wc.certify(CarryMachine(2, copies=2)), id="certified-copies"),
    pytest.param(lambda: wc.certify(CarryMachine(1, fault="drops")), id="addition-counterexample"),
    pytest.param(lambda: wc.certify(CarryMachine(2, fault="snap-tie")), id="near-tie"),
    pytest.param(
        lambda: wc.certify(TaskMachine(wt.task_named("modular-sum-4"), 1, copies=2), wt.task_named("modular-sum-4")),
        id="modular-sum",
    ),
    pytest.param(lambda: wc.certify(wc.NetworkMachine(wr.build(wt.ADDITION, 1, 4, 3, seed=0))), id="network"),
]


@pytest.mark.parametrize("make", CERTIFICATES)
def test_an_old_certificate_migrates_to_what_certify_writes_now(make: Any):
    certificate = as_written(make().to_json())
    old = old_certificate(certificate)
    assert old != certificate
    assert migrate.migrate_certificate(old, {}) == certificate


def test_a_migrated_certificate_names_its_migrated_network_and_migrates_once():
    certificate = {"network": "n.json", "sha256": "a" * 64, **as_written(wc.certify(CarryMachine(1)).to_json())}
    migrated = migrate.migrate_certificate(old_certificate(certificate), {"a" * 64: "b" * 64})
    assert migrated["sha256"] == "b" * 64
    assert migrate.migrate_certificate(migrated, {"b" * 64: "c" * 64}) is migrated


def test_e1s_certificate_keeps_its_names_and_reads_out_the_carry():
    certificate = as_written(wc.certify(CarryMachine(1, fault="drops")).to_json())
    old = old_certificate(certificate)
    del old["task"]
    old["wrong_digits"], old["wrong_carries"] = old.pop("wrong_outputs"), old.pop("wrong_states")
    migrated = migrate.migrate_certificate(old, {})
    assert "task" not in migrated and "wrong_outputs" not in migrated
    assert migrated["wrong_carries"] == certificate["wrong_states"]
    assert migrated["factors"] == certificate["factors"]
    assert migrated["reachable"] == certificate["reachable"]


def old_result() -> dict[str, Any]:
    tests: dict[str, Any] = {"8": {"passes": False, "reach": None, "codes": [0, 2], "properties": {}, "seconds": 1.0}}
    step: dict[str, Any] = {
        "step": 1,
        "epochs": 3,
        "tests": tests,
        "break": 0,
        "codes_used": [0, 1, 2],
        "train_seconds": [1.0],
    }
    fields = {"key": "curriculum-k1-s16-m8-seed0", "task": "modular-sum-4", "arm": "curriculum", "k": 1}
    return {**fields, "s": 16, "m": 8, "seed": 0, "steps": [step], "break": 0, "works": False, "seconds": 2.0}


def test_an_old_result_gains_its_cells_fields_and_tuple_codes_in_todays_order():
    migrated = migrate.migrate_result(old_result())
    assert list(migrated) == [
        *["key", "task", "arm", "k", "s", "m", "factors", "cross", "reads", "seed", "steps", "break", "works"],
        *["certified", "certified_step", "stopped", "seconds"],
    ]
    assert (migrated["factors"], migrated["cross"], migrated["reads"]) == ([[16, 8]], False, "output")
    assert (migrated["certified"], migrated["certified_step"], migrated["stopped"]) == (False, None, False)
    [step] = migrated["steps"]
    assert step["codes_used"] == [[0], [1], [2]]
    assert step["tests"]["8"]["codes"] == [[0], [2]]
    assert migrate.migrate_result(migrated) == migrated


def test_a_continuous_runs_result_keeps_no_codes():
    result = old_result()
    result["m"] = None
    result["steps"][0]["codes_used"] = None
    result["steps"][0]["tests"]["8"]["codes"] = None
    migrated = migrate.migrate_result(result)
    assert migrated["factors"] == [[16, None]]
    assert migrated["steps"][0]["codes_used"] is None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_network(path: Path) -> None:
    """A network saved in the layout before the factored state: its one factor's fields in its recurrence."""
    wr.build(wt.ADDITION, 1, 4, 3, seed=0).save(str(path))
    state = json.loads(path.read_text())
    recurrence = state["recurrence"]
    [factor] = recurrence.pop("factors")
    del recurrence["cross"]
    state["recurrence"] = {**factor, **recurrence}
    path.write_text(json.dumps(state))


@pytest.fixture
def study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An old study's files: a network, its step record and result, its certificate and a manifest."""
    monkeypatch.chdir(tmp_path)
    root = Path("study")
    models = root / "models" / "curriculum-k1-s4-m3-seed0"
    models.mkdir(parents=True)
    (root / "results").mkdir()
    (root / "certificates").mkdir()
    network = models / "step-1.network.json"
    old_network(network)
    result = old_result()
    (models / "step-1.json").write_text(json.dumps(result["steps"][0], indent=1))
    (root / "results" / "curriculum-k1-s4-m3-seed0.json").write_text(json.dumps(result, indent=1))
    old_certificate_file = {
        "network": str(network),
        "sha256": sha256(network),
        "snap_ratio": 1.0,
        **old_certificate(as_written(wc.certify(CarryMachine(1)).to_json())),
    }
    (root / "certificates" / "curriculum-k1-s4-m3-seed0-step-1.json").write_text(json.dumps(old_certificate_file))
    files = sorted(path for path in root.rglob("*.json"))
    (root / "study.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in files))
    (root / "notes.json").write_text(json.dumps({"lengths": {}}))
    return root


def test_a_tree_migrates_every_file_once_and_its_manifest(study: Path):
    manifest = (study / "study.sha256").read_text()
    changed = migrate.migrate_tree([study])
    network = study / "models" / "curriculum-k1-s4-m3-seed0" / "step-1.network.json"
    assert {path.name for path, _, _ in changed} == {
        "step-1.network.json",
        "step-1.json",
        "curriculum-k1-s4-m3-seed0.json",
        "curriculum-k1-s4-m3-seed0-step-1.json",
        "study.sha256",
    }
    load_network(str(network))
    certificate = json.loads((study / "certificates" / "curriculum-k1-s4-m3-seed0-step-1.json").read_text())
    assert certificate["sha256"] == sha256(network)
    assert (study / "study.before-factored-state.sha256").read_text() == manifest
    for line in (study / "study.sha256").read_text().splitlines():
        digest, name = line.split("  ")
        assert sha256(study / name) == digest
    assert json.loads((study / "notes.json").read_text()) == {"lengths": {}}
    assert migrate.migrate_tree([study]) == []


def test_recertify_finds_a_certificate_the_same_as_its_networks_and_a_changed_one_not(tmp_path: Path):
    network = tmp_path / "step-1.network.json"
    wr.build(wt.ADDITION, 1, 4, 3, seed=0).save(str(network))
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(wc.certify_file(network, snap_ratio=1.0), indent=1))
    assert migrate.recertify(path) == []
    assert migrate.main(["recertify", str(path)]) == 0
    changed = json.loads(path.read_text())
    changed["wrong_states"] += 1
    path.write_text(json.dumps(changed))
    assert migrate.recertify(path) == ["wrong_states"]
    assert migrate.main(["recertify", str(path)]) == 1
