"""
Rewrites files written before the factored state (the factored-state workplan, D9: old files are
migrated, not read through shims) into its layout, in place, printing each file's old and new
sha256.

A saved window-recurrent network (format 2): its "recurrence", {"state_size": s, "codes": ...,
"cell": ..., "pooling": ...}, becomes {"factors": [{"state_size": s, "codes": ...}], "cross":
false, "cell": ..., "pooling": ...}, the one factor it had. Its weights and optimizer state are
unchanged: a factor's parts are in the order the network's parts were (pooling, cell, codes,
initial state), so the file loads as the same network, by bits.

A study's result (results/<key>.json) and its step records (models/<key>/step-<j>.json), as the
harness writes them since stage 7: a code is a tuple, a code per factor, so "codes_used" and each
test's "codes" become lists of one-code lists; a result gains its cell's "factors" ([[s, m]]),
"cross" (false) and "reads" ("output", the pooling's reading before D10), after "m", and
"certified" (false), "certified_step" (null) and "stopped" (false), after "works": what today's
harness writes for a run without certify_steps or stop_at_certified, as every old run was (a
network certified later, by `certify`, has its certificate file). "peak_mb" stays absent:
unmeasured.

A certificate (window_recurrent_certificate's JSON, stage 6): "codes" a count per factor, a state a
list of codes ("start", "reachable", the counterexample's "codes", the margins' "code", the
minimised machine's states), "phi" a list of [state, target] pairs, and "factors", each factor's
readout (D9), computed from the certificate's own reachable states and phi as certify computes it.
E1's certificates keep their names (wrong_digits, wrong_carries, carry_in, addition) and their
missing "task" (addition). A certificate's "sha256", its network file's, becomes the migrated
network's.

`tree` migrates every such file under the roots: the networks first, then the step records, results
and certificates, then every manifest (*.sha256) listing a migrated file, its old hashes replaced
and the old manifest kept beside it as <name>.before-factored-state.sha256. It writes the table of
every changed file's old and new sha256 (stage 5's hash table). A file already migrated is left
alone, so a second run changes nothing.

    .venv/bin/python scripts/data/migrate_factored_state.py network FILE...
    .venv/bin/python scripts/data/migrate_factored_state.py tree ROOT... --table FILE
    .venv/bin/python scripts/data/migrate_factored_state.py recertify CERTIFICATE...

`recertify` is stage 5's check: each certificate's network (its "network" path, from the working
directory) certified again at its snap ratio, against its task, and compared with the certificate
field by field (all but "seconds"); it prints each one's differences and fails on any.

Stage 4 migrated the saved-model fixtures with `network`; stage 5 migrates the studies' files with
`tree`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_tasks as wt

BEFORE = ".before-factored-state"


def migrate_network(state: dict[str, Any]) -> dict[str, Any]:
    """A format-2 network's state with its recurrence in the factored layout; other shapes unchanged."""
    recurrence = state.get("recurrence")
    if recurrence is None or "factors" in recurrence:
        return state
    factored = {
        "factors": [{"state_size": recurrence["state_size"], "codes": recurrence["codes"]}],
        "cross": False,
        "cell": recurrence["cell"],
        "pooling": recurrence["pooling"],
    }
    return {key: factored if key == "recurrence" else value for key, value in state.items()}


def _codes(codes: list[Any] | None) -> list[Any] | None:
    """A list of codes as one-factor states; None (a continuous state) and states already tuples unchanged."""
    if codes is None:
        return None
    return [[code] if isinstance(code, int) else code for code in codes]


def migrate_step(step: dict[str, Any]) -> dict[str, Any]:
    """A step record with its codes as one-factor states."""
    tests = {n: {**test, "codes": _codes(test["codes"])} for n, test in step["tests"].items()}
    return {**step, "tests": tests, "codes_used": _codes(step["codes_used"])}


def _inserted(value: dict[str, Any], after: str, fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, field in value.items():
        out[key] = field
        if key == after:
            out.update(fields)
    return out


def migrate_result(result: dict[str, Any]) -> dict[str, Any]:
    """A run's result in stage 7's layout (the module's docstring); one already migrated unchanged."""
    if "factors" in result:
        return result
    migrated = _inserted(result, "m", {"factors": [[result["s"], result["m"]]], "cross": False, "reads": "output"})
    migrated = _inserted(migrated, "works", {"certified": False, "certified_step": None, "stopped": False})
    if "steps" in migrated:
        migrated["steps"] = [migrate_step(step) for step in migrated["steps"]]
    return migrated


def migrate_certificate(certificate: dict[str, Any], networks: Mapping[str, str]) -> dict[str, Any]:
    """
    A certificate in stage 6's layout (the module's docstring), its network's sha256 through networks
    (old to new); one already migrated unchanged.
    """
    if not isinstance(certificate["codes"], int):
        return certificate
    reachable = [(code,) for code in certificate["reachable"]]
    phi = {(int(code),): target for code, target in certificate["phi"].items()}
    target = wt.task_named(certificate.get("task", wt.ADDITION.name)).machine
    out = dict(certificate)
    if "sha256" in certificate:  # certify_file's: its network's
        out["sha256"] = networks.get(certificate["sha256"], certificate["sha256"])
    out["codes"], out["start"] = [certificate["codes"]], [certificate["start"]]
    out["reachable"] = [list(state) for state in reachable]
    out["phi"] = [[list(state), target_state] for state, target_state in phi.items()]
    counterexample = certificate["counterexample"]
    if counterexample is not None:
        out["counterexample"] = {**counterexample, "codes": _codes(counterexample["codes"])}
    for margin in ("snap", "lead"):
        code = certificate[margin]["code"]
        out[margin] = {**certificate[margin], "code": None if code is None else [code]}
    out["minimised"] = [_codes(block) for block in certificate["minimised"]]
    readouts = [asdict(readout) for readout in wc.readout(reachable, phi, target)]
    return _inserted(out, "minimised", {"factors": readouts})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite(path: Path, migrate: Callable[[Any], Any], indent: int | None) -> tuple[str, str]:
    """path rewritten by migrate, as its writer writes it; its old and new sha256."""
    old = _sha256(path)
    value = json.loads(path.read_text())
    migrated = migrate(value)
    if migrated != value:
        path.write_text(json.dumps(migrated, indent=indent))
    return old, _sha256(path)


def migrate_network_file(path: Path) -> tuple[str, str]:
    """path rewritten by migrate_network, as save() writes (model_io.py); its old and new sha256."""
    return _rewrite(path, migrate_network, None)


def _kind(value: Any) -> str | None:
    """What a study's JSON file is: a certificate, a result or a step record; None for anything else."""
    if not isinstance(value, dict):
        return None
    if "reachable" in value and "phi" in value:
        return "certificate"
    if "key" in value and "k" in value and ("steps" in value or "error" in value):
        return "result"
    if "step" in value and "codes_used" in value and "tests" in value:
        return "step"
    return None


def migrate_manifest(path: Path, hashes: Mapping[str, str]) -> bool:
    """
    A sha256sum manifest with its migrated files' hashes replaced (hashes: old to new), the old one
    kept beside it (<name>.before-factored-state.sha256, never overwritten); whether it changed.
    """
    lines = path.read_text().splitlines(keepends=True)
    migrated = [hashes[line[:64]] + line[64:] if line[:64] in hashes else line for line in lines]
    if migrated == lines:
        return False
    kept = path.with_name(path.stem + BEFORE + path.suffix)
    if not kept.exists():
        kept.write_text("".join(lines))
    path.write_text("".join(migrated))
    return True


def migrate_tree(roots: Sequence[Path]) -> list[tuple[Path, str, str]]:
    """Every file under roots migrated (the module's docstring); each changed file's path, old and new sha256."""
    files = sorted({path for root in roots for path in root.rglob("*") if path.is_file()})
    changed: list[tuple[Path, str, str]] = []

    def note(path: Path, old: str, new: str) -> None:
        if old != new:
            changed.append((path, old, new))

    for path in files:
        if path.name.endswith(".network.json"):
            note(path, *migrate_network_file(path))
    networks = {old: new for _, old, new in changed}
    for path in files:
        if path.suffix != ".json" or path.name.endswith(".network.json") or BEFORE in path.name:
            continue
        kind = _kind(json.loads(path.read_text()))
        if kind == "certificate":
            note(path, *_rewrite(path, lambda c: migrate_certificate(c, networks), 1))
        elif kind == "result":
            note(path, *_rewrite(path, migrate_result, 1))
        elif kind == "step":
            note(path, *_rewrite(path, migrate_step, 1))
    hashes = {old: new for _, old, new in changed}
    for path in files:
        if path.suffix == ".sha256" and BEFORE not in path.name:
            old = _sha256(path)
            if migrate_manifest(path, hashes):
                note(path, old, _sha256(path))
    return changed


def recertify(path: Path) -> list[str]:
    """Where a certificate and its network's certificate made now differ (the module's docstring)."""
    certificate = json.loads(path.read_text())
    task = certificate.get("task", wt.ADDITION.name)
    fresh = wc.certify_file(Path(certificate["network"]), task, certificate.get("snap_ratio", wc.SNAP_RATIO))
    if "snap_ratio" not in certificate:  # E1's first certificates, from before it was recorded: the default
        del fresh["snap_ratio"]
    fresh = json.loads(json.dumps(fresh))  # as written: its states lists
    if "task" not in certificate:  # E1's, from before the task field: its names are stage 2's
        fresh = _stage2_names(fresh)
    return [
        key
        for key in sorted(certificate.keys() | fresh.keys())
        if key != "seconds" and certificate.get(key) != fresh.get(key)
    ]


def _stage2_names(certificate: dict[str, Any]) -> dict[str, Any]:
    """
    A certificate under E1's names (wrong_digits, wrong_carries, carry_in, addition), without what
    E1's lack: the task and the counterexample's window after.
    """
    names = {"wrong_outputs": "wrong_digits", "wrong_states": "wrong_carries"}
    out = {names.get(key, key): value for key, value in certificate.items() if key != "task"}
    counterexample = out["counterexample"]
    if counterexample is not None:
        kinds = {"outputs": "digits", "state": "carry"}
        out["counterexample"] = {
            {"state_in": "carry_in", "case": "addition"}.get(key, key): kinds.get(value, value)
            if key == "kind"
            else value
            for key, value in counterexample.items()
            if key != "after"
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    network = commands.add_parser("network", help="saved window-recurrent networks (format 2)")
    network.add_argument("files", nargs="+", type=Path)
    tree = commands.add_parser("tree", help="every network, result, step record, certificate and manifest")
    tree.add_argument("roots", nargs="+", type=Path)
    tree.add_argument("--table", type=Path, required=True, help="each changed file's path, old and new sha256")
    again = commands.add_parser("recertify", help="each certificate's network certified again and compared")
    again.add_argument("certificates", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "recertify":
        failed = 0
        for path in arguments.certificates:
            differs = recertify(path)
            failed += bool(differs)
            print(f"{path}: {'differs in ' + ', '.join(differs) if differs else 'the same'}")
        print(f"{len(arguments.certificates) - failed} of {len(arguments.certificates)} the same")
        return 1 if failed else 0
    if arguments.command == "network":
        for path in arguments.files:
            old, new = migrate_network_file(path)
            print(f"{path}  {old}  {new}{'  (unchanged)' if old == new else ''}")
        return 0
    changed = migrate_tree(arguments.roots)
    arguments.table.write_text("".join(f"{path}\t{old}\t{new}\n" for path, old, new in changed))
    print(f"{len(changed)} files migrated; table in {arguments.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
