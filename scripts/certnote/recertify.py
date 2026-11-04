"""
The certificate note's recertification under the witness (indrajala-research
docs/workplans/certificate-note-workplan.md, stage 1, D5).

Every network certified on disk (the inventory, stage 0) is certified again with its Table emitted,
and the table checked by the independent checker (certificate_check.py), structure and fidelity.

    python scripts/certnote/recertify.py plan data/certificate-note/inventory.csv --root data/window-recurrent \\
        --out data/certificate-note/recertify/jobs.jsonl
    python scripts/certnote/recertify.py run data/certificate-note/recertify/jobs.jsonl \\
        --out data/certificate-note/recertify --workers 2 [--only-columns 1] [--limit N]

plan resolves each certified network (by sha256, where its certificates name one) to a file on disk:
the certificate's network path, as written or under the root; the file beside the certificate
(X.certificate.json's X.network.json); a certify CLI name (<run>-step-<j>.json for
<run>/step-<j>.network.json); or any network file of the same name whose sha256 matches. A network
whose certificates name no sha256 (addition's oldest) is taken from its path alone, and its
recertification must reproduce the stored certificate. One job per network and verdict it was
certified in (strict, snap ratio 1.0; tie-branched, above 1.0), plus E6's linear-window networks
(D9) at the tie-branched ratio whatever their stored verdicts. A job names the stored certificate
it must reproduce, preferring one made on x86-64 (the aarch64 recertifications are other copies).

run certifies each job (a family network, whose certificate has a slot, through its QueryMachine:
the encoding is not recorded, so every arm's queries the network's width allows are tried, the
number that reproduce the stored certificate counted, and the first that does kept), emits its table to
tables/<sha256[:16]>-<ratio>.npz, checks it, and appends one line to results.jsonl: the job, the
adapter, whether the stored certificate was reproduced (certified, reachable, phi, the wrong
counts, the near ties, the minimised machine) and the margins' differences from it, the table's
sizes, certify's seconds and the checker's report. A job already in results.jsonl is skipped
(resume by rerunning).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certificate_check as cc  # scripts/ isn't a package
from indrajala_ml.measurement.worker_threads import worker_threads

STRICT, BRANCHED = 1.0, 1.05  # the two verdicts' snap ratios (window_recurrent_certificate.SNAP_RATIO)
E6 = "e6-linear"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stored(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    return value


def resolve(
    certificate: Path, stored: dict[str, Any], root: Path, by_name: dict[str, list[Path]], digests: dict[Path, str]
) -> Path | None:
    """The network file a stored certificate is of (the module's docstring), or None."""
    want = stored.get("sha256")

    def matches(path: Path) -> bool:
        if not path.is_file():
            return False
        if want is None:
            return True
        if path not in digests:
            digests[path] = sha256(path)
        return digests[path] == want

    named = stored.get("network")
    candidates: list[Path] = []
    if named:
        p = Path(named)
        candidates += [p, root / p, root.parent.parent / p]
        parts = p.parts
        if "window-recurrent" in parts:
            candidates.append(root.joinpath(*parts[parts.index("window-recurrent") + 1 :]))
    name = certificate.name
    if name.endswith(".certificate.json"):
        candidates.append(certificate.with_name(name.removesuffix(".certificate.json") + ".network.json"))
    if found := re.fullmatch(r"(.+)-(step-\d+)\.json", name):
        candidates += [p for p in by_name.get(f"{found[2]}.network.json", []) if p.parent.name == found[1]]
    for path in candidates:
        if matches(path):
            return path
    if want is not None and named:
        for path in by_name.get(Path(named).name, []):
            if matches(path):
                return path
    return None


def plan(inventory: Path, root: Path, out: Path) -> list[dict[str, Any]]:
    """The jobs (the module's docstring), written to out as JSON lines."""
    rows = list(csv.DictReader(inventory.open()))
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*.network.json"):
        by_name[path.name].append(path)
    digests: dict[Path, str] = {}
    jobs: dict[tuple[str, float], dict[str, Any]] = {}
    unresolved: list[str] = []
    for r in rows:
        e6 = E6 in r["file"]
        if r["certified"] != "True" and not e6:
            continue
        certificate = Path(r["file"])
        stored = _stored(certificate)
        network = resolve(certificate, stored, root, by_name, digests)
        if network is None:
            unresolved.append(r["file"])
            continue
        identity = stored.get("sha256") or digests.get(network) or sha256(network)
        ratio = STRICT if float(stored.get("snap_ratio", BRANCHED)) <= 1.0 else BRANCHED
        verdicts = [ratio] if r["certified"] == "True" else []
        if e6 and BRANCHED not in verdicts:
            verdicts.append(BRANCHED)
        for v in verdicts:
            key = (identity, v)
            reproduce = str(certificate) if v == ratio else None
            job = jobs.get(key)
            if job is None or (reproduce and (job["reproduce"] is None or "aarch64" in job["reproduce"])):
                jobs[key] = {
                    "id": f"{identity[:16]}-{v}",
                    "sha256": identity,
                    "network": str(network),
                    "task": stored.get("task") or "addition",
                    "family": "slot" in stored,
                    "snap_ratio": v,
                    "columns": stored["columns"],
                    "reproduce": reproduce,
                    "e6": e6,
                }
    out.parent.mkdir(parents=True, exist_ok=True)
    listed = sorted(jobs.values(), key=lambda j: (j["columns"], j["task"], j["id"]))
    out.write_text("".join(json.dumps(j) + "\n" for j in listed))
    print(
        f"{len(listed)} jobs ({len({j['sha256'] for j in listed})} networks); {len(unresolved)} certificates unresolved"
    )
    for f in unresolved[:20]:
        print(f"  unresolved: {f}")
    return listed


REPRODUCED = ("certified", "reachable", "phi", "wrong_outputs", "wrong_states", "minimised")


def reproduces(fresh: dict[str, Any], stored: dict[str, Any]) -> bool:
    """Whether a fresh certificate's discrete verdict is the stored one's (the margins' near ties too)."""
    fresh = json.loads(json.dumps(fresh))  # tuples as the stored JSON's lists
    old = dict(stored)
    # addition's oldest certificates name the counts in the carry's terms (the certificate's docstring)
    old.setdefault("wrong_outputs", old.get("wrong_digits"))
    old.setdefault("wrong_states", old.get("wrong_carries"))
    same = all(fresh.get(key) == old.get(key) for key in REPRODUCED if key in old)
    return same and all(fresh[m]["short"] == old[m]["short"] for m in ("snap", "lead") if m in old)


def _candidates(job: dict[str, Any], width: int) -> list[Any]:
    """A family network's encodings whose window is the network's, the likeliest first."""
    from indrajala_studies import modular_family as mf

    task = job["task"]
    arms = [mf.FAMILY, mf.QUOTIENT, mf.SUM, mf.PRODUCT, mf.STATE, mf.SUFFIX, mf.TOKEN, mf.PLANTED, mf.COVER]
    found: dict[tuple[int, ...], Any] = {}
    if task.startswith("modular-division-"):
        p = int(task.rsplit("-", 1)[1])
        for vocabulary in (mf.VOCABULARY, mf.WIDE):
            for queries in arms:
                if all(q in vocabulary for q in queries):
                    encoding = mf.Encoding(p, queries, vocabulary)
                    if encoding.width == width:
                        found.setdefault(tuple(encoding.query_tokens), encoding)
    else:
        for queries in ((), mf.PLANTED, mf.COVER):
            try:
                encoding = mf.target_named(task, queries)
            except AssertionError, KeyError, ValueError:
                continue
            if encoding.width == width:
                found.setdefault(tuple(encoding.query_tokens), encoding)
    return list(found.values())


def recertify(job: dict[str, Any], out: Path) -> dict[str, Any]:
    """One job: certify with the table emitted, reproduce the stored certificate, check the table."""
    from indrajala_ml.model.persistence.load_network import load_network

    from indrajala_studies import modular_family as mf
    from indrajala_studies import window_recurrent_certificate as wc

    network = Path(job["network"])
    table = out / "tables" / f"{job['id']}.npz"
    table.parent.mkdir(parents=True, exist_ok=True)
    stored = _stored(Path(job["reproduce"])) if job["reproduce"] else None
    begun = time.perf_counter()
    tried = matching = 0
    if job["family"]:
        # every candidate tried, so an ambiguous one (two encodings reproducing) is counted; the
        # first that reproduces is kept, its table emitted
        width = load_network(str(network)).columns
        candidates = _candidates(job, width)
        assert candidates, f"no encoding of {job['task']} has the network's window ({width})"
        kept = None
        for encoding in candidates:
            tried += 1
            try:
                fresh = mf.certify_file(network, encoding, job["task"], job["snap_ratio"])
            except ValueError:  # a query token outside the network's vocabulary: not its encoding
                continue
            if stored is None or reproduces(fresh, stored):
                matching += 1
                kept = kept or encoding
        encoding = kept or candidates[0]
        fresh = mf.certify_file(network, encoding, job["task"], job["snap_ratio"], emit=table)
        adapter = list(encoding.query_tokens)
    else:
        fresh, adapter = wc.certify_file(network, job["task"], job["snap_ratio"], emit=table), None
        tried = matching = 1
    certify_seconds = time.perf_counter() - begun
    reproduced = None if stored is None else reproduces(fresh, stored)
    margins = (
        {}
        if stored is None
        else {m: abs(fresh[m]["smallest"] - stored[m]["smallest"]) for m in ("snap", "lead") if m in stored}
    )
    report = cc.check(table)
    return {
        "job": job,
        "query_tokens": adapter,
        "encodings_tried": tried,
        "encodings_reproducing": matching,
        "reproduced": reproduced,
        "margin_differences": margins,
        "certified": fresh["certified"],
        "reachable": len(fresh["reachable"]),
        "minimised": len(fresh["minimised"]),
        "near_ties": fresh["snap"]["short"],
        "split_ties": fresh.get("split_ties", 0),
        "evaluations": fresh["evaluations"],
        "certify_seconds": certify_seconds,  # every encoding tried, for a family network
        "seconds": fresh["seconds"],  # the emitted certificate's own search
        "check": report,
    }


def run(jobs_file: Path, out: Path, workers: int, only_columns: int | None, limit: int | None) -> int:
    jobs = [json.loads(line) for line in jobs_file.read_text().splitlines() if line.strip()]
    if only_columns is not None:
        jobs = [j for j in jobs if j["columns"] == only_columns]
    results = out / "results.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    done: set[str] = (
        {json.loads(line)["job"]["id"] for line in results.read_text().splitlines() if line.strip()}
        if results.exists()
        else set()
    )
    todo = [j for j in jobs if j["id"] not in done][:limit]
    print(f"{len(todo)} jobs to run ({len(done)} done)", flush=True)
    failures = 0
    context = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=context, **worker_threads(workers)) as pool:
        futures = {pool.submit(recertify, job, out): job for job in todo}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001  (a job's failure is recorded, the rest go on)
                failures += 1
                print(f"{job['id']}: error {error!r}", flush=True)
                with (out / "errors.jsonl").open("a") as f:
                    f.write(json.dumps({"job": job, "error": repr(error)}) + "\n")
                continue
            with results.open("a") as f:
                f.write(json.dumps(result) + "\n")
            check = result["check"]
            print(
                f"{job['id']} {job['task']}: certified {result['certified']}, reproduced {result['reproduced']}, "
                f"table {'valid' if check['valid'] else 'INVALID'} (agrees {check['agrees']}), "
                f"{check['entries']} entries, {result['certify_seconds']:.1f} + "
                f"{check['structure_seconds'] + check.get('fidelity_seconds', 0):.1f} s",
                flush=True,
            )
    print(f"recertify ended: {len(todo) - failures} done, {failures} errors", flush=True)
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("inventory", type=Path)
    p.add_argument("--root", type=Path, required=True, help="the data directory the inventory read")
    p.add_argument("--out", type=Path, required=True)
    r = commands.add_parser("run")
    r.add_argument("jobs", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--only-columns", type=int, help="this window size's jobs only")
    r.add_argument("--limit", type=int, help="at most this many jobs")
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan(args.inventory, args.root, args.out)
        return 0
    return run(args.jobs, args.out, args.workers, args.only_columns, args.limit)


if __name__ == "__main__":
    sys.exit(main())
