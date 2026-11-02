"""
The certificate note's cost reading (indrajala-research docs/workplans/certificate-note-workplan.md,
stage 4): what certifying and checking cost against the size of what is enumerated, from stage 1's
recertification (recertify.py's results.jsonl), beside the target-free proof's test set.

The certificate enumerates every reachable state under every window: R x |Sigma|^k entries, R the
reachable joint codes. Its cost is that count's, a forward pass an entry; the checker's fidelity
pass is as many again. Without the target, the proof needs Chow's test set for a state bound
(chow.test_set_size), which grows as |Sigma|^(bound - n + 1) in the slack bound - n: read here at
the exact bound n (the minimised machine's states) and at slack 1 and 2.

    python scripts/certnote/certificate_cost.py RESULTS.jsonl [...] [--json OUT]

Per family (task and window size): the jobs, R and the entries (min-max), certify's and the
checker's seconds (min-max), the seconds per thousand entries (median), the table's bytes per entry
(median), and the target-free test set's words at slack 0, 1 and 2.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from indrajala_studies import chow
from indrajala_studies import window_tasks as wt


def family(task: str) -> str:
    """A task's family: random machines and planted ones pooled by their size, others by name."""
    if task.startswith("random-planted-"):
        return "random-planted-" + task.split("-")[2]
    if task.startswith("random-"):
        return "random-" + task.split("-")[1]
    return task


def rows(paths: list[Path]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["job"]["id"]] = r  # a rerun's line replaces the earlier one
    return list(out.values())


def readings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(family(r["job"]["task"]), int(r["job"]["columns"]))].append(r)
    table: list[dict[str, Any]] = []
    for (name, k), rs in sorted(groups.items()):
        machine = wt.task_named(rs[0]["job"]["task"]).machine
        n = machine.states
        entries = [r["check"]["entries"] for r in rs]
        certify = [r.get("seconds", r["certify_seconds"]) for r in rs]  # the emitted certificate's search
        check = [r["check"]["structure_seconds"] + r["check"].get("fidelity_seconds", 0.0) for r in rs]
        table.append(
            {
                "family": name,
                "columns": k,
                "jobs": len(rs),
                "certified": sum(r["certified"] for r in rs),
                "valid": sum(r["check"]["valid"] for r in rs),
                "target_states": n,
                "windows": rs[0]["check"]["windows"],
                "reachable": [min(r["reachable"] for r in rs), max(r["reachable"] for r in rs)],
                "entries": [min(entries), max(entries)],
                "certify_seconds": [min(certify), max(certify)],
                "check_seconds": [min(check), max(check)],
                "certify_ms_per_kilo_entry": statistics.median(1e6 * c / e for c, e in zip(certify, entries)),
                "check_ms_per_kilo_entry": statistics.median(1e6 * c / e for c, e in zip(check, entries)),
                "bytes_per_entry": statistics.median(r["check"]["bytes"] / r["check"]["entries"] for r in rs),
                "chow_words": [chow.test_set_size(machine, n + slack) for slack in (0, 1, 2)] if k == 1 else None,
            }
        )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    table = readings(rows(args.results))
    for t in table:
        chow_words = "-" if t["chow_words"] is None else "/".join(f"{w:,}" for w in t["chow_words"])
        print(
            f"{t['family']} k {t['columns']}: {t['jobs']} jobs ({t['certified']} certified, {t['valid']} valid), "
            f"n {t['target_states']}, R {t['reachable'][0]}-{t['reachable'][1]}, entries "
            f"{t['entries'][0]:,}-{t['entries'][1]:,}, certify {t['certify_seconds'][0]:.2f}-"
            f"{t['certify_seconds'][1]:.2f} s, check {t['check_seconds'][0]:.2f}-{t['check_seconds'][1]:.2f} s, "
            f"{t['certify_ms_per_kilo_entry']:.0f} / {t['check_ms_per_kilo_entry']:.0f} ms per 1k entries, "
            f"{t['bytes_per_entry']:.1f} B/entry; Chow words {chow_words}"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
