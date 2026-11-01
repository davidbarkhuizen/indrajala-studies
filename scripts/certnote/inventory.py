"""
The certificate note's inventory (indrajala-research docs/workplans/certificate-note-workplan.md, stage 0).

Reads every certificate JSON under a data directory (window_recurrent_certificate.Certificate's
to_json, with the network's path and sha256 that the study scripts add) and writes one row per
certificate file: its verdict (strict, snap ratio 1.0, or tie-branched, above 1.0), the task, the
window's columns, each factor's codes, the reachable and minimised state counts, the evaluations
(windows run) and seconds, the margins (the smallest snap ratio and logit lead, the near ties) and
whether it certified, and whether the network's file is still on disk.

    python scripts/certnote/inventory.py DATA_DIR --out data/certificate-note/inventory.csv

The summary printed after: certificate files, distinct networks (by sha256), certified networks in
both verdicts, and per task the reachable states, evaluations and seconds of the certified ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

FIELDS = [
    "file", "network", "network_on_disk", "sha256", "verdict", "task", "columns", "codes", "reachable",
    "minimised", "evaluations", "seconds", "snap_smallest", "snap_short", "lead_smallest", "lead_short",
    "split_ties", "wrong_outputs", "wrong_states", "certified",
]


def row(path: Path, d: dict[str, Any], root: Path) -> dict[str, Any]:
    network = d.get("network") or ""
    on_disk = bool(network) and any(p.exists() for p in (Path(network), root.parent.parent / network, root / network))
    return {
        "file": str(path),
        "network": network,
        "network_on_disk": on_disk,
        "sha256": d.get("sha256", ""),
        "verdict": "strict" if float(d.get("snap_ratio", 1.05)) <= 1.0 else "branched",
        "task": d.get("task", ""),
        "columns": d.get("columns"),
        "codes": "x".join(str(c) for c in d.get("codes", [])),
        "reachable": len(d.get("reachable", [])),
        "minimised": len(d.get("minimised", [])),
        "evaluations": d.get("evaluations"),
        "seconds": d.get("seconds"),
        "snap_smallest": (d.get("snap") or {}).get("smallest"),
        "snap_short": (d.get("snap") or {}).get("short"),
        "lead_smallest": (d.get("lead") or {}).get("smallest"),
        "lead_short": (d.get("lead") or {}).get("short"),
        "split_ties": d.get("split_ties", 0),
        "wrong_outputs": d.get("wrong_outputs"),
        "wrong_states": d.get("wrong_states"),
        "certified": d.get("certified"),
    }


def rows(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.rglob("*.json")):
        if path.stat().st_size > 200_000_000:
            continue
        try:
            with path.open() as f:
                head = f.read(4096)
            if '"reachable"' not in head and '"phi"' not in head and '"wrong_outputs"' not in head:
                # a certificate's keys come early; skip networks, logs and reports cheaply
                if '"snap_ratio"' not in head:
                    continue
            d = json.loads(path.read_text())
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and "reachable" in d and "certified" in d:
            out.append(row(path, d, root))
    return out


def summary(table: list[dict[str, Any]]) -> str:
    networks = {r["sha256"] for r in table if r["sha256"]}
    by_verdict: dict[str, set[str]] = defaultdict(set)
    for r in table:
        if r["certified"]:
            by_verdict[r["verdict"]].add(r["sha256"] or r["file"])
    both = by_verdict["strict"] & by_verdict["branched"]
    lines = [
        f"certificate files: {len(table)}; distinct networks: {len(networks)}",
        f"certified strict: {len(by_verdict['strict'])}; tie-branched: {len(by_verdict['branched'])}; both: {len(both)}",
        "per task, certified tie-branched (distinct networks): count, reachable min-max, minimised, "
        "evaluations min-max, seconds min-max, smallest snap ratio, near ties max",
    ]
    tasks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in table:
        if r["certified"] and r["verdict"] == "branched":
            tasks[r["task"]][r["sha256"] or r["file"]] = r
    for task, rs in sorted(tasks.items()):
        v = list(rs.values())

        def span(k: str, fmt: str = "{}") -> str:
            xs = [x[k] for x in v if x[k] is not None]
            return f"{fmt.format(min(xs))}-{fmt.format(max(xs))}" if xs else "-"

        lines.append(
            f"  {task}: {len(v)}, reachable {span('reachable')}, minimised {span('minimised')}, "
            f"evaluations {span('evaluations')}, seconds {span('seconds', '{:.2f}')}, "
            f"snap {min(x['snap_smallest'] for x in v):.4f}, ties <= {max(x['snap_short'] for x in v)}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    table = rows(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    print(summary(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
