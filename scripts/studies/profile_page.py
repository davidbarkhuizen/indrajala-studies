"""
A crystallisation's course as one self-contained page (the state formation workplan's option 1): each
profiled run's epochs from its log.jsonl, read as a time series over the epochs trained, its phases
(warm-up, heat, hold, cool, hard) shaded: each factor's temperature, heat capacity and energy
(`thermals`, D32), its codes' groups (D20) and coupling (D25), and the run's screen with its
diagnoses' state share and certificate. Two views: stacked panels on one epoch axis, and an overlay of
one factor's quantities, each rescaled to its own range; and each factor's phase diagram (its codes
against temperature, from the run's saved step networks, beside the melting curve).

    .venv/bin/python scripts/studies/profile_page.py --out page.html NAME=path/to/log.jsonl ...
"""

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from indrajala_studies import modular_family as mf


def course(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A run's epochs as columns (x the epochs trained so far), its diagnoses and its marks."""
    epochs = [r for r in rows if r.get("epoch", 0) > 0 and "screen" in r and "end" not in r]
    factors = max((len(r["anneal"]["temperature"]) for r in epochs if "anneal" in r), default=0)
    columns: dict[str, Any] = {key: [] for key in ("x", "step", "epoch", "phase", "screen", "held")}
    for key in ("T", "C", "E", "G", "L"):
        columns[key] = [[] for _ in range(factors)]
    diagnoses: list[dict[str, Any]] = []
    marks: list[dict[str, Any]] = []
    trained = {id(r) for r in epochs}
    x = 0
    for r in rows:
        if id(r) in trained:
            x += 1
            a = r.get("anneal", {})
            t = a.get("temperature", [None] * factors)
            phase = "warm-up" if r.get("warm") else a.get("phase") or ("hard" if a else "train")
            if phase == "cool" and all(v is None for v in t):
                phase = "hard"
            columns["x"].append(x)
            columns["step"].append(r["step"])
            columns["epoch"].append(r["epoch"])
            columns["phase"].append(phase)
            columns["screen"].append(r["screen"])
            columns["held"].append(a.get("held"))
            soft = "phase" in a
            for key, logged in (("T", "temperature"), ("C", "capacity"), ("E", "energy"), ("G", "groups")):
                values = a.get(logged) if soft else None
                for g in range(factors):
                    columns[key][g].append(None if values is None or t[g] is None else values[g])
            couplings = a.get("coupling")
            for g in range(factors):
                columns["L"][g].append(None if couplings is None else couplings[g])
        elif "certified" in r:
            machine: dict[str, Any] = r.get("machine") or {}
            diagnoses.append(
                {
                    "x": x,
                    "step": r["step"],
                    "share": r.get("state_share"),
                    "certified": r["certified"],
                    "end": r.get("end"),
                    "minimised": r.get("minimised"),
                    "codes": machine.get("codes"),
                    "pairs": machine.get("pairs"),
                    "right": machine.get("right"),
                    "in_use": [f.get("codes") for f in r.get("factors", [])],
                }
            )
        elif "warm_up_end" in r:
            marks.append({"x": x, "label": f"warm-up ends (step {r['step']})"})
        elif r.get("crucible") == "grown" and r.get("epoch") == 0 and not any(m["label"] == "growth" for m in marks):
            marks.append({"x": x, "label": "growth"})
    return {"factors": factors, "columns": columns, "diagnoses": diagnoses, "marks": marks}


def diagram(rows: list[dict[str, Any]], directory: Path) -> list[dict[str, Any]]:
    """
    Each factor's phase diagram (Rose's bifurcation diagram): its codes at each profiled step's end
    (the step's saved network), projected on the first principal axis of all of them, against the
    temperature its last epoch trained at; with the melting curve measured at the warm-up's end (D21)
    and training's own groups (D20) against temperature.
    """
    start = next((r for r in rows if "warm_up_end" in r), None)
    if start is None:
        return []
    last: dict[int, dict[str, Any]] = {}  # each step's last profiled epoch's anneal
    for r in rows:
        if r.get("epoch", 0) > 0 and "phase" in r.get("anneal", {}):
            last[r["step"]] = r["anneal"]
    factors = len(start["warm_up_end"])
    codes: list[list[tuple[float, int, np.ndarray]]] = [[] for _ in range(factors)]
    for step, a in sorted(last.items()):
        path = directory / f"step-{step}.network.json"
        if not path.exists():
            certified = directory / f"certified-step-{step}.network.json"
            if not certified.exists():
                continue
            path = certified
        net = mf.load_network(str(path))
        for g, (factor, t) in enumerate(zip(net.recurrence_layer.factors, a["temperature"], strict=True)):
            if t is not None and factor.codes is not None:
                codes[g].append((float(t), step, np.array(factor.codes.E.tolist(), dtype=float)))
    out: list[dict[str, Any]] = []
    for g, made in enumerate(start["warm_up_end"]):
        if not codes[g]:
            out.append({})
            continue
        pooled = np.concatenate([E for _, _, E in codes[g]])
        centre = pooled.mean(axis=0)
        _, s, vt = np.linalg.svd(pooled - centre, full_matrices=False)
        axis = vt[0]
        points = [[round(t, 6), step, round(float(y), 5)] for t, step, E in codes[g] for y in (E - centre) @ axis]
        training = [[a["temperature"][g], a["groups"][g]] for a in last.values() if a["temperature"][g] is not None]
        out.append(
            {
                "points": points,
                "explained": round(float(s[0] ** 2 / max(float((s**2).sum()), 1e-300)), 3),
                "melting": made.get("melting_point"),
                "critical": made.get("critical_temperature"),
                "curve": made.get("transitions") or [],
                "training": training,
            }
        )
    return out


def splits(rows: list[dict[str, Any]]) -> list[list[list[Any]]]:
    """
    Each factor's splits in training's cool (its groups rising, D20) beside the melting curve's
    prediction (D21, at the warm-up's end): [observed T, groups reached, predicted T, nucleation], the
    prediction the curve's temperature where it first holds as many groups (None if it never does).
    """
    start = next((r for r in rows if "warm_up_end" in r), None)
    if start is None:
        return []
    cool = [r["anneal"] for r in rows if r.get("anneal", {}).get("phase") == "cool" and "groups" in r["anneal"]]
    out: list[list[list[Any]]] = []
    for g, made in enumerate(start["warm_up_end"]):
        curve: list[list[float]] = made.get("transitions") or []
        found: list[list[Any]] = []
        before = None
        for a in cool:
            t, n = a["temperature"][g], a["groups"][g]
            if t is None or n is None:
                continue
            if before is not None and n > before:
                predicted = next((c[0] for c in curve if c[2] >= n), None)
                found.append([t, n, predicted, not found])
            before = n
        out.append(found)
    return out


def steps(rows: list[dict[str, Any]], settle: int | None) -> list[list[list[Any]]]:
    """
    Each factor's temperature steps in the cool: [T, epochs at it, the capacity's mean there, d ln T to
    the next, left at the cap] (the cap read only for an adaptive cool, `settle` its epochs).
    """
    cool = [r["anneal"] for r in rows if r.get("anneal", {}).get("phase") == "cool" and "capacity" in r["anneal"]]
    factors = len(cool[0]["temperature"]) if cool else 0
    out: list[list[list[Any]]] = []
    for g in range(factors):
        levels: list[tuple[float, list[float]]] = []
        for a in cool:
            t, c = a["temperature"][g], a["capacity"][g]
            if t is None or c is None:
                break
            if levels and levels[-1][0] == t:
                levels[-1][1].append(c)
            else:
                levels.append((t, [c]))
        out.append(
            [
                [t, len(cs), sum(cs) / len(cs), math.log(t / after), settle is not None and len(cs) >= settle]
                for (t, cs), (after, _) in itertools.pairwise(levels)
            ]
        )
    return out


def identity(directory: Path) -> dict[str, Any]:
    """The run's identity from its checkpoint: the target's states, the cool's shape and its pace and settle."""
    path = directory / "checkpoint.json"
    run: dict[str, Any] = json.loads(path.read_text())["run"] if path.exists() else {}
    warm = str(run.get("warm_up", ""))
    found = {k: re.search(rf"\b{k}=([^,)]+)", warm) for k in ("shape", "pace", "settle")}
    shape = found["shape"].group(1).strip("'") if found["shape"] else None
    return {
        "q": run.get("target_states"),
        "shape": shape,
        "pace": float(found["pace"].group(1)) if found["pace"] else None,
        "settle": int(found["settle"].group(1)) if found["settle"] and shape == "adaptive" else None,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="NAME=LOG", help="a name for the run and its log.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Crystallisation Profiles")
    args = parser.parse_args(argv)
    runs: list[dict[str, Any]] = []
    for spec in args.runs:
        name, _, path = spec.partition("=")
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        reading = mf.heat_summary(mf.heat_course(rows))
        directory = Path(path).parent
        known = identity(directory)
        runs.append(
            {
                "name": name,
                "reading": reading,
                **known,
                **course(rows),
                "diagram": diagram(rows, directory),
                "splits": splits(rows),
                "steps": steps(rows, known["settle"]),
            }
        )
    template = (Path(__file__).parent / "profile_page.html").read_text()
    page = template.replace("__TITLE__", args.title).replace(
        "__DATA__", json.dumps({"runs": runs}, separators=(",", ":"))
    )
    args.out.write_text(page)
    print(f"{args.out}: {len(runs)} runs, {len(page) // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
