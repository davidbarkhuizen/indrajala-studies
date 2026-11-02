"""
The certificate note's interval bound (indrajala-research docs/workplans/certificate-note-workplan.md,
stage 3, D7): which of a certificate's decisions hold for the network's real-valued function, not
only for its floating-point evaluation.

certify reads each (state, window) entry's labels and snaps off one floating-point evaluation, with
fixed margins (a snap ratio of 1.05 branched, a logit lead of 1e-6). Here the same window is
evaluated in midpoint-radius interval arithmetic from the same exact inputs (the window's tokens and
the state's codebook rows), enclosing the exact real value of every activation:

- a sum of n terms of floats carries the order-free error bound gamma_n = n u / (1 - n u) times the
  sum of the terms' magnitudes (u = 2^-53), so any summation order, blocking or tiling is covered;
- every other elementary result (a product, a difference, a radius computed) is inflated by u of
  its magnitude, and exp, tanh, sigmoid, sqrt and a reciprocal by ULPS ulps (an assumption on the
  maths library, stated with the result);
- a softmax is shifted by an exact constant (the real function doesn't depend on it), so its max
  isn't an interval.

Then for every entry, with the table's recorded decisions:

- a label is sound when its logit's interval lies above every other class's (softmax is monotone);
- a snap is sound when the nearest code's squared-distance interval lies below every other code's;
  otherwise the codes whose intervals overlap the nearest's are the bound's branches, and the
  entry is covered when those are among the table's next states (its snap and its near ties'
  branches, D5), so the tie-branched certificate holds whatever the rounding.

The float evaluation itself must lie inside every interval (the bound's own check, reported as
outside: 0 for a sound implementation).

    python scripts/certnote/certificate_bound.py TABLE.npz [...] --out bound.jsonl [--batch 4096]

One JSON line a table: the entries; labels sound, undecided and contradicted (decided otherwise:
a bug); snaps sound, covered (the overlap within the table's branches) and uncovered; the float
values outside their intervals; and the largest relative radius of the cells' outputs and the logits.
Networks whose window this pass doesn't implement (window branches) are reported as unsupported.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certificate_check as cc  # scripts/ isn't a package

U = 2.0**-53  # the unit roundoff of float64
ULPS = 4  # exp, tanh, sigmoid, sqrt and reciprocals within this many ulps (the maths library's, assumed)
TINY = 1e-300  # an absolute floor on every radius (underflow)

Float = npt.NDArray[np.float64]


def gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


class I:
    """A midpoint-radius interval array: every value x with |x - m| <= r."""

    __slots__ = ("m", "r")

    def __init__(self, m: Any, r: Any = 0.0) -> None:
        self.m = np.asarray(m, dtype=np.float64)
        self.r = np.broadcast_to(np.asarray(r, dtype=np.float64), self.m.shape).copy()

    @staticmethod
    def rounded(m: Float, r: Float) -> I:
        """
        m computed in floats, r the exact error before m's rounding: both inflated. A value the
        floats can't hold (an overflow, a NaN) is the whole line: m 0, r inf.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            r = r * (1.0 + 4.0 * U) + U * np.abs(m) + TINY
            bad = ~(np.isfinite(m) & np.isfinite(r))
        return I(np.where(bad, 0.0, m), np.where(bad, np.inf, r))

    @staticmethod
    def span(lo: Float, hi: Float) -> I:
        with np.errstate(over="ignore", invalid="ignore"):
            return I.rounded((lo + hi) / 2.0, (hi - lo) / 2.0)

    @property
    def lo(self) -> Float:
        with np.errstate(invalid="ignore"):
            return self.m - self.r - U * np.abs(self.m - self.r)

    @property
    def hi(self) -> Float:
        with np.errstate(invalid="ignore"):
            return self.m + self.r + U * np.abs(self.m + self.r)

    @property
    def bounded(self) -> npt.NDArray[np.bool_]:
        return np.isfinite(self.r)

    def __getitem__(self, key: Any) -> I:
        return I(self.m[key], self.r[key])

    def reshape(self, *shape: int) -> I:
        return I(self.m.reshape(*shape), self.r.reshape(*shape))

    def __add__(self, other: I | Float | float) -> I:
        if isinstance(other, I):
            return I.rounded(self.m + other.m, self.r + other.r)
        return I.rounded(self.m + other, self.r)

    def __sub__(self, other: I | Float | float) -> I:
        if isinstance(other, I):
            return I.rounded(self.m - other.m, self.r + other.r)
        return I.rounded(self.m - other, self.r)

    def __mul__(self, other: I | Float | float) -> I:
        if isinstance(other, I):
            r = np.abs(self.m) * other.r + self.r * np.abs(other.m) + self.r * other.r
            return I.rounded(self.m * other.m, r)
        return I.rounded(self.m * other, self.r * np.abs(other))

    def square(self) -> I:
        lo, hi = self.lo, self.hi
        top = np.maximum(lo * lo, hi * hi)
        bottom = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(lo * lo, hi * hi))
        return I.span(bottom * (1 - 2 * U), top * (1 + 2 * U))

    def sum(self, axis: int) -> I:
        n = self.m.shape[axis]
        return I.rounded(self.m.sum(axis=axis), self.r.sum(axis=axis) + gamma(n) * np.abs(self.m).sum(axis=axis))


def monotone(x: I, f: Any) -> I:
    """An increasing function's image, each end within ULPS ulps."""
    with np.errstate(over="ignore", invalid="ignore"):
        a, b = f(x.lo), f(x.hi)
        e = ULPS * U
        return I.span(a - e * np.abs(a) - TINY, b + e * np.abs(b) + TINY)


def reciprocal(x: I) -> I:
    lo, hi = x.lo, x.hi
    if not (lo > 0).all():
        raise Unsupported("a reciprocal of an interval reaching 0 (a layer norm's variance unbounded)")
    a, b = 1.0 / hi, 1.0 / lo
    e = ULPS * U
    return I.span(a - e * a, b + e * b)


def dot(x: I, W: Float) -> I:
    """x (..., k) times an exact W (k, j)."""
    k = W.shape[0]
    aW = np.abs(W)
    return I.rounded(x.m @ W, x.r @ aW + gamma(k) * (np.abs(x.m) @ aW))


def dot2(x: I, y: I) -> I:
    """x (..., n, k) times y (..., k, j), both intervals."""
    k = x.m.shape[-1]
    ax, ay = np.abs(x.m), np.abs(y.m)
    r = ax @ y.r + x.r @ ay + x.r @ y.r + gamma(k) * (ax @ ay)
    return I.rounded(x.m @ y.m, r)


def exact_dot(X: Float, W: Float) -> I:
    """Two exact float arrays' product, its rounding bounded."""
    return I.rounded(X @ W, gamma(W.shape[0]) * (np.abs(X) @ np.abs(W)))


def softmax(z: I) -> I:
    """
    The softmax over the last axis as p_i = 1 / sum_j exp(z_j - z_i), each difference an interval
    of its own: e_i doesn't appear twice, so no dependency widens it (the naive e / sum(e) does).
    """
    d = I.rounded(z.m[..., np.newaxis, :] - z.m[..., :, np.newaxis], z.r[..., np.newaxis, :] + z.r[..., :, np.newaxis])
    e = monotone(d, np.exp)  # (..., i, j); the diagonal exactly 1 in the real function
    n = z.m.shape[-1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lo = np.maximum(e.lo, 0.0).sum(axis=-1) * (1 - gamma(n))
        hi = e.hi.sum(axis=-1) * (1 + gamma(n))
        a, b = 1.0 / hi, 1.0 / np.maximum(lo, 1.0)  # the sum has its own term 1: at least 1
        eps = ULPS * U
        return I.span(np.clip(a - eps * a, 0.0, 1.0), np.clip(b + eps * b, 0.0, 1.0))


def relu(x: Float) -> Float:
    return np.maximum(x, 0.0)


def sigmoid(x: Float) -> Float:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


class Unsupported(Exception):
    pass


def layer_norm(x: I, layer: Any) -> I:
    d = x.m.shape[-1]
    mu = x.sum(-1) * (1.0 / d)
    c = x - I(mu.m[..., np.newaxis], mu.r[..., np.newaxis])
    var = c.square().sum(-1) * (1.0 / d)
    std = monotone(var + layer.epsilon, np.sqrt)
    inv = reciprocal(std)
    return c * I(inv.m[..., np.newaxis], inv.r[..., np.newaxis]) * layer.gamma + layer.beta


def attention(x: I, layer: Any) -> I:
    if getattr(layer, "_rotary", False) or not layer.causal:
        raise Unsupported("rotary or non-causal attention")
    T = x.m.shape[1]
    h, dk = layer.heads, layer.key_size
    Q = dot(x, layer.Wq.T) + layer.bq
    K = dot(x, layer.Wk.T) + layer.bk
    V = dot(x, layer.Wv.T) + layer.bv
    heads: list[I] = []
    for j in range(h):
        cols = slice(j * dk, (j + 1) * dk)
        q, k, v = Q[:, :, cols], K[:, :, cols], V[:, :, cols]
        rows: list[I] = []
        for t in range(T):
            s = dot2(q[:, t : t + 1, :], I(np.swapaxes(k.m[:, : t + 1], 1, 2), np.swapaxes(k.r[:, : t + 1], 1, 2)))
            p = softmax(s * (1.0 / layer.scale))
            rows.append(dot2(p, v[:, : t + 1]))  # (N, 1, dk)
        heads.append(I(np.concatenate([r.m for r in rows], 1), np.concatenate([r.r for r in rows], 1)))
    H = I(np.concatenate([a.m for a in heads], 2), np.concatenate([a.r for a in heads], 2))
    return dot(H, layer.Wo.T) + layer.bo


class Bound:
    """The numpy copy of a network, its window evaluated in intervals from exact tokens and codes."""

    def __init__(self, network_path: Path) -> None:
        from indrajala_ml.model.persistence.load_network import load_network

        text = network_path.read_text()
        marker = '"implementation": "'
        start = text.index(marker) + len(marker)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / network_path.name
            target.write_text(text[:start] + "numpy" + text[text.index('"', start) :])
            self.net = load_network(str(target))
        names = {type(layer).__name__ for layer in self.net.layers}
        known = {
            "EmbeddingArrayLayer",
            "PositionArrayLayer",
            "StateInputArrayLayer",
            "ForkArrayLayer",
            "AddArrayLayer",
            "LayerNormArrayLayer",
            "AttentionArrayLayer",
            "TokenDenseArrayLayer",
            "AffineArrayLayer",
            "TokenSoftmaxArrayLayer",
        }
        if names - known:
            raise Unsupported(f"layers {sorted(names - known)}")
        rec = self.net.recurrence_layer
        if rec.cross or len(set(self.net._pooling_reads)) != 1:
            raise Unsupported("a crossed recurrence or per-factor reads (window branches)")
        self.E = [np.asarray(f.codes.E, dtype=np.float64) for f in rec.factors]

    def window(self, X: npt.NDArray[np.int64], C: Float) -> tuple[I, I, list[I]]:
        """Tokens X (N, T) and exact states C (N, S): the logits (N, T, classes), the cells (N, S), each factor's squared distances."""
        net = self.net
        T = X.shape[1]
        acts: list[I] = []
        x: I | None = None
        fork: list[I] = []
        logits: I | None = None
        for layer in net.layers:
            kind = type(layer).__name__
            if kind == "EmbeddingArrayLayer":
                x = I(layer.E[X])
            elif kind == "PositionArrayLayer":
                assert x is not None
                x = x + layer.P.reshape(1, T, -1)
            elif kind == "StateInputArrayLayer":
                assert x is not None
                p = exact_dot(C, layer.P.T)
                x = x + I(p.m[:, np.newaxis, :], p.r[:, np.newaxis, :])
            elif kind == "ForkArrayLayer":
                assert x is not None
                fork.append(x)
            elif kind == "AddArrayLayer":
                assert x is not None
                x = x + fork.pop()
            elif kind == "LayerNormArrayLayer":
                assert x is not None
                x = layer_norm(x, layer)
            elif kind == "AttentionArrayLayer":
                assert x is not None
                x = attention(x, layer)
            elif kind in ("TokenDenseArrayLayer", "AffineArrayLayer"):
                assert x is not None
                z = dot(x, layer.W.T) + layer.b
                x = monotone(z, relu) if getattr(layer, "activation", "linear") == "relu" else z
            elif kind == "TokenSoftmaxArrayLayer":
                assert x is not None
                z = dot(x, layer.W.T) + layer.b
                logits, x = z, softmax(z)
            assert x is not None
            acts.append(x)
        assert logits is not None
        tokens: I = ([I(np.zeros(0)), *acts])[int(net._pooling_reads[0])]  # activations[0] is the input
        cells: list[I] = []
        distances: list[I] = []
        rec = net.recurrence_layer
        for g, f in enumerate(rec.factors):
            c = C[:, rec.starts[g] : rec.starts[g] + rec.sizes[g]]
            s = dot(tokens, f.pooling.q[:, np.newaxis])[..., 0]  # (N, T)
            p = softmax(s)
            u = dot2(I(p.m[:, np.newaxis, :], p.r[:, np.newaxis, :]), tokens)[:, 0, :]
            cell = f.cell
            ss = cell.state_size
            A = dot(u, cell.W.T) + cell.b
            rz = A[:, : 2 * ss] + exact_dot(c, cell.U[: 2 * ss].T)
            r = monotone(rz[:, :ss], sigmoid)
            z = monotone(rz[:, ss:], sigmoid)
            rc = r * c
            ht = monotone(A[:, 2 * ss :] + dot(rc, cell.U[2 * ss :].T), np.tanh)
            hcell = z * c + (I(np.ones_like(z.m)) - z) * ht
            cells.append(hcell)
            E = self.E[g]
            diff = I(hcell.m[:, np.newaxis, :] - E[np.newaxis], hcell.r[:, np.newaxis, :])
            distances.append(I.rounded(diff.m, diff.r + U * np.abs(diff.m)).square().sum(-1))
        H = I(np.concatenate([h.m for h in cells], 1), np.concatenate([h.r for h in cells], 1))
        return logits, H, distances


def bound_table(path: Path, batch: int = 4096) -> dict[str, Any]:
    """One table's report (the module's docstring)."""
    begun = time.perf_counter()
    meta, a = cc.load(path)
    report: dict[str, Any] = {"table": str(path), "task": meta["task"], "network": meta["network"]}
    try:
        bound = Bound(Path(meta["network"]))
    except Unsupported as reason:
        return {**report, "unsupported": str(reason)}
    from indrajala_ml.model.persistence.load_network import load_network

    real = load_network(meta["network"])  # the float evaluation, as certified
    adapter = meta["adapter"]
    queries = list(adapter.get("query_tokens", [])) if adapter["kind"] == "query" else []
    slot = adapter.get("slot") if adapter["kind"] == "query" else None
    R, G = a["states"].shape
    W = cc.windows(meta["columns"], meta["tokens"])
    codes = list(meta["codes"])
    branches = set(
        zip(
            (a["branch_rows"] * len(W) + a["branch_windows"]).tolist(),
            cc.keys(a["branch_next"], codes).tolist(),
            strict=True,
        )
    )
    counts = dict.fromkeys(
        ("entries", "labels_sound", "labels_undecided", "labels_contradicted", "snaps_sound", "snaps_covered"), 0
    )
    counts |= {"snaps_uncovered": 0, "outside": 0}
    widest_cell = widest_logit = 0.0
    for i in range(0, R * len(W), batch):
        e = np.arange(i, min(i + batch, R * len(W)))
        r, w = e // len(W), e % len(W)
        states = a["states"][r].astype(np.int64)
        C = np.concatenate([E[states[:, g]] for g, E in enumerate(bound.E)], axis=1)
        X = np.concatenate([W[w], np.tile(np.array(queries, dtype=np.int64), (len(e), 1))], axis=1)
        logits, H, distances = bound.window(X, C)
        # the float evaluation inside its intervals: the cells, and the probabilities' softmax interval
        output, cells, _ = real.window_step(
            real.backend.matrix(X.astype(np.float64).tolist()), real.backend.matrix(C.tolist())
        )
        cells_f = np.array(cells.tolist(), dtype=np.float64)
        probs_f = np.array(output.tolist(), dtype=np.float64).reshape(len(e), X.shape[1], -1)
        probs = softmax(logits)
        counts["outside"] += int(((cells_f < H.lo) | (cells_f > H.hi)).sum())
        counts["outside"] += int(((probs_f < probs.lo) | (probs_f > probs.hi)).sum())
        widest_cell = max(widest_cell, float((H.r / np.maximum(np.abs(H.m), 1e-300)).max()))
        widest_logit = max(widest_logit, float((logits.r / np.maximum(np.abs(logits.m), 1e-300)).max()))
        # the labels: the recorded one's logit above every other's
        L = logits if slot is None else logits[:, slot : slot + 1, :]
        recorded = a["labels"][r, w].astype(np.int64)  # (n, k)
        top_lo = np.take_along_axis(L.lo, recorded[..., np.newaxis], axis=-1)[..., 0]
        others_hi = L.hi.copy()
        np.put_along_axis(others_hi, recorded[..., np.newaxis], -np.inf, axis=-1)
        sound = (top_lo > others_hi.max(axis=-1)).all(axis=1)
        # decided otherwise: another class's lower end above every other class's upper end
        best = np.argmax(L.m, axis=-1)
        best_lo = np.take_along_axis(L.lo, best[..., np.newaxis], axis=-1)[..., 0]
        rest_hi = L.hi.copy()
        np.put_along_axis(rest_hi, best[..., np.newaxis], -np.inf, axis=-1)
        contradicted = ((best != recorded) & (best_lo > rest_hi.max(axis=-1))).any(axis=1)
        counts["entries"] += len(e)
        counts["labels_sound"] += int(sound.sum())
        counts["labels_contradicted"] += int(contradicted.sum())
        counts["labels_undecided"] += int((~sound & ~contradicted).sum())
        # the snaps: per factor, the codes whose distance interval overlaps the nearest's
        options: list[npt.NDArray[np.bool_]] = []
        unique = np.ones(len(e), dtype=bool)
        for d in distances:
            nearest = np.argmin(d.m, axis=1)
            nearest_hi = np.take_along_axis(d.hi, nearest[:, np.newaxis], axis=1)
            overlap = d.lo <= nearest_hi
            unique &= overlap.sum(axis=1) == 1
            options.append(overlap)
        snapped = a["next"][r, w].astype(np.int64)
        for j in range(len(e)):
            if unique[j]:
                state = [int(np.flatnonzero(o[j])[0]) for o in options]
                if state == snapped[j].tolist():
                    counts["snaps_sound"] += 1
                    continue
            covered = True
            for state in (
                np.array(np.meshgrid(*[np.flatnonzero(o[j]) for o in options], indexing="ij")).reshape(G, -1).T
            ):
                key = int(cc.keys(state[np.newaxis, :], codes)[0])
                if key != int(cc.keys(snapped[j : j + 1], codes)[0]) and (int(e[j]), key) not in branches:
                    covered = False
                    break
            counts["snaps_covered" if covered else "snaps_uncovered"] += 1
    return {
        **report,
        **counts,
        "widest_cell": widest_cell,
        "widest_logit": widest_logit,
        "ulps": ULPS,
        "seconds": time.perf_counter() - begun,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("tables", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4096)
    args = parser.parse_args(argv)
    with args.out.open("a") as f:
        for path in args.tables:
            report = bound_table(path, args.batch)
            f.write(json.dumps(report) + "\n")
            if "unsupported" in report:
                print(f"{path.name}: unsupported ({report['unsupported']})", flush=True)
                continue
            print(
                f"{path.name} {report['task']}: {report['entries']} entries; labels sound {report['labels_sound']}, "
                f"undecided {report['labels_undecided']}, contradicted {report['labels_contradicted']}; snaps sound "
                f"{report['snaps_sound']}, covered {report['snaps_covered']}, uncovered {report['snaps_uncovered']}; "
                f"outside {report['outside']}; widest {report['widest_cell']:.1e} / {report['widest_logit']:.1e}; "
                f"{report['seconds']:.1f} s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
