"""
The certificate note's affine bound (indrajala-research docs/workplans/certificate-note-workplan.md,
stage 3, D7; the owner's D13, 2026-11-02: build affine arithmetic for the deep division networks
before release).

certificate_bound.py's interval pass loses every correlation: each layer's radius grows by |W|'s
row sums, about a thousandfold an attention block, so it decides little on deep windows. Here
each value is an affine form (Stolfi & de Figueiredo's affine arithmetic): a centre, signed
coefficients on noise symbols shared across the entry's values, and an independent remainder r,

    x = c + sum_s A_s e_s + [-r, r],  every e_s in [-1, 1].

It encloses every evaluation of the window whose elementary operations each err within the model
certificate_bound.py states (u = 2^-53 a result, gamma_n a sum of n terms in any order, ULPS ulps
for exp, tanh, the logistic function, sqrt and reciprocals), the exact real-valued function among
them. The rules:

- a linear map carries the coefficients through exactly (W A), so errors that cancel downstream
  cancel here; the remainder goes through |W|;
- a product of two forms keeps both first-order terms and bounds the second-order one by the radii's
  product; a sum's and a product's own rounding go into the remainder;
- a nonlinear f is linearised at the centre, f(c) + f'(c)(x - c), with the remainder bounded by half
  of |f''|'s maximum over the input's range times its radius squared (ReLU across 0 by the usual
  chord);
- after every layer each value's remainder becomes a fresh symbol of its own, so the next layer
  carries it correlated (the folding that keeps the forms first-order exact);
- every rounding of this module's own arithmetic is charged to the remainder too;
- the symbols are capped (--symbols K): after a fold, each entry keeps the K symbols with the most
  coefficient mass over every live form (the residual stream and the forks), and every dropped
  coefficient's magnitude joins its value's remainder (still an enclosure, only looser). Without the
  cap a fold adds T d symbols a layer, and a deep window's forms outgrow memory.

The decisions are read on differences, which are linear in the final features, so their forms stay
tight: a label is sound when its logit's lead over every other class's has a positive lower end past
the float softmax's own rounding; a snap is sound when every other code's squared distance less the
snapped code's, 2 h.(E_* - E_c) + |E_c|^2 - |E_*|^2, has a lower end above the float distances'
rounding. Otherwise the codes whose difference may be non-positive are the bound's branches, and the
entry is covered when they are among the table's (D5).

    python scripts/certnote/certificate_affine.py TABLE.npz [...] --out affine.jsonl \
        [--batch 16] [--symbols 1024] [--memory-gb 2.5]

A form's coefficients take batch x T x d x K x 8 bytes; --memory-gb caps the address space, so an
over-large run ends in a MemoryError rather than the machine's OOM killer.

One JSON line a table, certificate_bound.py's counts (outside: the float cells outside their forms'
ranges), and the largest symbol count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certificate_bound as cb  # the loader and the error model
import certificate_check as cc

U = cb.U
ULPS = cb.ULPS
Float = npt.NDArray[np.float64]
Fn = Callable[[Float], Float]
Bound2 = Callable[[Float, Float], Float]  # |f''| bounded over [lo, hi]


def gamma(n: int) -> float:
    return cb.gamma(n)


class AF:
    """Affine forms over an entry batch: c (N, *shape), A (N, *shape, S), r (N, *shape)."""

    __slots__ = ("A", "c", "r")

    def __init__(self, c: Float, A: Float | None = None, r: Float | None = None) -> None:
        self.c = np.asarray(c, dtype=np.float64)
        self.A = np.zeros((*self.c.shape, 0)) if A is None else A  # pyright: ignore[reportConstantRedefinition]
        self.r = np.zeros(self.c.shape) if r is None else r

    @property
    def S(self) -> int:
        return int(self.A.shape[-1])

    def coefficients(self) -> Float:
        return np.abs(self.A).sum(axis=-1)

    @property
    def rad(self) -> Float:
        return self.coefficients() + self.r

    @property
    def lo(self) -> Float:
        return self.c - self.rad * (1 + 4 * U)

    @property
    def hi(self) -> Float:
        return self.c + self.rad * (1 + 4 * U)

    def own(self) -> Float:
        """The rounding of this module's arithmetic for the form, charged to the remainder."""
        return 4 * U * (np.abs(self.c) + self.coefficients())


def _pad(A: Float, S: int) -> Float:
    if A.shape[-1] == S:
        return A
    return np.concatenate([A, np.zeros((*A.shape[:-1], S - A.shape[-1]))], axis=-1)


def _aligned(x: AF, y: AF) -> tuple[Float, Float]:
    S = max(x.S, y.S)
    return _pad(x.A, S), _pad(y.A, S)


def add(x: AF, y: AF) -> AF:
    Ax, Ay = _aligned(x, y)
    out = AF(x.c + y.c, Ax + Ay, x.r + y.r)
    out.r = out.r + U * (np.abs(x.c) + x.rad + np.abs(y.c) + y.rad) + out.own()  # the network's add, then ours
    return out


def neg(x: AF) -> AF:
    return AF(-x.c, -x.A, x.r.copy())


def sub(x: AF, y: AF) -> AF:
    return add(x, neg(y))


def scale(x: AF, k: Any) -> AF:
    """x times an exact constant (broadcast over the entry's shape)."""
    k = np.asarray(k, dtype=np.float64)
    out = AF(x.c * k, x.A * k[..., np.newaxis], x.r * np.abs(k))
    out.r = out.r + U * np.abs(k) * (np.abs(x.c) + x.rad) + out.own()
    return out


def shift(x: AF, b: Any) -> AF:
    """x plus an exact constant."""
    out = AF(x.c + b, x.A.copy(), x.r.copy())
    out.r = out.r + U * (np.abs(x.c) + x.rad + np.abs(b)) + out.own()
    return out


def linear(x: AF, W: Float, b: Float | None = None) -> AF:
    """x (N, ..., d) times an exact W (d, k), plus an exact b: the network's dot in any order."""
    d = W.shape[0]
    aW = np.abs(W)
    c = x.c @ W
    A = np.einsum("...ds,dk->...ks", x.A, W)
    r = x.r @ aW + gamma(d) * ((np.abs(x.c) + x.rad) @ aW)  # the remainder through |W|; the network's dot
    r = r + gamma(d) * ((np.abs(x.c) + x.coefficients()) @ aW)  # ours
    out = AF(c, A, r)
    return out if b is None else shift(out, b)


def exact_linear(X: Float, W: Float) -> AF:
    """Two exact float arrays' product: the network's rounding of it as the remainder."""
    return AF(X @ W, None, 2 * gamma(W.shape[0]) * (np.abs(X) @ np.abs(W)))


def mul(x: AF, y: AF) -> AF:
    Ax, Ay = _aligned(x, y)
    cx, cy = x.c[..., np.newaxis], y.c[..., np.newaxis]
    r = np.abs(x.c) * y.r + np.abs(y.c) * x.r + x.rad * y.rad
    r = r + U * (np.abs(x.c) + x.rad) * (np.abs(y.c) + y.rad)  # the network's product
    out = AF(x.c * y.c, cx * Ay + cy * Ax, r)
    out.r = out.r + out.own()
    return out


def total(x: AF, axis: int) -> AF:
    """The sum over an axis of the entry's shape (not the symbols'): the network's in any order."""
    n = x.c.shape[axis]
    a = axis if axis >= 0 else axis - 0  # an axis of c; A's is the same index
    out = AF(x.c.sum(axis=a), x.A.sum(axis=a if a >= 0 else a - 1), x.r.sum(axis=a))
    out.r = (
        out.r + gamma(n) * (np.abs(x.c) + x.rad).sum(axis=a) + gamma(n) * (np.abs(x.c) + x.coefficients()).sum(axis=a)
    )
    return out


def unary(x: AF, f: Fn, df: Fn, f2: Bound2) -> AF:
    """f linearised at the centre; f2(lo, hi) bounds |f''| over the range."""
    lo, hi = x.lo, x.hi
    f0, a = f(x.c), df(x.c)
    rad = x.rad
    delta = 0.5 * f2(lo, hi) * rad * rad
    library = ULPS * U * (np.abs(f0) + np.abs(a) * rad + delta)  # f within ULPS ulps of its value
    out = AF(f0, x.A * a[..., np.newaxis], np.abs(a) * x.r + delta + library)
    out.r = out.r + out.own()
    return out


def relu(x: AF) -> AF:
    lo, hi = x.lo, x.hi
    on, off = lo >= 0, hi <= 0
    cross = ~on & ~off
    span = np.where(cross, hi - lo, 1.0)
    alpha = np.where(on, 1.0, np.where(off, 0.0, hi / span))
    beta = np.where(cross, -alpha * lo / 2, 0.0)
    out = AF(alpha * x.c + beta, x.A * alpha[..., np.newaxis], alpha * x.r + beta)
    out.r = out.r + out.own()
    return out


def _exp(x: AF) -> AF:
    def f(v: Float) -> Float:
        return np.exp(v)

    def f2(lo: Float, hi: Float) -> Float:
        return np.exp(hi)

    with np.errstate(over="ignore"):
        return unary(x, f, f, f2)


def _tanh(x: AF) -> AF:
    def f(v: Float) -> Float:
        return np.tanh(v)

    def df(v: Float) -> Float:
        return 1 - np.tanh(v) ** 2

    def f2(lo: Float, hi: Float) -> Float:
        return np.full_like(lo, 0.7699)  # max |tanh''|

    return unary(x, f, df, f2)


def _sigmoid(x: AF) -> AF:
    def f(v: Float) -> Float:
        return cb.sigmoid(v)

    def df(v: Float) -> Float:
        return f(v) * (1 - f(v))

    def f2(lo: Float, hi: Float) -> Float:
        return np.full_like(lo, 0.0963)  # max |sigmoid''|

    return unary(x, f, df, f2)


def _sqrt(x: AF) -> AF:
    if not (x.lo > 0).all():
        raise cb.Unsupported("a square root of a range reaching 0")

    def f(v: Float) -> Float:
        return np.sqrt(v)

    def df(v: Float) -> Float:
        return 0.5 / np.sqrt(v)

    def f2(lo: Float, hi: Float) -> Float:
        return 0.25 * lo**-1.5

    return unary(x, f, df, f2)


def _reciprocal(x: AF) -> AF:
    if not (x.lo > 0).all():
        raise cb.Unsupported("a reciprocal of a range reaching 0")

    def f(v: Float) -> Float:
        return 1 / v

    def df(v: Float) -> Float:
        return -1 / (v * v)

    def f2(lo: Float, hi: Float) -> Float:
        return 2 / lo**3

    return unary(x, f, df, f2)


def fold(x: AF) -> AF:
    """Each value's remainder as a fresh symbol of its own."""
    N = x.c.shape[0]
    E = int(np.prod(x.c.shape[1:]))
    A = np.zeros((N, E, x.S + E))
    A[:, :, : x.S] = x.A.reshape(N, E, x.S)
    idx = np.arange(E)
    A[:, idx, x.S + idx] = x.r.reshape(N, E)
    return AF(x.c, A.reshape(*x.c.shape, x.S + E), np.zeros(x.c.shape))


def condense(forms: list[AF], cap: int) -> list[AF]:
    """The live forms on each entry's cap heaviest symbols; every dropped coefficient's |.| to its remainder.

    One selection for all the forms, so the symbols they share stay aligned (a form's columns past its
    own S are zeros)."""
    S = max(f.S for f in forms)
    if S <= cap:
        return forms
    N = forms[0].c.shape[0]
    mass = np.zeros((N, S))
    for f in forms:
        mass[:, : f.S] += np.abs(f.A).reshape(N, -1, f.S).sum(axis=1)
    keep = np.sort(np.argpartition(-mass, cap - 1, axis=1)[:, :cap], axis=1)  # (N, cap)
    dropped = np.ones((N, S), dtype=bool)
    np.put_along_axis(dropped, keep, False, axis=1)
    out: list[AF] = []
    for f in forms:
        flat = _pad(f.A, S).reshape(N, -1, S)
        A = np.take_along_axis(flat, keep[:, np.newaxis, :], axis=2).reshape(*f.c.shape, cap)
        lost = (np.abs(flat) * dropped[:, np.newaxis, :]).sum(axis=2).reshape(f.c.shape) * (1 + 2 * gamma(S))
        g = AF(f.c, A, f.r + lost)
        g.r = g.r + 4 * U * np.abs(g.r)  # ours
        out.append(g)
    return out


def take(x: AF, index: Any, axis: int) -> AF:
    """Values along an axis of the entry's shape (axis >= 1)."""
    return AF(np.take(x.c, index, axis=axis), np.take(x.A, index, axis=axis), np.take(x.r, index, axis=axis))


def expand(x: AF, axis: int) -> AF:
    """A new axis of length 1 in the entry's shape (a negative axis counts from the shape's end, not the symbols')."""
    a = axis if axis >= 0 else x.c.ndim + 1 + axis
    return AF(np.expand_dims(x.c, a), np.expand_dims(x.A, a), np.expand_dims(x.r, a))


def broadcast(x: AF, shape: tuple[int, ...]) -> AF:
    return AF(
        np.broadcast_to(x.c, shape).copy(),
        np.broadcast_to(x.A, (*shape, x.S)).copy(),
        np.broadcast_to(x.r, shape).copy(),
    )


def softmax_last(z: AF) -> AF:
    """p_i = 1 / (1 + sum_(j != i) exp(z_j - z_i)) over the last axis (a few values: tokens)."""
    n = z.c.shape[-1]
    parts: list[AF] = []
    for i in range(n):
        zi = expand(take(z, i, -1 + z.c.ndim), -1)  # (..., 1)
        d = sub(z, broadcast(zi, z.c.shape))  # (..., n): z_j - z_i
        e = _exp(d)
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        s = shift(total(take(e, np.flatnonzero(keep), z.c.ndim - 1), -1), 1.0) if n > 1 else AF(np.ones(z.c.shape[:-1]))
        parts.append(_reciprocal(s))
    return stack(parts, -1)


def stack(parts: list[AF], axis: int) -> AF:
    S = max(p.S for p in parts)
    ax = axis if axis >= 0 else parts[0].c.ndim + 1 + axis
    return AF(
        np.stack([p.c for p in parts], axis=ax),
        np.stack([_pad(p.A, S) for p in parts], axis=ax),
        np.stack([p.r for p in parts], axis=ax),
    )


def concat(parts: list[AF], axis: int) -> AF:
    S = max(p.S for p in parts)
    return AF(
        np.concatenate([p.c for p in parts], axis=axis),
        np.concatenate([_pad(p.A, S) for p in parts], axis=axis),
        np.concatenate([p.r for p in parts], axis=axis),
    )


def layer_norm(x: AF, layer: Any) -> AF:
    d = x.c.shape[-1]
    mu = scale(total(x, -1), 1.0 / d)
    centred = sub(x, broadcast(expand(mu, -1), x.c.shape))
    var = scale(total(mul(centred, centred), -1), 1.0 / d)
    inv = _reciprocal(_sqrt(shift(var, layer.epsilon)))
    return shift(scale(mul(centred, broadcast(expand(inv, -1), x.c.shape)), layer.gamma), layer.beta)


def attention(x: AF, layer: Any) -> AF:
    if getattr(layer, "_rotary", False) or not layer.causal:
        raise cb.Unsupported("rotary or non-causal attention")
    T = x.c.shape[1]
    h, dk = int(layer.heads), int(layer.key_size)
    Q, K, V = (linear(x, W.T, b) for W, b in ((layer.Wq, layer.bq), (layer.Wk, layer.bk), (layer.Wv, layer.bv)))
    heads: list[AF] = []
    for j in range(h):
        cols = np.arange(j * dk, (j + 1) * dk)
        q, k, v = take(Q, cols, 2), take(K, cols, 2), take(V, cols, 2)
        rows: list[AF] = []
        for t in range(T):
            keys = take(k, np.arange(t + 1), 1)  # (N, t+1, dk)
            qt = broadcast(expand(take(q, t, 1), 1), keys.c.shape)
            scores = scale(total(mul(qt, keys), -1), 1.0 / layer.scale)  # (N, t+1)
            p = softmax_last(scores)
            vals = take(v, np.arange(t + 1), 1)
            pv = mul(broadcast(expand(p, -1), vals.c.shape), vals)
            rows.append(total(pv, 1))  # (N, dk)
        heads.append(stack(rows, 1))  # (N, T, dk)
    return linear(concat(heads, 2), layer.Wo.T, layer.bo)


class Affine:
    """The window in affine forms, from exact tokens and codes (cb.Bound's numpy copy of the network)."""

    def __init__(self, network_path: Path, cap: int = 1024) -> None:
        self.bound = cb.Bound(network_path)
        self.net = self.bound.net
        self.E = self.bound.E
        self.cap = cap
        self.symbols = 0
        self.condensed = 0

    def window(self, X: npt.NDArray[np.int64], C: Float) -> tuple[AF, list[AF]]:
        net = self.net
        T = X.shape[1]
        read = int(net._pooling_reads[0])  # the pooled tokens: [input, *layers][read]
        if read < 1:
            raise cb.Unsupported("pooling on the input")
        tokens: AF | None = None
        x: AF | None = None
        fork: list[AF] = []
        logits: AF | None = None
        for i, layer in enumerate(net.layers, start=1):
            kind = type(layer).__name__
            if kind == "EmbeddingArrayLayer":
                x = AF(layer.E[X])
            elif kind == "PositionArrayLayer":
                assert x is not None
                x = shift(x, layer.P.reshape(1, T, -1))
            elif kind == "StateInputArrayLayer":
                assert x is not None
                p = exact_linear(C, layer.P.T)
                x = add(x, broadcast(expand(p, 1), x.c.shape))
            elif kind == "ForkArrayLayer":
                assert x is not None
                fork.append(x)
            elif kind == "AddArrayLayer":
                assert x is not None
                x = add(x, fork.pop())
            elif kind == "LayerNormArrayLayer":
                assert x is not None
                x = layer_norm(x, layer)
            elif kind == "AttentionArrayLayer":
                assert x is not None
                x = attention(x, layer)
            elif kind in ("TokenDenseArrayLayer", "AffineArrayLayer"):
                assert x is not None
                z = linear(x, layer.W.T, layer.b)
                x = relu(z) if getattr(layer, "activation", "linear") == "relu" else z
            elif kind == "TokenSoftmaxArrayLayer":
                assert x is not None
                logits = linear(x, layer.W.T, layer.b)
                x = logits  # the probabilities aren't read downstream: the decision is on the logits
            else:
                raise cb.Unsupported(kind)
            assert x is not None
            if kind not in ("ForkArrayLayer", "TokenSoftmaxArrayLayer"):
                x = fold(x)
                if x.S > self.cap:
                    self.condensed += 1
                    x, *fork = condense([x, *fork], self.cap)
            self.symbols = max(self.symbols, x.S)
            if i == read:
                tokens = x
        assert logits is not None and tokens is not None
        rec = net.recurrence_layer
        cells: list[AF] = []
        for g, f in enumerate(rec.factors):
            c = C[:, rec.starts[g] : rec.starts[g] + rec.sizes[g]]
            s = take(linear(tokens, f.pooling.q[:, np.newaxis]), 0, 2)  # (N, T)
            p = softmax_last(s)
            u = total(mul(broadcast(expand(p, -1), tokens.c.shape), tokens), 1)  # (N, d)
            cell = f.cell
            ss = cell.state_size
            A = linear(u, cell.W.T, cell.b)
            gates = add(take(A, np.arange(2 * ss), 1), exact_linear(c, cell.U[: 2 * ss].T))
            r = _sigmoid(take(gates, np.arange(ss), 1))
            z = _sigmoid(take(gates, np.arange(ss, 2 * ss), 1))
            rc = scale(r, c)
            ht = _tanh(add(take(A, np.arange(2 * ss, 3 * ss), 1), linear(rc, cell.U[2 * ss :].T)))
            one_minus_z = shift(neg(z), 1.0)
            cells.append(add(scale(z, c), mul(one_minus_z, ht)))
        self.symbols = max(self.symbols, max(c.S for c in cells))
        return logits, cells


def label_margin(z: AF) -> Float:
    """The float softmax's own rounding as a logit lead: ratios of exp within ULPS + 2 ulps each, the shifts' rounding."""
    size = np.abs(z.c).max(axis=-1, keepdims=True) + z.rad.max(axis=-1, keepdims=True)
    return (4 * (ULPS + 2) * U * 1.01) + 8 * U * size


def affine_table(path: Path, batch: int = 16, cap: int = 1024) -> dict[str, Any]:
    begun = time.perf_counter()
    meta, a = cc.load(path)
    report: dict[str, Any] = {"table": str(path), "task": meta["task"], "network": meta["network"]}
    try:
        model = Affine(Path(meta["network"]), cap)
    except cb.Unsupported as reason:
        return {**report, "unsupported": str(reason)}
    from indrajala_ml.model.persistence.load_network import load_network

    real = load_network(meta["network"])
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
        (
            "entries",
            "labels_sound",
            "labels_undecided",
            "labels_contradicted",
            "snaps_sound",
            "snaps_covered",
            "snaps_uncovered",
            "outside",
        ),
        0,
    )
    for i in range(0, R * len(W), batch):
        e = np.arange(i, min(i + batch, R * len(W)))
        r, w = e // len(W), e % len(W)
        states = a["states"][r].astype(np.int64)
        C = np.concatenate([E[states[:, g]] for g, E in enumerate(model.E)], axis=1)
        X = np.concatenate([W[w], np.tile(np.array(queries, dtype=np.int64), (len(e), 1))], axis=1)
        logits, cells = model.window(X, C)
        _, cells_f, _ = real.window_step(
            real.backend.matrix(X.astype(np.float64).tolist()), real.backend.matrix(C.tolist())
        )
        H = concat(cells, 1)
        Hf = np.array(cells_f.tolist(), dtype=np.float64)
        counts["outside"] += int(((Hf < H.lo) | (Hf > H.hi)).sum())
        counts["entries"] += len(e)
        # the labels: each column's recorded label's lead over every other class
        L = logits if slot is None else take(logits, np.array([slot]), 1)
        recorded = a["labels"][r, w].astype(np.int64)  # (n, k)
        top = AF(
            np.take_along_axis(L.c, recorded[..., np.newaxis], axis=-1),
            np.take_along_axis(L.A, recorded[..., np.newaxis, np.newaxis], axis=-2),
            np.take_along_axis(L.r, recorded[..., np.newaxis], axis=-1),
        )
        lead = sub(broadcast(top, L.c.shape), L)  # (n, k, classes): z_top - z_j
        margin = label_margin(L)
        mine = np.zeros(L.c.shape, dtype=bool)
        np.put_along_axis(mine, recorded[..., np.newaxis], True, axis=-1)
        sound = ((lead.lo > margin) | mine).all(axis=-1).all(axis=-1)
        contradicted = ((-lead.hi > margin) & ~mine).any(axis=-1).any(axis=-1)
        counts["labels_sound"] += int(sound.sum())
        counts["labels_contradicted"] += int(contradicted.sum())
        counts["labels_undecided"] += int((~sound & ~contradicted).sum())
        # the snaps: per factor, each other code's squared distance less the snapped one's
        snapped = a["next"][r, w].astype(np.int64)
        possible: list[npt.NDArray[np.bool_]] = []
        for g, (h, E) in enumerate(zip(cells, model.E, strict=True)):
            star = snapped[:, g]
            hE = linear(h, E.T)  # (n, m)
            hstar = AF(
                np.take_along_axis(hE.c, star[:, np.newaxis], axis=1),
                np.take_along_axis(hE.A, star[:, np.newaxis, np.newaxis], axis=1),
                np.take_along_axis(hE.r, star[:, np.newaxis], axis=1),
            )
            norms = (E * E).sum(axis=1)
            diff = shift(
                scale(sub(broadcast(hstar, hE.c.shape), hE), 2.0), norms[np.newaxis, :] - norms[star][:, np.newaxis]
            )
            # the float distances' own rounding: gamma_(s+2) of each squared distance, bounded above
            centre_d = ((h.c[:, np.newaxis, :] - E[np.newaxis]) ** 2).sum(axis=2)
            upper = (np.sqrt(centre_d) + np.sqrt((h.rad**2).sum(axis=1))[:, np.newaxis]) ** 2
            slack = gamma(E.shape[1] + 2) * (upper + np.take_along_axis(upper, star[:, np.newaxis], axis=1)) * 1.01
            other = np.ones(diff.c.shape, dtype=bool)
            np.put_along_axis(other, star[:, np.newaxis], False, axis=1)
            possible.append(~(diff.lo > slack) & other | ~other)
        for j in range(len(e)):
            options = [np.flatnonzero(p[j]) for p in possible]
            if all(len(o) == 1 for o in options):
                counts["snaps_sound"] += 1
                continue
            covered = True
            star_key = int(cc.keys(snapped[j : j + 1], codes)[0])
            for state in np.array(np.meshgrid(*options, indexing="ij")).reshape(G, -1).T:
                key = int(cc.keys(state[np.newaxis, :], codes)[0])
                if key != star_key and (int(e[j]), key) not in branches:
                    covered = False
                    break
            counts["snaps_covered" if covered else "snaps_uncovered"] += 1
    return {
        **report,
        **counts,
        "symbols": model.symbols,
        "cap": cap,
        "condensed": model.condensed,
        "batch": batch,
        "ulps": ULPS,
        "seconds": time.perf_counter() - begun,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("tables", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--symbols", type=int, default=1024, help="the symbol cap K an entry")
    parser.add_argument("--memory-gb", type=float, default=2.5, help="the address-space cap (0: none)")
    args = parser.parse_args(argv)
    if args.memory_gb > 0:
        import resource

        limit = int(args.memory_gb * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    with args.out.open("a") as f:
        for path in args.tables:
            report = affine_table(path, args.batch, args.symbols)
            f.write(json.dumps(report) + "\n")
            f.flush()
            if "unsupported" in report:
                print(f"{path.name}: unsupported ({report['unsupported']})", flush=True)
                continue
            print(
                f"{path.name} {report['task']}: {report['entries']} entries; labels sound {report['labels_sound']}, "
                f"undecided {report['labels_undecided']}, contradicted {report['labels_contradicted']}; snaps sound "
                f"{report['snaps_sound']}, covered {report['snaps_covered']}, uncovered {report['snaps_uncovered']}; "
                f"outside {report['outside']}; {report['symbols']} symbols ({report['condensed']} condensed); "
                f"{report['seconds']:.1f} s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
