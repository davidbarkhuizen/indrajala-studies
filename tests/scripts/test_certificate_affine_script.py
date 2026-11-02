"""
scripts/certnote/certificate_affine.py (the certificate note's D13): the affine primitives enclose
every point their input forms describe, condensing the symbols keeps that enclosure for forms that
share symbols, and on real networks the float evaluation lies inside every form and no recorded
decision is contradicted, with the symbol cap forcing condensation.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "certnote"))
import certificate_affine as ca  # scripts/ isn't a package


def form(rng: np.random.Generator, shape: tuple[int, ...], symbols: int, r: float = 1e-3) -> ca.AF:
    return ca.AF(rng.normal(size=shape), 1e-2 * rng.normal(size=(*shape, symbols)), np.full(shape, r))


def point(x: ca.AF, e: Any, rng: np.random.Generator) -> Any:
    """A value x describes: its symbols at e, its remainder anywhere in [-r, r]."""
    return x.c + (x.A * e).sum(axis=-1) + x.r * rng.uniform(-1, 1, size=x.c.shape)


def inside(x: ca.AF, values: Any) -> bool:
    return bool(((x.lo <= values) & (values <= x.hi)).all())


def test_the_primitives_enclose_every_point_of_their_inputs():
    rng = np.random.default_rng(0)
    x, y = form(rng, (3, 5), 4), form(rng, (3, 5), 4)
    W = rng.normal(size=(5, 2))
    for _ in range(100):
        e = rng.uniform(-1, 1, size=4)
        px, py = point(x, e, rng), point(y, e, rng)
        assert inside(ca.add(x, y), px + py)
        assert inside(ca.sub(x, x), np.zeros_like(px)) and ca.sub(x, x).rad.max() < 1e-2  # correlation kept
        assert inside(ca.mul(x, y), px * py)
        assert inside(ca.linear(x, W), px @ W)
        assert inside(ca._exp(x), np.exp(px))
        assert inside(ca._tanh(x), np.tanh(px))
        assert inside(ca.relu(x), np.maximum(px, 0))
        assert inside(ca.total(x, -1), px.sum(axis=-1))
        assert inside(ca.softmax_last(x), np.exp(px) / np.exp(px).sum(axis=-1, keepdims=True))
        assert inside(ca.fold(x), px)


def test_condensing_keeps_shared_symbols_aligned():
    rng = np.random.default_rng(1)
    x, y = form(rng, (2, 6), 12), form(rng, (2, 6), 9)  # y's columns past 9 are zeros
    cx, cy = ca.condense([x, y], 3)
    assert cx.S == cy.S == 3
    for _ in range(100):
        e = rng.uniform(-1, 1, size=12)
        px, py = point(x, e, rng), point(ca.AF(y.c, ca._pad(y.A, 12), y.r), e, rng)
        assert inside(cx, px) and inside(cy, py)
        assert inside(ca.sub(cx, cy), px - py)


@pytest.mark.parametrize("cap", [1024, 16])
def test_a_networks_float_evaluation_lies_inside_its_forms(tmp_path: Path, cap: int):
    network = wr.build(wt.ADDITION, 2, 4, 4, seed=1, window=wr.TRANSFORMER)
    path = tmp_path / "network.json"
    network.save(str(path))
    emitted = tmp_path / "table.npz"
    wc.certify_file(path, emit=emitted)
    report = ca.affine_table(emitted, batch=32, cap=cap)
    assert "unsupported" not in report
    assert report["outside"] == 0 and report["labels_contradicted"] == 0
    assert report["entries"] == report["labels_sound"] + report["labels_undecided"]
    assert report["entries"] == report["snaps_sound"] + report["snaps_covered"] + report["snaps_uncovered"]
    assert report["symbols"] <= cap
    if cap == 16:
        assert report["condensed"] > 0


def test_a_family_network_with_two_factors_is_bounded_through_its_queries(tmp_path: Path):
    from indrajala_studies import modular_family as mf

    encoding = mf.Encoding(5, mf.FAMILY)
    path = tmp_path / "family.network.json"
    mf.network(encoding, [(4, 4), (4, 4)], 0).save(str(path))
    emitted = tmp_path / "family.npz"
    mf.certify_file(path, encoding, "modular-division-5", emit=emitted)
    report = ca.affine_table(emitted, cap=64)
    assert report["outside"] == 0 and report["labels_contradicted"] == 0
