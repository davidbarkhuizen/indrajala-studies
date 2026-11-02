"""
scripts/certnote/certificate_bound.py (the certificate note's D7): the interval primitives enclose
their exact values, and on real networks the float evaluation lies inside every interval and no
recorded decision is contradicted.
"""

import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "certnote"))
import certificate_bound as cb  # scripts/ isn't a package


def inside(x: cb.I, values: Any) -> bool:
    return bool(((x.lo <= values) & (values <= x.hi)).all())


def test_a_dot_product_encloses_its_exact_value():
    rng = np.random.default_rng(0)
    x, W = rng.normal(size=(5, 40)), rng.normal(size=(40, 3))
    got = cb.dot(cb.I(x), W)
    for i in range(5):
        for j in range(3):
            exact = sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(x[i], W[:, j], strict=True))
            assert got.lo[i, j] <= float(exact) <= got.hi[i, j]


def test_interval_inputs_enclose_every_point_inside_them():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(4, 6))
    x = cb.I(m, 1e-3)
    for _ in range(50):
        point = m + rng.uniform(-1e-3, 1e-3, size=m.shape)
        assert inside(cb.softmax(x), np.exp(point) / np.exp(point).sum(axis=1, keepdims=True))
        assert inside(x.square(), point * point)
        assert inside(cb.monotone(x, np.tanh), np.tanh(point))
        assert inside(x.sum(-1), point.sum(axis=1))


@pytest.mark.parametrize("window", [wr.TRANSFORMER, "linear"])
def test_a_networks_float_evaluation_lies_inside_its_intervals(tmp_path: Path, window: str):
    network = wr.build(wt.ADDITION, 2, 4, 4, seed=1, window=window)
    path = tmp_path / "network.json"
    network.save(str(path))
    emitted = tmp_path / "table.npz"
    wc.certify_file(path, emit=emitted)
    report = cb.bound_table(emitted, batch=101)
    assert "unsupported" not in report
    assert report["outside"] == 0 and report["labels_contradicted"] == 0
    assert report["entries"] == report["labels_sound"] + report["labels_undecided"]
    assert report["entries"] == report["snaps_sound"] + report["snaps_covered"] + report["snaps_uncovered"]


def test_a_shallow_networks_decisions_are_all_sound(tmp_path: Path):
    # E6's linear window: no attention, the intervals stay narrow, every label and snap decided
    network = wr.build(wt.ADDITION, 1, 4, 4, seed=2, window="linear")
    path = tmp_path / "linear.network.json"
    network.save(str(path))
    emitted = tmp_path / "linear.npz"
    wc.certify_file(path, snap_ratio=1.0, emit=emitted)
    report = cb.bound_table(emitted)
    assert report["widest_cell"] < 1e-6
    assert report["labels_sound"] + report["labels_undecided"] == report["entries"]
    assert report["snaps_sound"] >= report["entries"] - report["snaps_uncovered"]


def test_a_family_network_with_two_factors_is_bounded_through_its_queries(tmp_path: Path):
    from indrajala_studies import modular_family as mf

    encoding = mf.Encoding(5, mf.FAMILY)
    path = tmp_path / "family.network.json"
    mf.network(encoding, [(4, 4), (4, 4)], 0).save(str(path))
    emitted = tmp_path / "family.npz"
    mf.certify_file(path, encoding, "modular-division-5", emit=emitted)
    report = cb.bound_table(emitted)
    assert report["outside"] == 0 and report["labels_contradicted"] == 0
