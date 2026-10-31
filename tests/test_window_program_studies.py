# pyright: reportPrivateUsage=false
"""
The studies' networks on the Rust window program (moved from
tests/model/networks/test_window_program_network.py, the research-separation workplan, D2): the
study's windows and the division family's network, built by the studies' own builders, are by
bits the per-layer path's, and so are their certificates and training steps; and those builders
build the library presets' networks, which ml's test checks.
"""

import random
from typing import Any

import numpy as np
import pytest
from indrajala_ml.model.layers.array.array_backend import RUST
from indrajala_ml.model.networks.window_recurrent_array_network import per_layer_path
from indrajala_ml.model.specs.window_recurrence import OUTPUT

from indrajala_studies import modular_family as mf
from indrajala_studies import window_recurrent_certificate as cert
from indrajala_studies import window_recurrent_study as wrs
from indrajala_studies import window_tasks as wt
from tests.helpers import (
    FAMILY_TOKENS,
    FAMILY_WIDTH,
    adder,
    assert_program_is_per_layer,
    bits,
    family,
)


def _snapshot_bits(network: Any) -> list[list[Any]]:
    return [[bits(a) for a in entry] for entry in network.snapshot()]


@pytest.mark.parametrize("window", list(wrs.WINDOWS))
def test_the_studys_builder_builds_the_presets_network(window: str):
    assert _snapshot_bits(wrs.build(wt.ADDITION, 1, 16, 16, seed=3, window=window)) == _snapshot_bits(
        adder(window, 1, 16, 16, seed=3)
    )


def test_the_familys_builder_builds_the_presets_network():
    encoding = mf.Encoding(7, ("sum", "difference"))
    assert (encoding.tokens, encoding.width, encoding.classes) == (FAMILY_TOKENS, FAMILY_WIDTH, 8)
    network = mf.network(encoding, [(16, 32), (16, 32)], seed=2, reads=OUTPUT)
    assert _snapshot_bits(network) == _snapshot_bits(family(seed=2))


@pytest.mark.parametrize("window", list(wrs.WINDOWS))
def test_the_studys_windows_are_the_per_layer_path(window: str):
    # E2b''s cell (the linear window), k 1, a state of 16 features and 16 codes, and the others
    network = wrs.build(wt.ADDITION, 1, 16, 16, seed=3, window=window)
    rng = np.random.default_rng(5)
    assert_program_is_per_layer(network, RUST.matrix(rng.integers(0, 27, (6, 40)).astype(float).tolist()))


def test_the_division_family_is_the_per_layer_path():
    # the fork probe's cell: p 7, two factors of 32 codes, the window's output read
    encoding = mf.Encoding(7, ("sum", "difference"))
    network = mf.network(encoding, [(16, 32), (16, 32)], seed=2, reads=OUTPUT)
    rng = random.Random(6)
    rows = [encoding.example(mf.columns(20, rng))[0] for _ in range(4)]
    assert_program_is_per_layer(network, RUST.matrix([list(row) for row in rows]))


def test_a_certificate_is_the_per_layer_paths():
    network = wrs.build(wt.ADDITION, 1, 4, 8, seed=4, window="linear")
    program = cert.certify(cert.NetworkMachine(network)).to_json()
    with per_layer_path(network):
        per_layer = cert.certify(cert.NetworkMachine(network)).to_json()
    program.pop("seconds")
    per_layer.pop("seconds")
    assert program == per_layer


def test_the_studys_and_the_familys_training_steps_are_the_per_layer_paths():
    encoding = mf.Encoding(7, ("sum", "difference"))
    rng = random.Random(9)
    family = [mf.network(encoding, [(16, 32), (16, 32)], seed=2, reads=OUTPUT) for _ in range(2)]
    examples = [encoding.example(mf.columns(15, rng)) for _ in range(6)]
    linear = [wrs.build(wt.ADDITION, 1, 16, 16, seed=3, window="linear") for _ in range(2)]
    digits = np.random.default_rng(10)
    sums = [
        (tuple(float(v) for v in digits.integers(0, 27, 19)), tuple(int(v) for v in digits.integers(0, 3, 19)))
        for _ in range(8)
    ]
    for (program, per_layer), rows in ((family, examples), (linear, sums)):
        program.learn_batch(0.003, rows)
        with per_layer_path(per_layer):
            per_layer.learn_batch(0.003, rows)
        assert [[bits(a) for a in entry] for entry in program.snapshot()] == [
            [bits(a) for a in entry] for entry in per_layer.snapshot()
        ]
