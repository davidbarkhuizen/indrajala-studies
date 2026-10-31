"""
What the study tests share: ml's tests/studies/helpers.py (the carry machine as a test double, the
smallest study settings a real network's run takes), moved with them; and, copied from ml's tests
at d126315 (the studies-repo workplan's D5), `bits` (tests/helpers.py) and the window program's
helpers and preset networks (tests/model/networks/test_window_program_network.py).
"""

from typing import Any, cast

import numpy as np
from indrajala_ml.model.layers.array.array_backend import RUST
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork, per_layer_path
from indrajala_ml.model.specs import window_presets as wp
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import OUTPUT, AttentionPooling, Codes, Factor, WindowRecurrence

from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt

ADDITION = wt.ADDITION

# a real network's run in seconds: one epoch of 64 examples a step, two steps, the tests' smallest
# reads; a test changes what it needs with dataclasses.replace
TINY = wr.Settings(
    epoch_examples=64,
    max_epochs=1,
    patience=1,
    screen=20,
    steps=2,
    test_lengths=(3,),
    sampled=50,
    relation_sampled=10,
    per_stratum=5,
    codes_cases=20,
    seeds=1,
)


class CarryTable:
    """The carry machine at k 1 as a WindowMachine, with flip (state, window) handing on the next carry instead."""

    def __init__(self, flip: tuple[int, int] | None = None) -> None:
        self.flip = flip

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return [3]

    def start(self) -> tuple[tuple[int, ...], list[np.ndarray]]:
        return (0,), [np.array([0.0, 1.0, 1.0])]

    def step(self, state: tuple[int, ...], windows: np.ndarray) -> wc.Steps:
        (carry,) = state
        labels, after = ADDITION.machine.windows(windows)
        digits, nexts = labels[carry], after[carry].copy()
        if self.flip is not None and self.flip[0] == carry:
            nexts[windows[:, 0] == self.flip[1]] = (nexts[windows[:, 0] == self.flip[1]] + 1) % 3
        distances = np.ones((len(windows), 3))
        distances[np.arange(len(windows)), nexts] = 0.0
        return wc.Steps(digits, np.full(digits.shape, 5.0), nexts[:, np.newaxis], [distances])


def bits(value: Any) -> Any:
    """
    Every float in value as float.hex, through either backend's arrays, lists, tuples and dicts:
    two values' bits are equal exactly when their floats are, bit for bit (0.0 and -0.0 differ,
    NaNs match), and pytest shows a readable diff. Tuples become lists and dict keys strings, as in
    a JSON file of bits (tests/saved_model_fixtures.py's).
    """
    to_list = getattr(value, "tolist", None)
    if to_list is not None:
        return bits(to_list())
    if isinstance(value, list | tuple):
        return [bits(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
    if isinstance(value, dict):
        return {str(key): bits(item) for key, item in cast("dict[Any, Any]", value).items()}
    if isinstance(value, float):
        return value.hex()
    return value


# the studies' cells from the library's presets: the window-recurrent adder's 27 column tokens and 3
# digits; the division family at p 7, the sum and difference queried (11 tokens, 3 a column, 8 classes)
PRESETS = {"transformer": wp.transformer_window, "linear": wp.linear_window, "ffn": wp.ffn_window}
FAMILY_TOKENS, FAMILY_WIDTH = 11, 3


def adder(window: str, k: int, s: int, m: int, seed: int) -> Any:
    recurrence = WindowRecurrence((Factor(s, Codes(m)),))
    return WindowRecurrentRustArrayNetwork.randomized((k,), PRESETS[window](27, 3), recurrence, Adam(), seed=seed)


def family(seed: int) -> Any:
    recurrence = WindowRecurrence((Factor(16, Codes(32)),) * 2, False, pooling=AttentionPooling(OUTPUT))
    specs = wp.transformer_window(FAMILY_TOKENS, 8)
    return WindowRecurrentRustArrayNetwork.randomized((FAMILY_WIDTH,), specs, recurrence, Adam(), seed=seed)


def program_passes(network: Any, X: Any) -> dict[str, Any]:
    # every inference pass's results, as bits and codes
    outputs, states, cells = network.window_states(X)
    result: dict[str, Any] = {
        "forward_windows": bits(network.forward_windows(X)),
        "window_states": [bits(outputs), [bits(s) for s in states], [bits(c) for c in cells]],
    }
    if network.recurrence.snapped:
        # a machine's: every factor snapped
        result["state_codes"] = network.state_codes(X)
        output, cell, codes = network.window_step(RUST.columns(X, 0, network.columns), network._first_state(X.shape[0]))
        result["window_step"] = [bits(output), bits(cell), codes]
    return result


def assert_program_is_per_layer(network: Any, X: Any) -> None:
    program = program_passes(network, X)
    with per_layer_path(network):
        per_layer = program_passes(network, X)
    assert program == per_layer
