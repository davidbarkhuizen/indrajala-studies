"""The collapse trigger (indrajala_studies/collapse.py)."""

import pytest

from indrajala_studies.collapse import Trigger

KINDS = {"quotient-0": 1.0, "quotient-nonzero": 0.5, "sum": 0.9}


def test_the_default_is_the_forks_old_trigger_the_pooled_screen_below_0_6_of_its_peak():
    trigger = Trigger()
    assert trigger.value(0.7, KINDS) == 0.7
    assert trigger.fires(0.59, 1.0) and not trigger.fires(0.6, 1.0)
    for score, peak in ((0.3, 0.5), (0.31, 0.5), (0.2999, 0.5)):
        assert trigger.fires(score, peak) == (score < (1 - 0.4) * peak)  # the fork's old rule, D1


def test_the_signals_read_the_pooled_share_the_kinds_mean_or_the_worst_kind():
    assert Trigger(signal="mean").value(0.9, KINDS) == pytest.approx(0.8)
    assert Trigger(signal="worst").value(0.9, KINDS) == 0.5
    assert Trigger(signal="worst").value(0.9, {}) == 0.9  # no kinds: the pooled share


def test_a_fall_fires_more_than_its_threshold_below_the_peak_once_the_peak_reaches_the_floor():
    trigger = Trigger(rule="fall", threshold=0.2, floor=0.8)
    assert trigger.fires(0.69, 0.9) and not trigger.fires(0.71, 0.9)
    assert not trigger.fires(0.1, 0.79)  # below the floor: still learning, no collapse
    assert trigger.describe() == "pooled 0.2 below the peak, once the peak is 0.8"


def test_an_unknown_signal_or_rule_is_refused():
    with pytest.raises(AssertionError):
        Trigger(signal="median")
    with pytest.raises(AssertionError):
        Trigger(rule="slope")
