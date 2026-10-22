import math

import numpy as np
import pytest

from robust_visual_perception.config import FilterConfig
from robust_visual_perception.sequence_metrics import (
    SWEEP_SETTINGS,
    adopt_setting,
    refilter,
    stream_metrics,
)


def _states(code: str) -> list[str]:
    return [{"S": "STOP", "C": "CLEAR", "U": "UNKNOWN"}[c] for c in code.split()]


GT = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0], bool)


def test_hand_example():
    m = stream_metrics(_states("S C C S S S C S S S C C"), GT, fps=6, merge_gap=1)
    assert (m["tp"], m["fp"], m["fn"]) == (4, 3, 1)
    assert m["precision"] == pytest.approx(4 / 7)
    assert m["recall"] == pytest.approx(4 / 5)
    assert m["accuracy"] == pytest.approx(8 / 12)
    assert m["transitions"] == 5
    assert m["transitions_per_min"] == pytest.approx(150.0)
    assert m["change_fraction"] == pytest.approx(5 / 11)
    assert (m["n_events"], m["events_detected"], m["events_missed"]) == (2, 2, 0)
    assert m["onset_mean_frames"] == pytest.approx(0.5)
    assert m["onset_mean_s"] == pytest.approx(0.5 / 6)
    assert m["release_mean_frames"] == pytest.approx(0.5)
    assert m["release_n"] == 2
    assert m["false_stop_episodes"] == 1


def test_never_stopping_misses_everything():
    m = stream_metrics(_states(" ".join("C" * 12)), GT, fps=6, merge_gap=1)
    assert m["events_missed"] == 2
    assert m["events_detected"] == 0
    assert math.isnan(m["onset_mean_frames"])
    assert math.isnan(m["release_mean_frames"])
    assert m["transitions"] == 0
    assert m["recall"] == 0.0


def test_release_held_into_next_event_is_not_counted():
    m = stream_metrics(_states("C C C S S S S S S S C C"), GT, fps=6, merge_gap=1)
    assert m["release_n"] == 1
    assert m["release_mean_frames"] == pytest.approx(0.0)


def test_unknown_counts_as_stop_and_is_reported():
    m = stream_metrics(_states("C C U S S C C C S S C C"), GT, fps=6, merge_gap=1)
    assert m["n_unknown"] == 1
    assert m["onset_mean_frames"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(1.0)


def test_refilter_matches_persistence_filter():
    out = refilter(_states("S S S C C C C C"), FilterConfig(3, 5, 5))
    assert out == _states("C C S S S S S C")


def test_sweep_settings_match_design():
    pairs = {(s.n_on, s.k_window) for s in SWEEP_SETTINGS}
    assert pairs == {(2, 3), (3, 5), (4, 6), (5, 8)}
    assert {s.m_off for s in SWEEP_SETTINGS} == {3, 5, 8}
    assert len(SWEEP_SETTINGS) == 12


def _summary(setting, missed, tpm, onset):
    return {
        "setting": setting,
        "events_missed": missed,
        "transitions_per_min": tpm,
        "onset_mean_s": onset,
    }


def test_default_kept_unless_dominated():
    default = FilterConfig(3, 5, 5)
    other = FilterConfig(2, 3, 3)
    rows = [_summary(default, 1, 10.0, 0.3), _summary(other, 0, 12.0, 0.2)]
    assert adopt_setting(rows, default) == default


def test_dominating_setting_is_adopted_fewest_transitions_first():
    default = FilterConfig(3, 5, 5)
    a = FilterConfig(4, 6, 5)
    b = FilterConfig(5, 8, 8)
    rows = [
        _summary(default, 1, 10.0, 0.3),
        _summary(a, 1, 8.0, 0.3),
        _summary(b, 0, 6.0, 0.25),
    ]
    assert adopt_setting(rows, default) == b


def test_equal_setting_does_not_dominate():
    default = FilterConfig(3, 5, 5)
    rows = [
        _summary(default, 1, 10.0, 0.3),
        _summary(FilterConfig(2, 3, 3), 1, 10.0, 0.3),
    ]
    assert adopt_setting(rows, default) == default
