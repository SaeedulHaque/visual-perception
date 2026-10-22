import pytest

from robust_visual_perception.decision import State
from robust_visual_perception.temporal_filter import PersistenceFilter

CODES = {"C": State.CLEAR, "S": State.STOP, "U": State.UNKNOWN}


def run(inputs: str, **kwargs) -> str:
    filt = PersistenceFilter(**kwargs)
    out = [filt.update(CODES[c]) for c in inputs.split()]
    return " ".join(s.value[0] for s in out)


def test_stop_needs_n_on_stop_frames():
    assert run("S S S") == "C C S"


def test_stop_counts_within_window():
    assert run("S C S C S") == "C C C C S"


def test_isolated_stops_never_assert_with_small_window():
    assert run("S C C S C C S", n_on=2, k_window=2) == "C C C C C C C"


def test_release_needs_m_off_consecutive_clear():
    assert run("S S S C C C C C") == "C C S S S S S C"


def test_stop_resets_release_count():
    assert run("S S S C C S C C C C C") == "C C S S S S S S S S C"


def test_unknown_passes_through():
    assert run("U") == "U"


def test_stop_held_through_gap_until_fresh_clears():
    assert run("S S S U C C C C C") == "C C S U S S S S C"


def test_unknown_clears_window():
    assert run("S S U S") == "C C U C"


def test_reset_restores_initial_state():
    filt = PersistenceFilter()
    for _ in range(3):
        filt.update(State.STOP)
    filt.reset()
    assert filt.update(State.CLEAR) is State.CLEAR
    assert filt.state is State.CLEAR


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_on": 0},
        {"n_on": 6, "k_window": 5},
        {"m_off": 0},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        PersistenceFilter(**kwargs)


def test_release_clears_window_so_old_stops_do_not_retrigger():
    assert run("S S C C C S", n_on=2, k_window=8, m_off=3) == "C S S S C C"
