import math

import numpy as np
import pytest

from robust_visual_perception.sim_eval import (
    agreement,
    final_pose,
    join_live,
    parse_run_name,
    spread,
    stream_row,
)

# Bag frames at 10 fps; GT-STOP in frames 3-6.
FRAMES = {
    "frame": np.arange(1, 11),
    "stamp_ns": np.arange(10, dtype=np.int64) * 100_000_000,
    "gt_stop": np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
}


def live_row(stamp_ns, baseline, filtered=None):
    return {
        "stamp_ns": str(stamp_ns),
        "baseline_state": baseline,
        "filtered_state": filtered or baseline,
        "latency_ms": "20.0",
        "age_sim_ms": "6.0",
        "callback_ms": "25.0",
    }


def test_join_live_matches_frames_by_stamp_and_skips_unknown_stamps():
    rows = [
        live_row(200_000_000, "STOP"),
        live_row(400_000_000, "STOP"),
        live_row(123, "STOP"),  # not in the bag
        live_row(900_000_000, "CLEAR"),
    ]
    joined = join_live(rows, FRAMES)
    assert joined["stamp_ns"].tolist() == [200_000_000, 400_000_000, 900_000_000]
    assert joined["gt_stop"].tolist() == [1, 1, 0]
    assert joined["baseline"] == ["STOP", "STOP", "CLEAR"]
    assert joined["unmatched"] == 1


def test_stream_row_uses_the_fps_of_the_processed_frames():
    # Node processed every second frame: 5 frames over 0.8 s -> 5 fps.
    stamps = FRAMES["stamp_ns"][::2]
    gt = FRAMES["gt_stop"][::2].astype(bool)
    states = ["CLEAR", "STOP", "STOP", "CLEAR", "CLEAR"]
    row = stream_row(states, gt, stamps)
    assert row["fps"] == pytest.approx(5.0)
    assert row["transitions"] == 2
    assert row["transitions_per_min"] == pytest.approx(2 / (5 / 5.0 / 60))
    assert row["events_detected"] == 1 and row["events_missed"] == 0


def test_agreement_on_common_stamps():
    live = {
        "stamp_ns": np.array([1, 2, 3, 4]),
        "baseline": ["STOP", "CLEAR", "STOP", "CLEAR"],
    }
    replay = {1: "STOP", 2: "STOP", 3: "STOP", 5: "CLEAR"}
    value, n = agreement(live, replay)
    assert n == 3
    assert value == pytest.approx(2 / 3)


def test_agreement_without_common_frames_is_nan():
    value, n = agreement({"stamp_ns": np.array([1]), "baseline": ["STOP"]}, {2: "STOP"})
    assert n == 0 and math.isnan(value)


def test_final_pose_reads_the_last_sample():
    poses = np.array(
        [
            [0.0, 0, 0, 0, 0, 0, 0, 1],
            [5.0, 1.5, -0.2, 0.01, 0, 0, math.sin(0.25), math.cos(0.25)],
        ]
    )
    x, y, yaw = final_pose(poses)
    assert (x, y) == (1.5, -0.2)
    assert yaw == pytest.approx(0.5)


def test_parse_run_name():
    assert parse_run_name("static_r2") == ("static", 2)
    assert parse_run_name("scripted_r10") == ("scripted", 10)
    with pytest.raises(ValueError):
        parse_run_name("static")


def test_spread_over_repeats():
    rows = [
        {"mode": "static", "method": "filtered", "recall": 0.8, "transitions": 4},
        {"mode": "static", "method": "filtered", "recall": 0.9, "transitions": 6},
        {"mode": "static", "method": "filtered", "recall": 1.0, "transitions": 5},
        {"mode": "static", "method": "baseline", "recall": 0.5, "transitions": 9},
    ]
    out = spread(rows, ["mode", "method"], ["recall", "transitions"])
    filtered = next(r for r in out if r["method"] == "filtered")
    assert filtered["n_repeats"] == 3
    assert filtered["recall_mean"] == pytest.approx(0.9)
    assert filtered["recall_min"] == pytest.approx(0.8)
    assert filtered["recall_max"] == pytest.approx(1.0)
    assert filtered["transitions_mean"] == pytest.approx(5.0)
    baseline = next(r for r in out if r["method"] == "baseline")
    assert baseline["n_repeats"] == 1


def test_spread_treats_a_missing_metric_as_nan():
    rows = [
        {"mode": "static", "source": "replay", "recall": 1.0},
        {"mode": "static", "source": "replay", "recall": 0.5},
    ]
    (out,) = spread(rows, ["mode", "source"], ["recall", "agreement"])
    assert out["recall_mean"] == pytest.approx(0.75)
    assert math.isnan(out["agreement_mean"])
