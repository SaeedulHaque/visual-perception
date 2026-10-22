import pytest

from robust_visual_perception.decision import (
    CORRIDOR_ROI,
    Roi,
    State,
    baseline_decision,
)
from robust_visual_perception.detector import (
    Detection,
    DetectionResult,
    DetectionStatus,
)

HW = (100, 200)  # height, width


def _result(*dets: Detection) -> DetectionResult:
    return DetectionResult.from_detections(list(dets), 1.0, HW)


def _det(conf: float, box=(80, 30, 120, 95), name="person", cid=0) -> Detection:
    return Detection(cid, name, conf, *box)


def test_threshold_boundary():
    assert baseline_decision(_result(_det(0.5))).state is State.STOP
    assert baseline_decision(_result(_det(0.49))).state is State.CLEAR


def test_empty_result_is_clear():
    decision = baseline_decision(_result())
    assert decision.state is State.CLEAR
    assert decision.reason is None


def test_other_classes_are_ignored():
    car = _det(0.9, name="car", cid=2)
    assert baseline_decision(_result(car)).state is State.CLEAR


def test_missing_frame_is_unknown():
    decision = baseline_decision(None)
    assert decision.state is State.UNKNOWN
    assert decision.reason == "missing_frame"


def test_detector_error_is_unknown():
    error = DetectionResult(DetectionStatus.ERROR, [], 0.0, HW, "boom")
    decision = baseline_decision(error)
    assert decision.state is State.UNKNOWN
    assert decision.reason == "detector_error"


def test_roi_accepts_tall_box_in_corridor():
    # centre x = 100 (0.5 W), centre y = 62.5, height 65 (0.65 H)
    assert baseline_decision(_result(_det(0.9)), roi=CORRIDOR_ROI).state is (State.STOP)


def test_roi_rejects_box_outside_corridor():
    left = _det(0.9, box=(5, 30, 35, 95))  # centre x = 20 (0.1 W)
    assert baseline_decision(_result(left), roi=CORRIDOR_ROI).state is State.CLEAR


def test_roi_rejects_short_box():
    short = _det(0.9, box=(90, 60, 110, 80))  # height 20 (0.2 H)
    assert baseline_decision(_result(short), roi=CORRIDOR_ROI).state is State.CLEAR


def test_roi_rejects_box_centred_above_band():
    high = _det(0.9, box=(90, 0, 110, 40))  # centre y = 20 (0.2 H), height 0.4
    assert baseline_decision(_result(high), roi=CORRIDOR_ROI).state is State.CLEAR


def test_corridor_roi_matches_design():
    assert Roi(0.30, 0.70, 0.30, 1.00, 0.25) == CORRIDOR_ROI


@pytest.mark.parametrize(
    "args",
    [(0.7, 0.3, 0.0, 1.0, 0.0), (0.0, 1.0, 0.5, 0.4, 0.0), (0.0, 1.2, 0.0, 1.0, 0.0)],
)
def test_roi_validates_fractions(args):
    with pytest.raises(ValueError):
        Roi(*args)
