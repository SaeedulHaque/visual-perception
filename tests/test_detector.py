import numpy as np
import pytest

from robust_visual_perception.detector import (
    Detection,
    DetectionResult,
    DetectionStatus,
    UltralyticsDetector,
    boxes_to_detections,
)

NAMES = {0: "person", 2: "car"}


@pytest.mark.parametrize(
    "conf,box",
    [
        (1.2, (0, 0, 10, 10)),
        (-0.1, (0, 0, 10, 10)),
        (0.5, (10, 0, 10, 10)),
        (0.5, (0, 10, 10, 5)),
    ],
)
def test_detection_rejects_invalid_values(conf, box):
    with pytest.raises(ValueError):
        Detection(0, "person", conf, *box)


def test_detection_accepts_valid_values():
    det = Detection(0, "person", 0.5, 1, 2, 3, 4)
    assert det.width == 2
    assert det.height == 2


def test_boxes_are_clipped_and_degenerate_boxes_dropped():
    xyxy = np.array([[-5, -5, 50, 50], [50, 50, 60, 60], [5, 5, 20, 30]], float)
    conf = np.array([0.9, 0.8, 0.3])
    cls = np.array([0, 2, 2])
    dets = boxes_to_detections(xyxy, conf, cls, NAMES, image_hw=(40, 40))
    assert len(dets) == 2
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == (0, 0, 40, 40)
    assert dets[0].class_name == "person"
    assert dets[1].class_name == "car"
    assert dets[1].confidence == pytest.approx(0.3)


def test_empty_boxes_give_empty_list():
    dets = boxes_to_detections(
        np.zeros((0, 4)), np.zeros(0), np.zeros(0), NAMES, image_hw=(40, 40)
    )
    assert dets == []


def test_result_status_from_detections():
    assert DetectionResult.from_detections([], 1.0, (4, 4)).status is (
        DetectionStatus.EMPTY
    )
    det = Detection(0, "person", 0.5, 1, 2, 3, 4)
    assert DetectionResult.from_detections([det], 1.0, (4, 4)).status is (
        DetectionStatus.OK
    )


def test_missing_weights_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch"):
        UltralyticsDetector(tmp_path / "custom_model.pt", imgsz=320)
