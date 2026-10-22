import numpy as np
import pytest

from robust_visual_perception.detection_store import (
    COCO80_TO_91,
    DetectionStore,
    load_detections,
)
from robust_visual_perception.detector import Detection


def test_store_round_trip_in_coco_result_format(tmp_path):
    store = DetectionStore()
    store.add("clean", 320, 139, [Detection(0, "person", 0.9, 10, 20, 40, 80)])
    store.add("clean", 320, 285, [Detection(2, "car", 0.4, 0, 0, 5, 5)])
    store.add("blur_k7", 320, 139, [])
    store.save(tmp_path)
    dets = load_detections(tmp_path, "clean", 320)
    np.testing.assert_allclose(
        dets,
        [[139, 10, 20, 30, 60, 0.9, 1], [285, 0, 0, 5, 5, 0.4, 3]],
        rtol=1e-6,
    )
    empty = load_detections(tmp_path, "blur_k7", 320)
    assert empty.shape == (0, 7)


def test_class_index_maps_to_coco_category_id():
    assert COCO80_TO_91[0] == 1  # person
    assert COCO80_TO_91[2] == 3  # car
    assert COCO80_TO_91[79] == 90  # toothbrush


def test_boxes_have_positive_size(tmp_path):
    store = DetectionStore()
    store.add("clean", 640, 1, [Detection(0, "person", 0.5, 1, 1, 1.5, 3)])
    store.save(tmp_path)
    dets = load_detections(tmp_path, "clean", 640)
    assert (dets[:, 3] > 0).all() and (dets[:, 4] > 0).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_detections(tmp_path, "clean", 320)
