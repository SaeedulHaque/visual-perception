import math

import numpy as np
import pytest
from pycocotools.coco import COCO

from robust_visual_perception.metrics import (
    box_iou,
    coco_map,
    decision_metrics,
    greedy_match,
    latency_stats,
    person_pr,
    ratio,
    retention,
)


def test_box_iou_known_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])


def test_box_iou_empty_shapes():
    assert box_iou(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_greedy_match_prefers_higher_score():
    gt = np.array([[0, 0, 10, 10]], float)
    preds = np.array([[0, 0, 10, 9], [0, 0, 10, 10]], float)
    scores = np.array([0.6, 0.9])
    assert greedy_match(preds, scores, gt, 0.5) == [(1, 0)]


def test_person_pr_counts():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    preds = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.7, 0.8])
    counts = person_pr(preds, scores, gt, np.zeros((0, 4)))
    assert (counts.tp, counts.fp, counts.fn) == (2, 1, 0)
    assert counts.matched_conf_sum == pytest.approx(1.6)


def test_person_pr_applies_conf_threshold():
    gt = np.array([[0, 0, 10, 10]], float)
    counts = person_pr(gt.copy(), np.array([0.49]), gt, np.zeros((0, 4)))
    assert (counts.tp, counts.fp, counts.fn) == (0, 0, 1)


def test_prediction_inside_crowd_region_is_ignored():
    crowd = np.array([[40, 40, 100, 100]], float)
    preds = np.array([[50, 50, 60, 60]], float)
    counts = person_pr(preds, np.array([0.9]), np.zeros((0, 4)), crowd)
    assert (counts.tp, counts.fp, counts.fn) == (0, 0, 0)


def test_retention_counts_clean_detections_kept():
    clean = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    degraded = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    kept, total = retention(clean, np.array([0.9, 0.8]), degraded, np.array([0.7, 0.3]))
    assert (kept, total) == (1, 2)


def test_retention_with_no_clean_detections():
    assert retention(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)), np.zeros(0)) == (
        0,
        0,
    )


def test_ratio_with_zero_denominator_is_nan():
    assert math.isnan(ratio(0, 0))
    assert ratio(1, 4) == 0.25


def test_decision_metrics_hand_example():
    pred = np.array([1, 1, 0, 0, 1, 0], bool)
    gt = np.array([1, 0, 1, 0, 1, 0], bool)
    m = decision_metrics(pred, gt)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_stop_rate"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["n"] == 6


def test_decision_metrics_without_positives_gives_nan_recall():
    m = decision_metrics(np.zeros(3, bool), np.zeros(3, bool))
    assert math.isnan(m["recall"])
    assert math.isnan(m["precision"])
    assert m["false_stop_rate"] == 0.0


def test_latency_stats_small_sample_has_no_p95():
    s = latency_stats([10.0, 20.0, 30.0])
    assert s["n"] == 3
    assert s["mean_ms"] == pytest.approx(20.0)
    assert s["median_ms"] == pytest.approx(20.0)
    assert math.isnan(s["p95_ms"])
    assert s["throughput_ips"] == pytest.approx(50.0)


def test_latency_stats_p95_from_100_samples():
    s = latency_stats(list(range(1, 101)))
    assert s["p95_ms"] == pytest.approx(np.percentile(range(1, 101), 95))


def test_latency_stats_empty():
    s = latency_stats([])
    assert s["n"] == 0
    assert math.isnan(s["mean_ms"])


def _tiny_coco() -> COCO:
    coco = COCO()
    coco.dataset = {
        "images": [
            {"id": 1, "width": 100, "height": 100},
            {"id": 2, "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 50],
                "area": 1500,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 3,
                "bbox": [40, 40, 40, 30],
                "area": 1200,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
    }
    coco.createIndex()
    return coco


def test_coco_map_perfect_predictions():
    dets = np.array([[1, 10, 10, 30, 50, 0.9, 1], [2, 40, 40, 40, 30, 0.8, 3]])
    m = coco_map(_tiny_coco(), dets, [1, 2])
    assert m["map"] == pytest.approx(1.0)
    assert m["map50"] == pytest.approx(1.0)
    assert m["person_ap"] == pytest.approx(1.0)


def test_coco_map_without_detections_is_zero():
    m = coco_map(_tiny_coco(), np.zeros((0, 7)), [1, 2])
    assert m["map"] == 0.0
    assert m["person_ap"] == 0.0
