"""Image-level metrics: box matching, person precision/recall, retention,
decision correctness, latency statistics and COCO mAP.

Boxes are N x 4 arrays in ``[x1, y1, x2, y2]`` pixel coordinates. A ratio
with a zero denominator is NaN, never 0, so "no data" cannot look like
"perfect" or "zero" performance.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PERSON_CATEGORY_ID = 1


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between N boxes ``a`` and M boxes ``b`` -> N x M."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    inter = _intersection(a, b)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> list[tuple[int, int]]:
    """One-to-one matching: predictions in descending score order each take
    the unmatched GT box with the highest IoU, if that IoU >= ``iou_thr``.
    Returns ``(pred_index, gt_index)`` pairs."""
    iou = box_iou(pred_boxes, gt_boxes)
    free = np.ones(iou.shape[1], bool)
    pairs = []
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        if not free.any():
            break
        candidates = np.where(free, iou[p], -1.0)
        g = int(np.argmax(candidates))
        if candidates[g] >= iou_thr:
            pairs.append((int(p), g))
            free[g] = False
    return pairs


@dataclass(frozen=True)
class PRCounts:
    tp: int
    fp: int
    fn: int
    matched_conf_sum: float


def person_pr(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    crowd_boxes: np.ndarray,
    conf_thr: float = 0.5,
    iou_thr: float = 0.5,
) -> PRCounts:
    """TP/FP/FN for one image at a fixed confidence threshold.

    An unmatched prediction that lies at least half inside a crowd region is
    ignored rather than counted as a false positive.
    """
    keep = np.asarray(pred_scores) >= conf_thr
    boxes = np.asarray(pred_boxes, float).reshape(-1, 4)[keep]
    scores = np.asarray(pred_scores, float)[keep]
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    pairs = greedy_match(boxes, scores, gt_boxes, iou_thr)
    matched = {p for p, _ in pairs}
    unmatched = [i for i in range(len(boxes)) if i not in matched]

    fp = len(unmatched)
    crowd_boxes = np.asarray(crowd_boxes, float).reshape(-1, 4)
    if unmatched and len(crowd_boxes):
        u = boxes[unmatched]
        area = (u[:, 2] - u[:, 0]) * (u[:, 3] - u[:, 1])
        inside = _intersection(u, crowd_boxes).max(axis=1) / area
        fp = int((inside < 0.5).sum())
    return PRCounts(
        tp=len(pairs),
        fp=fp,
        fn=len(gt_boxes) - len(pairs),
        matched_conf_sum=float(sum(scores[p] for p in matched)),
    )


def retention(
    clean_boxes: np.ndarray,
    clean_scores: np.ndarray,
    degraded_boxes: np.ndarray,
    degraded_scores: np.ndarray,
    conf_thr: float = 0.5,
    iou_thr: float = 0.5,
) -> tuple[int, int]:
    """How many confident clean-image detections survive degradation.

    Returns ``(kept, total)``: total = clean detections >= ``conf_thr``; kept =
    those matched one-to-one at IoU >= ``iou_thr`` by a degraded detection
    >= ``conf_thr``. A stability measure, not recall (no ground truth).
    """
    clean = np.asarray(clean_boxes, float).reshape(-1, 4)[
        np.asarray(clean_scores) >= conf_thr
    ]
    keep = np.asarray(degraded_scores) >= conf_thr
    degraded = np.asarray(degraded_boxes, float).reshape(-1, 4)[keep]
    scores = np.asarray(degraded_scores, float)[keep]
    return len(greedy_match(degraded, scores, clean, iou_thr)), len(clean)


def decision_metrics(pred_stop: np.ndarray, gt_stop: np.ndarray) -> dict[str, float]:
    """Binary STOP-vs-CLEAR correctness (STOP is the positive class)."""
    pred = np.asarray(pred_stop, bool)
    gt = np.asarray(gt_stop, bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_stop_rate": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, len(pred)),
        "n": len(pred),
    }


def latency_stats(values_ms: Sequence[float]) -> dict[str, float]:
    """n, mean, median, p95 (only when n >= 100) and sequential throughput."""
    values = np.asarray(values_ms, float)
    n = len(values)
    if n == 0:
        nan = math.nan
        return {"n": 0, "mean_ms": nan, "median_ms": nan, "p95_ms": nan,
                "throughput_ips": nan}  # fmt: skip
    mean = float(values.mean())
    return {
        "n": n,
        "mean_ms": mean,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)) if n >= 100 else math.nan,
        "throughput_ips": ratio(1000.0, mean),
    }


def coco_map(coco_gt, detections: np.ndarray, image_ids: Sequence[int]) -> dict:
    """COCO mAP@[.5:.95] and mAP@.5 over all classes, plus person AP,
    restricted to ``image_ids``.

    ``detections`` is N x 7 ``[image_id, x, y, w, h, score, category_id]``.
    """
    from pycocotools.cocoeval import COCOeval

    detections = np.asarray(detections, float).reshape(-1, 7)
    if len(detections) == 0:
        return {"map": 0.0, "map50": 0.0, "person_ap": 0.0, "person_ap50": 0.0}

    def evaluate(cat_ids: list[int] | None) -> tuple[float, float]:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(detections)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.imgIds = list(image_ids)
            if cat_ids is not None:
                ev.params.catIds = cat_ids
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        # pycocotools reports -1 when there is no GT for the selection.
        return tuple(float(s) if s >= 0 else math.nan for s in ev.stats[:2])

    map_, map50 = evaluate(None)
    person_ap, person_ap50 = evaluate([PERSON_CATEGORY_ID])
    return {"map": map_, "map50": map50, "person_ap": person_ap,
            "person_ap50": person_ap50}  # fmt: skip
