"""Evaluate a finished COCO image run: one aggregate row per (condition, imgsz).

Reads only what the run saved (``config.yaml``, ``per_frame.csv`` and the
stored detections), so metrics can be recomputed without re-running the model.
"""

from __future__ import annotations

import contextlib
import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from robust_visual_perception.coco_subset import read_manifest
from robust_visual_perception.config import load_config
from robust_visual_perception.detection_store import load_detections
from robust_visual_perception.metrics import (
    PERSON_CATEGORY_ID,
    coco_map,
    decision_metrics,
    latency_stats,
    person_pr,
    ratio,
    retention,
)

AGGREGATE_COLUMNS = [
    "condition",
    "kind",
    "param",
    "imgsz",
    "n_images",
    "n_person_images",
    "map",
    "map50",
    "person_ap",
    "person_ap50",
    "person_precision",
    "person_recall",
    "person_tp",
    "person_fp",
    "person_fn",
    "mean_matched_conf",
    "retention",
    "retention_n",
    "stop_precision",
    "stop_recall",
    "false_stop_rate",
    "stop_accuracy",
    "n_unknown",
    "latency_n",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "throughput_ips",
    "degrade_mean_ms",
]

Boxes = tuple[np.ndarray, np.ndarray]  # (xyxy N x 4, scores N)
EMPTY: Boxes = (np.zeros((0, 4)), np.zeros(0))


def _load_coco(instances_path: Path):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(instances_path))


def _person_gt(coco, image_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-crowd and crowd person boxes (xyxy) for one image."""
    boxes: dict[bool, list[list[float]]] = {False: [], True: []}
    for ann in coco.imgToAnns.get(image_id, []):
        if ann["category_id"] != PERSON_CATEGORY_ID:
            continue
        x, y, w, h = ann["bbox"]
        boxes[bool(ann.get("iscrowd", 0))].append([x, y, x + w, y + h])
    return (
        np.array(boxes[False], float).reshape(-1, 4),
        np.array(boxes[True], float).reshape(-1, 4),
    )


def _person_by_image(detections: np.ndarray) -> dict[int, Boxes]:
    persons = detections[detections[:, 6] == PERSON_CATEGORY_ID]
    out: dict[int, Boxes] = {}
    for image_id in np.unique(persons[:, 0]).astype(int):
        rows = persons[persons[:, 0] == image_id]
        xyxy = rows[:, 1:5].copy()
        xyxy[:, 2:] += xyxy[:, :2]
        out[int(image_id)] = (xyxy, rows[:, 5])
    return out


def _read_per_frame(run_dir: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with (run_dir / "per_frame.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            groups[(row["condition"], int(row["imgsz"]))].append(row)
    return groups


def _floats(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(r[column]) for r in rows if r[column]]


def evaluate_image_run(run_dir: Path, instances_path: Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if cfg.manifest is None:
        raise ValueError("image evaluation needs a run with a COCO manifest")
    image_ids = [e.image_id for e in read_manifest(cfg.manifest)]
    coco = _load_coco(instances_path)
    gt = {i: _person_gt(coco, i) for i in image_ids}
    per_frame = _read_per_frame(run_dir)
    threshold = cfg.decision.conf_threshold

    rows = []
    for imgsz in cfg.model.imgsz:
        clean = _person_by_image(load_detections(run_dir, "clean", imgsz))
        for condition in cfg.conditions:
            dets = load_detections(run_dir, condition.name, imgsz)
            persons = _person_by_image(dets)
            tp = fp = fn = kept = total = 0
            conf_sum = 0.0
            for image_id in image_ids:
                boxes, scores = persons.get(image_id, EMPTY)
                gt_boxes, crowd = gt[image_id]
                counts = person_pr(boxes, scores, gt_boxes, crowd, threshold, 0.5)
                tp, fp, fn = tp + counts.tp, fp + counts.fp, fn + counts.fn
                conf_sum += counts.matched_conf_sum
                k, t = retention(*clean.get(image_id, EMPTY), boxes, scores, threshold)
                kept, total = kept + k, total + t

            frames = per_frame[(condition.name, imgsz)]
            # UNKNOWN counts as STOP: a consumer treats it as stop-with-reason.
            pred_stop = np.array([r["baseline_state"] != "CLEAR" for r in frames])
            gt_stop = np.array([r["gt_state"] == "STOP" for r in frames])
            decision = decision_metrics(pred_stop, gt_stop)
            latency = latency_stats(_floats(frames, "latency_ms"))
            degrade = _floats(frames, "degrade_ms")
            ap = coco_map(coco, dets, image_ids)

            rows.append(
                {
                    "condition": condition.name,
                    "kind": condition.kind,
                    "param": "" if condition.param is None else condition.param,
                    "imgsz": imgsz,
                    "n_images": len(frames),
                    "n_person_images": int(gt_stop.sum()),
                    "map": ap["map"],
                    "map50": ap["map50"],
                    "person_ap": ap["person_ap"],
                    "person_ap50": ap["person_ap50"],
                    "person_precision": ratio(tp, tp + fp),
                    "person_recall": ratio(tp, tp + fn),
                    "person_tp": tp,
                    "person_fp": fp,
                    "person_fn": fn,
                    "mean_matched_conf": ratio(conf_sum, tp),
                    "retention": ratio(kept, total),
                    "retention_n": total,
                    "stop_precision": decision["precision"],
                    "stop_recall": decision["recall"],
                    "false_stop_rate": decision["false_stop_rate"],
                    "stop_accuracy": decision["accuracy"],
                    "n_unknown": sum(r["baseline_state"] == "UNKNOWN" for r in frames),
                    "latency_n": latency["n"],
                    "latency_mean_ms": latency["mean_ms"],
                    "latency_median_ms": latency["median_ms"],
                    "latency_p95_ms": latency["p95_ms"],
                    "throughput_ips": latency["throughput_ips"],
                    "degrade_mean_ms": float(np.mean(degrade)) if degrade else np.nan,
                }
            )
    return rows


def write_aggregate(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()}
            )
