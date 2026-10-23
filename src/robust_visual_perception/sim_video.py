"""Annotated frames of a simulation run, for watching what the node saw.

Each frame gets the ROI band, the GT box, the node's published detections
and a banner below the image with the time, the per-frame state, the
filtered state and the GT state. Frames the node did not process (dropped
under load) keep the last published state, shown greyed out as "held".
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from robust_visual_perception.decision import Roi

BANNER_H = 40
# BGR.
COLORS = {
    "roi": (255, 200, 0),
    "gt": (0, 200, 0),
    "target": (0, 165, 255),
    "weak": (160, 160, 160),
    "STOP": (0, 0, 200),
    "CLEAR": (0, 140, 0),
    "UNKNOWN": (110, 110, 110),
    "held": (70, 70, 70),
    "time": (30, 30, 30),
}
DETECTION_COLUMNS = ["stamp_ns", "class_name", "score", "x1", "y1", "x2", "y2"]


class DetectionRow(NamedTuple):
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def write_detections_csv(path: Path, rows: Iterable[tuple[int, DetectionRow]]) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DETECTION_COLUMNS)
        writer.writerows([stamp, *det] for stamp, det in rows)


def read_detections_csv(path: Path) -> dict[int, list[DetectionRow]]:
    """Published detections per image stamp (ns)."""
    out: dict[int, list[DetectionRow]] = defaultdict(list)
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["stamp_ns"])].append(
                DetectionRow(
                    r["class_name"],
                    float(r["score"]),
                    *(float(r[k]) for k in ("x1", "y1", "x2", "y2")),
                )
            )
    return dict(out)


def read_gt_boxes(path: Path) -> dict[int, tuple[float, float, float, float] | None]:
    """GT box per image stamp from a sequence's frames.csv (None: no box)."""
    out = {}
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            box = [r[k] for k in ("x1", "y1", "x2", "y2")]
            out[int(r["stamp_ns"])] = tuple(float(v) for v in box) if all(box) else None
    return out


def _state_key(state: str) -> str:
    return state.split(":", 1)[0]


def _cell(canvas: np.ndarray, index: int, color, text: str) -> None:
    h, w = canvas.shape[0] - BANNER_H, canvas.shape[1]
    x0, x1 = index * w // 4, (index + 1) * w // 4
    canvas[h:, x0:x1] = color
    cv2.putText(
        canvas,
        text,
        (x0 + 8, h + BANNER_H // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _box(canvas, box, color, thickness=2) -> None:
    x1, y1, x2, y2 = (round(v) for v in box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)


def draw_overlay(
    image: np.ndarray,
    *,
    t_s: float,
    gt_box: Sequence[float] | None,
    detections: Sequence[DetectionRow],
    roi: Roi,
    target_class: str,
    conf_threshold: float,
    baseline: str,
    filtered: str,
    gt_stop: bool,
    processed: bool,
) -> np.ndarray:
    """The frame with overlays and a state banner below it (new array)."""
    h, w = image.shape[:2]
    canvas = np.zeros((h + BANNER_H, w, 3), np.uint8)
    canvas[:h] = image
    for fraction in (roi.x_min, roi.x_max):
        x = round(fraction * w)
        cv2.line(canvas, (x, 0), (x, h - 1), COLORS["roi"], 1)
    if gt_box is not None:
        _box(canvas, gt_box, COLORS["gt"])
    for det in detections:
        strong = det.class_name == target_class and det.score >= conf_threshold
        _box(canvas, det[2:], COLORS["target" if strong else "weak"])
        cv2.putText(
            canvas,
            f"{det.class_name} {det.score:.2f}",
            (round(det.x1), max(12, round(det.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLORS["target" if strong else "weak"],
            1,
            cv2.LINE_AA,
        )

    _cell(canvas, 0, COLORS["time"], f"t {t_s:6.2f} s")
    for index, (name, state) in enumerate(
        (("frame", baseline), ("filtered", filtered)), 1
    ):
        key = _state_key(state)
        if processed:
            _cell(canvas, index, COLORS[key], f"{name} {state}")
        else:
            _cell(canvas, index, COLORS["held"], f"{name} {state} (held)")
    gt = "STOP" if gt_stop else "CLEAR"
    _cell(canvas, 3, COLORS[gt], f"GT {gt}")
    return canvas
