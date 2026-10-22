"""Per-frame safety decision: CLEAR, STOP or UNKNOWN.

This is an image-space heuristic: "a confident person detection is in view
(optionally inside a corridor region)". It says nothing about physical
distance and is not collision avoidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from robust_visual_perception.detector import (
    Detection,
    DetectionResult,
    DetectionStatus,
)


class State(StrEnum):
    CLEAR = "CLEAR"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Roi:
    """Region of interest as fractions of image width and height.

    A box counts when its centre lies in [x_min, x_max] x [y_min, y_max] and
    its height is at least ``min_height`` of the image height.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    min_height: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max, self.min_height)
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"ROI fractions must be in [0, 1], got {values}")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI needs x_min < x_max and y_min < y_max")

    def contains(self, det: Detection, image_hw: tuple[int, int]) -> bool:
        box = np.array([[det.x1, det.y1, det.x2, det.y2]])
        return bool(self.mask(box, image_hw)[0])

    def mask(self, boxes_xyxy: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
        """Vectorised ``contains`` for N x 4 boxes (used for ground truth)."""
        height, width = image_hw
        b = np.asarray(boxes_xyxy, float).reshape(-1, 4)
        cx = (b[:, 0] + b[:, 2]) / 2 / width
        cy = (b[:, 1] + b[:, 3]) / 2 / height
        return (
            (self.x_min <= cx)
            & (cx <= self.x_max)
            & (self.y_min <= cy)
            & (cy <= self.y_max)
            & (b[:, 3] - b[:, 1] >= self.min_height * height)
        )


# Middle 40 % of the width, lower 70 % of the height, box at least 25 % of
# the image height (design section 6.3).
CORRIDOR_ROI = Roi(0.30, 0.70, 0.30, 1.00, 0.25)


@dataclass(frozen=True)
class Decision:
    state: State
    reason: str | None = None


def baseline_decision(
    result: DetectionResult | None,
    target_class: str = "person",
    conf_threshold: float = 0.5,
    roi: Roi | None = None,
) -> Decision:
    """STOP if any ``target_class`` detection has confidence >= threshold.

    ``result=None`` means the frame never arrived (MISSING).
    """
    if result is None:
        return Decision(State.UNKNOWN, "missing_frame")
    if result.status is DetectionStatus.ERROR:
        return Decision(State.UNKNOWN, "detector_error")
    for det in result.detections:
        if det.class_name != target_class or det.confidence < conf_threshold:
            continue
        if roi is None or roi.contains(det, result.image_hw):
            return Decision(State.STOP)
    return Decision(State.CLEAR)
