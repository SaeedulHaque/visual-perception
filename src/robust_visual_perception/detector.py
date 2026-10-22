"""Detector interface and the Ultralytics YOLO wrapper.

The rest of the pipeline only sees the ``Detector`` protocol and the
``DetectionResult`` type. ``UltralyticsDetector`` is the only code that knows
about the vendor library, so another detector can be swapped in later.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class DetectionStatus(StrEnum):
    OK = "OK"  # at least one detection
    EMPTY = "EMPTY"  # inference ran, nothing found
    ERROR = "ERROR"  # inference raised; see DetectionResult.error


@dataclass(frozen=True)
class Detection:
    """One box in original-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError(
                f"box must have x1 < x2 and y1 < y2, got "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class DetectionResult:
    status: DetectionStatus
    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    image_hw: tuple[int, int] = (0, 0)
    error: str | None = None

    @classmethod
    def from_detections(
        cls, detections: list[Detection], latency_ms: float, image_hw: tuple[int, int]
    ) -> DetectionResult:
        status = DetectionStatus.OK if detections else DetectionStatus.EMPTY
        return cls(status, detections, latency_ms, image_hw)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> DetectionResult: ...


def boxes_to_detections(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    names: Mapping[int, str],
    image_hw: tuple[int, int],
) -> list[Detection]:
    """Convert raw box arrays to ``Detection`` objects.

    Boxes are clipped to the image. Boxes with zero width or height after
    clipping are dropped.
    """
    height, width = image_hw
    detections = []
    for (x1, y1, x2, y2), score, class_id in zip(xyxy, conf, cls, strict=True):
        x1, x2 = float(np.clip(x1, 0, width)), float(np.clip(x2, 0, width))
        y1, y2 = float(np.clip(y1, 0, height)), float(np.clip(y2, 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_id)
        detections.append(
            Detection(
                class_id,
                names.get(class_id, str(class_id)),
                float(np.clip(score, 0.0, 1.0)),
                x1,
                y1,
                x2,
                y2,
            )
        )
    return detections


class UltralyticsDetector:
    """Ultralytics YOLO model running batch-1 inference.

    ``conf`` and ``max_det`` are the inference-time limits. They are kept low
    and high (0.001, 300) so that one stored pass serves both mAP evaluation
    and the 0.5-threshold decision rule.
    """

    def __init__(
        self,
        weights: str | Path,
        imgsz: int,
        conf: float = 0.001,
        max_det: int = 300,
        device: str = "cpu",
    ) -> None:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        weights = Path(weights)
        if not weights.is_file() and weights.name not in GITHUB_ASSETS_NAMES:
            raise FileNotFoundError(
                f"weights not found: {weights}. Official names such as "
                f"'yolo26n.pt' are fetched automatically; for other models, "
                f"place the file at this path first."
            )
        weights.parent.mkdir(parents=True, exist_ok=True)

        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names: dict[int, str] = dict(self.model.names)
        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det
        self.device = device

    def detect(self, image: np.ndarray) -> DetectionResult:
        image_hw = (int(image.shape[0]), int(image.shape[1]))
        start = time.perf_counter_ns()
        try:
            result = self.model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.conf,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            latency_ms = (time.perf_counter_ns() - start) / 1e6
            logger.exception("inference failed")
            return DetectionResult(
                DetectionStatus.ERROR, [], latency_ms, image_hw, repr(exc)
            )
        latency_ms = (time.perf_counter_ns() - start) / 1e6

        boxes = result.boxes
        detections = boxes_to_detections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            self.names,
            image_hw,
        )
        return DetectionResult.from_detections(detections, latency_ms, image_hw)
