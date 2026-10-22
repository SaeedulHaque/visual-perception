"""Test doubles shared by several test modules."""

from __future__ import annotations

import numpy as np

from robust_visual_perception.detector import (
    Detection,
    DetectionResult,
    DetectionStatus,
)


def person(confidence: float, box=(10.0, 10.0, 20.0, 40.0)) -> Detection:
    return Detection(0, "person", confidence, *box)


class FakeDetector:
    """Replays a script of detection lists or exceptions, one per call.

    An exception entry becomes an ERROR result, as in the real wrapper.
    When the script runs out, the last entry is repeated.
    """

    def __init__(self, script: list[list[Detection] | Exception]):
        self.script = script
        self.calls = 0

    def detect(self, image: np.ndarray) -> DetectionResult:
        entry = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        hw = (int(image.shape[0]), int(image.shape[1]))
        if isinstance(entry, Exception):
            return DetectionResult(DetectionStatus.ERROR, [], 0.0, hw, str(entry))
        status = DetectionStatus.OK if entry else DetectionStatus.EMPTY
        return DetectionResult(status, list(entry), 1.0, hw)
