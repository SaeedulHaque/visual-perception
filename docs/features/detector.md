# Detector

Module: `src/robust_visual_perception/detector.py`
Tests: `tests/test_detector.py` (no weights needed), plus the smoke test for
the real model.

## Why a wrapper

The pipeline should not depend on one vendor's result objects. The decision
logic, metrics and the ROS node only see three plain types:

```python
class DetectionStatus(StrEnum):
    OK  # at least one box
    EMPTY  # inference ran, nothing found
    ERROR  # inference raised; message in DetectionResult.error


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float  # validated to [0, 1]
    x1: float
    y1: float  # top-left, original-image pixels
    x2: float
    y2: float  # bottom-right, x1 < x2 and y1 < y2


@dataclass(frozen=True)
class DetectionResult:
    status: DetectionStatus
    detections: list[Detection]
    latency_ms: float
    image_hw: tuple[int, int]  # (height, width) of the input image
    error: str | None
```

`Detector` is a `Protocol` with one method, `detect(image) -> DetectionResult`.
Anything with that method works, which is how the tests use a scripted
`FakeDetector` instead of a neural network.

## The model: YOLO26n

- Ultralytics YOLO26, nano size, pretrained on COCO (80 classes, `person`
  is class 0). About 5.5 MB of weights. I picked the nano model because the
  whole study runs on a laptop CPU and the question is robustness of a
  small, deployable model, not peak accuracy.
- The weights file `models/yolo26n.pt` is downloaded by Ultralytics on first
  use and is git-ignored. A non-official file name that does not exist
  raises `FileNotFoundError` with a hint instead of a vague download error.
- Ultralytics is AGPL-3.0, which is why this repository is AGPL-3.0 too.

## What `detect` does

1. Calls `model.predict(image, imgsz, conf, max_det, device="cpu")` on one
   BGR image (batch size 1, like a live camera).
2. Times that call with `time.perf_counter_ns()`. This includes Ultralytics'
   preprocessing (letterbox resize to `imgsz`, normalisation), the forward
   pass and postprocessing. Image loading and degradation are not included.
3. Converts boxes back to plain `Detection`s, clipping to the image and
   dropping boxes that become zero-area after clipping.
4. Any exception becomes `status=ERROR` with the message, and is logged with
   its traceback. It is never silently swallowed, and the run continues.

## Two thresholds, on purpose

Inference runs at `conf=0.001` and `max_det=300`. That keeps almost every
candidate box, which COCO mAP needs (it integrates over the full
precision-recall curve). The decision rule then applies its own threshold
of 0.5 on top. One stored inference pass serves both.

## `imgsz`

YOLO resizes the longer image side to `imgsz` (with padding) before the
network. Smaller `imgsz` is faster but loses small objects. The experiments
compare 320, 416 and 640.

## Observed while building

The first `detect` call on a fresh model is far slower than the rest
(one-off setup inside PyTorch and Ultralytics). This is why the runner
performs 10 untimed warm-up calls before timing anything.
