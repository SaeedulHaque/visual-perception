"""Run one image experiment: frames x input sizes x conditions -> per_frame.csv.

``process_frame`` is the single per-frame step (degrade, detect, decide)
shared by the image runner and the sequence runner.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from robust_visual_perception.coco_subset import read_manifest
from robust_visual_perception.config import ExperimentConfig
from robust_visual_perception.decision import (
    Decision,
    Roi,
    State,
    baseline_decision,
)
from robust_visual_perception.degradation import Condition, apply
from robust_visual_perception.detection_store import DetectionStore
from robust_visual_perception.detector import (
    DetectionResult,
    DetectionStatus,
    Detector,
)
from robust_visual_perception.image_source import (
    Frame,
    FrameStatus,
    iter_image_files,
    list_image_files,
)
from robust_visual_perception.reporting import (
    PerFrameWriter,
    prepare_run_dir,
    write_resources,
    write_run_metadata,
)

logger = logging.getLogger(__name__)

DetectorFactory = Callable[[int], Detector]


@dataclass(frozen=True)
class FrameOutput:
    decision: Decision
    result: DetectionResult | None  # None when the detector was not called
    row: dict[str, Any]  # per-frame CSV fields filled by this step


def safe_detect(detector: Detector, image: np.ndarray) -> DetectionResult:
    """A detector should report failures itself; this guards against one that
    raises anyway, so one bad frame cannot end the run."""
    try:
        return detector.detect(image)
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR row
        logger.exception("detector raised")
        hw = (int(image.shape[0]), int(image.shape[1]))
        return DetectionResult(DetectionStatus.ERROR, [], 0.0, hw, repr(exc))


def process_frame(
    frame: Frame,
    condition: Condition,
    detector: Detector,
    *,
    target_class: str,
    conf_threshold: float,
    noise_key: str,
    noise_seed: int,
    roi: Roi | None = None,
    injected: str | None = None,
) -> FrameOutput:
    """Degrade, detect and decide for one frame under one condition.

    ``injected`` simulates a fault without calling the detector:
    ``"MISSING"`` (frame lost) or ``"ERROR"`` (detector failure).
    """
    load_ms = "" if frame.load_ms is None else f"{frame.load_ms:.3f}"
    if frame.status is FrameStatus.MISSING or injected == "MISSING":
        decision = baseline_decision(None)
        return FrameOutput(decision, None, {"status": "MISSING", **_state(decision)})
    if injected == "ERROR":
        hw = (int(frame.image.shape[0]), int(frame.image.shape[1]))
        result = DetectionResult(DetectionStatus.ERROR, [], 0.0, hw, "injected")
        decision = baseline_decision(result)
        row = {"status": "ERROR", "load_ms": load_ms, **_state(decision)}
        return FrameOutput(decision, result, row)

    start = time.perf_counter_ns()
    image = apply(condition, frame.image, noise_key, noise_seed)
    degrade_ms = (time.perf_counter_ns() - start) / 1e6
    result = safe_detect(detector, image)
    decision = baseline_decision(result, target_class, conf_threshold, roi)
    target_confs = [
        d.confidence for d in result.detections if d.class_name == target_class
    ]
    row = {
        "status": result.status.value,
        "n_detections": len(result.detections),
        "n_target": sum(c >= conf_threshold for c in target_confs),
        "max_target_conf": f"{max(target_confs):.4f}" if target_confs else "",
        "latency_ms": f"{result.latency_ms:.3f}",
        "load_ms": load_ms,
        "degrade_ms": f"{degrade_ms:.3f}",
        **_state(decision),
    }
    return FrameOutput(decision, result, row)


def _state(decision: Decision) -> dict[str, str]:
    return {
        "baseline_state": decision.state.value,
        "state_reason": decision.reason or "",
    }


def warm_up(detectors: dict[int, Detector], image: np.ndarray, n: int) -> None:
    """Untimed calls so one-off setup cost does not enter the latency data."""
    for detector in detectors.values():
        for _ in range(n):
            safe_detect(detector, image)


def _frame_paths(cfg: ExperimentConfig) -> tuple[list[Path], dict[str, str]]:
    """Image paths in run order, and ground-truth state per frame id."""
    image_dir = Path(cfg.image_dir)
    if cfg.manifest is None:
        return list_image_files(image_dir), {}
    entries = read_manifest(cfg.manifest)
    paths = [image_dir / e.file_name for e in entries]
    gt = {
        Path(e.file_name).stem: (State.STOP if e.n_person > 0 else State.CLEAR).value
        for e in entries
    }
    return paths, gt


def run_image_experiment(
    cfg: ExperimentConfig,
    detector_factory: DetectorFactory,
    command: str,
    overwrite: bool = False,
) -> Path:
    """Run every (frame, imgsz, condition) and write ``per_frame.csv``.

    Each image is read once. With a manifest (COCO), all detections are also
    stored for mAP evaluation. Missing frames and detector errors become
    ``UNKNOWN`` rows; they never stop the run. Returns the run directory.
    """
    start = time.perf_counter()
    run_dir = prepare_run_dir(cfg.output_root, cfg.run_name, overwrite)
    write_run_metadata(run_dir, cfg, command)
    paths, gt = _frame_paths(cfg)
    logger.info(
        "%d images, %d conditions, imgsz %s",
        len(paths),
        len(cfg.conditions),
        list(cfg.model.imgsz),
    )

    detectors = {imgsz: detector_factory(imgsz) for imgsz in cfg.model.imgsz}
    store = DetectionStore() if cfg.manifest is not None else None
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    warmed = False
    n_rows = 0
    latencies: dict[tuple[str, int], list[float]] = defaultdict(list)
    stops: dict[tuple[str, int], int] = defaultdict(int)

    with PerFrameWriter(run_dir / "per_frame.csv") as writer:
        for index, frame in enumerate(iter_image_files(paths)):
            if not warmed and frame.status is FrameStatus.OK:
                warm_up(detectors, frame.image, cfg.warmup)
                warmed = True
            for imgsz, detector in detectors.items():
                for condition in cfg.conditions:
                    out = process_frame(
                        frame,
                        condition,
                        detector,
                        target_class=cfg.decision.target_class,
                        conf_threshold=cfg.decision.conf_threshold,
                        noise_key=frame.frame_id,
                        noise_seed=cfg.noise_seed,
                    )
                    writer.write(
                        {
                            "frame_id": frame.frame_id,
                            "condition": condition.name,
                            "imgsz": imgsz,
                            "gt_state": gt.get(frame.frame_id, ""),
                            **out.row,
                        }
                    )
                    n_rows += 1
                    key = (condition.name, imgsz)
                    stops[key] += out.decision.state is State.STOP
                    result = out.result
                    if (
                        result is not None
                        and result.status is not DetectionStatus.ERROR
                    ):
                        latencies[key].append(result.latency_ms)
                        if store is not None:
                            store.add(*key, int(frame.frame_id), result.detections)
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (index + 1) % 100 == 0:
                logger.info("%d / %d images", index + 1, len(paths))

    if store is not None:
        store.save(run_dir)
    write_resources(run_dir, peak_rss, time.perf_counter() - start, n_rows)
    for (name, imgsz), values in sorted(latencies.items(), key=lambda kv: kv[0][1]):
        logger.info(
            "%-14s imgsz=%-4d n=%-4d STOP=%-4d mean latency %.1f ms",
            name,
            imgsz,
            len(values),
            stops[(name, imgsz)],
            float(np.mean(values)),
        )
    return run_dir
