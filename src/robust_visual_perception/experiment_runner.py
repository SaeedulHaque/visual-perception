"""Run one image experiment: frames x conditions x input sizes -> per_frame.csv."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from robust_visual_perception.coco_subset import read_manifest
from robust_visual_perception.config import ExperimentConfig
from robust_visual_perception.decision import State, baseline_decision
from robust_visual_perception.degradation import apply
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
    write_run_metadata,
)

logger = logging.getLogger(__name__)

DetectorFactory = Callable[[int], Detector]


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


def _safe_detect(detector: Detector, image: np.ndarray) -> DetectionResult:
    """A detector should report failures itself; this guards against one that
    raises anyway, so one bad frame cannot end the run."""
    try:
        return detector.detect(image)
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR row
        logger.exception("detector raised")
        hw = (int(image.shape[0]), int(image.shape[1]))
        return DetectionResult(DetectionStatus.ERROR, [], 0.0, hw, repr(exc))


def _warm_up(detector: Detector, frames: list[Frame], n: int) -> None:
    first = next((f for f in frames if f.status is FrameStatus.OK), None)
    if first is None:
        return
    for _ in range(n):
        _safe_detect(detector, first.image)


def run_image_experiment(
    cfg: ExperimentConfig,
    detector_factory: DetectorFactory,
    command: str,
    overwrite: bool = False,
) -> Path:
    """Run every (imgsz, frame, condition) and write ``per_frame.csv``.

    Returns the run directory. Missing frames and detector errors become
    ``UNKNOWN`` rows; they never stop the run.
    """
    run_dir = prepare_run_dir(cfg.output_root, cfg.run_name, overwrite)
    write_run_metadata(run_dir, cfg, command)
    paths, gt = _frame_paths(cfg)
    frames = list(iter_image_files(paths))
    n_missing = sum(f.status is FrameStatus.MISSING for f in frames)
    logger.info(
        "%d frames (%d missing), %d conditions, imgsz %s",
        len(frames),
        n_missing,
        len(cfg.conditions),
        list(cfg.model.imgsz),
    )

    target = cfg.decision.target_class
    threshold = cfg.decision.conf_threshold
    latencies: dict[tuple[str, int], list[float]] = defaultdict(list)
    stops: dict[tuple[str, int], int] = defaultdict(int)

    with PerFrameWriter(run_dir / "per_frame.csv") as writer:
        for imgsz in cfg.model.imgsz:
            detector = detector_factory(imgsz)
            _warm_up(detector, frames, cfg.warmup)
            for frame in frames:
                for condition in cfg.conditions:
                    row = {
                        "frame_id": frame.frame_id,
                        "condition": condition.name,
                        "imgsz": imgsz,
                        "gt_state": gt.get(frame.frame_id, ""),
                    }
                    if frame.status is FrameStatus.MISSING:
                        decision = baseline_decision(None)
                        row.update(status=FrameStatus.MISSING.value)
                    else:
                        image = apply(
                            condition, frame.image, frame.frame_id, cfg.noise_seed
                        )
                        result = _safe_detect(detector, image)
                        decision = baseline_decision(result, target, threshold)
                        target_confs = [
                            d.confidence
                            for d in result.detections
                            if d.class_name == target
                        ]
                        row.update(
                            status=result.status.value,
                            n_detections=len(result.detections),
                            n_target=sum(c >= threshold for c in target_confs),
                            max_target_conf=(
                                f"{max(target_confs):.4f}" if target_confs else ""
                            ),
                            latency_ms=f"{result.latency_ms:.3f}",
                        )
                        if result.status is not DetectionStatus.ERROR:
                            latencies[(condition.name, imgsz)].append(result.latency_ms)
                    row.update(
                        baseline_state=decision.state.value,
                        state_reason=decision.reason or "",
                    )
                    stops[(condition.name, imgsz)] += decision.state is State.STOP
                    writer.write(row)

    for (name, imgsz), values in latencies.items():
        logger.info(
            "%-14s imgsz=%-4d n=%-4d STOP=%-4d mean latency %.1f ms",
            name,
            imgsz,
            len(values),
            stops[(name, imgsz)],
            float(np.mean(values)),
        )
    return run_dir
