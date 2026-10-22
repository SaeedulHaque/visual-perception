"""Run the sequence experiment: MOT17 frames x conditions -> per_frame.csv.

For every frame and condition: degrade, detect, decide with the corridor ROI,
then pass the decision through that condition's temporal filter. Ground truth
comes from the GT-STOP rule on the sequence's gt.txt.
"""

from __future__ import annotations

import logging
import time
import zlib
from pathlib import Path

import numpy as np
import psutil

from robust_visual_perception.config import FaultConfig, SequenceExperimentConfig
from robust_visual_perception.decision import State
from robust_visual_perception.experiment_runner import (
    DetectorFactory,
    process_frame,
    warm_up,
)
from robust_visual_perception.image_source import FrameStatus, iter_image_files
from robust_visual_perception.mot17 import gt_stop_states, parse_gt, parse_seqinfo
from robust_visual_perception.reporting import (
    PerFrameWriter,
    prepare_run_dir,
    write_resources,
    write_run_metadata,
)
from robust_visual_perception.temporal_filter import PersistenceFilter

logger = logging.getLogger(__name__)


def fault_for(key: str, fault: FaultConfig) -> str | None:
    """Injected fault for one frame (``"MISSING"``, ``"ERROR"`` or ``None``).

    Keyed by the frame, not by the condition, so every condition loses the
    same frames.
    """
    if fault.missing_rate <= 0 and fault.error_rate <= 0:
        return None
    u = np.random.default_rng(fault.seed + zlib.crc32(key.encode())).random()
    if u < fault.missing_rate:
        return "MISSING"
    if u < fault.missing_rate + fault.error_rate:
        return "ERROR"
    return None


def run_sequence_experiment(
    cfg: SequenceExperimentConfig,
    detector_factory: DetectorFactory,
    command: str,
    overwrite: bool = False,
) -> Path:
    start = time.perf_counter()
    run_dir = prepare_run_dir(cfg.output_root, cfg.run_name, overwrite)
    write_run_metadata(run_dir, cfg, command)
    (imgsz,) = cfg.model.imgsz
    detector = detector_factory(imgsz)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    warmed = False
    n_rows = 0

    with PerFrameWriter(run_dir / "per_frame.csv") as writer:
        for name in cfg.sequences:
            seq_dir = Path(cfg.data_root) / name
            info = parse_seqinfo((seq_dir / "seqinfo.ini").read_text())
            gt = parse_gt((seq_dir / "gt" / "gt.txt").read_text())
            gt_states = gt_stop_states(gt, info, cfg.roi)
            # One path per frame number, so a missing file stays in place.
            paths = [
                seq_dir / info.img_dir / f"{i:06d}{info.img_ext}"
                for i in range(1, info.length + 1)
            ]
            filters = {
                c.name: PersistenceFilter(
                    cfg.filter.n_on, cfg.filter.k_window, cfg.filter.m_off
                )
                for c in cfg.conditions
            }
            logger.info("%s: %d frames at %.0f fps", name, info.length, info.fps)
            for index, frame in enumerate(iter_image_files(paths, fps=info.fps)):
                if not warmed and frame.status is FrameStatus.OK:
                    warm_up({imgsz: detector}, frame.image, cfg.warmup)
                    warmed = True
                key = f"{name}/{frame.frame_id}"
                injected = fault_for(key, cfg.fault)
                gt_state = State.STOP if gt_states[index] else State.CLEAR
                for condition in cfg.conditions:
                    out = process_frame(
                        frame,
                        condition,
                        detector,
                        target_class=cfg.decision.target_class,
                        conf_threshold=cfg.decision.conf_threshold,
                        noise_key=key,
                        noise_seed=cfg.noise_seed,
                        roi=cfg.roi,
                        injected=injected,
                    )
                    filtered = filters[condition.name].update(out.decision.state)
                    writer.write(
                        {
                            "sequence": name,
                            "frame_id": frame.frame_id,
                            "condition": condition.name,
                            "imgsz": imgsz,
                            "gt_state": gt_state.value,
                            "filtered_state": filtered.value,
                            **out.row,
                        }
                    )
                    n_rows += 1
                peak_rss = max(peak_rss, process.memory_info().rss)
                if (index + 1) % 100 == 0:
                    logger.info("%s: %d / %d frames", name, index + 1, info.length)

    write_resources(run_dir, peak_rss, time.perf_counter() - start, n_rows)
    return run_dir
