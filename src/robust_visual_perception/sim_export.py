"""Turn a recorded simulation run into a sequence in MOT17 layout (no ROS).

The exported folder has what the Phase 2 sequence runner reads:
``seqinfo.ini``, ``img1/000001.png ...`` and ``gt/gt.txt``, plus
``frames.csv`` (frame number, image stamp, GT box and GT-STOP). Each frame's
GT is computed from that frame's own stamp, so a frame missing from the bag
cannot shift the alignment of the others.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from robust_visual_perception.decision import Roi
from robust_visual_perception.mot17 import SeqInfo
from robust_visual_perception.sim_geometry import (
    SIM_ROI,
    WAFFLE_PI_CAMERA_OFFSET,
    ActorPath,
    CameraModel,
    PersonVolume,
    camera_pose,
    interpolate_pose,
    person_box,
)

FRAMES_COLUMNS = ["frame", "stamp_ns", "x1", "y1", "x2", "y2", "gt_stop"]


def gt_boxes(
    stamps_s: np.ndarray,
    pose_stamps_s: np.ndarray,
    poses: np.ndarray,
    camera: CameraModel,
    path: ActorPath,
    person: PersonVolume,
    camera_offset: Sequence[float] = WAFFLE_PI_CAMERA_OFFSET,
) -> list[np.ndarray | None]:
    """GT person box per image stamp (None when the person is not in view)."""
    boxes = []
    for t in np.asarray(stamps_s, float):
        robot = interpolate_pose(pose_stamps_s, poses, float(t))
        x, y, _ = path.position(float(t))
        boxes.append(
            person_box((x, y), camera_pose(robot, camera_offset), camera, person)
        )
    return boxes


def frame_rate(stamps_ns: np.ndarray) -> float:
    """Average frames per second over the recording."""
    stamps = np.asarray(stamps_ns, np.int64)
    if len(stamps) < 2:
        return float("nan")
    return (len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1e9)


def stamp_gaps(stamps_ns: np.ndarray) -> int:
    """Number of frames missing, judged from the median frame interval."""
    diffs = np.diff(np.asarray(stamps_ns, np.int64))
    if not len(diffs):
        return 0
    step = np.median(diffs)
    return int(np.sum(np.rint(diffs / step) - 1))


def write_sequence(
    seq_dir: Path,
    name: str,
    stamps_ns: np.ndarray,
    boxes: Sequence[np.ndarray | None],
    camera: CameraModel,
    roi: Roi = SIM_ROI,
) -> SeqInfo:
    """Write seqinfo.ini, gt/gt.txt and frames.csv for frames 1..N.

    Images go to ``seq_dir/img1/`` and are written by the caller.
    """
    seq_dir = Path(seq_dir)
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "img1").mkdir(exist_ok=True)
    fps = frame_rate(stamps_ns)
    info = SeqInfo(
        name=name,
        fps=round(fps, 3) if np.isfinite(fps) else 0.0,
        length=len(stamps_ns),
        width=camera.width,
        height=camera.height,
        img_dir="img1",
        img_ext=".png",
    )
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={info.name}\n"
        f"imDir={info.img_dir}\n"
        f"frameRate={info.fps}\n"
        f"seqLength={info.length}\n"
        f"imWidth={info.width}\n"
        f"imHeight={info.height}\n"
        f"imExt={info.img_ext}\n"
    )
    hw = (camera.height, camera.width)
    gt_lines = []
    with (seq_dir / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAMES_COLUMNS)
        writer.writeheader()
        for i, (stamp, box) in enumerate(zip(stamps_ns, boxes, strict=True), 1):
            row = {"frame": i, "stamp_ns": int(stamp), "gt_stop": 0}
            if box is not None:
                x1, y1, x2, y2 = (float(v) for v in box)
                row.update(
                    x1=f"{x1:.3f}", y1=f"{y1:.3f}", x2=f"{x2:.3f}", y2=f"{y2:.3f}"
                )
                row["gt_stop"] = int(roi.mask(np.array([box]), hw)[0])
                # MOT17 columns: frame, id, x, y, w, h, considered, class,
                # visibility. No occlusion in this world, so visibility is 1.
                gt_lines.append(
                    f"{i},1,{x1:.3f},{y1:.3f},{x2 - x1:.3f},{y2 - y1:.3f},1,1,1.0"
                )
            writer.writerow(row)
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    return info


def read_frames_csv(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "frame": np.array([int(r["frame"]) for r in rows], int),
        "stamp_ns": np.array([int(r["stamp_ns"]) for r in rows], np.int64),
        "gt_stop": np.array([int(r["gt_stop"]) for r in rows], int),
    }


def build_sequence(
    seq_dir: Path,
    path: ActorPath,
    person: PersonVolume,
    roi: Roi = SIM_ROI,
    camera_offset: Sequence[float] = WAFFLE_PI_CAMERA_OFFSET,
) -> SeqInfo:
    """GT files for a sequence folder holding the raw export of a bag
    (``stamps.csv``, ``poses.csv``, ``camera.json``)."""
    seq_dir = Path(seq_dir)
    stamps = np.loadtxt(
        seq_dir / "stamps.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2
    )[:, 1]
    poses = np.loadtxt(seq_dir / "poses.csv", delimiter=",", skiprows=1, ndmin=2)
    camera = CameraModel(**json.loads((seq_dir / "camera.json").read_text()))
    boxes = gt_boxes(
        stamps / 1e9, poses[:, 0], poses[:, 1:], camera, path, person, camera_offset
    )
    return write_sequence(seq_dir, seq_dir.name, stamps, boxes, camera, roi)
