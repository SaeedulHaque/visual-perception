"""Render a simulation run as an annotated video, and optionally a still sheet.

  uv run python scripts/render_sim_video.py --run teleop_r1
  uv run python scripts/render_sim_video.py --run teleop_r1 \
      --sheet results/sim/summary/figures/frames_teleop_r1.png

Needs the run's exported sequence (frames, frames.csv from make_sim_gt.py,
detections.csv from the bag export) and the node's live_frames.csv. The
video goes to data/raw/sim/videos/<run>.mp4 (git-ignored).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from robust_visual_perception.decision import Roi
from robust_visual_perception.sim_video import (
    draw_overlay,
    read_detections_csv,
    read_gt_boxes,
)


def frames_of(seq_dir: Path, results_dir: Path):
    """Annotated frames in order, with their stamps and GT-STOP flags."""
    params = json.loads((results_dir / "node_params.json").read_text())
    roi = Roi(*params["roi"])
    gt_boxes = read_gt_boxes(seq_dir / "frames.csv")
    with (seq_dir / "frames.csv").open(newline="") as f:
        gt_stop = {int(r["stamp_ns"]): r["gt_stop"] == "1" for r in csv.DictReader(f)}
    with (seq_dir / "stamps.csv").open(newline="") as f:
        stamps = [int(r["stamp_ns"]) for r in csv.DictReader(f)]
    with (results_dir / "live_frames.csv").open(newline="") as f:
        live = {int(r["stamp_ns"]): r for r in csv.DictReader(f)}
    detections_path = seq_dir / "detections.csv"
    if not detections_path.exists():
        raise SystemExit(f"{detections_path} missing: re-export the bag")
    detections = read_detections_csv(detections_path)

    baseline = filtered = "UNKNOWN:waiting"
    for i, stamp in enumerate(stamps, 1):
        row = live.get(stamp)
        if row is not None:
            baseline, filtered = row["baseline_state"], row["filtered_state"]
        image = cv2.imread(str(seq_dir / "img1" / f"{i:06d}.png"))
        yield (
            stamp,
            gt_stop[stamp],
            draw_overlay(
                image,
                t_s=stamp / 1e9,
                gt_box=gt_boxes[stamp],
                detections=detections.get(stamp, []),
                roi=roi,
                target_class=params["target_class"],
                conf_threshold=params["conf_threshold"],
                baseline=baseline,
                filtered=filtered,
                gt_stop=gt_stop[stamp],
                processed=row is not None,
            ),
        )


def sheet(frames: list[np.ndarray], columns: int = 3) -> np.ndarray:
    """Frames at half size in a grid."""
    small = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)) for f in frames]
    while len(small) % columns:
        small.append(np.zeros_like(small[0]))
    rows = [np.hstack(small[i : i + columns]) for i in range(0, len(small), columns)]
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--sequences-root", type=Path, default=Path("data/raw/sim/sequences")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/sim"))
    parser.add_argument("--out", type=Path, help="video path (.mp4)")
    parser.add_argument("--sheet", type=Path, help="also write a still sheet (.png)")
    parser.add_argument("--sheet-frames", type=int, default=6)
    args = parser.parse_args()

    seq_dir = args.sequences_root / args.run
    out = args.out or Path("data/raw/sim/videos") / f"{args.run}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = json.loads((seq_dir / "export.json").read_text())["fps"]
    n_frames = json.loads((seq_dir / "export.json").read_text())["frames"]
    # Sheet frames: frames the node processed, evenly spaced over the run,
    # half of them GT-STOP when there are any, so the sheet shows both states.
    picks: set[int] = set()
    if args.sheet:
        with (args.results_root / args.run / "live_frames.csv").open(newline="") as f:
            processed = {int(r["stamp_ns"]) for r in csv.DictReader(f)}
        with (seq_dir / "frames.csv").open(newline="") as f:
            rows = [
                (i, r["gt_stop"] == "1")
                for i, r in enumerate(csv.DictReader(f))
                if int(r["stamp_ns"]) in processed
            ]
        stop = [i for i, is_stop in rows if is_stop]
        clear = [i for i, is_stop in rows if not is_stop]
        half = args.sheet_frames // 2
        for group, count in ((stop, half), (clear, args.sheet_frames - half)):
            if group:
                idx = np.linspace(0, len(group) - 1, count + 2)[1:-1]
                picks.update(group[round(j)] for j in idx)

    writer = None
    stills = []
    for i, (_, _, frame) in enumerate(frames_of(seq_dir, args.results_root / args.run)):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        writer.write(frame)
        if i in picks:
            stills.append(frame)
    writer.release()
    print(f"{out}: {n_frames} frames at {fps} fps")
    if args.sheet:
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.sheet), sheet(stills))
        print(f"{args.sheet}: {len(stills)} frames")


if __name__ == "__main__":
    main()
