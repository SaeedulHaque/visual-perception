"""Check the simulation ground truth against detections (a sanity check only).

    uv run python scripts/check_sim_gt.py data/raw/sim/sequences/static_r1 ...

On every ``--step``-th frame: the IoU between the GT box and the best
`person` detection (confidence >= 0.5), and whether GT-STOP and the
detection-based STOP agree. Nothing is tuned on this; it shows whether the
projected GT boxes sit where the rendered person is. Writes
results/sim/summary/gt_check.json and an overlay figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from robust_visual_perception.decision import baseline_decision
from robust_visual_perception.detector import UltralyticsDetector
from robust_visual_perception.metrics import box_iou
from robust_visual_perception.sim_geometry import SIM_ROI

GT_COLOUR = (122, 175, 27)  # aqua #1baf7a, BGR
DET_COLOUR = (52, 104, 235)  # orange #eb6834, BGR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sequences", type=Path, nargs="+")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--weights", default="models/yolo26n.pt")
    parser.add_argument("--out", type=Path, default=Path("results/sim/summary"))
    args = parser.parse_args()

    detector = UltralyticsDetector(args.weights, 416)
    ious, agree, n_checked = [], 0, 0
    candidates = {"1": [], "0": []}  # by gt_stop: (seq, frame, gt box, detection)
    for seq in args.sequences:
        with (seq / "frames.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows[:: args.step]:
            image = cv2.imread(str(seq / "img1" / f"{int(row['frame']):06d}.png"))
            result = detector.detect(image)
            stop = baseline_decision(result, "person", 0.5, SIM_ROI).state.value
            n_checked += 1
            agree += (stop == "STOP") == (row["gt_stop"] == "1")
            persons = [
                d
                for d in result.detections
                if d.class_name == "person" and d.confidence >= 0.5
            ]
            if not row["x1"] or not persons:
                continue
            gt = np.array([[float(row[k]) for k in ("x1", "y1", "x2", "y2")]])
            boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in persons])
            iou = float(box_iou(gt, boxes).max())
            ious.append(iou)
            best = boxes[int(box_iou(gt, boxes).argmax())]
            candidates[row["gt_stop"]].append((seq, int(row["frame"]), gt[0], best))

    ious = np.array(ious)
    summary = {
        "sequences": [s.name for s in args.sequences],
        "step": args.step,
        "n_checked": n_checked,
        "n_frames": len(ious),
        "iou_median": float(np.median(ious)),
        "iou_p10": float(np.percentile(ious, 10)),
        "state_agreement": agree / n_checked,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gt_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Two GT-STOP frames and two frames whose GT box is not STOP (far away or
    # at the image border), evenly spaced through the checked frames.
    picked = []
    for key in ("1", "0"):
        items = candidates[key]
        picked += [items[i] for i in np.linspace(0, len(items) - 1, 2).astype(int)]
    tiles = []
    for seq, frame, gt, det in picked:
        tile = cv2.imread(str(seq / "img1" / f"{frame:06d}.png"))
        label = f"{seq.name} frame {frame}"
        cv2.rectangle(
            tile, tuple(map(int, gt[:2])), tuple(map(int, gt[2:])), GT_COLOUR, 2
        )
        cv2.rectangle(
            tile, tuple(map(int, det[:2])), tuple(map(int, det[2:])), DET_COLOUR, 2
        )
        cv2.putText(
            tile,
            f"{label}  green: GT  orange: detection",
            (8, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if tiles:
        while len(tiles) < 4:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        figures = args.out / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(figures / "gt_check.png"), grid)


if __name__ == "__main__":
    main()
