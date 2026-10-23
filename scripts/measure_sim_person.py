"""Measure the simulated person's size from a static run (no detector).

    uv run python scripts/measure_sim_person.py data/raw/sim/sequences/static_r1

Background: the median of frames where the person is well outside the
camera's view. Measurement frames: the person on the far crossing, close to
the image centre, so the whole body is visible. Each silhouette is
back-projected at the person's known distance; the medians become the
person volume in configs/sim/person.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from robust_visual_perception.sim_geometry import (
    ActorPath,
    CameraModel,
    camera_pose,
    interpolate_pose,
)
from robust_visual_perception.sim_measure import silhouette_box, size_from_box


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--world", type=Path, default=Path("configs/sim/world.yaml"))
    parser.add_argument("--out", type=Path, default=Path("configs/sim/person.yaml"))
    parser.add_argument(
        "--far-x", type=float, default=10.0, help="x of the far crossing (m)"
    )
    args = parser.parse_args()

    seq = args.sequence
    path = ActorPath.from_config(yaml.safe_load(args.world.read_text())["actor"])
    camera = CameraModel(**json.loads((seq / "camera.json").read_text()))
    stamps = np.loadtxt(
        seq / "stamps.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2
    )
    poses = np.loadtxt(seq / "poses.csv", delimiter=",", skiprows=1, ndmin=2)
    half_fov = math.atan(camera.cx / camera.fx)

    background_frames, measure_frames = [], []
    cam = None
    for frame, stamp_ns in stamps:
        t = stamp_ns / 1e9
        cam = camera_pose(interpolate_pose(poses[:, 0], poses[:, 1:], t))
        x, y, _ = path.position(t)
        local = np.linalg.inv(cam) @ np.array([x, y, 0.0, 1.0])
        angle = math.atan2(local[1], local[0])
        if local[0] < 0 or abs(angle) > half_fov + math.radians(5):
            background_frames.append(frame)
        elif abs(x - args.far_x) < 1e-6 and abs(local[1]) < 1.0:
            measure_frames.append((frame, local[0]))

    def image(frame: int) -> np.ndarray:
        return cv2.imread(str(seq / "img1" / f"{frame:06d}.png"))

    step = max(1, len(background_frames) // 25)
    background = np.median(
        np.stack([image(f) for f in background_frames[::step]]), axis=0
    ).astype(np.uint8)
    camera_height = float(cam[2, 3])
    sizes = []
    for frame, distance in measure_frames:
        box = silhouette_box(image(frame), background)
        if box is not None:
            sizes.append(size_from_box(box, distance, camera_height, camera))
    if not sizes:
        raise SystemExit("no measurement frames with a silhouette")
    sizes = np.array(sizes)
    z_min, z_max, radius = np.median(sizes, axis=0)
    print(
        f"{len(background_frames[::step])} background frames, "
        f"{len(sizes)} measurement frames"
    )
    for name, column in zip(("z_min", "z_max", "half_width"), sizes.T, strict=True):
        print(
            f"{name}: median {np.median(column):.3f} m, "
            f"range {column.min():.3f} to {column.max():.3f} m"
        )

    args.out.write_text(
        "# Person volume for ground truth: a vertical cylinder at the actor's\n"
        "# position. Measured by scripts/measure_sim_person.py from\n"
        f"# {seq} ({len(sizes)} silhouettes on the far crossing,\n"
        "# background subtraction, medians); never from detector output.\n"
        + yaml.safe_dump(
            {
                "person": {
                    "z_min": round(float(z_min), 3),
                    "z_max": round(float(z_max), 3),
                    "radius": round(float(radius), 3),
                }
            },
            sort_keys=False,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
