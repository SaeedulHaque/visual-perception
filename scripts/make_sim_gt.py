"""Compute ground truth for exported simulation runs.

    uv run python scripts/make_sim_gt.py data/raw/sim/sequences/static_r1 ...

Each folder holds the raw export of one bag (stamps.csv, poses.csv,
camera.json, img1/). Writes seqinfo.ini, gt/gt.txt and frames.csv, so the
folder can be replayed with scripts/run_sequence_experiment.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from robust_visual_perception.sim_export import build_sequence, read_frames_csv
from robust_visual_perception.sim_geometry import ActorPath, PersonVolume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sequences", type=Path, nargs="+")
    parser.add_argument("--world", type=Path, default=Path("configs/sim/world.yaml"))
    parser.add_argument("--person", type=Path, default=Path("configs/sim/person.yaml"))
    args = parser.parse_args()
    path = ActorPath.from_config(yaml.safe_load(args.world.read_text())["actor"])
    person = PersonVolume(**yaml.safe_load(args.person.read_text())["person"])
    for seq in args.sequences:
        info = build_sequence(seq, path, person)
        gt_stop = read_frames_csv(seq / "frames.csv")["gt_stop"]
        print(
            f"{info.name}: {info.length} frames at {info.fps} fps, "
            f"GT-STOP in {gt_stop.sum()} frames"
        )


if __name__ == "__main__":
    main()
