"""Generate the Gazebo world from configs/sim/world.yaml.

    uv run python scripts/make_sim_world.py

Writes ros2_ws/src/rvp_ros/worlds/inspection.sdf. A test checks that the
committed world matches the config, so re-run this after editing the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from robust_visual_perception.sim_geometry import ActorPath
from robust_visual_perception.sim_world import world_sdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/sim/world.yaml"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ros2_ws/src/rvp_ros/worlds/inspection.sdf"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    path = ActorPath.from_config(config["actor"])
    robot = config["robot"]
    pose = (robot["x"], robot["y"], robot["yaw"])
    args.out.write_text(world_sdf(path, config["waypoint_dt"], pose))
    print(f"wrote {args.out} (loop period {path.period:.1f} s)")


if __name__ == "__main__":
    main()
