import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from robust_visual_perception.sim_geometry import ActorPath
from robust_visual_perception.sim_world import ROBOT_NAME, WORLD_NAME, world_sdf

ROOT = Path(__file__).resolve().parents[1]
PATH = ActorPath(corners=((3.0, -3.0), (3.0, 3.0)), speed=1.0, turn_s=0.5)


def test_world_has_the_systems_the_simulation_needs():
    world = ET.fromstring(world_sdf(PATH, 0.25)).find("world")
    assert world.get("name") == WORLD_NAME
    plugins = {p.get("filename") for p in world.findall("plugin")}
    assert {
        "gz-sim-physics-system",
        "gz-sim-user-commands-system",
        "gz-sim-scene-broadcaster-system",
        "gz-sim-sensors-system",
        "gz-sim-imu-system",
    } <= plugins


def test_actor_waypoints_match_the_path():
    world = ET.fromstring(world_sdf(PATH, 0.25)).find("world")
    trajectory = world.find("actor/script/trajectory")
    points = trajectory.findall("waypoint")
    expected = PATH.waypoints(0.25)
    assert len(points) == len(expected)
    for element, (t, x, y, z, yaw) in zip(points, expected, strict=True):
        assert float(element.findtext("time")) == pytest.approx(t, abs=1e-6)
        pose = [float(v) for v in element.findtext("pose").split()]
        assert pose == pytest.approx([x, y, z, 0.0, 0.0, yaw], abs=1e-6)
    assert world.findtext("actor/script/loop") == "true"
    assert world.findtext("actor/script/auto_start") == "true"


def test_world_includes_the_robot_at_its_start_pose():
    world = ET.fromstring(world_sdf(PATH, 0.25, (1.0, -2.0, 0.5))).find("world")
    (robot,) = world.findall("include")
    assert robot.findtext("uri") == "model://turtlebot3_waffle_pi"
    assert robot.findtext("name") == ROBOT_NAME
    pose = [float(v) for v in robot.findtext("pose").split()]
    assert pose == pytest.approx([1.0, -2.0, 0.01, 0.0, 0.0, 0.5])
    (plugin,) = robot.findall("plugin")
    assert plugin.get("filename") == "gz-sim-pose-publisher-system"
    assert plugin.findtext("publish_model_pose") == "true"
    assert plugin.findtext("use_pose_vector_msg") == "false"


def test_committed_world_matches_the_config():
    config = yaml.safe_load((ROOT / "configs/sim/world.yaml").read_text())
    path = ActorPath.from_config(config["actor"])
    committed = ROOT / "ros2_ws/src/rvp_ros/worlds/inspection.sdf"
    robot = config["robot"]
    expected = world_sdf(
        path, config["waypoint_dt"], (robot["x"], robot["y"], robot["yaw"])
    )
    assert committed.read_text() == expected
