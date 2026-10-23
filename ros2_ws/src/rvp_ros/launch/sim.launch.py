"""One simulation run: Gazebo (headless, starts paused), bridge, perception
node, command guard, optional scripted drive, bag recorder and run monitor.

  ros2 launch rvp_ros sim.launch.py mode:=static run_name:=static_r1

Modes: static (robot parked), scripted (drive profile), teleop (keyboard,
run teleop_twist_keyboard separately). The run monitor unpauses the world
once all nodes are ready and ends the launch after ``duration`` sim seconds;
if any node exits early, the whole run is shut down.

``gui:=true`` also starts the Gazebo GUI client, to watch a teleop run. It
needs an X display and draws on the GPU, so the measured runs leave it off.
Closing its window does not end the run.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SHARE = get_package_share_directory("rvp_ros")
TB3_MODELS = os.path.join(get_package_share_directory("turtlebot3_gazebo"), "models")
BAG_TOPICS = [
    "/camera/image_raw",
    "/camera/camera_info",
    "/rvp/robot_pose",
    "/odom",
    "/cmd_vel",
    "/rvp/cmd_vel",
    "/perception/detections",
    "/perception/safety_state",
]


def _setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    mode = arg("mode")
    if mode not in ("static", "scripted", "teleop"):
        raise RuntimeError(f"mode must be static, scripted or teleop, got {mode}")
    run_name = arg("run_name")
    out_dir = os.path.join(arg("output_root"), run_name)
    bag_dir = os.path.join(arg("bag_root"), run_name)
    os.makedirs(arg("bag_root"), exist_ok=True)
    sim_time = {"use_sim_time": True}

    gazebo = ExecuteProcess(
        cmd=[
            "gz",
            "sim",
            "-s",
            "--headless-rendering",
            "--seed",
            arg("seed"),
            "-v",
            "2",
            arg("world"),
        ],
        output="screen",
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": os.path.join(SHARE, "config", "bridge.yaml")}],
        output="screen",
    )
    perception = Node(
        package="rvp_ros",
        executable="perception_node",
        parameters=[sim_time, {"weights": arg("weights"), "output_dir": out_dir}],
        output="screen",
    )
    guard = Node(
        package="rvp_ros",
        executable="cmd_guard",
        parameters=[sim_time],
        output="screen",
    )
    # Wall-clock receive times: with --use-sim-time the recorder would wait
    # for /clock, which the paused world does not publish yet. Everything
    # downstream uses the message header stamps (sim time) instead.
    recorder = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--storage-preset-profile",
            "zstd_fast",
            "-o",
            bag_dir,
            "--topics",
            *BAG_TOPICS,
        ],
        output="screen",
    )
    monitor = Node(
        package="rvp_ros",
        executable="run_monitor",
        parameters=[
            {
                "duration_s": float(arg("duration")),
                "camera_subscribers": 2,
                "wait_for_command": mode == "scripted",
                "output_dir": out_dir,
            }
        ],
        output="screen",
    )
    actions = [gazebo, bridge, perception, guard, recorder, monitor]
    if mode == "scripted":
        actions.append(
            Node(
                package="rvp_ros",
                executable="scripted_drive",
                parameters=[sim_time, {"profile": arg("profile")}],
                output="screen",
            )
        )

    # The monitor ends the run; any other process ending early ends it too.
    for process in [p for p in actions if p is not recorder]:
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=process,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason=f"{process.name} exited"))
                    ],
                )
            )
        )
    # Added after the exit handlers: closing the window leaves the run going.
    if arg("gui") == "true":
        actions.append(ExecuteProcess(cmd=["gz", "sim", "-g", "-v", "2"]))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="static"),
            DeclareLaunchArgument("run_name", default_value="static_r1"),
            DeclareLaunchArgument("duration", default_value="120.0"),
            DeclareLaunchArgument("seed", default_value="7"),
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("output_root", default_value="/work/results/sim"),
            DeclareLaunchArgument("bag_root", default_value="/work/data/raw/sim/bags"),
            DeclareLaunchArgument("weights", default_value="/work/models/yolo26n.pt"),
            DeclareLaunchArgument(
                "world", default_value=os.path.join(SHARE, "worlds", "inspection.sdf")
            ),
            DeclareLaunchArgument(
                "profile",
                default_value=os.path.join(SHARE, "config", "drive_profile.yaml"),
            ),
            AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", TB3_MODELS),
            OpaqueFunction(function=_setup),
        ]
    )
