"""Run monitor: start the paused simulation once every node is ready, and
end the run after ``duration_s`` of sim time.

Ready means: the camera topic has at least ``camera_subscribers`` subscribers
(the perception node subscribes only after its detector warm-up, and the bag
recorder) and, when ``wait_for_command`` is set, /rvp/cmd_vel has a
publisher. Then the world is unpaused, so every run starts from sim time 0
with the same state. When the sim clock passes ``duration_s`` the monitor
writes run_info.json and exits; the launch file then shuts everything down.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock as ClockMsg


class RunMonitor(Node):
    def __init__(self) -> None:
        super().__init__("run_monitor")
        p = self.declare_parameters(
            "",
            [
                ("duration_s", 120.0),
                ("world", "inspection"),
                ("camera_topic", "/camera/image_raw"),
                ("camera_subscribers", 2),
                ("wait_for_command", False),
                ("ready_timeout_s", 180.0),
                ("output_dir", ""),
            ],
        )
        self.cfg = {param.name: param.value for param in p}
        self.sim_time = 0.0
        self.create_subscription(ClockMsg, "/clock", self._on_clock, 10)

    def _on_clock(self, msg: ClockMsg) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9

    def _ready(self) -> bool:
        topic = self.cfg["camera_topic"]
        if self.count_subscribers(topic) < self.cfg["camera_subscribers"]:
            return False
        if self.count_publishers(topic) < 1:
            return False
        return not self.cfg["wait_for_command"] or (
            self.count_publishers("/rvp/cmd_vel") >= 1
        )

    def _unpause(self) -> None:
        subprocess.run(
            [
                "gz",
                "service",
                "-s",
                f"/world/{self.cfg['world']}/control",
                "--reqtype",
                "gz.msgs.WorldControl",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "5000",
                "--req",
                "pause: false",
            ],
            check=True,
            capture_output=True,
        )

    def run(self) -> int:
        info = {"completed": False, "duration_s": self.cfg["duration_s"]}
        start = time.monotonic()
        while not self._ready():
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.monotonic() - start > self.cfg["ready_timeout_s"]:
                self.get_logger().error("nodes not ready in time; giving up")
                self._write(info)
                return 1
        info["ready_wait_s"] = round(time.monotonic() - start, 3)
        self.get_logger().info("all nodes ready; starting the simulation")
        self._unpause()
        wall_start = time.monotonic()
        sim_start = self.sim_time
        while self.sim_time < self.cfg["duration_s"]:
            rclpy.spin_once(self, timeout_sec=0.2)
        wall = time.monotonic() - wall_start
        info.update(
            completed=True,
            wall_s=round(wall, 3),
            sim_s=round(self.sim_time - sim_start, 3),
            real_time_factor=round((self.sim_time - sim_start) / wall, 4),
        )
        self._write(info)
        self.get_logger().info(f"run finished: {info}")
        return 0

    def _write(self, info: dict) -> None:
        if self.cfg["output_dir"]:
            out = Path(self.cfg["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "run_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    rclpy.init()
    node = RunMonitor()
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 1
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
