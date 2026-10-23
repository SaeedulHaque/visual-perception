"""Scripted drive: replay a fixed velocity profile against sim time.

Publishes TwistStamped on /rvp/cmd_vel (through cmd_guard to the robot).
The command at sim time t comes only from the profile, never from
perception output. Speeds are clamped; after the profile ends and on
shutdown, zero is sent.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from robust_visual_perception.drive_profile import DriveProfile, Limits


class ScriptedDrive(Node):
    def __init__(self) -> None:
        super().__init__("scripted_drive")
        self.declare_parameter("profile", "")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("max_linear", 0.15)
        self.declare_parameter("max_angular", 0.5)
        path = self.get_parameter("profile").value
        self.profile = DriveProfile.load(path)
        self.limits = Limits(
            self.get_parameter("max_linear").value,
            self.get_parameter("max_angular").value,
        )
        self.pub = self.create_publisher(TwistStamped, "/rvp/cmd_vel", 10)
        # Node clock follows /clock (use_sim_time), so the profile is keyed
        # to sim time and the timer does not fire while the sim is paused.
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._tick)
        self.get_logger().info(
            f"profile {path}: {len(self.profile.segments)} segments, "
            f"{self.profile.total_s:.1f} s"
        )

    def _tick(self) -> None:
        t = self.get_clock().now().nanoseconds / 1e9
        self.publish(*self.profile.command_at(t, self.limits))

    def publish(self, linear: float, angular: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ScriptedDrive()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            for _ in range(3):
                node.publish(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
