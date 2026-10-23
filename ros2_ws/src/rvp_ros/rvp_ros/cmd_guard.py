"""Command guard between any command source and the robot.

/rvp/cmd_vel (scripted drive or keyboard teleop) -> /cmd_vel (robot).
Clamps speeds and sends zero when no command has arrived for
``timeout_s`` on the steady clock, so the robot stops if the source dies.
Sends zero on shutdown.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from robust_visual_perception.drive_profile import CommandGuard, Limits


class CmdGuard(Node):
    def __init__(self) -> None:
        super().__init__("cmd_guard")
        self.declare_parameter("timeout_s", 0.5)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("max_linear", 0.15)
        self.declare_parameter("max_angular", 0.5)
        self.guard = CommandGuard(
            self.get_parameter("timeout_s").value,
            Limits(
                self.get_parameter("max_linear").value,
                self.get_parameter("max_angular").value,
            ),
        )
        self.steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.create_subscription(TwistStamped, "/rvp/cmd_vel", self._on_command, 10)
        self.create_timer(
            1.0 / self.get_parameter("rate_hz").value, self._tick, clock=self.steady
        )

    def _now(self) -> float:
        return self.steady.now().nanoseconds / 1e9

    def _on_command(self, msg: TwistStamped) -> None:
        self.guard.on_command(msg.twist.linear.x, msg.twist.angular.z, self._now())

    def _tick(self) -> None:
        self.publish(*self.guard.output(self._now()))

    def publish(self, linear: float, angular: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = CmdGuard()
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
