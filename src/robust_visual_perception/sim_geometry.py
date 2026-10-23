"""Ground-truth geometry for the Gazebo simulation (no ROS needed).

The walking person follows a scripted loop (``ActorPath``), and the robot's
true pose comes from the simulator. Projecting a simple person volume through
the camera model gives the GT box for any frame, and the same ROI rule as for
detections turns it into GT-STOP.

Frames follow Gazebo: x forward, y left, z up, for the world and for the
camera link (the camera looks along its +x axis).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robust_visual_perception.decision import Roi

# Design x band and minimum height. No vertical band: the camera sits about
# 0.1 m above the ground, so a person's box centre is always in the upper part
# of the image and the design's y >= 0.30 band would never fire.
SIM_ROI = Roi(0.30, 0.70, 0.0, 1.0, 0.25)

# camera_rgb_frame relative to the TurtleBot3 Waffle Pi model frame
# (turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf).
WAFFLE_PI_CAMERA_OFFSET = (0.076, 0.0, 0.093)


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class PersonVolume:
    """Vertical cylinder standing at the actor's position (world z range)."""

    z_min: float
    z_max: float
    radius: float


def pose_matrix(
    translation: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = [float(v) for v in translation]
    return T


def camera_pose(
    robot_T: np.ndarray, offset: Sequence[float] = WAFFLE_PI_CAMERA_OFFSET
) -> np.ndarray:
    """Camera pose in the world: the robot pose plus the fixed mount offset."""
    mount = np.eye(4)
    mount[:3, 3] = [float(v) for v in offset]
    return robot_T @ mount


def interpolate_pose(stamps: np.ndarray, poses: np.ndarray, t: float) -> np.ndarray:
    """Pose at time ``t`` from samples ``poses`` (N x 7: x, y, z, qx, qy, qz, qw).

    Linear in position, normalised linear in orientation (fine for the small
    steps between simulator pose messages). Outside the samples, the nearest
    sample is used.
    """
    stamps = np.asarray(stamps, float)
    poses = np.asarray(poses, float)
    if t <= stamps[0]:
        return pose_matrix(poses[0, :3], poses[0, 3:])
    if t >= stamps[-1]:
        return pose_matrix(poses[-1, :3], poses[-1, 3:])
    i = int(np.searchsorted(stamps, t, side="right")) - 1
    a = (t - stamps[i]) / (stamps[i + 1] - stamps[i])
    q0, q1 = poses[i, 3:], poses[i + 1, 3:]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    position = (1 - a) * poses[i, :3] + a * poses[i + 1, :3]
    return pose_matrix(position, (1 - a) * q0 + a * q1)


def person_box(
    person_xy: Sequence[float],
    camera_T: np.ndarray,
    camera: CameraModel,
    person: PersonVolume,
    n_points: int = 16,
    near: float = 0.05,
) -> np.ndarray | None:
    """Clipped image box (x1, y1, x2, y2) of the person, or None.

    None when any point of the volume is behind the camera (or closer than
    ``near``), or when nothing of the box is inside the image.
    """
    angles = np.arange(n_points) * 2 * math.pi / n_points
    px = person_xy[0] + person.radius * np.cos(angles)
    py = person_xy[1] + person.radius * np.sin(angles)
    points = np.concatenate(
        [
            np.stack([px, py, np.full(n_points, z), np.ones(n_points)], axis=1)
            for z in (person.z_min, person.z_max)
        ]
    )
    local = (np.linalg.inv(camera_T) @ points.T).T
    if np.any(local[:, 0] <= near):
        return None
    u = camera.cx - camera.fx * local[:, 1] / local[:, 0]
    v = camera.cy - camera.fy * local[:, 2] / local[:, 0]
    x1, x2 = np.clip([u.min(), u.max()], 0, camera.width)
    y1, y2 = np.clip([v.min(), v.max()], 0, camera.height)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], float)


def _wrap(angle: float) -> float:
    """Angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class _Phase:
    start: float
    duration: float
    xy0: tuple[float, float]
    xy1: tuple[float, float]
    yaw0: float
    yaw1: float  # unwrapped end yaw, so interpolation turns the short way


@dataclass(frozen=True)
class ActorPath:
    """Closed loop through ``corners``: walk each leg at ``speed`` (m/s), then
    turn on the spot for ``turn_s`` seconds to face the next leg.

    ``z`` is the actor's pose height (the mesh origin, not the feet).
    """

    corners: tuple[tuple[float, float], ...]
    speed: float
    turn_s: float
    z: float = 1.0

    def __post_init__(self) -> None:
        if len(self.corners) < 2:
            raise ValueError("an actor path needs at least two corners")
        if self.speed <= 0 or self.turn_s < 0:
            raise ValueError("speed must be > 0 and turn_s >= 0")
        for a, b in self._legs():
            if math.dist(a, b) == 0:
                raise ValueError(f"consecutive corners must differ, got {a}")

    @classmethod
    def from_config(cls, data: dict) -> ActorPath:
        return cls(
            corners=tuple((float(x), float(y)) for x, y in data["corners"]),
            speed=float(data["speed"]),
            turn_s=float(data["turn_s"]),
            z=float(data.get("z", 1.0)),
        )

    def _legs(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        n = len(self.corners)
        return [(self.corners[i], self.corners[(i + 1) % n]) for i in range(n)]

    def _phases(self) -> list[_Phase]:
        legs = self._legs()
        headings = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in legs]
        phases, t = [], 0.0
        for i, (a, b) in enumerate(legs):
            walk = math.dist(a, b) / self.speed
            phases.append(_Phase(t, walk, a, b, headings[i], headings[i]))
            t += walk
            turn = _wrap(headings[(i + 1) % len(legs)] - headings[i])
            phases.append(_Phase(t, self.turn_s, b, b, headings[i], headings[i] + turn))
            t += self.turn_s
        return phases

    @property
    def period(self) -> float:
        last = self._phases()[-1]
        return last.start + last.duration

    def position(self, t: float) -> tuple[float, float, float]:
        """(x, y, yaw) at time ``t`` (the loop repeats with ``period``)."""
        t = t % self.period
        for phase in self._phases():
            if phase.duration > 0 and t < phase.start + phase.duration:
                a = (t - phase.start) / phase.duration
                return (
                    phase.xy0[0] + a * (phase.xy1[0] - phase.xy0[0]),
                    phase.xy0[1] + a * (phase.xy1[1] - phase.xy0[1]),
                    _wrap(phase.yaw0 + a * (phase.yaw1 - phase.yaw0)),
                )
        first = self._phases()[0]
        return (*first.xy0, _wrap(first.yaw0))

    def waypoints(self, dt: float) -> list[tuple[float, float, float, float, float]]:
        """(t, x, y, z, yaw) at most ``dt`` apart, evenly spaced within each
        walk or turn, with every corner time included; the last waypoint
        closes the loop at ``period``."""
        points = []
        for phase in self._phases():
            if phase.duration <= 0:
                continue
            steps = max(1, math.ceil(phase.duration / dt - 1e-9))
            for k in range(steps):
                t = phase.start + k * phase.duration / steps
                x, y, yaw = self.position(t)
                points.append((t, x, y, self.z, yaw))
        period = self.period
        x, y, yaw = self.position(0.0)
        points.append((period, x, y, self.z, yaw))
        return points
