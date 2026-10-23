"""Scripted drive profile and command guard for the simulated robot (no ROS).

The profile is a list of ``{duration_s, linear_x, angular_z}`` segments,
replayed against sim time, so the trajectory does not depend on how fast the
simulation runs. The guard sits between any command source (the profile or
keyboard teleop) and the robot: it clamps speeds and sends zero when commands
stop arriving. Neither reads perception output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Limits:
    max_linear: float = 0.15  # m/s
    max_angular: float = 0.5  # rad/s


def clamp(linear: float, angular: float, limits: Limits) -> tuple[float, float]:
    return (
        max(-limits.max_linear, min(limits.max_linear, linear)),
        max(-limits.max_angular, min(limits.max_angular, angular)),
    )


@dataclass(frozen=True)
class Segment:
    duration_s: float
    linear_x: float
    angular_z: float

    def __post_init__(self) -> None:
        if not self.duration_s > 0:
            raise ValueError(f"segment duration must be > 0, got {self.duration_s}")


@dataclass(frozen=True)
class DriveProfile:
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a drive profile needs at least one segment")

    @classmethod
    def load(cls, path: Path) -> DriveProfile:
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            tuple(
                Segment(
                    float(s["duration_s"]),
                    float(s["linear_x"]),
                    float(s["angular_z"]),
                )
                for s in data["segments"]
            )
        )

    @property
    def total_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    def command_at(self, t: float, limits: Limits | None = None) -> tuple[float, float]:
        """(linear, angular) at ``t`` seconds after the start; zero outside
        the profile."""
        start = 0.0
        for segment in self.segments:
            if start <= t < start + segment.duration_s:
                return clamp(segment.linear_x, segment.angular_z, limits or Limits())
            start += segment.duration_s
        return 0.0, 0.0


class CommandGuard:
    """Clamp incoming commands; output zero once none arrived for ``timeout_s``."""

    def __init__(self, timeout_s: float = 0.5, limits: Limits | None = None) -> None:
        self.timeout_s = timeout_s
        self.limits = limits or Limits()
        self._last: tuple[float, float] | None = None
        self._last_time = 0.0

    def on_command(self, linear: float, angular: float, now: float) -> None:
        self._last = clamp(linear, angular, self.limits)
        self._last_time = now

    def output(self, now: float) -> tuple[float, float]:
        if self._last is None or now - self._last_time > self.timeout_s:
            return 0.0, 0.0
        return self._last
