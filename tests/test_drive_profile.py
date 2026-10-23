import pytest

from robust_visual_perception.drive_profile import (
    CommandGuard,
    DriveProfile,
    Limits,
    Segment,
    clamp,
)

PROFILE = DriveProfile(
    (Segment(2.0, 0.0, 0.0), Segment(3.0, 0.1, 0.0), Segment(1.0, 0.0, -0.3))
)


def test_command_follows_the_segments():
    assert PROFILE.total_s == 6.0
    assert PROFILE.command_at(1.0) == (0.0, 0.0)
    assert PROFILE.command_at(2.0) == (0.1, 0.0)
    assert PROFILE.command_at(4.9) == (0.1, 0.0)
    assert PROFILE.command_at(5.5) == (0.0, -0.3)


def test_command_is_zero_before_the_start_and_after_the_end():
    assert PROFILE.command_at(-1.0) == (0.0, 0.0)
    assert PROFILE.command_at(6.0) == (0.0, 0.0)
    assert PROFILE.command_at(100.0) == (0.0, 0.0)


def test_commands_above_the_limits_are_clamped():
    fast = DriveProfile((Segment(1.0, 0.5, -2.0),))
    assert fast.command_at(0.5) == (0.15, -0.5)
    assert fast.command_at(0.5, Limits(0.2, 1.0)) == (0.2, -1.0)


def test_clamp_keeps_the_sign():
    assert clamp(-0.4, 0.7, Limits()) == (-0.15, 0.5)
    assert clamp(0.05, -0.1, Limits()) == (0.05, -0.1)


def test_profile_from_yaml(tmp_path):
    path = tmp_path / "drive.yaml"
    path.write_text(
        "segments:\n"
        "  - {duration_s: 2.0, linear_x: 0.0, angular_z: 0.0}\n"
        "  - {duration_s: 3.0, linear_x: 0.1, angular_z: 0.0}\n"
        "  - {duration_s: 1.0, linear_x: 0.0, angular_z: -0.3}\n"
    )
    assert DriveProfile.load(path) == PROFILE


def test_segments_need_a_positive_duration():
    with pytest.raises(ValueError):
        Segment(0.0, 0.1, 0.0)
    with pytest.raises(ValueError):
        DriveProfile(())


def test_guard_sends_zero_before_any_command():
    assert CommandGuard().output(now=10.0) == (0.0, 0.0)


def test_guard_forwards_a_fresh_command_clamped():
    guard = CommandGuard(timeout_s=0.5)
    guard.on_command(0.3, 0.2, now=1.0)
    assert guard.output(now=1.4) == (0.15, 0.2)


def test_guard_stops_after_the_timeout():
    guard = CommandGuard(timeout_s=0.5)
    guard.on_command(0.1, 0.0, now=1.0)
    assert guard.output(now=1.5) == (0.1, 0.0)
    assert guard.output(now=1.51) == (0.0, 0.0)
    guard.on_command(0.1, 0.0, now=2.0)
    assert guard.output(now=2.1) == (0.1, 0.0)
