import math

import numpy as np
import pytest

from robust_visual_perception.decision import Roi
from robust_visual_perception.sim_geometry import (
    SIM_ROI,
    ActorPath,
    CameraModel,
    PersonVolume,
    camera_pose,
    interpolate_pose,
    person_box,
    pose_matrix,
)

CAMERA = CameraModel(fx=530.47, fy=530.47, cx=320.0, cy=240.0, width=640, height=480)
PERSON = PersonVolume(z_min=0.0, z_max=1.8, radius=0.25)


def camera_at(x=0.0, y=0.0, z=0.1, yaw=0.0):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return pose_matrix((x, y, z), q)


# --- projection -------------------------------------------------------------


def test_person_straight_ahead_projects_to_a_centred_box():
    box = person_box((3.0, 0.0), camera_at(), CAMERA, PERSON)
    x1, y1, x2, y2 = box
    assert (x1 + x2) / 2 == pytest.approx(320.0, abs=1e-6)
    # Head above the image top (clipped); feet: nearest point of the base
    # circle at x = 2.75 m, 0.1 m below the camera.
    assert y1 == 0.0
    assert y2 == pytest.approx(240 + 530.47 * 0.1 / 2.75, abs=1e-6)
    # Width: the tangent points of the circle, at most r / sqrt(d^2 - r^2).
    assert x2 - x1 <= 2 * 530.47 * 0.25 / math.sqrt(3.0**2 - 0.25**2) + 1e-6
    assert x2 - x1 > 2 * 530.47 * 0.25 / 3.0 - 1.0


def test_far_person_is_fully_visible_and_short():
    x1, y1, x2, y2 = person_box((10.0, 0.0), camera_at(), CAMERA, PERSON)
    assert y1 == pytest.approx(240 - 530.47 * 1.7 / 9.75, abs=1e-6)
    assert y2 == pytest.approx(240 + 530.47 * 0.1 / 9.75, abs=1e-6)


def test_person_behind_the_camera_has_no_box():
    assert person_box((-3.0, 0.0), camera_at(), CAMERA, PERSON) is None


def test_person_partly_behind_the_camera_has_no_box():
    assert person_box((0.1, 0.0), camera_at(), CAMERA, PERSON) is None


def test_person_outside_the_field_of_view_has_no_box():
    assert person_box((1.0, 5.0), camera_at(), CAMERA, PERSON) is None


def test_person_at_the_image_border_is_clipped():
    x1, y1, x2, y2 = person_box((3.0, 1.9), camera_at(), CAMERA, PERSON)
    assert x1 == 0.0
    assert 0.0 < x2 < 60.0
    assert x1 < x2 and y1 < y2


def test_camera_yaw_rotates_the_view():
    box = person_box((0.0, 3.0), camera_at(yaw=math.pi / 2), CAMERA, PERSON)
    assert (box[0] + box[2]) / 2 == pytest.approx(320.0, abs=1e-6)


def test_camera_pose_adds_the_mount_offset_in_the_robot_frame():
    robot = camera_at(x=1.0, y=2.0, z=0.0, yaw=math.pi / 2)
    cam = camera_pose(robot, (0.076, 0.0, 0.093))
    assert cam[:3, 3] == pytest.approx([1.0, 2.076, 0.093])


def test_sim_roi_has_no_vertical_band():
    assert Roi(0.30, 0.70, 0.0, 1.0, 0.25) == SIM_ROI
    near = person_box((3.0, 0.0), camera_at(), CAMERA, PERSON)
    far = person_box((10.0, 0.0), camera_at(), CAMERA, PERSON)
    assert SIM_ROI.mask(near[None], (480, 640))[0]
    assert not SIM_ROI.mask(far[None], (480, 640))[0]


# --- robot pose interpolation ------------------------------------------------


def test_interpolate_pose_is_linear_in_position():
    stamps = np.array([0.0, 1.0])
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]], float)
    T = interpolate_pose(stamps, poses, 0.25)
    assert T[:3, 3] == pytest.approx([0.5, 0.0, 0.0])


def test_interpolate_pose_outside_the_samples_uses_the_nearest():
    stamps = np.array([1.0, 2.0])
    poses = np.array([[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]], float)
    assert interpolate_pose(stamps, poses, 0.0)[0, 3] == 1.0
    assert interpolate_pose(stamps, poses, 9.0)[0, 3] == 2.0


# --- actor path ---------------------------------------------------------------

PATH = ActorPath(corners=((3.0, -3.0), (3.0, 3.0)), speed=1.0, turn_s=0.5, z=1.0)


def test_path_period_is_walking_plus_turning():
    assert PATH.period == pytest.approx(6.0 + 0.5 + 6.0 + 0.5)


def test_path_position_is_linear_on_a_leg():
    x, y, yaw = PATH.position(3.0)
    assert (x, y) == pytest.approx((3.0, 0.0))
    assert yaw == pytest.approx(math.pi / 2)


def test_path_keeps_position_while_turning():
    for t in (6.0, 6.2, 6.5):
        x, y, _ = PATH.position(t)
        assert (x, y) == pytest.approx((3.0, 3.0))
    assert PATH.position(6.5)[2] == pytest.approx(-math.pi / 2)


def test_path_is_periodic():
    for t in (0.3, 4.1, 6.3, 12.7):
        assert PATH.position(t + PATH.period) == pytest.approx(PATH.position(t))


def test_waypoints_include_every_corner_and_close_the_loop():
    points = PATH.waypoints(0.25)
    times = [p[0] for p in points]
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(PATH.period)
    assert np.all(np.diff(times) > 0)
    assert np.max(np.diff(times)) <= 0.25 + 1e-9
    for corner_time in (6.0, 6.5, 12.5):
        assert any(abs(t - corner_time) < 1e-9 for t in times)
    assert points[0][1:3] == pytest.approx(points[-1][1:3])
    assert all(p[3] == 1.0 for p in points)


def test_waypoints_follow_position():
    for t, x, y, _, yaw in PATH.waypoints(0.25):
        px, py, pyaw = PATH.position(t)
        assert (x, y) == pytest.approx((px, py))
        assert math.cos(yaw - pyaw) == pytest.approx(1.0)


def test_path_from_config():
    path = ActorPath.from_config(
        {"corners": [[3, -3], [3, 3]], "speed": 1.0, "turn_s": 0.5, "z": 1.0}
    )
    assert path == PATH


def test_path_needs_two_distinct_corners_and_positive_speed():
    with pytest.raises(ValueError):
        ActorPath(corners=((0.0, 0.0),), speed=1.0, turn_s=0.5)
    with pytest.raises(ValueError):
        ActorPath(corners=((0.0, 0.0), (0.0, 0.0)), speed=1.0, turn_s=0.5)
    with pytest.raises(ValueError):
        ActorPath(corners=((0.0, 0.0), (1.0, 0.0)), speed=0.0, turn_s=0.5)
