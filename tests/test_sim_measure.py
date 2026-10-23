import numpy as np
import pytest

from robust_visual_perception.sim_geometry import CameraModel
from robust_visual_perception.sim_measure import silhouette_box, size_from_box

CAMERA = CameraModel(500.0, 500.0, 320.0, 240.0, 640, 480)


def test_silhouette_box_finds_the_changed_region():
    background = np.full((480, 640, 3), 100, np.uint8)
    frame = background.copy()
    frame[50:300, 200:260] = (30, 160, 40)
    frame[400, 10] = 200  # a single noisy pixel is ignored
    assert silhouette_box(frame, background, threshold=25) == (200, 50, 260, 300)


def test_silhouette_box_none_without_change():
    background = np.full((480, 640, 3), 100, np.uint8)
    assert silhouette_box(background.copy(), background) is None


def test_size_from_box_back_projects_at_the_known_distance():
    # Camera at height 0.1 m; at 10 m, rows 100 and 250 are 2.8 m and -0.2 m
    # (relative to the optical axis: (240 - v) * 10 / 500).
    z_min, z_max, half_width = size_from_box((300, 100, 340, 250), 10.0, 0.1, CAMERA)
    assert z_max == pytest.approx(0.1 + 140 * 10 / 500)
    assert z_min == pytest.approx(0.1 - 10 * 10 / 500)
    assert half_width == pytest.approx(20 * 10 / 500)
