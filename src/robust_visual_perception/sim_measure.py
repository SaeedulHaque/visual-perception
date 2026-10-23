"""Measure the simulated person's size from rendered frames (no detector).

A frame with the person is compared with a background frame of the same
view; the changed pixels form the person's silhouette. Its image extent,
back-projected at the person's known distance, gives the height range and
half-width used for the ground-truth volume.
"""

from __future__ import annotations

import cv2
import numpy as np

from robust_visual_perception.sim_geometry import CameraModel


def silhouette_box(
    frame: np.ndarray, background: np.ndarray, threshold: int = 25
) -> tuple[int, int, int, int] | None:
    """(x1, y1, x2, y2) of the largest changed region, end-exclusive."""
    diff = cv2.absdiff(frame, background).max(axis=2)
    mask = (diff > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    x, y, w, h, _ = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    return int(x), int(y), int(x + w), int(y + h)


def size_from_box(
    box: tuple[int, int, int, int],
    distance: float,
    camera_height: float,
    camera: CameraModel,
) -> tuple[float, float, float]:
    """(z_min, z_max, half_width) in metres for a silhouette box seen at
    ``distance`` along the optical axis from a level camera."""
    x1, y1, x2, y2 = box
    z_max = camera_height + (camera.cy - y1) * distance / camera.fy
    z_min = camera_height + (camera.cy - y2) * distance / camera.fy
    half_width = (x2 - x1) / 2 * distance / camera.fx
    return z_min, z_max, half_width
