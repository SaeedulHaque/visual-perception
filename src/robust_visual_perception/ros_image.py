"""Convert raw ``sensor_msgs/Image`` fields to a BGR uint8 array (no ROS).

Done with numpy instead of cv_bridge, so the node does not depend on
cv_bridge's own NumPy and OpenCV build, and the conversion is tested here.
"""

from __future__ import annotations

import numpy as np

_CHANNELS = {"rgb8": 3, "bgr8": 3, "mono8": 1}


def image_to_bgr(
    data: bytes | bytearray | memoryview,
    width: int,
    height: int,
    step: int,
    encoding: str,
) -> np.ndarray:
    """H x W x 3 BGR copy of an image message's pixel buffer.

    ``step`` is the row length in bytes, which may include padding.
    """
    if encoding not in _CHANNELS:
        raise ValueError(
            f"unsupported image encoding {encoding!r}; expected one of "
            f"{sorted(_CHANNELS)}"
        )
    channels = _CHANNELS[encoding]
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(f"step {step} is shorter than a row ({row_bytes} bytes)")
    buffer = np.frombuffer(data, np.uint8)
    if buffer.size < step * height:
        raise ValueError(f"buffer has {buffer.size} bytes, expected {step * height}")
    rows = buffer[: step * height].reshape(height, step)[:, :row_bytes]
    pixels = rows.reshape(height, width, channels)
    if encoding == "rgb8":
        return np.ascontiguousarray(pixels[..., ::-1])
    if encoding == "mono8":
        return np.repeat(pixels, 3, axis=2)
    return pixels.copy()
