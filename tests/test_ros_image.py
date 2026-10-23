import numpy as np
import pytest

from robust_visual_perception.ros_image import image_to_bgr


def rgb_image():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0)  # red
    rgb[1, 2] = (0, 0, 255)  # blue
    return rgb


def test_rgb8_is_converted_to_bgr():
    out = image_to_bgr(rgb_image().tobytes(), 3, 2, 9, "rgb8")
    assert out.shape == (2, 3, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (0, 0, 255)
    assert tuple(out[1, 2]) == (255, 0, 0)


def test_bgr8_is_unchanged():
    bgr = rgb_image()
    assert np.array_equal(image_to_bgr(bgr.tobytes(), 3, 2, 9, "bgr8"), bgr)


def test_mono8_becomes_three_equal_channels():
    gray = np.array([[0, 128, 255], [1, 2, 3]], np.uint8)
    out = image_to_bgr(gray.tobytes(), 3, 2, 3, "mono8")
    assert out.shape == (2, 3, 3)
    assert np.array_equal(out[..., 0], gray) and np.array_equal(out[..., 2], gray)


def test_row_padding_is_skipped():
    rgb = rgb_image()
    padded = np.zeros((2, 12), np.uint8)  # 9 bytes of pixels + 3 of padding
    padded[:, :9] = rgb.reshape(2, 9)
    padded[:, 9:] = 77
    out = image_to_bgr(padded.tobytes(), 3, 2, 12, "rgb8")
    assert np.array_equal(out, rgb[..., ::-1])


def test_result_is_a_writable_copy():
    data = bytearray(rgb_image().tobytes())
    out = image_to_bgr(data, 3, 2, 9, "bgr8")
    out[0, 0] = 1
    assert data[0] == 255


@pytest.mark.parametrize("encoding", ["rgba8", "16UC1", "bayer_rggb8", "yuv422"])
def test_unsupported_encodings_are_rejected(encoding):
    with pytest.raises(ValueError, match="unsupported image encoding"):
        image_to_bgr(bytes(24), 3, 2, 12, encoding)


def test_short_buffer_or_stride_is_rejected():
    with pytest.raises(ValueError):
        image_to_bgr(bytes(10), 3, 2, 9, "rgb8")
    with pytest.raises(ValueError):
        image_to_bgr(bytes(18), 3, 2, 8, "rgb8")
