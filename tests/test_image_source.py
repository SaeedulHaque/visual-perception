from pathlib import Path

import cv2
import numpy as np
import pytest

from robust_visual_perception.image_source import (
    FrameStatus,
    ImageLoadError,
    iter_image_files,
    iter_image_folder,
    load_image,
)


def _write_png(path: Path, value: int = 100, gray: bool = False) -> Path:
    shape = (10, 12) if gray else (10, 12, 3)
    cv2.imwrite(str(path), np.full(shape, value, dtype=np.uint8))
    return path


def test_valid_png_gives_ok_frame(tmp_path):
    path = _write_png(tmp_path / "a.png")
    (frame,) = iter_image_files([path])
    assert frame.status is FrameStatus.OK
    assert frame.frame_id == "a"
    assert frame.image.shape == (10, 12, 3)
    assert frame.image.dtype == np.uint8
    assert frame.error is None


def test_grayscale_image_loads_as_three_channels(tmp_path):
    path = _write_png(tmp_path / "g.png", gray=True)
    img = load_image(path)
    assert img.shape == (10, 12, 3)


def test_missing_and_corrupt_files_give_missing_frames_and_iteration_continues(
    tmp_path,
):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(np.random.default_rng(0).bytes(200))
    good = _write_png(tmp_path / "good.png")
    frames = list(iter_image_files([tmp_path / "nope.png", corrupt, good]))
    assert [f.status for f in frames] == [
        FrameStatus.MISSING,
        FrameStatus.MISSING,
        FrameStatus.OK,
    ]
    for frame in frames[:2]:
        assert frame.image is None
        assert frame.error


def test_load_image_raises_with_reason(tmp_path):
    with pytest.raises(ImageLoadError, match="not found"):
        load_image(tmp_path / "nope.png")


def test_folder_is_sorted_and_filters_extensions(tmp_path):
    for name in ["003.png", "001.jpg", "002.jpeg"]:
        _write_png(tmp_path / name)
    (tmp_path / "notes.txt").write_text("x")
    ids = [f.frame_id for f in iter_image_folder(tmp_path)]
    assert ids == ["001", "002", "003"]


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_image_folder(tmp_path / "missing"))


def test_fps_gives_timestamps(tmp_path):
    paths = [_write_png(tmp_path / f"{i:03d}.png") for i in range(3)]
    stamps = [f.timestamp_s for f in iter_image_files(paths, fps=10)]
    assert stamps == pytest.approx([0.0, 0.1, 0.2])


def test_no_fps_gives_no_timestamp(tmp_path):
    (frame,) = iter_image_files([_write_png(tmp_path / "a.png")])
    assert frame.timestamp_s is None
