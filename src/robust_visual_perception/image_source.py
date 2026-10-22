"""Frame sources: iterate images from a file list or a folder.

A file that cannot be read becomes a ``Frame`` with ``status=MISSING`` and the
reason in ``error``. It is never silently skipped, so downstream code can
report it (the decision layer turns it into ``UNKNOWN``).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class FrameStatus(StrEnum):
    OK = "OK"
    MISSING = "MISSING"


class ImageLoadError(Exception):
    """Raised when an image file is absent or cannot be decoded."""


@dataclass(frozen=True)
class Frame:
    frame_id: str
    image: np.ndarray | None
    timestamp_s: float | None
    status: FrameStatus
    error: str | None = None
    load_ms: float | None = None


def load_image(path: Path) -> np.ndarray:
    """Read ``path`` as an HxWx3 uint8 BGR image (grayscale is expanded)."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"file not found: {path}")
    # IMREAD_COLOR always returns 3 channels, so grayscale JPEGs (COCO has
    # some) come back as HxWx3.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"could not decode image: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ImageLoadError(f"unexpected image format {image.shape} in {path}")
    return image


def iter_image_files(
    paths: Sequence[Path], fps: float | None = None
) -> Iterator[Frame]:
    """Yield one ``Frame`` per path, in the given order."""
    for index, path in enumerate(paths):
        path = Path(path)
        timestamp = index / fps if fps else None
        start = time.perf_counter_ns()
        try:
            image = load_image(path)
        except ImageLoadError as exc:
            yield Frame(path.stem, None, timestamp, FrameStatus.MISSING, str(exc))
            continue
        load_ms = (time.perf_counter_ns() - start) / 1e6
        yield Frame(path.stem, image, timestamp, FrameStatus.OK, load_ms=load_ms)


def list_image_files(folder: Path) -> list[Path]:
    """All ``.jpg``, ``.jpeg`` and ``.png`` files in ``folder``, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def iter_image_folder(folder: Path, fps: float | None = None) -> Iterator[Frame]:
    """Yield frames for all images in ``folder``, sorted by file name."""
    return iter_image_files(list_image_files(folder), fps=fps)
