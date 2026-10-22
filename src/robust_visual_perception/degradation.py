"""Synthetic image degradations: Gaussian blur, low light and sensor noise.

Every function takes an HxWx3 uint8 image, never modifies it, and returns a
new HxWx3 uint8 image. Noise is seeded per frame from the frame id, so the
same frame always gets the same noise regardless of processing order.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

ConditionKind = Literal["clean", "blur", "low_light", "noise"]

DEFAULT_NOISE_SEED = 1234


@dataclass(frozen=True)
class Condition:
    """One experimental condition, e.g. ``Condition("blur_k7", "blur", 7)``."""

    name: str
    kind: ConditionKind
    param: float | None

    def __post_init__(self) -> None:
        if self.kind not in ("clean", "blur", "low_light", "noise"):
            raise ValueError(f"unknown condition kind: {self.kind!r}")
        if self.kind == "clean" and self.param is not None:
            raise ValueError("condition 'clean' takes no parameter")
        if self.kind != "clean" and self.param is None:
            raise ValueError(f"condition kind {self.kind!r} needs a parameter")
        if self.kind == "blur" and (
            self.param != int(self.param) or self.param < 1 or self.param % 2 == 0
        ):
            raise ValueError(f"blur ksize must be a positive odd integer: {self.param}")
        if self.kind in ("low_light", "noise") and self.param < 0:
            raise ValueError(f"{self.kind} parameter must be >= 0: {self.param}")


DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    Condition("clean", "clean", None),
    Condition("blur_k7", "blur", 7),
    Condition("blur_k15", "blur", 15),
    Condition("blur_k31", "blur", 31),
    Condition("lowlight_g060", "low_light", 0.60),
    Condition("lowlight_g045", "low_light", 0.45),
    Condition("lowlight_g025", "low_light", 0.25),
    Condition("noise_s10", "noise", 10),
    Condition("noise_s20", "noise", 20),
    Condition("noise_s35", "noise", 35),
)


def _check_image(img: np.ndarray) -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must have shape HxWx3, got {img.shape}")


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Blur with a ``ksize`` x ``ksize`` Gaussian kernel (sigma chosen by OpenCV)."""
    _check_image(img)
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}")
    return cv2.GaussianBlur(img, (int(ksize), int(ksize)), 0)


def low_light(img: np.ndarray, gain: float) -> np.ndarray:
    """Scale every pixel by ``gain`` (0.25 = a quarter of the brightness)."""
    _check_image(img)
    if gain < 0:
        raise ValueError(f"gain must be >= 0, got {gain}")
    return _to_uint8(img.astype(np.float32) * np.float32(gain))


def gaussian_noise(
    img: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise with std ``sigma`` (8-bit intensity units)."""
    _check_image(img)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return _to_uint8(img.astype(np.float32) + noise)


def frame_seed(frame_id: str, base_seed: int) -> int:
    """Seed for one frame. CRC32 is stable across runs, unlike ``hash()``."""
    return base_seed + zlib.crc32(frame_id.encode("utf-8"))


def apply(
    condition: Condition,
    img: np.ndarray,
    frame_id: str,
    base_seed: int = DEFAULT_NOISE_SEED,
) -> np.ndarray:
    """Apply ``condition`` to ``img`` and return a new image."""
    _check_image(img)
    if condition.kind == "clean":
        return img.copy()
    if condition.kind == "blur":
        return gaussian_blur(img, int(condition.param))
    if condition.kind == "low_light":
        return low_light(img, float(condition.param))
    rng = np.random.default_rng(frame_seed(frame_id, base_seed))
    return gaussian_noise(img, float(condition.param), rng)
