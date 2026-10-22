import zlib

import numpy as np
import pytest

from robust_visual_perception.degradation import (
    DEFAULT_CONDITIONS,
    Condition,
    apply,
    frame_seed,
    gaussian_blur,
    gaussian_noise,
    low_light,
)


def _image(value: int = 128, shape: tuple[int, int, int] = (32, 48, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _random_image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (32, 48, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "fn",
    [
        lambda img: gaussian_blur(img, 7),
        lambda img: low_light(img, 0.45),
        lambda img: gaussian_noise(img, 20, np.random.default_rng(0)),
    ],
)
def test_degradations_return_new_uint8_array_and_keep_input(fn):
    img = _random_image()
    before = img.copy()
    out = fn(img)
    assert out is not img
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
    ],
)
def test_invalid_image_raises(bad):
    with pytest.raises(ValueError):
        low_light(bad, 0.5)


@pytest.mark.parametrize("ksize", [0, 4, -3])
def test_blur_rejects_bad_kernel(ksize):
    with pytest.raises(ValueError):
        gaussian_blur(_image(), ksize)


def test_blur_smooths_an_edge():
    img = _image(0)
    img[:, 24:] = 255
    out = gaussian_blur(img, 15)
    assert 0 < out[0, 24, 0] < 255


def test_low_light_scales_and_rounds():
    np.testing.assert_array_equal(low_light(_image(200), 0.5), _image(100))
    np.testing.assert_array_equal(low_light(_image(3), 0.5), _image(2))


def test_low_light_gain_one_is_identity():
    img = _random_image()
    np.testing.assert_array_equal(low_light(img, 1.0), img)


def test_negative_gain_and_sigma_raise():
    with pytest.raises(ValueError):
        low_light(_image(), -0.1)
    with pytest.raises(ValueError):
        gaussian_noise(_image(), -1, np.random.default_rng(0))


def test_noise_is_clipped_to_valid_range():
    out = gaussian_noise(_image(250), 35, np.random.default_rng(0))
    assert out.max() == 255
    assert out.min() >= 0
    assert (out != 250).any()


def test_frame_seed_uses_crc32():
    assert frame_seed("abc", 1234) == 1234 + zlib.crc32(b"abc")


def test_apply_noise_is_deterministic_per_frame_id():
    cond = Condition("noise_s20", "noise", 20)
    img = _image()
    a = apply(cond, img, "000001")
    b = apply(cond, img, "000001")
    c = apply(cond, img, "000002")
    np.testing.assert_array_equal(a, b)
    assert (a != c).any()


def test_apply_noise_depends_on_base_seed():
    cond = Condition("noise_s20", "noise", 20)
    img = _image()
    assert (
        apply(cond, img, "x", base_seed=1) != apply(cond, img, "x", base_seed=2)
    ).any()


def test_apply_clean_returns_equal_copy():
    img = _random_image()
    out = apply(Condition("clean", "clean", None), img, "f")
    assert out is not img
    np.testing.assert_array_equal(out, img)


def test_apply_dispatches_blur_and_low_light():
    img = _random_image()
    np.testing.assert_array_equal(
        apply(Condition("b", "blur", 7), img, "f"), gaussian_blur(img, 7)
    )
    np.testing.assert_array_equal(
        apply(Condition("l", "low_light", 0.6), img, "f"), low_light(img, 0.6)
    )


@pytest.mark.parametrize(
    "kind,param",
    [("fog", 1.0), ("blur", None), ("clean", 3.0), ("noise", None)],
)
def test_condition_validates_kind_and_param(kind, param):
    with pytest.raises(ValueError):
        Condition("bad", kind, param)


def test_default_conditions_match_design():
    names = [c.name for c in DEFAULT_CONDITIONS]
    assert names == [
        "clean",
        "blur_k7",
        "blur_k15",
        "blur_k31",
        "lowlight_g060",
        "lowlight_g045",
        "lowlight_g025",
        "noise_s10",
        "noise_s20",
        "noise_s35",
    ]
    params = {c.name: c.param for c in DEFAULT_CONDITIONS}
    assert params["blur_k31"] == 31
    assert params["lowlight_g025"] == 0.25
    assert params["noise_s35"] == 35


@pytest.mark.parametrize(
    "kind,param",
    [("blur", 7.5), ("blur", 8), ("blur", 0), ("low_light", -0.5), ("noise", -1)],
)
def test_condition_rejects_out_of_range_params(kind, param):
    with pytest.raises(ValueError):
        Condition("bad", kind, param)
