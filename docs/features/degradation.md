# Degradations

Module: `src/robust_visual_perception/degradation.py`
Tests: `tests/test_degradation.py`

## Why

A robot camera rarely sees clean images. Motion and focus errors blur the
image, dim scenes lower the signal, and small sensors add noise. I want to
know how much a pretrained detector degrades under each of these, at
controlled and repeatable strengths. Real degradations come mixed together
(a dark scene is also noisy and often blurred). Here each one is applied
alone, so its effect can be measured on its own. That is a deliberate
simplification and is listed as a limitation.

## The conditions

| Name | Kind | Parameter | What it models |
|---|---|---|---|
| `clean` | none | — | reference |
| `blur_k7`, `blur_k15`, `blur_k31` | Gaussian blur | kernel size 7 / 15 / 31 px | defocus, motion blur (isotropic approximation) |
| `lowlight_g060`, `lowlight_g045`, `lowlight_g025` | intensity gain | 0.60 / 0.45 / 0.25 | under-exposure |
| `noise_s10`, `noise_s20`, `noise_s35` | additive Gaussian noise | sigma 10 / 20 / 35 (8-bit units) | sensor read and shot noise (simplified) |

The levels were fixed in the design before any experiment ran, so they are
not tuned to make results look good or bad.

## How each one works

- **Blur** uses `cv2.GaussianBlur` with a square kernel of odd size `k` and
  sigma left to OpenCV (`0.3 * ((k - 1) * 0.5 - 1) + 0.8`). A larger kernel
  spreads each pixel over a wider area and removes fine edges, which is what
  a detector's early layers rely on.
- **Low light** multiplies every pixel by `gain` in float32, rounds, clips to
  [0, 255] and casts back to uint8. This is a pure exposure change. A real
  camera would also raise gain and add noise; that is not modelled here.
- **Noise** adds `N(0, sigma²)` independently to every pixel and channel,
  then rounds and clips. Clipping means bright and dark regions get
  slightly less noise than the nominal sigma.

## Reproducible noise

Noise must be identical every time the same frame is processed, and must
not depend on the order in which frames are processed. Each frame gets its
own random generator:

```
seed = base_seed + zlib.crc32(frame_id)
rng  = np.random.default_rng(seed)
```

`zlib.crc32` is used instead of Python's `hash()` because `hash()` of a
string is randomised per process (`PYTHONHASHSEED`), which would make runs
irreproducible. `base_seed` defaults to 1234.

## Guarantees

- Input must be HxWx3 uint8, otherwise `ValueError`.
- The input array is never modified; a new array is always returned (even
  for `clean`).
- Output is always HxWx3 uint8 in [0, 255].

## Usage

```python
from robust_visual_perception.degradation import DEFAULT_CONDITIONS, apply

for condition in DEFAULT_CONDITIONS:
    degraded = apply(condition, image_bgr, frame_id="000139")
```

## What I learned / notes

- Rounding with `np.rint` before the cast matters: a plain `astype(uint8)`
  truncates, which biases low-light images darker by up to one level.
- These are synthetic perturbations of already-processed JPEGs, not a
  physical camera model. A more realistic pipeline would degrade the raw
  sensor signal before demosaicing and compression.
