# Image source

Module: `src/robust_visual_perception/image_source.py`
Tests: `tests/test_image_source.py`

## Why

Every later stage needs the same thing: a stream of frames with an id, the
pixels, and an optional timestamp. Wrapping that in one small type keeps the
detector and decision code independent of where frames come from (COCO
images in Phase 1, MOT17 sequences in Phase 2, and in Phase 3 the frames of
a recorded bag, exported to an image folder in MOT17 layout and read like
any other sequence).

The other job of this module is to make bad input visible. In a robot, a
frame that never arrives is information: the system does not know what is
in front of it. So a file that cannot be read is not dropped; it becomes a
frame with status `MISSING`, and the decision layer turns that into
`UNKNOWN`.

## Types

```python
class FrameStatus(StrEnum):
    OK = "OK"
    MISSING = "MISSING"


@dataclass(frozen=True)
class Frame:
    frame_id: str  # file stem, e.g. "000000139"
    image: np.ndarray | None  # HxWx3 uint8 BGR, None when MISSING
    timestamp_s: float | None  # index / fps for sequences, None for photos
    status: FrameStatus
    error: str | None = None  # why the frame is MISSING
```

## Functions

- `load_image(path)` reads one file with OpenCV and returns HxWx3 uint8 BGR.
  It raises `ImageLoadError` with the reason (not found, could not decode).
- `iter_image_files(paths, fps=None)` yields one frame per path in the given
  order, turning load errors into `MISSING` frames.
- `list_image_files(folder)` returns every `.jpg`, `.jpeg` or `.png` in a
  folder, sorted by name; `iter_image_folder(folder, fps=None)` iterates
  them as frames. MOT17 frames are named
  `000001.jpg`, `000002.jpg`, so sorting by name is frame order.

## Details worth knowing

- **BGR, not RGB.** OpenCV loads channels as blue-green-red. Ultralytics
  expects BGR for numpy input, so no conversion is needed. Converting by
  mistake would swap red and blue and quietly lower accuracy.
- **Grayscale images.** A few COCO images are single-channel JPEGs.
  `cv2.IMREAD_COLOR` always returns three channels, so these load as
  HxWx3 with equal channels instead of crashing the degradation code.
- **Timestamps.** For a sequence, `timestamp_s = index / fps`. This is the
  nominal capture time, used later to express delays in seconds.
