# Problems I ran into, and what to avoid

A record of what went wrong or slowed me down while building this project,
how each problem showed up, what I changed, and what to do differently.
Grouped by phase; within a phase, roughly in the order I hit them.

## Environment and tooling

### No ROS 2 packages for the host OS
- **Symptom:** the machine runs Ubuntu 26.04. ROS 2 Jazzy ships binary
  packages for Ubuntu 24.04 only.
- **Fix:** ROS 2 and Gazebo will run in Docker (Phase 3); the core package
  never depends on ROS.
- **Avoid:** check the ROS distribution's supported OS list before planning
  a native install. Keep perception code ROS-free so it runs anywhere.

### System Python too new
- **Symptom:** system Python is 3.14. PyTorch, OpenCV and Ultralytics wheels
  are not guaranteed for it.
- **Fix:** `uv` downloads a standalone Python 3.12 for the project
  (`requires-python = ">=3.12,<3.13"`). The system Python is untouched.
- **Avoid:** never `sudo pip install`; pin the interpreter per project.

### Default PyTorch wheels are huge
- **Symptom:** the default `torch` wheel on Linux bundles CUDA libraries,
  several GB, useless on a laptop without an NVIDIA GPU.
- **Fix:** `pyproject.toml` points `torch` and `torchvision` at the PyTorch
  CPU index (`[tool.uv.sources]`). The whole environment is about 1.4 GB.
- **Avoid:** decide CPU vs CUDA wheels up front; switching later means a
  second environment (Phase 4 uses a separate one on the GPU machine).

### `uv run` exits with code 120 when output goes to a file
- **Symptom:** with `uv` installed from snap, `uv run python ... > /tmp/log`
  prints nothing and exits 120. The same command in a terminal or piped
  through `cat` works.
- **Cause:** snap confinement; the child process cannot write to the
  redirected file under `/tmp`.
- **Fix:** call `.venv/bin/python` directly when redirecting output, or
  install `uv` with the official installer instead of snap.

### License conflict
- **Symptom:** the repository started as MIT, but Ultralytics is AGPL-3.0.
- **Fix:** relicensed the repository as AGPL-3.0 in Phase 1.
- **Avoid:** check dependency licenses before choosing your own.

### Scripted edits failing after auto-formatting
- **Symptom:** search-and-replace edits on a file failed to apply after
  `ruff format` had rewrapped the lines, and one fetch script ran an old
  version because the edit silently did not happen.
- **Avoid:** re-read a file after formatting before editing it again, and
  make edit scripts fail loudly when the text to replace is not found.

## Phase 1: offline pipeline

### COCO downloads: slow and HTTP only
- **Symptom:** COCO annotations exist only inside a 240 MB archive (with
  train annotations), and the server answered at well under 1 MB/s. The
  image server does not support HTTPS.
- **Fix:** download the archive once, extract only `instances_val2017.json`,
  fetch the selected images one by one (not the 1 GB image zip), and record
  SHA-256 checksums of the annotation files in `data/README.md`.
- **Avoid:** start long downloads in the background early; verify files with
  checksums when the transport is not encrypted.

### Grayscale images in COCO
- **Symptom:** a few COCO JPEGs are single-channel. Code expecting HxWx3
  would crash or need special cases.
- **Fix:** `cv2.imread(..., cv2.IMREAD_COLOR)` always returns three channels.

### First inference is very slow
- **Symptom:** the first `predict` call on a fresh model takes far longer
  than the rest (one-off setup in PyTorch and Ultralytics).
- **Fix:** 10 untimed warm-up calls per detector before timing.
- **Avoid:** never include the first calls in latency statistics.

### Rounding bias in degradations
- **Symptom:** `array.astype(np.uint8)` truncates, so scaled images come out
  up to one level too dark.
- **Fix:** `np.rint` before `np.clip` and the cast. A test pins the exact
  values (`0.5 × 3 → 2`).

### Irreproducible noise from `hash()`
- **Risk:** seeding noise with Python's `hash(frame_id)` changes every
  process, because string hashing is randomised.
- **Fix:** `seed = base_seed + zlib.crc32(frame_id)`, stable everywhere and
  independent of processing order.

### Temporal filter re-triggering from stale history
- **Symptom:** found while writing the filter tests. With a window longer
  than the release count, old STOP frames still in the window re-asserted
  STOP on the next single STOP frame after a release.
- **Fix:** the window is cleared on release. Recorded in the design.
- **Avoid:** write the filter's behaviour as input → output strings first;
  the edge cases show up immediately.

### Silent type coercion in configs
- **Symptom (self-review):** a blur kernel of `7.5` in YAML would have been
  cast to 7 without a warning.
- **Fix:** `Condition` validates kernel sizes (positive odd integers) and
  non-negative gains and sigmas; config loading rejects unknown keys.
- **Avoid:** a typo or a wrong type that runs silently is worse than a crash.

### Half-written files
- **Risk:** an interrupted download or extraction leaves a truncated file
  that later looks valid.
- **Fix:** every download and extraction writes `<name>.part` and renames
  on success.

## Phase 2: experiments

### Holding all frames in memory
- **Symptom:** the Phase 1 runner loaded all images up front. Fine for 20
  COCO images, not for MOT17 sequences of up to 900 frames at 1920×1080.
- **Fix:** frames are streamed; each image is read once and dropped after
  its rows are written. A test asserts one read per image across sizes.

### MOT17 is 5.9 GB for under 0.3 GB of needed data
- **Symptom:** `MOT17.zip` holds train and test, each sequence three times
  (one copy per public detector). I needed GT for seven sequences and
  images for three.
- **Problems on the way:**
  - Wrapping HTTP range requests in `io.RawIOBase` failed inside `zipfile`
    with `NotImplementedError` (RawIOBase needs `readinto`). A plain
    seekable file object with `read`, `seek`, `tell` and `readinto` works.
  - Reading images one member at a time cost about 2 s per file in round
    trips.
- **Fix:** `RangeFile` lets `zipfile` read the remote directory; all images
  of one sequence sit in one contiguous byte span, fetched with a single
  range request (`extract_prefix`). 267 MB on disk.
- **Avoid:** before downloading a large archive, check `Accept-Ranges` with
  `curl -I`; partial reads are often possible.

### The design's corridor ROI qualified only one sequence
- **Symptom:** with the planned ROI, MOT17 sequences were either crowded
  (someone in the corridor 84 to 100 % of the time) or almost empty. The
  selection rule needs three sequences with 15 to 70 % occupancy.
- **Fix:** a fixed grid search on GT only, with the choice rule written in
  code before any model output on MOT17 existed (`mot17.choose_roi`). The
  full search table is committed (`data/manifests/mot17_roi_search.csv`).
- **Avoid:** compute dataset statistics before fixing thresholds that
  depend on them, and when you must adjust, make the adjustment a rule, not
  a choice.

### GT and detections did not share one geometry
- **Symptom:** MOT17 GT boxes can extend past the image border; detections
  are clipped to it. The same person could pass the ROI test on one side and
  fail on the other.
- **Fix:** GT boxes are clipped exactly like detections.

### GT flicker split events
- **Symptom:** a person whose visibility hovers around the 0.5 cut-off turns
  GT-STOP on and off, so one approach would count as many tiny events.
- **Fix:** GT events merge runs separated by fewer than 5 frames, fixed in
  the plan before any evaluation.

### Class ids that do not line up
- **Symptom:** YOLO uses 80 contiguous class indices; COCO category ids run
  from 1 to 90 with gaps. Feeding YOLO indices to `pycocotools` gives wrong
  per-class results without any error (person happens to be fine, others
  are not).
- **Fix:** an explicit 80 → 91 table, checked against Ultralytics' own.

### pycocotools quirks
- **Symptom:** it prints its summary to stdout on every call, and returns
  `-1` when a selection has no ground truth.
- **Fix:** stdout redirected during evaluation; `-1` becomes NaN, never 0.

### Detection storage size
- **Symptom:** inference at `conf=0.001` keeps about 110 boxes per image:
  about 2.4 million boxes for 22,500 inferences. COCO JSON would be large
  and slow.
- **Fix:** compressed `.npz` arrays in the exact layout `loadRes` accepts;
  not committed.

### Runtime estimates
- **Symptom:** I expected the sequence run to take more than an hour from
  early progress lines. It took about 16 minutes
  (`results/sequence_main/resources.json`). The 640×480 sequence is much
  faster than the 1080p ones, so early progress was a poor guide.
- **Avoid:** record wall time in every run (`resources.json`) and estimate
  from that, not from the first progress lines.

### Latency measurement order
- **Symptom:** the `clean` condition is consistently about 0.5 to 2 ms
  slower than the other conditions. It is always the first call a detector
  makes on a new image, which points to a measurement-order effect rather
  than image content.
- **Also:** a first run made while I was doing light work in parallel had
  latencies about 2 % higher than an idle rerun (accuracy identical).
- **Fix:** the committed run is the idle one; the effect is documented in
  `docs/limitations.md`; a dedicated benchmark that runs each setting in its
  own block is planned for Phase 4.
- **Avoid:** do nothing else on the machine during timed runs, and do not
  interleave settings you want to compare.

### A filter sweep that could not discriminate
- **Symptom:** on the tuning sequence (static camera, steady signal), all 12
  filter settings gave identical transitions per minute and missed events;
  only onset delay differed. The declared adoption rule therefore picked
  the fastest setting.
- **Lesson:** a tuning sequence must contain the behaviour you are tuning
  for. Choosing it by GT statistics alone does not guarantee that the
  detector flickers on it.

### Metric accounting that looks like a failure
- **Symptom:** in the fault-injection run, transitions per minute and
  false-STOP episodes jump.
- **Cause:** each UNKNOWN frame inside a CLEAR stretch adds two transitions
  and, because UNKNOWN counts as STOP, one false-STOP episode.
- **Lesson:** state how a metric treats special states, and explain it
  next to the result.

### Hard-coded numbers in generated summaries
- **Symptom:** the summary generator had "500 with people, 250 without" and
  the fault rates typed into its text. Its test passed anyway, because the
  test data used the same numbers.
- **Fix:** the counts now come from the aggregate CSV and the fault rates
  from the run's config; the test uses different numbers so it fails if
  anything is typed in again.
- **Avoid:** test data should never equal the real defaults.

### Shared defaults leaking between configs
- **Symptom:** the sequence config's default input size (416) silently
  became 640, because the shared parser built `ModelConfig()` with image
  defaults. Caught by a test.
- **Fix:** the parser takes per-experiment defaults.

### Figures that hide data
- **Symptom:** in the sweep plot, settings with identical results drew on
  top of each other, the default marker was hidden, and the title was cut.
- **Fix:** identical results share one point with a combined label; the
  default is drawn last; shorter title. Always open the figure and look.
