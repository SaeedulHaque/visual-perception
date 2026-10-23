# Phase 1 Plan: Offline Pipeline (MVP)

**Goal:** a working offline pipeline that takes COCO images, applies the ten
degradation conditions, runs YOLO26n on CPU, makes a per-frame CLEAR / STOP /
UNKNOWN decision and writes `per_frame.csv` plus run metadata. A smoke test
on 20 COCO images proves it end to end.

**Architecture:** a plain Python package `robust_visual_perception` under
`src/`, one module per pipeline stage, no ROS. Scripts in `scripts/` are thin
CLIs over the package. Every module except the Ultralytics wrapper is tested
with pytest and needs no model weights (a fake detector stands in).

**Tech stack:** Python 3.12 (uv-managed venv), PyTorch CPU wheels,
Ultralytics, OpenCV, NumPy, PyYAML, psutil, pytest, ruff.

**Spec:** `docs/design.md` (sections 3, 5, 6.1, 6.2 subset selection, 6.5).

**Done when:** the smoke test on 20 COCO images writes `per_frame.csv`;
`pytest` is green; the license is AGPL-3.0; every feature has a page in
`docs/features/`.

## Progress

Phase 1 is complete and merged into `main` (PR #2, branch
`feature/offline-pipeline`).

- [x] Task 1: project scaffold and license
- [x] Task 2: degradations
- [x] Task 3: image source
- [x] Task 4: detector interface and Ultralytics wrapper
- [x] Task 5: baseline decision and temporal filter
- [x] Task 6: config, environment and run outputs
- [x] Task 7: COCO subset selection and fetch script
- [x] Task 8: runner and smoke test
- [x] Done: the smoke test on 20 COCO images writes `per_frame.csv`
- [x] Done: `pytest` green (100 tests at the end of Phase 1)
- [x] Done: license switched to AGPL-3.0
- [x] Done: a page in `docs/features/` for every feature

## Global constraints

- Python 3.12 venv in `.venv/` via `uv`; host Python untouched; no `sudo pip`.
- PyTorch from `https://download.pytorch.org/whl/cpu`.
- Core package imports without ROS.
- Degradation levels exactly: blur ksize 7/15/31; low-light gain
  0.60/0.45/0.25; noise sigma 10/20/35; noise base seed 1234.
- Noise seed per frame = `base_seed + zlib.crc32(frame_id.encode())`.
- Decision threshold 0.5, target class `person`. Filter default n_on=3,
  k_window=5, m_off=5.
- Inference stored at `conf=0.001`, `max_det=300`; decision applies 0.5 on top.
- Latency: `time.perf_counter_ns` around the detector call only; 10 warm-up
  calls per imgsz excluded.
- No weights, datasets or run outputs committed (`.gitignore`).
- No metric written in docs unless a saved run produced it.

## Review focus

1. COCO contains grayscale JPEGs: they must load as HxWx3 uint8, not crash
   or be skipped. Test in Task 3.
2. A corrupt or truncated image file gives a MISSING row with a reason, and
   the run continues. Test in Task 3 and Task 8.
3. A detector exception mid-run gives an ERROR result and an UNKNOWN row,
   and the run continues. Test in Task 4 and Task 8.
4. A dropped connection during download leaves no half-written file that is
   later treated as valid (`.part` file, rename on success). Test in Task 7.
5. Re-running with an existing `run_name` never overwrites earlier results
   silently (error unless `--overwrite`). Test in Task 8.

---

### Task 1: Project scaffold and license

**Files:** create `pyproject.toml`, `.gitignore`, `src/robust_visual_perception/__init__.py`,
`tests/__init__.py` (empty), `docs/setup.md`; replace `LICENSE` with the AGPL-3.0 text.

- [x] `pyproject.toml`: package `robust-visual-perception`, `requires-python
  = ">=3.12,<3.13"`, src layout (hatchling), deps `numpy`, `opencv-python`,
  `pyyaml`, `psutil`, `torch`, `torchvision`, `ultralytics`; dev group
  `pytest`, `ruff`. `[tool.uv.sources]` pins torch/torchvision to the CPU
  index. Ruff line length 88.
- [x] `.gitignore`: `.venv/`, `__pycache__/`, `data/raw/`, `models/*.pt`,
  `results/*` except committed final runs, `*.json.gz`.
- [x] `uv sync` creates `.venv` with Python 3.12.
- [x] Verify: `uv run python -c "import torch, ultralytics, cv2; print(torch.__version__, torch.cuda.is_available())"`
  prints a `+cpu` version and `False`.
- [x] `docs/setup.md`: why 3.12, why CPU wheels, how to recreate the venv.
- [x] Suggested commits: `chore: switch license to AGPL-3.0` and `chore: add python project scaffold`
  (made as `301a2b4` and `d4b3c28`).

### Task 2: Degradations

**Files:** `src/robust_visual_perception/degradation.py`, `tests/test_degradation.py`,
`docs/features/degradation.md`.

**Produces:**
- `Condition(name: str, kind: Literal["clean","blur","low_light","noise"], param: float | None)` (frozen dataclass; validates kind and param).
- `DEFAULT_CONDITIONS: tuple[Condition, ...]` — the 10 conditions, names
  `clean, blur_k7, blur_k15, blur_k31, lowlight_g060, lowlight_g045, lowlight_g025, noise_s10, noise_s20, noise_s35`.
- `gaussian_blur(img, ksize: int) -> np.ndarray`, `low_light(img, gain: float) -> np.ndarray`,
  `gaussian_noise(img, sigma: float, rng: np.random.Generator) -> np.ndarray`,
  `frame_seed(frame_id: str, base_seed: int) -> int`,
  `apply(condition, img, frame_id: str, base_seed: int = 1234) -> np.ndarray`.

Tests (write first, see them fail, then implement):
- each function returns a new uint8 array with the input shape and leaves
  the input unchanged;
- non-uint8 or non-HxWx3 input raises `ValueError`; even ksize raises;
  negative gain/sigma raises;
- `low_light(full 200 image, 0.5)` is all 100; `gain=1.0` is identity;
- noise output stays in [0, 255] for a 250-valued image at sigma 35;
- `apply` with the same frame_id gives identical noise twice; two frame_ids differ;
- `frame_seed("abc", 1234) == 1234 + zlib.crc32(b"abc")`;
- `clean` returns an equal copy, not the same object;
- `len(DEFAULT_CONDITIONS) == 10` and names are unique.

Rounding: float32 math, `np.rint`, `np.clip(0, 255)`, then cast.
Suggested commit: `feat(degradation): add blur, low-light and noise conditions`.

### Task 3: Image source

**Files:** `src/robust_visual_perception/image_source.py`, `tests/test_image_source.py`,
`docs/features/image-source.md`.

**Produces:**
- `FrameStatus(StrEnum)`: `OK`, `MISSING`.
- `Frame(frame_id: str, image: np.ndarray | None, timestamp_s: float | None, status: FrameStatus, error: str | None = None)`.
- `load_image(path: Path) -> np.ndarray` (raises `ImageLoadError` with the reason).
- `iter_image_files(paths: Sequence[Path], fps: float | None = None) -> Iterator[Frame]`;
  frame_id = file stem; timestamp = index / fps when fps is given.
- `iter_image_folder(folder: Path, fps: float | None = None) -> Iterator[Frame]`;
  sorted `*.jpg|*.jpeg|*.png`; missing folder raises `FileNotFoundError`.

Tests: valid PNG gives OK frame HxWx3 uint8; a grayscale PNG loads as
HxWx3; a non-existent path and a file of random bytes each give MISSING with
a non-empty `error` and the iteration continues to the next file; folder
order is sorted; fps=10 gives timestamps 0.0, 0.1, ...

Suggested commit: `feat(image-source): add frame iterator with missing-frame status`.

### Task 4: Detector interface and Ultralytics wrapper

**Files:** `src/robust_visual_perception/detector.py`, `tests/test_detector.py`,
`tests/fakes.py`, `docs/features/detector.md`.

**Produces:**
- `DetectionStatus(StrEnum)`: `OK`, `EMPTY`, `ERROR`.
- `Detection(class_id: int, class_name: str, confidence: float, x1, y1, x2, y2: float)`;
  `__post_init__` raises `ValueError` unless `0 <= confidence <= 1`, `x1 < x2`, `y1 < y2`.
- `DetectionResult(status, detections: list[Detection], latency_ms: float, image_hw: tuple[int, int], error: str | None = None)`.
- `Detector` Protocol: `detect(image: np.ndarray) -> DetectionResult`.
- `boxes_to_detections(xyxy, conf, cls, names: Mapping[int, str], image_hw) -> list[Detection]`:
  clips boxes to the image, drops boxes that become zero-area.
- `UltralyticsDetector(weights: str | Path, imgsz: int, conf: float = 0.001, max_det: int = 300, device: str = "cpu")`.
  `detect` times `model.predict` with `perf_counter_ns`; any exception gives
  `status=ERROR` with the message, logged with `logger.exception`.
- `tests/fakes.py`: `FakeDetector(script: list[DetectionResult | Exception])`
  for later tasks (raises when the entry is an exception, wrapping like the real one).

Tests: Detection validation cases; `boxes_to_detections` clips (-5, -5, 50,
50) on a 40x40 image to (0, 0, 40, 40) and drops (50, 50, 60, 60); status
EMPTY when the list is empty. The real wrapper is exercised by the smoke
test, not unit tests (weights not required for pytest).

Suggested commit: `feat(detector): add detector protocol and ultralytics wrapper`.

### Task 5: Baseline decision and temporal filter

**Files:** `src/robust_visual_perception/decision.py`, `src/robust_visual_perception/temporal_filter.py`,
`tests/test_decision.py`, `tests/test_temporal_filter.py`,
`docs/features/decision.md`, `docs/features/temporal-filter.md`.

**Produces:**
- `State(StrEnum)`: `CLEAR`, `STOP`, `UNKNOWN`.
- `Roi(x_min: float, x_max: float, y_min: float, y_max: float, min_height: float)`,
  fractions of image size; `CORRIDOR_ROI = Roi(0.30, 0.70, 0.30, 1.00, 0.25)`.
- `Decision(state: State, reason: str | None)`.
- `baseline_decision(result: DetectionResult | None, target_class: str = "person", conf_threshold: float = 0.5, roi: Roi | None = None) -> Decision`.
  `None` gives `UNKNOWN("missing_frame")`; ERROR gives `UNKNOWN("detector_error")`.
  STOP when any target detection has conf ≥ threshold (and, with ROI, centre
  inside the box region and height ≥ `min_height * H`).
- `PersistenceFilter(n_on: int = 3, k_window: int = 5, m_off: int = 5)`
  with `update(state: State) -> State` and `reset()`. Validates
  `1 <= n_on <= k_window`, `m_off >= 1`. Semantics as design section 5.

Decision tests: threshold boundary (0.5 is STOP, 0.49 is CLEAR); wrong class
ignored; ROI excludes a box centred at x=0.1·W and a box shorter than 25 %;
None and ERROR give UNKNOWN with reasons.

Filter tests (input string to expected output, C/S/U):
- `S S S` → `C C S` (n_on=3);
- `S C S C S` → `C C C C S`;
- after STOP, `C C C C` stays STOP, 5th C gives CLEAR (m_off=5);
- STOP interrupted by `C C S` resets the release count;
- `U` gives UNKNOWN; while STOP, `S U C C C C` keeps STOP after the gap
  until 5 fresh C;
- window cleared by U: `S S U S` → `C C U C`;
- invalid parameters raise `ValueError`.

Suggested commits: `feat(decision): add per-frame stop rule with optional roi`,
`feat(temporal-filter): add n-of-k persistence filter`.

### Task 6: Config, environment and run outputs

**Files:** `src/robust_visual_perception/config.py`, `src/robust_visual_perception/environment.py`,
`src/robust_visual_perception/reporting.py`, `configs/smoke.yaml`,
`tests/test_config.py`, `tests/test_reporting.py`, `scripts/collect_env.py`,
`docs/features/config-and-outputs.md`.

**Produces:**
- `ModelConfig(weights="models/yolo26n.pt", imgsz=(640,), conf=0.001, max_det=300, device="cpu")`,
  `DecisionConfig(target_class="person", conf_threshold=0.5)`,
  `FilterConfig(n_on=3, k_window=5, m_off=5)`,
  `ExperimentConfig(run_name, image_dir, manifest: str | None, output_root="results", model, decision, filter, conditions: tuple[Condition, ...], noise_seed=1234, warmup=10)`.
- `load_config(path) -> ExperimentConfig` (unknown keys raise `ValueError`;
  missing `conditions` means `DEFAULT_CONDITIONS`), `config_to_dict(cfg) -> dict`.
- `collect_environment() -> dict`: python, platform, CPU model, logical
  cores, RAM, torch/ultralytics/opencv/numpy versions, torch threads, git
  commit and dirty flag.
- `PER_FRAME_COLUMNS` (design 6.5 plus `state_reason`), `PerFrameWriter`
  (context manager, `write(row: dict)` rejects unknown keys),
  `prepare_run_dir(output_root, run_name, overwrite: bool) -> Path`
  (raises `FileExistsError` if it exists and not overwrite),
  `write_run_metadata(run_dir, cfg, command: str)`.

Tests: YAML round-trip; unknown key raises; `PerFrameWriter` header equals
`PER_FRAME_COLUMNS`; unknown column raises; `prepare_run_dir` on an existing
directory raises without overwrite.

Suggested commit: `feat(config): add yaml config, environment capture and run outputs`.

### Task 7: COCO subset selection and fetch script

**Files:** `src/robust_visual_perception/coco_subset.py`, `scripts/fetch_coco_subset.py`,
`tests/test_coco_subset.py`, `data/README.md`, `docs/features/coco-subset.md`.

**Produces:**
- `SubsetImage(image_id: int, file_name: str, coco_url: str, n_person: int)`.
- `select_subset(instances: dict, n_person: int, n_negative: int, seed: int = 0) -> list[SubsetImage]`.
  Person image = ≥1 non-crowd person annotation; negative = no person
  annotation at all; crowd-only images are in neither pool. Each pool is
  sorted by id, permuted with `np.random.default_rng(seed)`, and the first
  n taken, so a smaller subset is a prefix of a larger one.
- `write_manifest(path, images)` / `read_manifest(path)`, CSV columns
  `image_id,file_name,coco_url,n_person`.
- `download_file(url, dest, retries=3)`: writes `dest.part`, renames on
  success, skips an existing `dest`.
- Script: downloads and caches `annotations_trainval2017.zip` in
  `data/raw/coco/`, extracts `instances_val2017.json`, writes
  `data/manifests/coco_val2017_<name>.csv`, fetches images to
  `data/raw/coco/val2017/`. Flags `--n-person --n-negative --seed --name`.

Tests: on a hand-made instances dict, pools are correct (crowd-only image
excluded); requesting more than available raises `ValueError`; the 2+1
subset is a prefix of the 3+2 subset; manifest round-trip; `download_file`
with a failing opener leaves no `dest` and no `.part`.

Suggested commit: `feat(data): add coco subset selection and fetch script`.

### Task 8: Runner and smoke test

**Files:** `src/robust_visual_perception/experiment_runner.py`, `scripts/run_smoke_test.py`,
`tests/test_experiment_runner.py`, `docs/features/smoke-test.md`; update `README.md`.

**Produces:**
- `run_image_experiment(cfg: ExperimentConfig, detector_factory: Callable[[int], Detector], command: str, overwrite: bool = False) -> Path`.
  For each imgsz: build the detector, run `cfg.warmup` untimed calls on the
  first valid frame, then for each condition and each frame: degrade, detect,
  decide, write one row. `gt_state` comes from the manifest (`STOP` if
  `n_person > 0`), empty when there is no manifest. `filtered_state` is empty
  for image runs. Prints a one-line summary per (condition, imgsz).
- Script: `--config configs/smoke.yaml --overwrite`.

Tests (FakeDetector, tmp dir, 3 tiny images, 2 conditions, 1 imgsz):
row count = 3 × 2; columns match; a corrupt image gives MISSING rows with
UNKNOWN/`missing_frame`; a raising detector call gives ERROR row with
UNKNOWN/`detector_error` and later rows still written; `config.yaml`,
`env.json`, `command.txt` exist; second run without overwrite raises.

Verification:
- `uv run pytest -q` all green.
- `uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke`
- `uv run python scripts/run_smoke_test.py --config configs/smoke.yaml`
  writes `results/smoke/per_frame.csv` with 20 × 10 rows.

Suggested commit: `feat(runner): add image experiment runner and smoke test`.

---

## Recreate Phase 1 and check it yourself

These steps rebuild Phase 1 from scratch on your machine and check each
result, not only the unit tests. They need Linux or macOS with
[`uv`](https://docs.astral.sh/uv/), about 2 GB of disk for the environment
and 300 MB for the COCO annotations.

### 1. Get the code at the Phase 1 state

```
git clone https://github.com/SaeedulHaque/visual-perception.git
cd visual-perception
git checkout 005e10a          # main after Phase 1 was merged
```

To run the same steps on the latest version instead, skip the checkout and
replace `scripts/run_smoke_test.py --config configs/smoke.yaml` below with
`scripts/run_image_experiment.py --config configs/smoke.yaml` (the script
was renamed in Phase 2).

### 2. Environment

```
uv sync
uv run python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
uv run python scripts/collect_env.py
```

Check: a torch version ending in `+cpu` (on Linux x86_64) and `False`.
`collect_env.py` prints the record every run saves as `env.json`: CPU,
cores, RAM, package versions, torch threads. Keep it; it describes the
device you are benchmarking.

### 3. Unit tests and lint

```
uv run pytest -v
uv run ruff check .
```

Check: 100 tests pass at this commit (more on later versions); ruff reports
`All checks passed!`. No model weights or network are needed.

### 4. Data: same images as the reference

```
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
git diff --exit-code data/manifests/coco_val2017_smoke.csv && echo "same 20 images"
sha256sum data/raw/coco/annotations_trainval2017.zip data/raw/coco/instances_val2017.json
ls data/raw/coco/val2017 | wc -l
```

Check:
- `git diff` prints nothing, then `same 20 images`: the seeded selection
  picked exactly the images in the committed manifest.
- Both checksums equal the ones in `data/README.md`.
- 20 image files.

### 5. Smoke run with the real model

```
uv run python scripts/run_smoke_test.py --config configs/smoke.yaml
```

The first run downloads `models/yolo26n.pt` (5.5 MB). Check the output:

```
wc -l results/smoke/per_frame.csv                       # 201 = header + 20 x 10 rows
cut -d, -f4 results/smoke/per_frame.csv | sort | uniq -c   # status column
ls results/smoke                                        # command.txt config.yaml env.json per_frame.csv
```

- 201 lines.
- Status is only `OK` (or `EMPTY`); no `ERROR` or `MISSING` rows.
  (At this commit `status` is column 4; from Phase 2 on it is column 5,
  because a `sequence` column comes first.)
- `config.yaml` equals `configs/smoke.yaml` with defaults filled in;
  `env.json` describes your machine.

### 6. Read the results

Decision vs ground truth per condition:

```
uv run python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("results/smoke/per_frame.csv")))
agree = collections.Counter((r["condition"], r["baseline_state"] == r["gt_state"]) for r in rows)
for cond in dict.fromkeys(r["condition"] for r in rows):
    print(f"{cond:15s} agree {agree[(cond, True)]:2d} / 20")
PY
```

Your device's detector latency at imgsz 640:

```
uv run python - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open("results/smoke/per_frame.csv")))
ms = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
print(f"n={len(ms)} mean={statistics.mean(ms):.1f} ms median={statistics.median(ms):.1f} ms")
PY
```

What to expect:
- The agreement counts should be the same on any machine with the same
  package versions, because the images, noise seeds and model are fixed.
  A different CPU or torch build can change a borderline confidence and
  flip an image now and then.
- Latency is your device's number. 200 timed calls is a quick indication
  only; the Phase 2 benchmark is the proper measurement.

### 7. Break it on purpose (optional)

```
echo junk > data/raw/coco/val2017/000000241668.jpg
uv run python scripts/run_smoke_test.py --config configs/smoke.yaml --overwrite
grep 000000241668 results/smoke/per_frame.csv | head -2      # MISSING, UNKNOWN, missing_frame
rm data/raw/coco/val2017/000000241668.jpg
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke   # restores it
```
