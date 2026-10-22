# Experiment runner and smoke test

Module: `src/robust_visual_perception/experiment_runner.py`
Script: `scripts/run_smoke_test.py`
Tests: `tests/test_experiment_runner.py`

## What the runner does

`run_image_experiment(cfg, detector_factory, command, overwrite=False)`:

1. Creates `results/<run_name>/` (refuses to replace an existing one unless
   `overwrite`) and writes `config.yaml`, `env.json`, `command.txt`.
2. Builds the frame list: from the manifest when one is given (which also
   provides the ground-truth state), otherwise every image in `image_dir`.
3. For each input size: builds a detector, runs `warmup` untimed calls,
   then for every frame and every condition: degrade, detect, decide, and
   write one row to `per_frame.csv`.
4. Logs one summary line per (condition, imgsz).

Each image is loaded once and degraded ten times, rather than reloaded per
condition. The same frames in the same order are used for every condition.

`detector_factory` is a function `imgsz -> Detector`. The script passes one
that builds `UltralyticsDetector`; the tests pass one that builds a scripted
`FakeDetector`, so the whole runner is tested without a model.

## Failure handling

| Situation | Row |
|---|---|
| image file missing or corrupt | `status=MISSING`, `UNKNOWN` / `missing_frame`, no latency |
| detector returns an error | `status=ERROR`, `UNKNOWN` / `detector_error` |
| detector raises despite its contract | same as above; the runner catches and logs it |

None of these stops the run. Every case has a test.

## Running the smoke test

```
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
uv run python scripts/run_smoke_test.py --config configs/smoke.yaml
```

Expected: `results/smoke/per_frame.csv` with a header and 200 rows
(20 images × 10 conditions × 1 input size), plus the three metadata files.
Re-running needs `--overwrite`.

The smoke test proves the pipeline runs end to end on real images with the
real model. Its STOP counts and latencies are printed for a sanity check
only; 20 images are far too few to support any claim, so they are not
reported as results. The real measurements are Phase 2.

## Reading `per_frame.csv` quickly

```
uv run python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("results/smoke/per_frame.csv")))
agree = collections.Counter(
    (r["condition"], r["baseline_state"] == r["gt_state"]) for r in rows
)
for (cond, ok), n in sorted(agree.items()):
    print(f"{cond:15s} {'agree' if ok else 'differ':6s} {n}")
PY
```
