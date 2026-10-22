# Phase 2 Plan: Experiments

**Goal:** run the COCO image experiment and the MOT17 sequence experiment
defined in the design, compute every metric from saved per-frame outputs,
and produce aggregate CSVs, figures and a generated summary.

**Architecture:** inference and evaluation are separate steps. A run script
does inference only and writes `per_frame.csv` (plus stored detections for
COCO). An evaluate script reads those files and writes `aggregate.csv`,
figures and `summary.md`. Metrics can be recomputed without re-running the
model, and every number in the summary comes from a CSV.

**Spec:** `docs/design.md` sections 6.2 to 6.5.

**Done when:** both experiments, the filter sweep and the fault-injection
run are complete; aggregate CSVs, figures and summaries are in `results/`;
each metric has a unit test on a hand-computed example; every new feature
has a page in `docs/features/`; `docs/limitations.md` exists.

## Global constraints (from the design, unchanged)

- Conditions: the ten design conditions; noise base seed 1234.
- COCO: 500 person + 250 negative images, seed 0; imgsz 320, 416, 640;
  inference `conf=0.001`, `max_det=300`; decision threshold 0.5; match IoU 0.5.
- MOT17 event rule: GT class 1, `considered=1`, visibility ≥ 0.5, box centre
  in `CORRIDOR_ROI`, box height ≥ 25 % of image height. Detections use the
  same geometry at confidence ≥ 0.5.
- Sequence selection uses GT only: occupancy 15–70 %, ≥ 3 GT events,
  prefer moving camera. One tuning and two evaluation sequences.
- Filter default `n_on=3, k_window=5, m_off=5`. Sweep on the tuning sequence
  only: n_on/k ∈ {2/3, 3/5, 4/6, 5/8} × m_off ∈ {3, 5, 8}.
- Sequence runs at imgsz 416. Fault run: 2 % MISSING, 1 % ERROR, seeded.
- Latency: detector call only, warm-up excluded, p95 only when n ≥ 100.
- No metric in docs unless produced by a saved run; thresholds not tuned.

## Decisions made in this plan

- **GT event:** a maximal run of GT-STOP frames, after merging runs separated
  by gaps shorter than 5 frames (GT flickers when a visibility value crosses
  0.5). Fixed before looking at any model output.
- **Release delay:** frames from GT event end to the first non-STOP output,
  counted only when it happens before the next GT event starts.
- **UNKNOWN in correctness metrics:** counted as STOP (a consumer treats
  UNKNOWN as stop-with-reason), and reported separately as a count.
- **Sweep adoption rule (declared before the sweep):** keep the default
  unless another setting dominates it on the tuning sequence, averaged over
  the ten conditions: no more missed events, no more transitions per
  minute, no larger mean onset delay, and strictly better on at least one.
  If several dominate, adopt the one with the fewest transitions per minute.
- **Sequence assignment rule:** among qualifying sequences, the two
  moving-camera sequences with the most GT events are evaluation sequences;
  the qualifying sequence with the most events among the rest is the tuning
  sequence. Camera motion per sequence from the MOT16 paper (moving: 05, 10,
  11, 13; static: 02, 04, 09).
- **Stored detections:** one compressed `.npz` per (condition, imgsz) holding
  an N×7 array `[image_id, x, y, w, h, score, coco_category_id]`, the format
  `pycocotools` `loadRes` accepts. Smaller than JSON for ~2.4 M boxes.
- **Crowd regions in person P/R:** an unmatched prediction whose area lies
  ≥ 50 % inside a crowd region is ignored, not a false positive (same idea
  as COCO evaluation).
- **MOT17 download:** the server supports HTTP range requests. Read the zip
  directory remotely, fetch GT for all seven train sequences, then fetch
  only the selected sequences' images (one detector copy, FRCNN) as one
  contiguous byte range each. About 0.4 GB instead of 5.9 GB.

## Review focus

1. A MOT17 frame listed in GT but missing on disk must give a MISSING row,
   not shift the frame alignment. Test in Task 5.
2. An image with no person boxes at all (negatives) must not divide by zero
   in retention or P/R; such cases give NaN with n=0. Test in Task 2.
3. A sequence condition where the model never fires STOP must still give
   valid event metrics (all missed, onset NaN). Test in Task 6.
4. An interrupted MOT17 fetch must not leave a partial sequence that later
   counts as complete. Test in Task 4.
5. Detections at the image border or with x1 == x2 after clipping must not
   reach pycocotools as zero-area boxes. Covered by Phase 1 clipping; the
   store test checks w, h > 0.

---

### Task 1: Streaming runner, timing columns, detection store

**Files:** modify `experiment_runner.py`, `reporting.py`; create
`detection_store.py`; tests `test_experiment_runner.py`, `test_reporting.py`,
`test_detection_store.py`; doc `features/detection-store.md`.

- Loop order becomes frame → imgsz → condition: each image is read once,
  all detectors are built and warmed up first, and frames are not held in
  memory (needed for 1080p MOT17 sequences).
- New per-frame columns `sequence`, `load_ms`, `degrade_ms`.
- `DetectionStore.add(condition, imgsz, image_id: int, detections)` and
  `save(run_dir)`; `load_detections(run_dir, condition, imgsz) -> np.ndarray`.
  Class index → COCO category id via `ultralytics.data.converter.coco80_to_coco91_class()`.
- Run directory gets `resources.json` with peak RSS (psutil) and wall time.

Tests: rows identical in content to before for the image runner; one
image read per frame; store round-trip; xywh conversion; w, h > 0.

### Task 2: Image metrics

**Files:** `metrics.py`, `tests/test_metrics.py`, `features/metrics.md`.

- `box_iou(a, b) -> np.ndarray`, `greedy_match(pred, scores, gt, iou_thr) -> list[tuple[int, int]]`.
- `person_pr(preds, gt, crowd, conf_thr=0.5, iou_thr=0.5) -> counts (tp, fp, fn, matched_conf_sum)`.
- `retention(clean, degraded, conf_thr=0.5, iou_thr=0.5) -> (matched, total)`.
- `decision_metrics(pred_stop, gt_stop) -> dict` (precision, recall, false-STOP rate, n).
- `latency_stats(values_ms) -> dict` (n, mean, median, p95 only if n ≥ 100, throughput).
- `coco_map(coco_gt, dets, image_ids) -> dict` (mAP, mAP50, person AP, person AP50).
- Ratios with zero denominator return NaN, never 0.

Tests on hand-computed examples: IoU of two known boxes; greedy match
prefers higher score; 2 GT + 3 preds gives tp=2, fp=1, fn=0; crowd-ignored
prediction; retention 1/2; decision metrics on a 6-frame example; p95 absent
for n < 100; `coco_map` on a tiny synthetic GT where predictions equal GT
gives AP 1.0.

### Task 3: Image experiment and evaluation

**Files:** `configs/image_experiment.yaml`, `scripts/run_image_experiment.py`,
`scripts/evaluate_image_experiment.py`, `image_eval.py` (logic), test
`test_image_eval.py`, doc `features/image-experiment.md`.

- Evaluate writes `aggregate.csv`, one row per (condition, imgsz): mAP,
  mAP50, person AP, person precision / recall, image-level STOP precision /
  recall / false-STOP rate, retention, mean matched confidence, latency
  stats, n images.
- Run the full experiment (22,500 inferences).

### Task 4: MOT17 data

**Files:** `mot17.py`, `remote_zip.py`, `scripts/fetch_mot17.py`,
`tests/test_mot17.py`, `tests/test_remote_zip.py`, `data/README.md`,
doc `features/mot17.md`.

- `parse_seqinfo`, `parse_gt`, `gt_stop_states(gt, info, roi, min_visibility)`,
  `find_events(states, merge_gap=5)`, `sequence_stats`, `select_sequences`.
- `RangeFile(url, opener)`: seekable read-only file over HTTP ranges with a
  local cache of one prefetched span; `fetch_members(zip, prefix, dest)`.
- Output: `data/manifests/mot17_sequences.csv` (stats and role of all seven
  sequences, committed); images and GT under `data/raw/mot17/`.
- A sequence directory is written to `<name>.part/` and renamed when
  complete.

### Task 5: Sequence runner

**Files:** `config.py` (`SequenceExperimentConfig`, `FaultConfig`),
`sequence_runner.py`, `scripts/run_sequence_experiment.py`,
`configs/sequence_experiment.yaml`, `configs/sequence_faults.yaml`,
`tests/test_sequence_runner.py`, doc `features/sequence-experiment.md`.

- For each sequence, frame, condition: degrade, detect, decide with
  `CORRIDOR_ROI`, filter (one filter per condition, reset per sequence).
- Rows carry `sequence`, `filtered_state`, `gt_state`.
- Fault injection keyed by `crc32(sequence/frame_id)`: identical across
  conditions.

### Task 6: Sequence metrics, sweep, evaluation

**Files:** `sequence_metrics.py`, `scripts/evaluate_sequence_experiment.py`,
`tests/test_sequence_metrics.py`; doc section in `features/metrics.md`.

- Per (sequence, condition, method): precision, recall, accuracy,
  transitions per minute, change fraction, events detected / missed, mean
  onset and release delay (frames and seconds), false-STOP episodes,
  UNKNOWN count.
- Sweep: re-filter the stored baseline states of the tuning sequence with
  each setting (identical to online filtering, no new inference); apply the
  adoption rule; write `sweep.csv`.

### Task 7: Figures and summaries

**Files:** `plots.py`, `scripts/plot_results.py`, `summary.py`.

- Image: AP and recall vs severity per degradation (one line per imgsz),
  retention vs severity, latency vs imgsz.
- Sequence: transitions per minute and onset delay, baseline vs filtered,
  per condition; sweep trade-off plot.
- `summary.md` tables are generated from the aggregate CSVs.

### Task 8: Runs, hand-check, docs

- Run everything; hand-check two COCO images and one MOT17 event against
  the metric output and record the check in `docs/features/metrics.md`.
- Update `docs/data-flow.md`, README status; add `docs/limitations.md`.

---

## Recreate Phase 2 and check it yourself

These steps reproduce every Phase 2 result on your machine and compare it
with the committed runs. The committed runs are the **reference**: they were
made on the machine described in each run's `env.json` (i7-13620H, CPU
only). Your runs go under new names, so the committed results stay intact.

Accuracy numbers (AP, precision, recall, events, delays) should match the
reference: data, seeds, thresholds and model are fixed. On the same package
versions and a similar CPU they match exactly. A different CPU family or
torch build can shift a few borderline confidences, which is why the
comparison uses a tolerance. Latency columns are your device's benchmark
and are shown next to the reference, never counted as a mismatch.

### 0. Setup

```
git clone https://github.com/SaeedulHaque/visual-perception.git
cd visual-perception
uv sync
uv run pytest
uv run python scripts/collect_env.py     # your device
```

### 1. Data (same images, same sequences)

```
uv run python scripts/fetch_coco_subset.py --n-person 500 --n-negative 250 --name main
uv run python scripts/fetch_mot17.py
git diff --exit-code data/manifests/ && echo "manifests identical"
```

Check: no diff. That proves your COCO subset, the MOT17 GT statistics, the
GT-only ROI search (`mot17_roi_search.csv`) and the sequence roles
(`mot17_sequences.csv`) are exactly the reference ones. Downloads: about
120 MB of COCO images and 270 MB of MOT17.

### 2. Quick device benchmark (a few minutes)

750 clean images at imgsz 320, 416 and 640 (2,250 timed inferences, so p95
is reported):

```
uv run python scripts/run_image_experiment.py --config configs/benchmark.yaml
uv run python scripts/evaluate_image_experiment.py --run results/benchmark
uv run python scripts/compare_runs.py --reference results/image_main --candidate results/benchmark
```

The comparison checks the three `clean` rows of the reference (the other 27
are listed as not compared) and prints:

```
mean detector latency (ms)      reference  candidate  ratio
  clean/320                         ...        ...       ...
```

A ratio below 1 means your device is faster than the reference machine.
`results/benchmark/aggregate.csv` also has median, p95 and throughput, and
`results/benchmark/resources.json` the peak memory and wall time.

GPU: with a CUDA build of torch (Phase 4), copy `configs/benchmark.yaml`,
set `device: cuda:0` and a new `run_name`, and run the same three commands.

### 3. Full image experiment

```
uv run python scripts/run_image_experiment.py --config configs/image_experiment.yaml --run-name my_image_main
uv run python scripts/evaluate_image_experiment.py --run results/my_image_main
uv run python scripts/plot_results.py --image-run results/my_image_main
uv run python scripts/compare_runs.py --reference results/image_main --candidate results/my_image_main
```

Check:
- `accuracy: MATCH` on all 30 rows.
- `results/my_image_main/summary.md` has the same tables and the same
  hypothesis-check lines as `results/image_main/summary.md` (latency
  numbers aside).
- The figures in `results/my_image_main/figures/` look like the reference
  ones.

Reference wall time is in `results/image_main/resources.json`.

### 4. Sequence experiment, sweep and fault injection

```
uv run python scripts/run_sequence_experiment.py --config configs/sequence_experiment.yaml --run-name my_sequence_main
uv run python scripts/run_sequence_experiment.py --config configs/sequence_faults.yaml --run-name my_sequence_faults
uv run python scripts/evaluate_sequence_experiment.py --run results/my_sequence_main
uv run python scripts/evaluate_sequence_experiment.py --run results/my_sequence_faults \
    --adopted results/my_sequence_main/adopted_filter.json
uv run python scripts/plot_results.py --sequence-run results/my_sequence_main \
    --fault-run results/my_sequence_faults
uv run python scripts/compare_runs.py --reference results/sequence_main --candidate results/my_sequence_main
uv run python scripts/compare_runs.py --reference results/sequence_faults --candidate results/my_sequence_faults
diff results/sequence_main/adopted_filter.json results/my_sequence_main/adopted_filter.json && echo "same filter adopted"
```

Check:
- `accuracy: MATCH` for both runs (90 and 6 rows).
- The same filter setting is adopted by the sweep.
- The fault run has the same UNKNOWN frame counts (the faults are seeded per
  frame, so they hit the same frames everywhere).

This is the long step: the reference wall time is in
`results/sequence_main/resources.json` (1080p frames are degraded ten times
each).

### 5. Check one number by hand

Pick any row of `results/my_image_main/aggregate.csv` and recompute it
from the saved files, as in the hand-checks in
`docs/features/image-experiment.md` and
`docs/features/sequence-experiment.md`. The metrics are plain functions in
`src/robust_visual_perception/metrics.py` and `sequence_metrics.py`, so a
check is a few lines of Python on `per_frame.csv` and the `.npz` files.

### What a mismatch means

| Output | Likely cause |
|---|---|
| manifest diff in step 1 | different annotation file (compare the SHA-256 in `data/README.md`) or a changed selection rule |
| accuracy differences of a few thousandths in a few rows | different CPU or torch build (float rounding near the 0.5 threshold) |
| large accuracy differences everywhere | different model weights or package versions; compare the two `env.json` files |
| latency ratio far from 1 | your device; that is the benchmark |
