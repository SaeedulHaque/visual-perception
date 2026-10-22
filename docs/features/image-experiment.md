# Image experiment (COCO)

Modules: `experiment_runner.py`, `detection_store.py`, `image_eval.py`
Scripts: `run_image_experiment.py`, `evaluate_image_experiment.py`,
`plot_results.py`
Config: `configs/image_experiment.yaml`
Tests: `test_experiment_runner.py`, `test_detection_store.py`,
`test_image_eval.py`
Results: `results/image_main/` (`summary.md`, `aggregate.csv`, `figures/`)

## Question

How much do blur, low light and noise reduce detection quality of a nano
detector, and does input resolution change that? (Design RQ1, RQ2, H1, H2.)

## Setup

- 750 COCO val2017 images: 500 with at least one non-crowd person, 250 with
  no person (manifest `data/manifests/coco_val2017_main.csv`, seed 0).
- 10 conditions × 3 input sizes (320, 416, 640) = 22,500 inferences.
- Inference at `conf=0.001`, `max_det=300` so mAP sees the full ranking;
  decision and precision/recall at confidence 0.5.

## Reproduce

```
uv run python scripts/fetch_coco_subset.py --n-person 500 --n-negative 250 --name main
uv run python scripts/run_image_experiment.py --config configs/image_experiment.yaml
uv run python scripts/evaluate_image_experiment.py --run results/image_main
uv run python scripts/plot_results.py --image-run results/image_main
```

Inference wall time is recorded in `results/image_main/resources.json`;
evaluation (30 COCOeval passes) takes a few minutes.

## Inference and evaluation are separate

The run writes only raw outputs:

- `per_frame.csv`: one row per (image, imgsz, condition) with status,
  counts, latency, degradation time and the decision.
- `detections_<condition>_<imgsz>.npz`: every box above the inference floor,
  as an N×7 array `[image_id, x, y, w, h, score, coco_category_id]`. That is
  the array layout `pycocotools` accepts directly. YOLO's 80 class indices
  are mapped to COCO's non-contiguous category ids (`COCO80_TO_91`; person
  is 1 in both). These files are large and git-ignored.

`evaluate_image_experiment.py` reads them and writes `aggregate.csv`, one
row per (condition, imgsz). Changing a metric means re-running the
evaluation, not the model.

## Loop order

Frame → input size → condition. Each image is read from disk once, and the
three detectors are built and warmed up (10 untimed calls each) before the
first frame. Frames are streamed, not held in memory.

## Hand-check

Two images checked by hand at imgsz 640 against the COCO annotations and the
stored boxes (confidence ≥ 0.5):

| Image | Condition | GT persons | Predictions | Best IoU per prediction | Expected TP/FP/FN | `person_pr` |
|---|---|---|---|---|---|---|
| 241668 | clean | 2 (heights 601, 544 px) | 0.79, 0.62 | 0.94, 0.70 | 2/0/0 | 2/0/0 |
| 241668 | blur_k31 | 2 | 0.79 | 0.68 | 1/0/1 | 1/0/1 |
| 441586 | clean | 5 (heights 255, 51, 20, 21, 46 px) | 0.88, 0.56 | 0.97, 0.84 | 2/0/3 | 2/0/3 |
| 441586 | blur_k31 | 5 | 0.82 | 0.90 | 1/0/4 | 1/0/4 |

Image 441586 also shows why person recall at 0.5 is well below 1 even on
clean images: three of its five people are 20 to 51 px tall, and the nano
model at this threshold does not find them.

## Reproducibility check

The experiment was run twice on the same machine (the first time with light
other work going on, the second with the machine idle; the second is the
committed run). `compare_runs.py --tolerance 0` on the two aggregates:
`rows compared: 30; largest accuracy difference: 0`, and the per-frame
columns other than timing (`status`, counts, confidences, decisions) are
byte-identical. Only latency differed, by about 2 %.

`configs/benchmark.yaml` (clean images only) reproduced the three `clean`
rows exactly as well. Anyone can run the same check; see the
"Recreate Phase 2" section of `docs/plans/phase-2-experiments.md`.

## Reading the results

`results/image_main/summary.md` holds the tables and the mechanical
hypothesis checks; `figures/` holds the plots. Points to keep in mind:

- mAP is computed on this 750-image subset, so it is not directly
  comparable to published full-val2017 numbers.
- Retention for `clean` is 1 by definition.
- The false-STOP rate is measured on 250 negative images, so each false
  STOP moves it by 0.004.
- Latency was measured with the three input sizes interleaved per image;
  see [limitations](../limitations.md).
