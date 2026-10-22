# How the project fits together

This page is the map. It explains what the project is for, how data moves
through it, how the parts depend on each other, and how each part is
tested. Each stage has a detailed page in [features/](features/README.md);
the reasons behind the choices are in [design.md](design.md).

## 1. What the project is about

A robot that stops for people needs a camera-based detector, and real
cameras do not deliver clean images. This project measures three things:

1. **Robustness.** How much a small pretrained detector (YOLO26n, on a
   laptop CPU) loses when images are blurred, darkened or noisy, at
   controlled strengths.
2. **Cost.** How inference latency depends on input resolution and hardware.
3. **Stability.** Whether a temporal filter turns a flickering per-frame
   STOP / CLEAR signal into a stable one, and how much delay it adds.

Everything is measured against ground truth (COCO, MOT17, and later the
simulator), and every reported number has to come from a saved run.

## 2. Project flow (phases)

```
Phase 0  Design           docs/design.md: questions, hypotheses, fixed parameters
   │
Phase 1  Offline pipeline  core package + tests + smoke test          ◄── done
   │
Phase 2  Experiments       COCO image experiment, MOT17 sequence experiment,  ◄── done
   │                       metrics, filter sweep, faults, plots, summaries
Phase 3  Robotics          ROS 2 Jazzy + Gazebo Harmonic in Docker; TurtleBot3
   │                       camera ─► perception node using the same core package
Phase 4  GPU comparison    same benchmark on a second machine (CPU vs GPU)
   │
Phase 5  Report            README results table, clean re-run of all commands
```

The offline pipeline comes first because it is where the measurements
happen. The ROS node in Phase 3 adds no new perception logic; it wraps the
same functions, so simulator results are directly comparable.

## 3. End-to-end data flow

Every experiment has three stages, each a separate script:

```
fetch (once)          run (inference only)              evaluate + plot
──────────────        ─────────────────────             ───────────────────────────
data/manifests/  ──►  results/<run>/per_frame.csv  ──►  results/<run>/aggregate.csv
data/raw/             results/<run>/detections_*.npz    results/<run>/figures/*.png
                      config.yaml env.json ...          results/<run>/summary.md
```

Metrics never run inside inference. The model runs once; metrics can be
recomputed and checked from the saved files.

### 3.1 Image experiment (COCO)

```
                    ┌──────────────────── one-time data preparation ────────────────────┐
                    │                                                                   │
 cocodataset.org ──►│ scripts/fetch_coco_subset.py                                      │
                    │   annotations zip ─► instances_val2017.json                       │
                    │   select_subset(seed 0) ─► data/manifests/coco_val2017_<name>.csv │
                    │   download selected images ─► data/raw/coco/val2017/*.jpg         │
                    └───────────────────────────────┬───────────────────────────────────┘
                                                    │ manifest + images
                                                    ▼
 configs/image_experiment.yaml ──► load_config() ──► ExperimentConfig
                                                    │
                                                    ▼
 ┌────────────────────────────── run_image_experiment() ───────────────────────────────┐
 │                                                                                     │
 │  manifest ─► image paths + gt_state (STOP if the image has a person)                │
 │                                                                                     │
 │  build one detector per imgsz, 10 untimed warm-up calls each                        │
 │  iter_image_files ──► Frame(frame_id, image, status OK | MISSING)   (streamed)      │
 │        │  for each frame (read once):                                               │
 │        │    for each imgsz (320, 416, 640):                                         │
 │        │      for each of the 10 conditions:   process_frame()                      │
 │        ▼                                                                            │
 │   ┌───────────┐ uint8  ┌───────────┐ DetectionResult ┌───────────┐ Decision         │
 │   │ apply()   │──────► │ detect()  │───────────────► │ baseline_ │───────┐          │
 │   │ degrade   │ image  │ YOLO26n   │ boxes, latency  │ decision  │ state │          │
 │   └───────────┘        └───────────┘ status          └───────────┘ reason│          │
 │                                                                          ▼          │
 │                                              PerFrameWriter.write(row) ─► CSV row   │
 │   all boxes ─► DetectionStore ─► detections_<condition>_<imgsz>.npz                 │
 │   MISSING frame ─────────────── skips degrade/detect ──► UNKNOWN / missing_frame    │
 └─────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 results/image_main/  config.yaml env.json command.txt resources.json per_frame.csv *.npz
                                                    │
 evaluate_image_experiment.py   image_eval.evaluate_image_run()
   stored boxes + COCO GT ─► coco_map (pycocotools), person_pr, retention
   per_frame.csv          ─► decision_metrics (image STOP), latency_stats
                                                    ▼
 aggregate.csv (one row per condition × imgsz) ─► plot_results.py ─► figures/, summary.md
```

### 3.2 Sequence experiment (MOT17)

```
 motchallenge.net MOT17.zip (5.9 GB, never fully downloaded)
   │ RangeFile: HTTP range requests, zipfile reads the archive directory remotely
   ├─► seqinfo.ini + gt/gt.txt of 7 training sequences
   │     gt_stop_states() with each ROI in the grid ─► mot17_roi_search.csv
   │     choose_roi() ─► ROI;  select_sequences() ─► mot17_sequences.csv (roles)
   └─► extract_prefix(): one byte-range request per selected sequence ─► img1/

 configs/sequence_experiment.yaml ──► SequenceExperimentConfig
   for each sequence (tuning MOT17-09, evaluation MOT17-05, MOT17-11):
     GT-STOP per frame from gt.txt (same ROI rule as the detections)
     one PersistenceFilter per condition, reset per sequence
     for each frame (one path per frame number, so gaps stay aligned):
       optional injected fault (MISSING / ERROR), same frames for every condition
       for each condition:
         process_frame(..., roi) ─► Decision ─► filter.update() ─► filtered state
         row: sequence, frame, condition, baseline, filtered, gt, latency
   ─► results/sequence_main/per_frame.csv

 evaluate_sequence_experiment.py   sequence_eval.evaluate_sequence_run()
   tuning sequence: refilter() baseline states with 12 settings ─► sweep.csv
                    adopt_setting() (dominance rule) ─► adopted_filter.json
   every (sequence, condition): stream_metrics() for baseline, filtered
                    (and filtered_adopted if the sweep changed the setting)
   ─► aggregate.csv ─► plot_results.py ─► figures/, summary.md
```

The fault run (`configs/sequence_faults.yaml`) is the same loop with 2 %
MISSING and 1 % ERROR frames, evaluated with the main run's adopted setting.

## 4. What travels between stages

| From → To | Type | Contents |
|---|---|---|
| image source → degradation | `Frame` | `frame_id`, HxWx3 uint8 BGR `image` (or `None`), `timestamp_s`, `status` OK / MISSING, `error` |
| degradation → detector | `np.ndarray` | a new HxWx3 uint8 image; the original is never modified |
| detector → decision | `DetectionResult` | `status` OK / EMPTY / ERROR, list of `Detection` (class, confidence, box in pixels), `latency_ms`, `image_hw`, `error` |
| decision → filter | `Decision` | `state` CLEAR / STOP / UNKNOWN, `reason` (`missing_frame`, `detector_error`) |
| filter → output | `State` | filtered CLEAR / STOP / UNKNOWN |
| decision → output | `FrameOutput` | the `Decision`, the `DetectionResult` (or `None`), and the CSV fields of this step (`process_frame`) |
| detector → store | `Detection` list | kept per (condition, imgsz) as N×7 arrays in COCO result layout |
| per-frame CSV → metrics | CSV rows, `.npz` arrays | the only inputs of the evaluation step |
| config → everything | `ExperimentConfig` / `SequenceExperimentConfig` | data paths, model settings, thresholds, filter, ROI, faults, conditions, seed, warm-up count |

Every type is a small frozen dataclass or enum. No stage receives a vendor
object (such as an Ultralytics `Results`), so stages can be tested, and
replaced, independently.

## 5. How the components depend on each other

```
                       scripts/run_image_experiment.py
                           │          │          │
                           ▼          ▼          ▼
                       config   experiment_runner   detector (UltralyticsDetector)
                         │        │  │  │  │  │  │
          ┌──────────────┘        │  │  │  │  │  └──► reporting ──► config
          ▼                       │  │  │  │  │            └─────► environment
     degradation ◄────────────────┘  │  │  │  └──► coco_subset  (read_manifest)
                                     │  │  └─────► image_source
                                     │  └────────► decision ──► detector (types only)
                                     └───────────► detector
                                                       ▲
     temporal_filter ──► decision (State)              │
                                                  only module that
     scripts/fetch_coco_subset.py ──► coco_subset imports ultralytics
     scripts/collect_env.py       ──► environment

 Phase 2 additions
     sequence_runner ──► experiment_runner.process_frame, mot17, temporal_filter,
                         reporting, config
     mot17           ──► decision (Roi)
     remote_zip         (stdlib only)
     detection_store ──► detector (types)
     metrics            (numpy, pycocotools)
     image_eval      ──► metrics, detection_store, coco_subset, config
     sequence_metrics ─► metrics, mot17 (events), temporal_filter
     sequence_eval   ──► sequence_metrics, mot17, config
     plots, summary     read aggregate CSVs only
```

Rules the layout follows:

- **Arrows point one way.** Low-level modules (`degradation`, `image_source`,
  `detector` types) import nothing from the package. Only
  `experiment_runner` knows about all stages.
- **One vendor boundary.** `UltralyticsDetector` is the only code that
  imports `ultralytics`. Everything else talks to the `Detector` protocol
  (`detect(image) -> DetectionResult`).
- **No ROS in the core.** The package imports and runs without ROS. The
  Phase 3 node will import these modules, not the other way round.
- **Scripts are thin.** They parse arguments, build objects, and call one
  package function. All logic that matters lives in `src/` where it is
  tested.

## 6. What each component does

| Component | Job | Key decisions | Page |
|---|---|---|---|
| `coco_subset` + `fetch_coco_subset.py` | build a reproducible image set with ground truth | person = non-crowd person box; negative = no person; crowd-only images excluded; seeded shuffle so the smoke set is a prefix of the main set; atomic downloads | [coco-subset](features/coco-subset.md) |
| `image_source` | turn files into frames | unreadable file → MISSING frame, never skipped; grayscale → 3 channels | [image-source](features/image-source.md) |
| `degradation` | apply one condition | 10 fixed conditions; noise seeded by `base_seed + crc32(frame_id)` | [degradation](features/degradation.md) |
| `detector` | run YOLO26n, time it | inference floor `conf=0.001`, `max_det=300`; exceptions → ERROR result; latency is only the detector call | [detector](features/detector.md) |
| `decision` | per-frame CLEAR / STOP / UNKNOWN | threshold 0.5 fixed a priori; optional corridor ROI; no data → UNKNOWN, never CLEAR | [decision](features/decision.md) |
| `temporal_filter` | stabilise the state over time | STOP when 3 of last 5 are STOP; release after 5 CLEAR in a row; UNKNOWN passes through and clears history | [temporal-filter](features/temporal-filter.md) |
| `config` | describe a run in YAML | unknown keys are errors, not silent defaults | [config-and-outputs](features/config-and-outputs.md) |
| `environment` | record machine and versions | CPU, RAM, package versions, torch threads, git commit and dirty flag | [config-and-outputs](features/config-and-outputs.md) |
| `reporting` | write run outputs | fixed CSV header; existing run directory never overwritten without `--overwrite` | [config-and-outputs](features/config-and-outputs.md) |
| `experiment_runner` | orchestrate one image run; `process_frame` is the shared per-frame step | each image read once and streamed; warm-up excluded; every failure becomes a row | [smoke-test](features/smoke-test.md) |
| `detection_store` | keep every box for mAP | N×7 `.npz` per (condition, imgsz) in pycocotools result layout | [image-experiment](features/image-experiment.md) |
| `metrics` | image metrics | greedy one-to-one matching at IoU 0.5; crowd regions ignored; NaN for empty denominators | [metrics](features/metrics.md) |
| `image_eval` | aggregate one image run | one row per (condition, imgsz); UNKNOWN counts as STOP | [image-experiment](features/image-experiment.md) |
| `remote_zip` | read a remote zip partially | range requests; one request per sequence span; `.part` then rename | [mot17](features/mot17.md) |
| `mot17` | GT-STOP, events, selection | GT boxes clipped like detections; events merge gaps < 5 frames; ROI grid search on GT only | [mot17](features/mot17.md) |
| `sequence_runner` | run sequences through the filter | one filter per condition, reset per sequence; seeded faults identical across conditions | [sequence-experiment](features/sequence-experiment.md) |
| `sequence_metrics` | stream metrics, sweep, adoption | onset/release delays, transitions, false-STOP episodes; dominance rule | [metrics](features/metrics.md) |
| `sequence_eval` | aggregate one sequence run | sweep on tuning sequence only; adopted setting applied unchanged | [sequence-experiment](features/sequence-experiment.md) |
| `plots`, `summary` | figures and `summary.md` | generated from CSVs only; mechanical hypothesis checks | [image-experiment](features/image-experiment.md) |

## 7. One frame's journey

Taking COCO image `000000241668` (two people in the manifest) through the
`blur_k7` condition at `imgsz=640`:

1. The manifest row says `n_person=2`, so `gt_state = STOP`.
2. `iter_image_files` loads `data/raw/coco/val2017/000000241668.jpg` into a
   `Frame` with status OK.
3. `apply(Condition("blur_k7", "blur", 7), image, "000000241668")` returns a
   blurred copy.
4. `UltralyticsDetector.detect` resizes to 640, runs the network, times
   the call and returns every box with confidence ≥ 0.001.
5. `baseline_decision` finds a `person` box with confidence ≥ 0.5 and
   returns `STOP`.
6. The runner writes one row. Real row from `results/image_main`
   (`sequence` is empty for image runs):

```
sequence,frame_id,condition,imgsz,status,n_detections,n_target,max_target_conf,latency_ms,load_ms,degrade_ms,baseline_state,state_reason,filtered_state,gt_state
,000000241668,blur_k7,640,OK,119,1,0.7418,31.488,2.741,0.822,STOP,,,STOP
```
7. Its boxes go to `detections_blur_k7_640.npz`; the evaluation later
   matches them against the image's two GT persons. The hand-check in
   [image-experiment.md](features/image-experiment.md) walks through this
   image under `clean` and `blur_k31`.

`n_detections` counts boxes above the 0.001 floor (mostly
low-confidence candidates kept for mAP), not people in the image.
`n_target=1` is the number of person boxes at ≥ 0.5. The clean version of
the same image has two, so the blur removed one confident detection. That
is the kind of change the experiments add up over hundreds of images.

If the file had been corrupt, steps 3 to 5 would be skipped and the row
would read `status=MISSING`, `baseline_state=UNKNOWN`,
`state_reason=missing_frame`.

## 8. Tests

```
uv run pytest            # 174 tests, a few seconds, no model weights or network needed
uv run ruff check .      # lint
```

| Test file | Tests | What it proves |
|---|---|---|
| `test_degradation.py` | 29 | output shape and dtype, input untouched, exact low-light values, noise clipping, noise identical per frame id and different across ids, invalid images and parameters rejected, the 10 conditions match the design |
| `test_image_source.py` | 8 | valid image loads, grayscale becomes 3 channels, missing and corrupt files become MISSING without stopping iteration, sorted folder order, timestamps from fps |
| `test_detector.py` | 9 | `Detection` validation, boxes clipped to the image, degenerate boxes dropped, OK vs EMPTY status, clear error for missing custom weights |
| `test_decision.py` | 13 | 0.5 is STOP and 0.49 is CLEAR, other classes ignored, missing frame and detector error give UNKNOWN with reasons, ROI accepts and rejects the right boxes |
| `test_temporal_filter.py` | 13 | the worked examples from the filter page: onset after 3 of 5, release after 5 CLEAR, release count reset by STOP, behaviour around UNKNOWN gaps, no re-trigger from stale history, invalid parameters |
| `test_config.py` | 9 | defaults match the design, nested sections parse, YAML round-trip (image and sequence configs), typos and missing keys raise, duplicate condition names raise, one imgsz for sequences |
| `test_reporting.py` | 7 | CSV header, unknown column rejected, existing run directory protected, metadata and resources files written, environment record complete |
| `test_coco_subset.py` | 7 | pool rules (crowd-only excluded), too-large requests raise, smaller subset is a prefix, seed matters, manifest round-trip, failed download leaves no files, existing files skipped |
| `test_experiment_runner.py` | 10 | one row per frame and condition, ground truth from manifest, warm-up calls not written, low-confidence person reported but CLEAR, MISSING and ERROR rows, run continues after a detector failure, existing run not overwritten, folder mode without manifest, each image read once across sizes, detections and resources saved |
| `test_detection_store.py` | 4 | round trip in COCO result layout, class index → COCO category id, positive box sizes, missing file raises |
| `test_metrics.py` | 16 | IoU values, greedy matching order, TP/FP/FN, threshold, crowd-ignored predictions, retention, NaN for empty ratios, decision metrics, latency stats and p95 rule, COCO mAP = 1 for perfect predictions and 0 without predictions |
| `test_image_eval.py` | 4 | a perfect fake detector scores 1 everywhere, a degraded condition losing the person, a false STOP on a negative image, column order |
| `test_mot17.py` | 11 | seqinfo parsing, all GT filters (considered, class, visibility), ROI geometry, GT clipping, event merging, selection rule, ROI choice rule |
| `test_remote_zip.py` | 5 | range reads behave like a file, zipfile lists and reads remote members, one request per extracted span, interrupted extract leaves nothing, existing destination skipped |
| `test_sequence_runner.py` | 7 | GT, baseline and filtered states per frame, ROI applied, missing image keeps alignment, full fault rates, faults identical across conditions, deterministic fault draw, filter reset between sequences |
| `test_sequence_metrics.py` | 9 | the worked example on the metrics page, never-stopping stream, release into the next event, UNKNOWN as STOP, `refilter` = online filter, sweep grid, adoption rule cases |
| `test_sequence_eval.py` | 3 | baseline and filtered rows, sweep only on the tuning sequence, adopted rows only when the setting changed |
| `test_summary_and_plots.py` | 3 | monotonic check text, image counts and fault rates read from data, H3 check text, figure files written |
| `test_compare.py` | 7 | timing columns recognised, identical results match while timing is reported separately, differences above tolerance reported, small ones accepted, only shared rows compared, NaN equals NaN, nothing in common is not a match |

**Checking against the reference runs.** `scripts/compare_runs.py`
compares a re-run's `aggregate.csv` with a committed run: accuracy must
agree within a tolerance, latency is printed side by side as a device
benchmark. Step-by-step recreation of each phase is at the end of each plan
in `docs/plans/`.

**How the tests avoid the model.** `tests/fakes.py` provides
`FakeDetector`, which replays a scripted list of detections or exceptions.
Because the runner only needs something with a `detect()` method, the
whole pipeline is tested with fake detections in milliseconds. The real
model is exercised by the smoke test.

**Smoke test (integration).** Real images, real model:

```
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
uv run python scripts/run_image_experiment.py --config configs/smoke.yaml
```

It passes when `results/smoke/per_frame.csv` has 200 rows (20 images × 10
conditions) and the three metadata files exist. Its numbers are a sanity
check, not results.

## 9. Reproducibility

| Source of variation | How it is fixed |
|---|---|
| which images | committed manifest, seeded selection |
| image content | SHA-256 of the annotation files in `data/README.md` |
| noise | per-frame seed from `crc32(frame_id)`, independent of processing order |
| faults | per-frame seed from `crc32(sequence/frame)`, identical across conditions |
| MOT17 sequences and ROI | GT-only rules in code; search table and roles committed in `data/manifests/` |
| metrics | recomputed from saved files by the evaluate scripts; summaries generated, never typed |
| parameters | YAML config, copied into each run directory |
| software | `uv.lock` pins every package; `env.json` records versions and git commit |
| hardware | `env.json` records CPU, cores, RAM, torch threads |
| thresholds | fixed in the design before any run, never tuned on evaluation data |

## 10. Where to start reading

1. [design.md](design.md): the questions and why the method looks like this.
2. This page: how the pieces connect.
3. [features/](features/README.md): one stage at a time.
   [troubleshooting.md](troubleshooting.md) lists the problems met on the
   way and how they were solved.
4. `src/robust_visual_perception/experiment_runner.py`: the whole pipeline
   in one function; `sequence_runner.py` for the video version.
5. `results/*/summary.md`: the generated results.
6. `tests/test_temporal_filter.py`: the filter's behaviour as readable
   input → output strings.
