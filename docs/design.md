# Design: Robust Visual Perception for Autonomous Inspection

Status: approved · 2026-10-22

## 1. Intent

A small, reproducible robotics-perception prototype that measures how a
lightweight pretrained detector behaves under controlled visual degradation,
and whether confidence-aware temporal filtering stabilises a perception-based
safety state, and at what response-delay cost.

It is the baseline for my robotics-perception work and a base for later
research. Its value is methodology, honesty and reproducibility, not code
volume. The offline experiment is completed before any robotics extension
starts.

### Research questions

1. How does blur, low illumination and sensor noise, at increasing severity,
   change detection quality (precision, recall, AP) of a nano detector?
2. How do inference latency and throughput depend on input resolution and
   hardware (CPU, optional GPU)?
3. Does N-of-K temporal filtering reduce unstable safety-state decisions, and
   what detection delay and missed-event cost does it introduce?

### Hypotheses (stated before any experiment is run)

- H1: AP and recall for `person` decrease monotonically with severity for all
  three degradations; low light and noise at high severity hurt more than blur.
- H2: Latency grows roughly with input pixel count; degradation type does not
  materially change latency.
- H3: Temporal filtering reduces state transitions per minute versus the
  per-frame baseline, increases mean event-onset delay by roughly the filter
  window, and can suppress short events.

Results are reported whether they support these or not.

## 2. Goals, non-goals, anti-goals

Goals
- Working offline pipeline: images/video → degradation → detector → decision
  → temporal filter → metrics, CSVs, plots, summary.
- Ground-truth-based metrics (COCO subset, MOT17 sequences).
- Unit tests for every core module; tests do not need model weights.
- One-command reproduction of every number in the README.
- Extension 1: ROS 2 Jazzy + Gazebo Harmonic in Docker; the camera of a
  simulated TurtleBot3 Waffle Pi feeds a perception node that publishes
  detections and a safety state.
- Extension 2: CPU vs GPU latency comparison on Machine B with identical
  inputs.

Non-goals
- Training or fine-tuning any model.
- Distance estimation from bounding boxes; SLAM; navigation; motion control.
- Autonomous or perception-driven motion: the perception node never
  publishes velocity commands. Motion comes only from human keyboard teleop
  (demo run) or a fixed, pre-written velocity script (scripted run), both
  independent of perception output (section 7).
- ONNX/TensorRT optimisation (future work).

Anti-goals (failure even if it "works")
- Any number in docs not produced by a saved run.
- Calling detection retention "recall", or stability "safety".
- Presenting the image-space STOP heuristic as collision avoidance.
- Tuning thresholds on evaluation data to make a method look better.
- Weeks of simulator troubleshooting at the expense of the offline results.
- Empty placeholder packages/sections to look complete.

## 3. Constraints and environment

Machine A (measured 2026-10-22): Ubuntu 26.04 LTS, kernel 7.0.0-31,
i7-13620H (10C/16T, AVX2, AVX-VNNI), 30 GiB RAM, 194 GB free, Intel Raptor
Lake UHD iGPU, Wayland, Docker present, no ROS/Gazebo, system Python 3.14.4,
`uv` present.

- Host Python stays untouched. Core uses a `uv`-managed **Python 3.12** venv
  (`.venv/`), because 3.14 wheel coverage for torch/opencv/ultralytics is not
  guaranteed. No `sudo pip`.
- PyTorch from the **CPU wheel index** (avoids multi-GB CUDA wheels) on
  Machine A.
- ROS 2 Jazzy has no native packages for 26.04, so ROS + Gazebo run only in a
  Docker image based on the official `ros:jazzy` image with `ros-jazzy-ros-gz`
  (Gazebo Harmonic). Host stays clean.
- Machine B (GTX 1650, WSL) is optional; nothing in the core depends on it.
- Core package must import and run without ROS installed.

## 4. Decisions made

| Topic | Decision |
|---|---|
| Detector | Ultralytics **YOLO26n** pretrained on COCO; fallback YOLO11n if YOLO26n does not install/run |
| License | Repo relicensed **MIT → AGPL-3.0** to match Ultralytics |
| Images | COCO val2017 subset with GT boxes |
| Video | MOT17 sequences with per-frame GT |
| ROS | Docker, ROS 2 Jazzy + Gazebo Harmonic, TurtleBot3 Waffle Pi in a custom world with a scripted walking person |
| GPU | Machine B comparison; the benchmark scripts are run there by hand |
| Git | One feature branch per phase, pushed to `origin` and merged after review |
| Commits | Conventional Commits, one logical change each |

## 5. Architecture

```
            ┌───────────── core package: robust_visual_perception (no ROS) ─────────────┐
 source ──► │ image_source ─► degradation ─► detector ─► decision ─► temporal_filter     │
 (COCO,     │      │              │             │            │            │             │
  MOT17,    │      └──────────────┴─────────────┴────────────┴────────────┴─► metrics   │
  rosbag    │                                                                 reporting │
  frames)   └──────────────────────────────────────────────────────────────────────────┘
                                   ▲ same functions imported by
 TB3 camera ───► ros_gz_bridge ─► perception_node (ROS 2 adapter, Docker only)
                                   ├─► /perception/detections  vision_msgs/Detection2DArray
                                   └─► /perception/safety_state std_msgs/String
```

### Modules (src/robust_visual_perception/)

- `config.py`: dataclasses loaded from YAML; the run's resolved config is
  saved next to its results.
- `image_source.py`: iterates `Frame(frame_id, image_bgr_uint8, timestamp_s,
  status, error)` from an image list, image folder or MOT sequence (an image
  folder with a frame rate). Validates HxWx3 uint8. A missing/corrupt file
  yields a `Frame` with `status=MISSING`, `image=None` and the reason in
  `error`; it is never silently skipped.
- `degradation.py`: pure functions, never modify input (copy), validate shape
  and dtype, clip to [0, 255], return uint8.
  - `gaussian_blur(img, ksize)`: odd ksize, sigma derived by OpenCV.
  - `low_light(img, gain)`: multiply in float32, clip, cast back.
  - `gaussian_noise(img, sigma, rng)`: additive N(0, sigma²) per pixel and
    channel. Seed per frame = `base_seed + zlib.crc32(frame_id)` (stable across runs,
    unlike Python `hash`),
    so results do not depend on processing order.
  - `apply(condition, img, frame_id)`: dispatch from a `Condition(name,
    kind, param)`.
- `detector.py`: `Detector` protocol: `detect(image) -> DetectionResult`.
  `DetectionResult(status: OK|EMPTY|ERROR, detections: list[Detection],
  latency_ms, error: str|None)`. `Detection(class_id, class_name, confidence,
  x1, y1, x2, y2)` in original-image pixels, validated (0 ≤ conf ≤ 1,
  x1 < x2, y1 < y2, clipped to image). `UltralyticsDetector` is the only
  vendor-specific code. Missing weights give an actionable error message.
  Exceptions from inference become `status=ERROR`, logged, never swallowed.
- `decision.py`: `baseline_decision(result, target_class, conf_threshold,
  roi=None) -> Decision(state, reason)` where `State ∈ {CLEAR, STOP,
  UNKNOWN}`. A missing frame (`result=None`) or a detector `ERROR` gives
  `UNKNOWN` with a reason. Optional image ROI (see 6.3).
- `temporal_filter.py`: `PersistenceFilter(n_on, k_window, m_off)`. STOP
  asserted when ≥ `n_on` of last `k_window` valid frames are STOP; released
  to CLEAR after `m_off` consecutive CLEAR frames. `UNKNOWN` input: the
  filter outputs `UNKNOWN` immediately (it never reports CLEAR through
  missing data) and clears its window and release counter. The last
  asserted state is kept, so a STOP that was active before a gap still needs
  `m_off` fresh CLEAR frames to release. On release the window is cleared,
  so old STOP frames cannot re-trigger STOP. Downstream consumers can treat
  `UNKNOWN` as stop-with-reason. Staleness in the ROS node is handled by its
  watchdog (section 7), not by the filter.
- `metrics.py`: COCO evaluation (via `pycocotools`), per-condition latency
  stats, detection retention, temporal stream metrics (6.3). Empty inputs
  raise or return NaN with an explicit `n`, never 0.
- `reporting.py`: CSV writing with documented columns, plots (matplotlib),
  Markdown summary generated from the aggregate CSVs.
- `experiment_runner.py`: orchestrates one experiment from a config file.

Scripts (`scripts/`): `fetch_coco_subset.py`, `fetch_mot17.py`,
`run_image_experiment.py` (also runs the smoke test), `run_sequence_experiment.py`,
`plot_results.py`, `bench_latency.py` (also used on Machine B),
`collect_env.py`.

## 6. Experimental design

### 6.1 Conditions (fixed before running; mid levels = original prompt values)

| Kind | Levels |
|---|---|
| clean | — |
| blur | ksize 7, 15, 31 |
| low light | gain 0.60, 0.45, 0.25 |
| noise | sigma 10, 20, 35 (8-bit scale), base seed 1234 |

10 conditions in total. These are synthetic settings, not claims about
real-world lighting or optics. Real low light also adds noise and motion blur.
This is a documented limitation.

Every condition uses the identical set and order of source frames.

### 6.2 Image experiment (COCO val2017)

- Subset: 500 images containing ≥1 non-crowd `person` + 250 images with no
  `person`, sampled with seed 0 from sorted image ids. Images are fetched
  individually by `coco_url` from `instances_val2017.json` (no 1 GB zip).
  Not committed. A manifest of image ids is committed.
- Inference is run once per (condition, imgsz) at `conf=0.001` and
  `max_det=300`, storing all detections, so a single pass serves mAP and the
  decision rule.
- imgsz ∈ {320, 416, 640}: 10 × 3 × 750 = 22,500 inferences.
- Metrics per (condition, imgsz):
  - COCO mAP@[.5:.95] and mAP@.5 over all 80 classes on the subset, plus
    `person` AP, with `pycocotools` restricted to the subset image ids.
    Subset mAP is not comparable to full-val2017 numbers.
  - Precision and recall for `person` at operating threshold 0.5, IoU 0.5.
  - Image-level STOP decision vs GT (any non-crowd person present; no ROI on
    COCO) at threshold 0.5:
    precision, recall, false-STOP rate on the 250 negatives.
  - Detection retention: share of clean-image `person` detections
    (conf ≥ 0.5) matched at IoU ≥ 0.5 by a `person` detection (conf ≥ 0.5)
    in the degraded version of the same image. Reported as a stability measure, never as recall.
  - Mean confidence of matched `person` detections.
- Thresholds 0.5 / IoU 0.5 are fixed a priori, not tuned.

### 6.3 Sequence experiment (MOT17, temporal filter)

- Data: MOT17 train split (has GT). Download is several GB; one detector-copy
  of each needed sequence is kept, the rest deleted. Not committed.
- Event definition (image-space, explicitly not distance): the frame is
  GT-STOP when a GT pedestrian (class 1, `considered=1`, visibility ≥ 0.5)
  has its box centre inside a central "path corridor" ROI (middle 40% of
  width, lower 70% of height) and box height ≥ 25% of image height. The
  prediction-side rule uses identical geometry on detections with
  conf ≥ 0.5.
- Sequence selection uses GT only (never model outputs): compute GT-STOP
  occupancy and number of GT events per sequence. Choose one **tuning**
  sequence and two **evaluation** sequences with occupancy between ~15% and
  ~70% and ≥ 3 GT events, preferring a moving-camera sequence for
  robot-like motion. If ROI parameters give no sequence meeting this, adjust
  ROI using GT statistics only and document it.
  *Applied in Phase 2:* the ROI above qualified one sequence. A fixed grid
  search on GT only chose `x 0.35–0.65, y ≥ 0.30, height ≥ 0.50`
  (procedure and table in `docs/features/mot17.md` and
  `data/manifests/mot17_roi_search.csv`). GT events merge runs separated by
  fewer than 5 frames.
- Filter parameters chosen a priori: `n_on=3, k_window=5, m_off=5`. A small
  sweep (n_on/k ∈ {2/3, 3/5, 4/6, 5/8}, m_off ∈ {3, 5, 8}) runs on the
  tuning sequence only. If a different setting is adopted, it is applied
  unchanged to the evaluation sequences, and all sweep results are reported.
- Run at imgsz 416 for all 10 conditions, baseline vs filtered.
- Metrics per (sequence, condition, method): frame-level precision/recall/
  accuracy vs GT-STOP; transitions per minute (fps from `seqinfo.ini`);
  fraction of frames with a state change; GT events detected / missed;
  onset delay (frames and seconds from GT event start to first STOP);
  release delay; false-STOP episodes. "Stable but wrong" is visible because
  correctness and stability are reported side by side.
- Fault injection (seeded): drop 2% of frames as MISSING and 1% as ERROR in
  one extra run, to show UNKNOWN handling. Reported separately.

### 6.4 Latency

- `time.perf_counter_ns` around the detector call only (preprocess, forward,
  postprocess). Image loading and degradation timed separately. Model loading
  is excluded. The first 10 inferences are warm-up and excluded.
- Report n, mean, median, p95 (only when n ≥ 100), throughput = 1000 /
  mean_ms in images/s (sequential, batch 1), torch thread count, CPU model,
  versions.
- CPU and memory: `psutil` process RSS sampled per run; no heavy monitoring.

### 6.5 Outputs (per run: `results/<run_name>/`)

- `config.yaml` (resolved), `env.json` (hardware, OS, package versions, git
  commit), `command.txt` (exact command).
- `per_frame.csv`: frame_id, condition, imgsz, status, n_detections,
  n_target, max_target_conf, latency_ms, baseline_state, filtered_state
  (sequence runs), gt_state (when available).
- `detections_<condition>_<imgsz>.json.gz`: COCO-format detections
  (git-ignored, large).
- `aggregate.csv`, `figures/*.png`, `summary.md`.
- Final runs referenced by the README are committed (CSVs, figures, summary);
  smoke runs are not.

## 7. Robotics extension (Phase 3, Docker)

- `docker/Dockerfile`: `ros:jazzy-ros-base` + `ros-jazzy-ros-gz`,
  `ros-jazzy-vision-msgs`, `ros-jazzy-cv-bridge`, TurtleBot3 simulation
  packages for Jazzy (binary packages if available, else the ROBOTIS
  `jazzy` branch of `turtlebot3_simulations` built with colcon),
  `ros-jazzy-teleop-twist-keyboard`; venv with
  `--system-site-packages` for torch-CPU + ultralytics + the core package.
  `docker/compose.yaml` passes `/dev/dri` (Intel GPU for Gazebo rendering);
  GUI forwarding is optional, so the camera sensor must work headless
  (`gz sim -s`).
- Robot: **TurtleBot3 Waffle Pi** (the TB3 model with an RGB camera),
  spawned into our own world through the ROBOTIS spawn launch file and model
  description. Chosen because it is a widely recognised research platform and
  a base for later SLAM / Nav2 / multi-sensor work, not because this project
  needs its LiDAR or odometry.
- World `ros2_ws/src/rvp_ros/worlds/inspection.sdf`: ground, a few static
  props, and a Gazebo **actor** walking a scripted looping path that crosses
  the robot's camera view. Stock TB3 worlds contain no people, so the actor
  is required either way.
- GT for reproducible runs: GT event intervals (actor
  inside the camera frustum and within the ROI band) are computed from the
  scripted actor trajectory, the robot pose and the camera transform and
  intrinsics from the TB3 model. Distance here comes from simulator pose,
  not from bounding boxes.
- Three sim runs:
  1. **Static run** (reproducible): robot parked at a fixed pose.
  2. **Scripted-motion run** (reproducible): `scripted_drive` node replays a
     fixed velocity profile from `config/drive_profile.yaml` (list of
     `{duration_s, linear_x, angular_z}` segments, e.g. slow forward pass
     toward the walking person, turn, return), keyed to sim time (`/clock`),
     so wall-clock speed does not change the trajectory. Safeguards: speed
     clamped to ≤ 0.15 m/s and ≤ 0.5 rad/s, zero command published at the
     end and on shutdown, command timeout so the robot stops if the script
     node dies. It never reads perception output.
  3. **Teleop demo run** (not reproducible): I drive the robot with
     `teleop_twist_keyboard` to show perception on a moving camera. Recorded
     as a rosbag and a video. GT metrics are computed for this one run (true
     pose is known) but reported separately as a single non-repeatable demo.
- GT for moving runs uses the robot's true pose from the simulator (bridged
  Gazebo pose topic), not `/odom`, so wheel-slip or physics drift between
  repeats does not corrupt GT: each frame's GT is computed from where the
  robot actually was.
- Reproducibility protocol for runs 1 and 2: each is repeated 3 times from a
  fresh container with the same world, seed and script. Reported: per-repeat
  metrics, their spread, and final-pose difference between repeats. Gazebo
  physics and ROS message timing are not bit-deterministic, so the claim is
  "repeatable within the measured spread", not "identical". Each run's
  rosbag is kept; replaying a bag through the offline runner is the
  bit-exact reproduction path (bags are not committed if large; their
  manifest and checksums are).
- Fallback: if the TB3 packages do not build or run on Jazzy within the
  troubleshooting cap, replace the robot with a fixed camera model in the
  same world; the rest of Phase 3 is unchanged.
- `perception_node`: subscribes to the bridged camera topic (actual name
  read from `ros2 topic list`), sensor-data QoS with depth 1 so stale frames
  are dropped rather than queued; calls the core detector + decision +
  filter; publishes `/perception/detections` (vision_msgs/Detection2DArray)
  and `/perception/safety_state` (std_msgs/String: `CLEAR|STOP|UNKNOWN:<reason>`);
  a watchdog timer publishes `UNKNOWN:stale` if no frame arrives in 0.5 s.
  Measures per-frame inference and end-to-end latency (image header stamp
  to publish, sim time).
- A rosbag of the camera topic is recorded and converted to frames, so the
  same offline runner (and Machine B) can replay the identical sequence.
- Troubleshooting cap: about 2 h on simulator/graphics issues. If exceeded,
  record the blocker in `docs/troubleshooting.md` and stop Phase 3.
- Risk: the detector may not recognise the rendered actor as `person`. If so,
  report that (it is a sim-to-real finding) and try one Fuel human model
  before stopping.

## 8. GPU extension (Phase 4, Machine B)

- `requirements-gpu.txt` and a short `docs/gpu.md`: inspect `nvidia-smi`
  in WSL, install CUDA torch wheel in a venv, check
  `torch.cuda.is_available()`. No CUDA toolkit install.
- Transfer the COCO subset manifest + images (or re-fetch by manifest) and
  run `bench_latency.py` with identical model, imgsz, conditions and
  thresholds on Machine B CPU and GPU. Results are labelled per machine and
  never presented as identical-hardware comparisons.
- If GPU is not visible within ~30 min, skip and say so.

## 9. Repository layout

```
README.md  LICENSE (AGPL-3.0)  pyproject.toml  requirements*.txt  .gitignore
configs/        image_experiment.yaml, sequence_experiment.yaml, smoke.yaml
data/           README.md (how to fetch), manifests/ (committed), raw/ (ignored)
models/         README.md (weights auto-downloaded, ignored)
scripts/        CLI entry points (section 5)
src/robust_visual_perception/   core package
tests/          pytest unit tests (fake detector, no weights)
results/        committed final runs only
docs/           design.md, methodology.md, limitations.md, troubleshooting.md, gpu.md
docker/         Dockerfile, compose.yaml          (Phase 3 only)
ros2_ws/src/rvp_ros/                               (Phase 3 only)
```

Created only when its phase is reached.

## 10. Phases and done-criteria

| Phase | Branch | Done when |
|---|---|---|
| 0 Design | `docs/design` | this doc approved |
| 1 Offline MVP | `feature/offline-pipeline` | smoke test on ~20 COCO images writes per_frame.csv; pytest green; license switched |
| 2 Experiments | `feature/experiments` | image + sequence experiments run; aggregate CSVs, plots, summary committed; numbers hand-checked on a small example |
| 3 ROS/Gazebo | `feature/ros2-sim` | container builds; TB3 Waffle Pi (or fixed-camera fallback) camera feeds the node; node publishes detections + safety state; static and scripted-motion runs repeated 3× with GT metrics, spread and latency; teleop demo recorded; or documented blocker |
| 4 GPU | `feature/gpu-bench` | Machine B numbers saved, or documented skip |
| 5 Docs | `docs/final-report` | README with real results table; clean-shell rerun of documented commands passes; one-page project summary |

Each phase starts with a short brief and ends with an outcome report. The
next phase starts only after that report is reviewed.

## 11. Details decided during implementation

Internal function signatures beyond those above, plot styling, CLI flag
names, logging format, exact Dockerfile layering, SDF world details (props,
camera pose, actor path), choice of which MOT17 sequences after the GT
statistics are computed (within 6.3 rules).
