# Perception node

Node: `ros2_ws/src/rvp_ros/rvp_ros/perception_node.py`
Core module: `ros_image.py` (test: `test_ros_image.py`)

## What it does

```
/camera/image_raw ──► image_to_bgr ──► process_frame (clean) ──► PersistenceFilter
 (sensor_msgs/Image)                    detect + baseline_decision      │
                                        with the sim ROI                 │
          /perception/detections ◄── Detection2DArray (image header) ◄──┤
          /perception/safety_state ◄── String: CLEAR | STOP | UNKNOWN:<reason>
          results/sim/<run>/live_frames.csv ◄── one row per processed frame
```

The node adds no perception logic of its own. Each frame goes through the
same `process_frame` step as the offline experiments (clean condition), so
a frame processed live and the same frame replayed offline give the same
per-frame decision. The node never publishes velocity commands.

| Setting | Value | Why |
|---|---|---|
| model | YOLO26n, imgsz 416, `conf=0.001`, `max_det=300` | identical to the offline runs, so live and replay are comparable frame by frame |
| decision | `person`, confidence ≥ 0.5, sim ROI | [sim-ground-truth.md](sim-ground-truth.md) |
| filter | `n_on=2, k_window=3, m_off=3` | the setting the Phase 2 procedure adopted; the baseline state is logged too |
| published detections | confidence ≥ 0.25 | keeps messages small; the decision still uses the full output |
| camera QoS | best effort, keep last, depth 1 | a frame that arrives while the previous one is processed replaces the queued one: stale frames are dropped, not queued |
| watchdog | 0.5 s on the steady clock | publishes `UNKNOWN:stale` when no frame arrived; on the wall clock so it still fires if Gazebo stops (sim time would stop too) |

The detector is warmed up (10 calls on a blank image) **before** the camera
subscription is created. The subscription is the run monitor's "ready"
signal, so the simulation only starts once the node can keep up.

## Image conversion without cv_bridge

`image_to_bgr(data, width, height, step, encoding)` turns the raw message
buffer into an H×W×3 BGR array:

- `rgb8` (what Gazebo sends) is reversed to BGR, `bgr8` is copied, and
  `mono8` is repeated into three channels.
- Rows may be padded (`step > width × channels`); the padding is skipped.
- Any other encoding, or a buffer shorter than `step × height`, raises
  `ValueError`. The node logs that and passes the frame on as MISSING, so
  the state becomes `UNKNOWN:missing_frame` instead of the node crashing.

Doing this in the core package keeps it testable on the host, and the node
does not depend on cv_bridge's NumPy and OpenCV build.

## Per-frame log

`live_frames.csv`, one row per frame the node processed:

| Column | Meaning |
|---|---|
| `stamp_ns`, `stamp_s` | image header stamp (sim time); the key for joining with GT |
| `status`, `n_detections`, `n_target`, `max_target_conf` | as in the offline per-frame CSV |
| `latency_ms` | detector call only (wall clock), as offline |
| `callback_ms` | whole callback: conversion, detection, decision, filter, publishing (wall clock) |
| `age_sim_ms` | the image's age when the callback starts: sim clock minus the image stamp (rendering, bridge and queueing) |
| `baseline_state`, `state_reason`, `filtered_state` | per-frame decision and filtered output |

`node_summary.json` holds the processed-frame count and the number of stale
episodes. `node_params.json` records the parameters the node ran with.

## Two clocks for latency

The node's sim clock comes from `/clock` messages. Those are handled by the
same single-threaded executor as the images, so the clock does not advance
while a frame is being processed. A "publish time minus stamp" in sim time
would therefore leave out the processing itself. (A first set of runs
measured it that way, and the result came out shorter than the inference
alone. Those runs were discarded and redone with the split below.)

- `age_sim_ms`: how old the image is when processing starts, on the sim
  clock.
- `callback_ms`: how long processing takes, on the wall clock.

With a real-time factor close to 1 (each run's factor is reported), their
sum approximates the delay from rendering to the published state.
