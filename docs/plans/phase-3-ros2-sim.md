# Phase 3 Plan: ROS 2 and Gazebo simulation

**Goal:** run the same detector, decision rule and temporal filter as a
ROS 2 node on the camera of a simulated TurtleBot3 Waffle Pi. A scripted
person walks through the robot's view. Ground truth comes from the
simulator, so the node's safety state can be scored with the Phase 2
metrics.

**Architecture:** everything ROS-related runs in one Docker image
(ROS 2 Jazzy, Gazebo Harmonic, TurtleBot3 packages). The ROS package
`rvp_ros` holds thin nodes. Everything that can be tested without ROS goes
in the core package and runs in the host test suite:
- ground-truth geometry
- image conversion
- the drive profile and command guard
- bag export
- evaluation

A recorded bag is exported to the same folder layout as a MOT17 sequence.
The Phase 2 sequence runner then replays it unchanged.

**Spec:** `docs/design.md` section 7 (and 6.3 for the metrics).

**Done when:**
- The image builds.
- The camera of the Waffle Pi feeds the node.
- The node publishes detections and a safety state.
- The static and the scripted-motion runs are each repeated 3 times, with GT
  metrics, their spread and latency.
- Bags are replayed through the offline runner.
- The teleop demo is ready to record.
- Every feature has a page in `docs/features/`.

## Progress

Implemented on `feature/ros2-sim`; not yet committed.

- [x] Task 1: container (`docker/`; `check_container.sh` passes: imports,
  233 core tests inside the container, headless camera frame)
- [x] Task 2: simulation ground truth (world generated from
  `configs/sim/world.yaml`; person size measured: `configs/sim/person.yaml`)
- [x] Task 3: image conversion and perception node
- [x] Task 4: scripted drive and command guard
- [x] Task 5: launch, recording, export
- [x] Task 6: simulation evaluation (`results/sim/summary/`)
- [x] Task 7: runs and docs
- [x] Done: the image builds
- [x] Done: the Waffle Pi camera feeds the node; the node publishes
  detections and a safety state
- [x] Done: static run ×3 and scripted run ×3, each in a fresh container,
  with GT metrics, spread and latency
- [x] Done: bags replayed through the offline runner; a second replay is
  identical (0 of 21,848 rows differ)
- [x] Done: GT check against detections (`results/sim/summary/gt_check.json`)
- [x] Done: teleop mode ready (launch, recording and export checked
  without a keyboard)
- [x] Done: teleop demo recorded (`results/sim/teleop_r1`), with the Gazebo
  window, and an annotated video and still sheet of it
- [x] Done: feature pages, data flow, README, limitations and
  troubleshooting updated
- [x] Done: committed and pushed to `feature/ros2-sim`

## Global constraints (from the design, unchanged)

- The perception node never publishes velocity commands. Motion comes only
  from the scripted drive or from keyboard teleop. Neither one reads
  perception output.
- Scripted drive limits: speed ≤ 0.15 m/s and ≤ 0.5 rad/s. A zero command
  is sent at the end and on shutdown, and a command timeout stops the robot
  if the script dies. The profile is keyed to sim time.
- Ground truth uses the simulator's true robot pose, not `/odom`. Distance
  comes from the simulator pose, never from bounding boxes.
- Each reproducible run is repeated 3 times, each from a fresh container.
  The claim is "repeatable within the measured spread", not "identical".
- Watchdog: `UNKNOWN:stale` when no frame has arrived for 0.5 s.
- Camera subscription: sensor-data QoS, depth 1 (stale frames are dropped).
- Troubleshooting cap: about 2 h on simulator or graphics problems. If it
  is exceeded, record the blocker and stop.
- No numbers in the docs unless they come from a saved run.

## Decisions made in this plan

Facts checked before writing this plan, in a scratch container:
- `ros-jazzy-turtlebot3-gazebo` 2.3.7 exists as a binary package, so no
  source build is needed.
- Headless rendering (`gz sim -s --headless-rendering`) publishes the
  640×480 camera at about 30 Hz.
- The Fuel walking actor is detected as `person` (confidence about 0.75 to
  0.9 on a few test frames).
- The robot's pose appears in `/world/<world>/dynamic_pose/info`; the
  actor's does not.
- Camera intrinsics: fx = fy = 530.47, cx = 320, cy = 240.

1. **No `cv_bridge`.** The node converts `sensor_msgs/Image` to a BGR
   array with numpy (`ros_image.py`, core package, tested on the host).
   This removes the dependency on cv_bridge's own NumPy and OpenCV build.
2. **NumPy 1.26 in the container.** The ROS Jazzy Python message packages
   are compiled against NumPy 1.x and do not load under NumPy 2. The core
   package's pin becomes `numpy>=1.26`, and the core test suite also runs
   inside the container, to show that the core code works on both.
3. **Actor ground truth from its own path.** Gazebo does not publish the
   actor's pose.
   - The path is defined once, in `configs/sim/actor_path.yaml`: corners,
     speed and turn time.
   - A script generates the world's waypoints from it, every 0.25 s.
     Waypoints that dense keep Gazebo's spline interpolation within a few
     centimetres of the straight-segment path the GT code uses.
   - A test checks that the committed world matches the generated one.
4. **Camera pose.** The camera pose is the robot's true pose (full 3D
   orientation; see the changes below for its source) combined with the
   `camera_rgb_frame` offset from the TB3 model file (0.076, 0, 0.093 m). The camera looks along the link's +x axis. The intrinsics are
   read from `camera_info` in the bag.
5. **Person volume.** The person is modelled as a vertical box, with height
   H and half-width r.
   - H and r are measured once from a rendered frame by background
     subtraction (silhouette extent at a known distance), not from detector
     output.
   - The GT box is the image box of the 8 projected corners, clipped to the
     image like detections.
   - A box exists only when all corners are in front of the camera.
   - No prop stands between the robot and the path, so there is no
     occlusion.
6. **Sim ROI: `x 0.30–0.70, y 0–1, height ≥ 0.25`.** The x band and the
   height come from the design (6.3). The vertical band is dropped. The
   camera sits about 0.1 m above the ground, so a person's box centre is
   always in the upper part of the image, and the design's `y ≥ 0.30` band
   would never fire. This follows from geometry and is fixed before any run.
7. **Filter setting 2/3/3.** This is the setting the Phase 2 procedure
   adopted. The node logs the baseline state as well, so both are scored.
8. **Same detector settings as offline:** imgsz 416, `conf=0.001`,
   `max_det=300`. Live and replayed decisions can then be compared frame by
   frame. Only detections with confidence ≥ 0.25 are published, to keep
   messages small.
9. **Watchdog on the steady (wall) clock.** If Gazebo stops, sim time stops
   too, and a sim-time timer would never fire.
10. **Command guard node.**
    - Both the scripted drive and teleop publish to `/rvp/cmd_vel`.
    - `cmd_guard` clamps the speeds and forwards to `/cmd_vel`. It sends
      zero when no command has arrived for 0.5 s (steady clock) and on
      shutdown.
    - The launch file shuts the whole simulation down if the drive or the
      guard exits early.
11. **Runs.**
    - Each run lasts 120 s of sim time with a fixed Gazebo seed. The path
      gives about 4 near crossings, which should be STOP, and 4 far
      crossings, which should be CLEAR.
    - The static run parks the robot at the origin, facing the path.
    - The scripted run drives toward the path, turns, waits, turns back and
      reverses.
12. **Replay.**
    - Bag → `data/raw/sim/<run>/` in MOT17 layout: PNG frames,
      `seqinfo.ini`, and `gt/gt.txt` from the projected GT boxes.
    - The existing `run_sequence_experiment.py` and evaluation then run on
      it (clean condition, filter 2/3/3).
    - Each bag is replayed twice, to check the replay is bit-exact. Live
      and replayed decisions are compared on the frames the node processed.
13. **Storage.** Bags (MCAP) go under `data/raw/sim/bags/`, which is not
    committed. Their sizes and SHA-256 checksums are committed in
    `data/manifests/sim_bags.csv`.
14. **Teleop demo.** The launch mode, the recording and the export are
    ready. I drive the run by hand, so it is listed as a manual step, not
    an automated one.

## Changed during implementation

- **Robot in the world file.** The robot is part of the world file
  instead of being spawned with the ROBOTIS launch file. It is the same
  model description. The world then loads paused with everything in place,
  and the run monitor unpauses it once all nodes are ready.
- **Robot pose source.** The bridged `dynamic_pose/info` messages carry no
  entity names or stamps. A PosePublisher added to the robot in the world
  file publishes its true pose (`/rvp/robot_pose`, sim time).
- **Recorder clock.** The bag recorder uses wall-clock receive times.
  With `--use-sim-time` it waits for `/clock`, which a paused world does
  not publish.
- **Latency split.** A sim-time "publish minus stamp" left out the
  processing, because the node's clock does not advance during a
  callback. The node logs the image's age at callback start (sim clock)
  and the callback time (wall clock). The first set of runs was discarded
  and all runs redone.
- **Export split.** The bag export in the container only extracts raw
  data. GT is computed on the host (`make_sim_gt.py`), after the person's
  size has been measured from a static run.
- **Gazebo window for teleop.** The first teleop attempt showed no window,
  since Gazebo runs headless. `RVP_GUI=1` (launch argument `gui:=true`)
  now also starts the Gazebo GUI client on the host's X display. Only
  teleop runs allow it, so the measured runs are unchanged.
- **Annotated video.** Not planned. The bag export also writes the node's
  published detections (`detections.csv`), and `render_sim_video.py`
  draws them with the GT box and the states on every frame
  ([sim-video.md](../features/sim-video.md)).

## Review focus

1. A bag with a dropped camera frame must not shift the frame-to-GT
   alignment in the exported sequence. Export writes GT per frame from that
   frame's own stamp, and the stamp gaps are reported. Test in Task 5.
2. A person partly behind the camera, or at the image border, must give no
   box or a clipped box, never a box with inverted corners. Test in Task 2.
3. A drive profile with a speed above the limits must be clamped. A profile
   that ends must leave the robot at zero. Test in Task 4.
4. An image encoding other than `rgb8`/`bgr8`, or a row stride with padding,
   must be converted correctly or rejected with a clear error. Test in
   Task 3.
5. Repeats with different frame counts (the node drops frames under load)
   must still be scored. Metrics use per-run fps from the stamps. Test in
   Task 6.

---

### Task 1: Container

**Files:**
- `docker/Dockerfile`, `docker/compose.yaml`, `.dockerignore`
- `pyproject.toml` (numpy pin)
- `docker/check_container.sh`
- doc `features/ros2-container.md`

- Base `ros:jazzy-ros-base`. Packages: `ros-jazzy-ros-gz`,
  `ros-jazzy-turtlebot3-gazebo`, `ros-jazzy-vision-msgs`,
  `ros-jazzy-teleop-twist-keyboard`, `ros-jazzy-rosbag2-storage-mcap`,
  `python3-venv`.
- Fuel actor fetched at build time, so runs do not depend on the network.
- A venv with `--system-site-packages` holds torch CPU, ultralytics and the
  core dependencies, with the NumPy version constrained to the system's
  1.26.4.
- The repository is mounted at `/work`. The container runs as the host user
  so result files are not owned by root. `/dev/dri` is passed through, with
  the video and render groups.
- `check_container.sh` checks:
  - imports (rclpy, sensor_msgs, the core package, ultralytics)
  - that the core pytest suite passes in the container
  - that a headless camera frame arrives

### Task 2: Simulation ground truth (core)

**Files:**
- `sim_geometry.py`, `tests/test_sim_geometry.py`
- `configs/sim/actor_path.yaml`, `scripts/make_sim_world.py`
- `ros2_ws/src/rvp_ros/worlds/inspection.sdf`
- doc `features/sim-ground-truth.md`

- `ActorPath.from_config(dict)`, with `position(t) -> (x, y, yaw)` (periodic)
  and `waypoints(dt) -> list[(t, x, y, z, yaw)]`.
- `CameraModel(fx, fy, cx, cy, width, height)`, and
  `camera_pose(robot_pose, offset) -> 4x4`.
- `person_box(person_xy, camera_T, model, height, half_width) -> np.ndarray | None`
  returns a clipped xyxy box.
- `world_sdf(path, ...) -> str` is used by the script. A test compares it
  with the committed world.

Tests:
- the person straight ahead at 3 m projects to a centred box, with the
  height computed by hand
- a person behind the camera gives `None`
- a person at the image border gives a box clipped to the border
- `position` is linear on a segment and periodic
- turns keep the position fixed

### Task 3: Image conversion and perception node

**Files:**
- `ros_image.py`, `tests/test_ros_image.py`
- `ros2_ws/src/rvp_ros/` (`package.xml`, `setup.py`,
  `rvp_ros/perception_node.py`)
- `ros2_ws/src/rvp_ros/config/bridge.yaml`
- doc `features/perception-node.md`

- `image_to_bgr(data, width, height, step, encoding) -> np.ndarray` covers
  `rgb8`, `bgr8` and `mono8`, with row padding. Other encodings raise
  `ValueError`.
- The node:
  - builds the detector once and warms it up
  - per frame: detect, `baseline_decision` with the sim ROI, filter
  - publishes `Detection2DArray` (with the image's header) and
    `std_msgs/String` (`CLEAR`, `STOP` or `UNKNOWN:<reason>`)
  - runs the watchdog
  - writes `live_frames.csv` (stamp, status, counts, inference ms, sim-time
    end-to-end ms, baseline and filtered state)

### Task 4: Scripted drive and command guard

**Files:**
- `drive_profile.py`, `tests/test_drive_profile.py`
- `rvp_ros/scripted_drive.py`, `rvp_ros/cmd_guard.py`
- `config/drive_profile.yaml`
- doc `features/scripted-drive.md`

- `DriveProfile.load(path)` and `command_at(t) -> (v, w)`, zero after the
  end.
- `clamp(v, w, limits)`.
- `CommandGuard(timeout_s, limits)`, with `on_command(v, w, now)` and
  `output(now) -> (v, w)`.

### Task 5: Launch, recording, export

**Files:**
- `launch/sim.launch.py`, `rvp_ros/run_monitor.py`, `rvp_ros/export_bag.py`
- `sim_export.py`, `tests/test_sim_export.py`
- `docker/sim_run.sh`, `scripts/run_sim.py`
- doc `features/sim-runs.md`

- Launch arguments: `mode:=static|scripted|teleop`, `duration`, `run_dir`,
  `seed`.
- The launch starts:
  - Gazebo (headless) with the world
  - the robot spawn and the bridge
  - the perception node
  - the guard, plus the drive in scripted mode
  - `ros2 bag record` (MCAP)
  - the run monitor, which exits at the given sim time and so ends the
    launch
- Export writes frames, stamps, robot poses and camera info.
  `sim_export.write_sequence(...)` writes the MOT17 layout with GT from the
  geometry.
- `run_sim.py --mode static --repeats 3` starts one fresh container per
  repeat.

### Task 6: Simulation evaluation

**Files:**
- `sim_eval.py`, `tests/test_sim_eval.py`
- `scripts/evaluate_sim.py`, `configs/sim_replay.yaml`
- doc section in `features/sim-runs.md`

- Live metrics per repeat and method use `stream_metrics` with the run's
  own fps.
- Latency: inference ms and end-to-end ms on sim time. Also reported: the
  processed-frame fraction, the real-time factor and the final robot pose.
- Spread over repeats: mean, min, max.
- Agreement: share of node-processed frames where the live and replayed
  baseline states are equal.
- Writes `results/sim/summary/` (`repeats.csv`, `spread.csv`,
  `summary.md`, figures).

### Task 7: Runs and docs

- Measure the person's size. Generate the world. Run static ×3 and scripted
  ×3. Export, replay twice, evaluate.
- GT check: overlay GT boxes on clean frames, and record the IoU with
  detections (as a check only, nothing is tuned).
- Update `docs/data-flow.md`, the README, `docs/limitations.md` and
  `docs/troubleshooting.md`. Add the "Recreate Phase 3" section, with the
  commands as run.

---

## Recreate Phase 3 and check it yourself

These steps rebuild the container, record your own runs, and compare them
with the committed ones in `results/sim/`. The reference runs were made on
the machine described in each run's `env.json` (i7-13620H with its Intel
iGPU, Docker 29, Ubuntu 26.04 host).

What should match and what should not:

| Output | Expected |
|---|---|
| GT events per run, GT-STOP frames per static run | identical (the person's loop and the robot's start are fixed) |
| replay results of a given bag | identical, every time, on the same software |
| live recall, precision, onset delay | close to the reference, within the spread of its three repeats; your CPU decides how many frames the node can process |
| frames processed, inference and callback time, real-time factor | your device; this is the benchmark |
| final pose after the scripted drive | within a few centimetres of the reference |

### 0. Prerequisites

Docker with Compose v2, about 7 GB of disk for the image (6.7 GB here), and the Phase 1
setup (`uv sync`, model weights in `models/`). A GPU is not required:
Gazebo renders on the Intel iGPU through `/dev/dri`. Without a usable GPU,
set `RVP_SOFTWARE_RENDERING=1` (slower).

### 1. Build and check the container

```
docker compose -f docker/compose.yaml build
HOST_UID=$(id -u) HOST_GID=$(id -g) RENDER_GID=$(getent group render | cut -d: -f3) \
    docker compose -f docker/compose.yaml run --rm -T sim docker/check_container.sh
```

Check: the package versions print with `numpy 1.26.4`, the core tests all
pass inside the container, and the last line is
`camera frame received (640 px wide)`.

### 2. Record runs

Use your own run-name prefix, so the committed runs stay intact:

```
uv run python scripts/run_sim.py --mode static --repeats 3 --prefix mystatic
uv run python scripts/run_sim.py --mode scripted --repeats 3 --prefix myscripted
```

Each repeat simulates 120 s; `run_info.json` records its wall time. Check:
- every repeat prints `exit 0`
- `results/sim/mystatic_r1/run_info.json` says `"completed": true`, with a
  real-time factor; compare it with `results/sim/static_r1/run_info.json`
- `data/raw/sim/sequences/mystatic_r1/export.json` reports 0 missing frames

If a run fails, its container log is in `data/raw/sim/logs/<run>.log`.

### 3. Ground truth and replay

The size of the person is already in `configs/sim/person.yaml`. To measure
it yourself from your own static run, without overwriting the committed
file:

```
uv run python scripts/measure_sim_person.py data/raw/sim/sequences/mystatic_r1 --out /tmp/person.yaml
diff configs/sim/person.yaml /tmp/person.yaml
```

Only the comment line naming the source run should differ.

Then compute GT and replay your bags:

```
uv run python scripts/make_sim_gt.py data/raw/sim/sequences/my*_r*
```

The printed GT-STOP frame counts for your static runs should equal the
reference (`grep -c ',1$' results/sim/static_r1/gt_frames.csv`). Then copy
`configs/sim_replay.yaml`, set `run_name: my_replay` and
`sequences: [mystatic_r1, ..., myscripted_r3]`, and run:

```
uv run python scripts/run_sequence_experiment.py --config my_sim_replay.yaml
uv run python scripts/evaluate_sequence_experiment.py --run results/sim/my_replay
```

### 4. Evaluate and compare

`evaluate_sim.py` scores every run under `results/sim/` that has a node
log. To score only yours, point it at a copy:

```
mkdir -p /tmp/mysim && cp -r results/sim/my* /tmp/mysim/
uv run python scripts/evaluate_sim.py --results /tmp/mysim --replay my_replay --replay-check none
```

Read `/tmp/mysim/summary/summary.md` next to `results/sim/summary/summary.md`:
- GT events and events detected per run should be the same
- recall and onset delay should be close (within the spread of three
  repeats)
- `live = replay` should be 1.000, or very close
- the latency and frames-processed rows are your device's numbers

### 5. Replay the committed bags (bit-exact path)

The committed results come from bags that are not in the repository (see
`data/manifests/sim_bags.csv` for their checksums). With a copy of the bags
in `data/raw/sim/bags/`, extract them in the container and replay:

```
for run in static_r1 static_r2 static_r3 scripted_r1 scripted_r2 scripted_r3; do
  HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/compose.yaml run --rm -T sim \
    ros2 run rvp_ros export_bag --bag /work/data/raw/sim/bags/$run --out /work/data/raw/sim/sequences/$run
done
uv run python scripts/make_sim_gt.py data/raw/sim/sequences/static_r* data/raw/sim/sequences/scripted_r*
uv run python scripts/run_sequence_experiment.py --config configs/sim_replay.yaml --run-name my_replay
uv run python scripts/evaluate_sequence_experiment.py --run results/sim/my_replay
uv run python scripts/compare_runs.py --reference results/sim/sim_replay --candidate results/sim/my_replay
```

On the same package versions the replay matches exactly. The committed
second replay (`results/sim/sim_replay_check`) was compared this way:
`rows compared: 12; largest accuracy difference: 0`.

### 6. Teleop demo (by hand)

```
xhost +SI:localuser:$(whoami)      # once per login, for the Gazebo window
HOST_UID=$(id -u) HOST_GID=$(id -g) RENDER_GID=$(getent group render | cut -d: -f3) RVP_GUI=1 \
    docker compose -f docker/compose.yaml run --rm sim docker/sim_run.sh teleop teleop_r1 120
```

Drive with the keys `teleop_twist_keyboard` prints (`i` forward, `j`/`l`
turn, `k` stop). The guard caps the speed at 0.15 m/s and 0.5 rad/s, and
stops the robot 0.5 s after the last key. `teleop_twist_keyboard` sends
one command per key press, so hold a key to keep moving; a single tap
moves the robot for about 0.5 s. The arrow keys do nothing. The Gazebo
window takes the keyboard focus when it opens: click the terminal before
driving. Do not press Ctrl-C: it ends the whole run early. The run ends after 120 s of sim time and is exported like the
others. Then:

```
uv run python scripts/make_sim_gt.py data/raw/sim/sequences/teleop_r1
uv run python scripts/evaluate_sim.py
```

It is scored live only, as a single demo run. To watch it:

```
uv run python scripts/render_sim_video.py --run teleop_r1 \
    --sheet results/sim/summary/figures/frames_teleop_r1.png
```

Check: the video is in `data/raw/sim/videos/teleop_r1.mp4`, and the
sheet shows six frames with green GT boxes, orange detections and the
state banner.
