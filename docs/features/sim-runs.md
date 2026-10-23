# Simulation runs, replay and evaluation

Launch: `ros2_ws/src/rvp_ros/launch/sim.launch.py`
Nodes: `run_monitor.py`, `export_bag.py` (plus the perception node, drive
and guard)
Scripts: `docker/sim_run.sh` (in the container), `scripts/run_sim.py`,
`scripts/make_sim_gt.py`, `scripts/evaluate_sim.py`,
`scripts/write_sim_manifest.py`
Core: `sim_eval.py`, `sim_report.py` (test: `test_sim_eval.py`)
Configs: `configs/sim_replay.yaml`
Results: `results/sim/`

## One run

`docker/sim_run.sh <mode> <run_name>` runs inside a fresh container:

1. `sim.launch.py` starts:
   - Gazebo headless with the inspection world, **paused**, with seed 7
   - the bridge (`config/bridge.yaml`)
   - the perception node
   - the command guard, plus the scripted drive in `scripted` mode
   - `ros2 bag record` (MCAP, zstd)
   - the run monitor
2. The run monitor waits until the camera topic has its two subscribers
   (the perception node after warm-up, and the recorder). In scripted mode
   it also waits for the drive's publisher. Then it unpauses the world.
   Every run therefore starts at sim time 0 with the person at the start
   of the loop and the robot at the origin.
3. At 120 s of sim time the monitor writes `run_info.json` (wall time,
   real-time factor) and exits. The launch file then stops everything; the
   recorder closes the bag cleanly. If any other node exits early, the run
   is shut down too, and `run_info.json` does not say `completed`.
4. `export_bag` extracts frames, stamps, robot poses and camera info from
   the bag ([sim-ground-truth.md](sim-ground-truth.md)).

| Mode | Robot | Reproducible |
|---|---|---|
| `static` | parked at the origin, facing the loop | yes |
| `scripted` | the drive profile ([scripted-drive.md](scripted-drive.md)) | yes |
| `teleop` | driven by hand with `teleop_twist_keyboard` | no: one demo run |

What the bag holds: `/camera/image_raw`, `/camera/camera_info`,
`/rvp/robot_pose`, `/odom`, `/cmd_vel`, `/rvp/cmd_vel`,
`/perception/detections` and `/perception/safety_state`. The recorder uses
wall-clock receive times: with `--use-sim-time` it would wait for `/clock`,
which the paused world does not publish yet. Everything downstream uses the
message header stamps, which are sim time.

## Repeats

```
uv run python scripts/run_sim.py --mode static --repeats 3
uv run python scripts/run_sim.py --mode scripted --repeats 3
```

Each repeat is a new `docker compose run --rm` container, so no Gazebo or
ROS state carries over. Gazebo physics and ROS message timing are not
bit-deterministic, so the claim is "repeatable within the measured spread",
not "identical". `run_sim.py` also writes `command.txt` and `env.json`
(package versions inside the container, the image id and the host's git
state) into `results/sim/<run>/`.

## Replay

```
uv run python scripts/make_sim_gt.py data/raw/sim/sequences/*_r*
uv run python scripts/run_sequence_experiment.py --config configs/sim_replay.yaml
uv run python scripts/run_sequence_experiment.py --config configs/sim_replay.yaml --run-name sim_replay_check
uv run python scripts/evaluate_sequence_experiment.py --run results/sim/sim_replay
uv run python scripts/evaluate_sequence_experiment.py --run results/sim/sim_replay_check
uv run python scripts/compare_runs.py --reference results/sim/sim_replay --candidate results/sim/sim_replay_check
```

The exported folders are in the MOT17 layout, so the Phase 2 sequence
runner replays them unchanged: every recorded frame, clean condition, the
same model, threshold, ROI and filter as the node. The replay is the
bit-exact reproduction path; running it twice checks that.

The replay runs on the host venv, while the live node ran in the container
(NumPy 1.26, OpenCV 4.11 against NumPy 2.5, OpenCV 5.0 on the host). The
live-versus-replay agreement therefore also shows whether the two
environments give the same per-frame decisions.

## Evaluation

```
uv run python scripts/check_sim_gt.py data/raw/sim/sequences/static_r1 data/raw/sim/sequences/scripted_r1
uv run python scripts/evaluate_sim.py
```

For each run, source and method:

| Source | Frames scored |
|---|---|
| live | the frames the node processed (it drops frames when busy), joined to GT by image stamp |
| replay | every recorded frame |

| Method | States |
|---|---|
| baseline | per-frame decision |
| filtered | 2/3/3 persistence filter |

Metrics are the Phase 2 stream metrics: precision, recall, accuracy,
transitions per minute, GT events detected or missed, onset delay and
false-STOP episodes. UNKNOWN counts as STOP. fps comes from the scored
frames' own stamps, so a live run with dropped frames is still scored
correctly. GT events merge runs split by fewer than 5 frames, as in Phase 2.

Also per run:
- the share of recorded frames the node processed
- inference and callback time (wall clock) and the image's age when
  processing starts (sim clock)
- the real-time factor
- the final robot pose
- the share of processed frames where the live and replayed per-frame
  states agree

Outputs in `results/sim/summary/`: `repeats.csv`, `spread.csv` (mean, min
and max over the repeats), `replay_check.json`, `gt_check.json`,
`summary.md` and `figures/`.

## Results

All numbers are from `results/sim/summary/summary.md` (3 repeats per mode,
mean with the min–max range). Read them with the simulation part of
[limitations.md](../limitations.md).

**The detector and the decision rule on the robot camera.**
- Every GT event was detected in every run: 4 per static run, 2 per
  scripted run (the drive turns the camera away from the near crossing
  for part of the run).
- No false-STOP episodes.
- Per-frame STOP recall on the live node was 0.964 (static) and 0.969
  (scripted), with precision 1.000. The missed frames are at the edges of
  each event. In `static_r1` and `scripted_r1`, every missed frame has its
  GT box centre at 0.30 or 0.70 of the image width, exactly on the ROI
  border. The person is detected there (confidence about 0.8), but the
  detected box's centre falls just outside the band.

**The filter (2/3/3) had nothing to stabilise.**
- The baseline did not flicker in these runs. Transitions per minute were
  the same with and without the filter (4.0 static, 2.0 scripted, live).
- What the filter did add is its cost:
  - mean onset delay went from 0.03 s to 0.06 s (static) and from 0.03 s
    to 0.07 s (scripted)
  - precision went from 1.000 to 0.966 and 0.955, because STOP is held for
    a few frames after the person leaves the ROI
- This is the "delay cost without a stability gain" side of H3. The MOT17
  results remain the evidence for the gain.

**Live equals replay.**
- On every frame the node processed, its per-frame state equalled the
  replay's: agreement 1.000 in all six runs. This held although the node
  ran in the container (NumPy 1.26, OpenCV 4.11) and the replay on the host
  (NumPy 2.5, OpenCV 5.0).
- A second replay of the same bags gave 0 differing rows out of 21,848.
- So the bag is a bit-exact reproduction path, and the live node is the
  offline pipeline.

**Load.**
- With Gazebo, the bridge and the recorder on the same CPU, inference in
  the node averaged 28.3 ms (static). The idle offline image run measured
  18.7 ms at the same input size (`results/image_main`, clean, 416).
- The whole callback averaged 31.1 ms. That is close to the 33 ms frame
  interval, so the node processed 89–92 % of the frames and skipped the
  rest (depth-1 queue, by design).
- Images were on average 11–13 ms old (sim time) when processing
  started.
- The real-time factor was 0.98.

**Repeatability.**
- Static runs: the same GT (170 GT-STOP frames per run) and the same event
  counts. Recall varied between 0.962 and 0.968 (live), because a
  different subset of frames was skipped in each run.
- Scripted runs: the final pose after the drive differed between repeats
  by up to 1.1 cm in x, 3.8 cm in y and 0.04 rad in yaw. That is physics
  and timing variation, recorded rather than hidden.

## Teleop demo

```
HOST_UID=$(id -u) HOST_GID=$(id -g) RENDER_GID=$(getent group render | cut -d: -f3) \
    docker compose -f docker/compose.yaml run --rm sim docker/sim_run.sh teleop teleop_r1 120
```

This needs an interactive terminal: no `-T`. The simulation runs in the
background and `teleop_twist_keyboard` takes the keyboard. Its commands go
through the command guard (speed limits, and a stop after 0.5 s without a
key). The run ends after 120 s of sim time like the others, and is
exported the same way. It is scored with the same code, but reported on its
own as a single non-repeatable demo.
