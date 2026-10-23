# visual-perception

A reproducible robotics perception project evaluating lightweight object
detection under visual degradation, with temporal decision filtering,
latency analysis, and optional ROS 2/Gazebo integration.

**Question:** how does a small pretrained detector (YOLO26n, CPU) degrade
under blur, low light and sensor noise, and does a temporal filter make a
perception-based safety state more stable, at what delay cost?

How everything fits together (data flow, components, tests):
[docs/data-flow.md](docs/data-flow.md). Design and methodology:
[docs/design.md](docs/design.md).

## Status

| Phase | Content | State |
|---|---|---|
| 1 | Offline pipeline: degradations, detector, decision, temporal filter, run outputs, smoke test | done |
| 2 | COCO image experiment and MOT17 sequence experiment, filter sweep, fault injection | done |
| 3 | ROS 2 Jazzy + Gazebo Harmonic (Docker), TurtleBot3 camera feeding a perception node, repeated runs, bag replay | done (teleop demo still to record) |
| 4 | CPU vs GPU latency on a second machine | planned |
| 5 | Final report | planned |

Results are generated from saved runs, never typed by hand:

- [results/image_main/summary.md](results/image_main/summary.md): COCO,
  750 images × 10 conditions × 3 input sizes
- [results/sequence_main/summary.md](results/sequence_main/summary.md):
  MOT17, temporal filter, sweep and fault injection
- [results/sim/summary/summary.md](results/sim/summary/summary.md):
  TurtleBot3 simulation, live node and bag replay, 3 repeats each of a
  static and a scripted-motion run ([what the runs show](docs/features/sim-runs.md#results))

Read them together with [docs/limitations.md](docs/limitations.md).
Problems I ran into while building this, and what to avoid:
[docs/troubleshooting.md](docs/troubleshooting.md). A results
table in this README follows in Phase 5.

## Pipeline

```
image source ─► degradation ─► detector ─► decision ─► temporal filter ─► per_frame.csv
 (COCO, MOT17)   (blur, low      (YOLO26n)   (CLEAR /     (N-of-K +
                  light, noise)               STOP /       release
                                              UNKNOWN)     hysteresis)
```

Each stage has its own page in [docs/features/](docs/features/README.md).

In simulation (Phase 3), a ROS 2 node runs the same per-frame step on the
camera of a TurtleBot3 Waffle Pi in Gazebo, while a scripted person walks
past:

```
Gazebo camera ─► perception_node ─► /perception/detections, /perception/safety_state
                   (same process_frame, sim ROI, 2/3/3 filter)
rosbag ─► MOT17-layout export + GT from the simulator ─► offline sequence runner
```

The ROS parts run in Docker ([ros2-container.md](docs/features/ros2-container.md)).

## Quick start

```
uv sync                                    # Python 3.12 venv, CPU PyTorch
uv run pytest                              # unit tests, no model weights needed
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
uv run python scripts/run_image_experiment.py --config configs/smoke.yaml
```

The smoke test writes `results/smoke/per_frame.csv` (20 images × 10
conditions). Details: [docs/setup.md](docs/setup.md),
[docs/features/smoke-test.md](docs/features/smoke-test.md).

Full experiments: [image experiment](docs/features/image-experiment.md#reproduce),
[sequence experiment](docs/features/sequence-experiment.md#reproduce).

## Reproduce the results and benchmark your device

Step-by-step recreation of each phase, with a check after every step:

- Phase 1: [docs/plans/phase-1-offline-pipeline.md](docs/plans/phase-1-offline-pipeline.md#recreate-phase-1-and-check-it-yourself)
- Phase 2: [docs/plans/phase-2-experiments.md](docs/plans/phase-2-experiments.md#recreate-phase-2-and-check-it-yourself)
- Phase 3: [docs/plans/phase-3-ros2-sim.md](docs/plans/phase-3-ros2-sim.md#recreate-phase-3-and-check-it-yourself)

Quick device benchmark (750 COCO images, three input sizes), compared with
the reference machine:

```
uv run python scripts/fetch_coco_subset.py --n-person 500 --n-negative 250 --name main
uv run python scripts/run_image_experiment.py --config configs/benchmark.yaml
uv run python scripts/evaluate_image_experiment.py --run results/benchmark
uv run python scripts/compare_runs.py --reference results/image_main --candidate results/benchmark
```

`compare_runs.py` checks that your accuracy numbers match the committed run
and prints your latency next to the reference.

## Layout

```
configs/    experiment configs (YAML); configs/sim/ for the simulation world
data/       manifests (committed) and raw data (git-ignored)
docker/     ROS 2 Jazzy + Gazebo image, compose file, run scripts
docs/       design, setup, per-feature notes, plans
models/     detector weights (downloaded, git-ignored)
results/    run outputs (final runs only are committed)
scripts/    command-line entry points
ros2_ws/    ROS 2 package rvp_ros: nodes, launch file, generated world
src/robust_visual_perception/   core package (no ROS dependency)
tests/      pytest unit tests (no ROS needed)
```

## Scope and limits

- The STOP state is an image-space heuristic ("a confident person detection
  is in view"). It does not estimate distance and is not collision
  avoidance.
- Degradations are synthetic and applied one at a time.
- No training or fine-tuning; the detector is used as shipped.

## License

AGPL-3.0, matching Ultralytics, which this project depends on.
