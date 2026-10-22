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
| 2 | COCO image experiment and MOT17 sequence experiment | planned |
| 3 | ROS 2 Jazzy + Gazebo Harmonic (Docker), TurtleBot3 camera feeding a perception node | planned |
| 4 | CPU vs GPU latency on a second machine | planned |
| 5 | Final report | planned |

No results yet. Numbers will appear here only from saved runs in
`results/`.

## Pipeline

```
image source ─► degradation ─► detector ─► decision ─► temporal filter ─► per_frame.csv
 (COCO, MOT17)   (blur, low      (YOLO26n)   (CLEAR /     (N-of-K +
                  light, noise)               STOP /       release
                                              UNKNOWN)     hysteresis)
```

Each stage has its own page in [docs/features/](docs/features/README.md).

## Quick start

```
uv sync                                    # Python 3.12 venv, CPU PyTorch
uv run pytest                              # unit tests, no model weights needed
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
uv run python scripts/run_smoke_test.py --config configs/smoke.yaml
```

The smoke test writes `results/smoke/per_frame.csv` (20 images × 10
conditions). Details: [docs/setup.md](docs/setup.md),
[docs/features/smoke-test.md](docs/features/smoke-test.md).

## Layout

```
configs/    experiment configs (YAML)
data/       manifests (committed) and raw data (git-ignored)
docs/       design, setup, per-feature notes, plans
models/     detector weights (downloaded, git-ignored)
results/    run outputs (final runs only are committed)
scripts/    command-line entry points
src/robust_visual_perception/   core package (no ROS dependency)
tests/      pytest unit tests
```

## Scope and limits

- The STOP state is an image-space heuristic ("a confident person detection
  is in view"). It does not estimate distance and is not collision
  avoidance.
- Degradations are synthetic and applied one at a time.
- No training or fine-tuning; the detector is used as shipped.

## License

AGPL-3.0, matching Ultralytics, which this project depends on.
