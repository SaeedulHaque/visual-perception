# Features

One page per part of the pipeline, in data-flow order.

| Page | Module |
|---|---|
| [COCO subset](coco-subset.md) | `coco_subset.py`, `scripts/fetch_coco_subset.py` |
| [MOT17 data and selection](mot17.md) | `mot17.py`, `remote_zip.py`, `scripts/fetch_mot17.py` |
| [Image source](image-source.md) | `image_source.py` |
| [Degradations](degradation.md) | `degradation.py` |
| [Detector](detector.md) | `detector.py` |
| [Baseline decision](decision.md) | `decision.py` |
| [Temporal filter](temporal-filter.md) | `temporal_filter.py` |
| [Config and outputs](config-and-outputs.md) | `config.py`, `environment.py`, `reporting.py` |
| [Runner and smoke test](smoke-test.md) | `experiment_runner.py`, `scripts/run_image_experiment.py` |
| [Metrics](metrics.md) | `metrics.py`, `sequence_metrics.py` |
| [Image experiment](image-experiment.md) | `detection_store.py`, `image_eval.py` |
| [Sequence experiment](sequence-experiment.md) | `sequence_runner.py`, `sequence_eval.py` |
| Figures and summaries | `plots.py`, `summary.py`, `scripts/plot_results.py` |
| Comparing with the reference runs | `compare.py`, `scripts/compare_runs.py` (see the "Recreate" sections in `docs/plans/`) |
| [ROS 2 container](ros2-container.md) | `docker/` |
| [Simulation world and ground truth](sim-ground-truth.md) | `sim_geometry.py`, `sim_world.py`, `sim_export.py`, `sim_measure.py` |
| [Perception node](perception-node.md) | `ros_image.py`, `rvp_ros/perception_node.py` |
| [Scripted drive and command guard](scripted-drive.md) | `drive_profile.py`, `rvp_ros/scripted_drive.py`, `rvp_ros/cmd_guard.py` |
| [Simulation runs, replay and evaluation](sim-runs.md) | `sim.launch.py`, `run_monitor.py`, `export_bag.py`, `sim_eval.py`, `sim_report.py` |
