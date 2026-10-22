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
