# Configuration and run outputs

Modules: `config.py`, `environment.py`, `reporting.py`
Tests: `tests/test_config.py`, `tests/test_reporting.py`

## Why

Every number this project reports must be traceable to a saved run: which
data, which model, which thresholds, which machine, which commit, which
command. So a run is fully described by one YAML file, and the run directory
stores a copy of the resolved config next to the results.

## Config file

Example: `configs/smoke.yaml`.

```yaml
run_name: smoke                    # results/<run_name>/
image_dir: data/raw/coco/val2017
manifest: data/manifests/coco_val2017_smoke.csv   # optional; gives GT and order
output_root: results
model:
  weights: models/yolo26n.pt
  imgsz: [640]                     # one pass per size
  conf: 0.001                      # inference floor (keeps candidates for mAP)
  max_det: 300
  device: cpu
decision:
  target_class: person
  conf_threshold: 0.5              # decision threshold, fixed a priori
filter:                            # used by sequence runs
  n_on: 3
  k_window: 5
  m_off: 5
noise_seed: 1234
warmup: 10                         # untimed calls per imgsz
conditions:                        # optional; default = the ten design conditions
  - {name: clean, kind: clean}
  - {name: blur_k7, kind: blur, param: 7}
```

Loading is strict. An unknown key anywhere (a typo such as `imgsize`), a
missing `run_name` / `image_dir`, or two conditions with the same name
raise `ValueError`. A silent default would be worse than an error, because
the run would look fine and measure the wrong thing.

## Run directory

`results/<run_name>/` contains:

| File | Content |
|---|---|
| `config.yaml` | the resolved config (defaults filled in) |
| `env.json` | Python, OS, CPU model, cores, RAM, package versions, torch thread count, CUDA flag, git commit and dirty flag |
| `command.txt` | the exact command line |
| `per_frame.csv` | one row per (frame, condition, imgsz) |

If the directory already exists, the run stops with `FileExistsError`. It
is replaced only with `--overwrite`, so earlier results are never lost by
accident.

`python scripts/collect_env.py` prints the same environment record.

## `per_frame.csv` columns

| Column | Meaning |
|---|---|
| `frame_id` | image file stem (COCO image id, zero-padded) |
| `condition` | condition name, e.g. `blur_k15` |
| `imgsz` | detector input size |
| `status` | `OK`, `EMPTY` (no boxes), `ERROR` (detector failed) or `MISSING` (frame unreadable) |
| `n_detections` | boxes kept at the inference floor (`conf ≥ 0.001`) |
| `n_target` | target-class boxes with confidence ≥ decision threshold |
| `max_target_conf` | highest target-class confidence (any value), empty if none |
| `latency_ms` | detector call time, `perf_counter_ns` |
| `baseline_state` | per-frame decision: `CLEAR`, `STOP`, `UNKNOWN` |
| `state_reason` | `missing_frame` or `detector_error` when `UNKNOWN` |
| `filtered_state` | temporal-filter output (sequence runs only) |
| `gt_state` | ground-truth state when known (COCO: `STOP` if the image has a non-crowd person) |

`n_detections` is large at the 0.001 floor (often hundreds of mostly
low-confidence boxes). It is not a count of objects in the image.

## Notes

- The `git_dirty` flag in `env.json` shows whether the run used
  uncommitted code. Final runs should come from a clean tree.
- Detections themselves are stored in Phase 2
  (`detections_<condition>_<imgsz>.json.gz`), when mAP is computed.
