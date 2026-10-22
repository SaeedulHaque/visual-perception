# Sequence experiment (MOT17, temporal filter)

Modules: `sequence_runner.py`, `sequence_metrics.py`, `sequence_eval.py`
Scripts: `run_sequence_experiment.py`, `evaluate_sequence_experiment.py`,
`plot_results.py`
Configs: `configs/sequence_experiment.yaml`, `configs/sequence_faults.yaml`
Tests: `test_sequence_runner.py`, `test_sequence_metrics.py`,
`test_sequence_eval.py`
Results: `results/sequence_main/`, `results/sequence_faults/`

## Question

Does the N-of-K filter make the STOP / CLEAR signal more stable on video,
how much later does it react, and does it miss events? (Design RQ3, H3.)

## Setup

- Sequences: MOT17-09 (tuning), MOT17-05 and MOT17-11 (evaluation), selected
  from GT only ([mot17.md](mot17.md)).
- All ten conditions, imgsz 416, corridor ROI from the GT-only search.
- Per frame and condition: degrade → detect → `baseline_decision` with the
  ROI → that condition's `PersistenceFilter`. Each condition has its own
  filter, reset at the start of every sequence.
- Online filter: the design default `n_on=3, k_window=5, m_off=5`.

## Reproduce

```
uv run python scripts/fetch_mot17.py
uv run python scripts/run_sequence_experiment.py --config configs/sequence_experiment.yaml
uv run python scripts/run_sequence_experiment.py --config configs/sequence_faults.yaml
uv run python scripts/evaluate_sequence_experiment.py --run results/sequence_main
uv run python scripts/evaluate_sequence_experiment.py --run results/sequence_faults \
    --adopted results/sequence_main/adopted_filter.json
uv run python scripts/plot_results.py --sequence-run results/sequence_main \
    --fault-run results/sequence_faults
```

## Filter sweep

The filter only sees the per-frame state, so filtering stored baseline
states offline gives exactly what the online filter would have output. The
sweep therefore needs no new inference: `refilter` replays the tuning
sequence's baseline states through each of the 12 settings
(n_on/k ∈ {2/3, 3/5, 4/6, 5/8} × m_off ∈ {3, 5, 8}).

Adoption rule, fixed before the sweep ran: keep 3/5/5 unless another
setting **dominates** it on the tuning sequence (averaged over the ten
conditions): no more missed events, no more transitions per minute, no
longer onset delay, and strictly better on at least one. If several
dominate, take the one with the fewest transitions per minute. The adopted
setting is then applied unchanged to the evaluation sequences
(`filtered_adopted` rows, if it differs from the default).

## Fault injection

`sequence_faults` reruns the evaluation sequences (clean condition) with 2 %
of frames dropped (MISSING) and 1 % detector failures (ERROR). Faults are
drawn per frame from `seed + crc32("sequence/frame")`, so they are
reproducible and the same frames fail under every condition. Expected
behaviour: each fault frame is UNKNOWN in both baseline and filtered
output, the filter clears its window, and an active STOP is kept until
fresh CLEAR frames arrive.

## Outputs

| File | Content |
|---|---|
| `per_frame.csv` | per (sequence, frame, condition): baseline, filtered and GT state, latency |
| `aggregate.csv` | per (sequence, condition, method) stream metrics ([metrics.md](metrics.md)) |
| `sweep.csv`, `sweep_summary.csv` | sweep per condition, and averaged per setting |
| `adopted_filter.json` | default and adopted setting |
| `summary.md`, `figures/` | generated tables, mechanical H3 checks, plots |

## Hand-check

MOT17-05, clean condition, second GT event (frames 160 to 234), states read
from `per_frame.csv` (S = STOP, C = CLEAR), frames 156 to 240:

```
GT        CCCCSSSSSSSS ... SSSSSSCCCCCC
baseline  SSSSSSSSSSSS ... SSSSSSCCCCCC
2/3/3     SSSSSSSSSSSS ... SSSSSSSSCCCC
3/5/5     SSSSSSSSSSSS ... SSSSSSSSSSCC
```

- The detector already reports STOP four frames before the GT rule fires
  (a confident person box is in the corridor before any GT box meets all
  GT conditions), so onset delay is 0 for all three.
- After the event, baseline clears on the first frame, 2/3/3 two frames
  later and 3/5/5 four frames later: release delay = `m_off − 1` frames, as
  the filter definition says. `stream_metrics` on this event gives the same
  values (onset 0 / 0 / 0, release 0 / 2 / 4 frames).

## Reading the results

`results/sequence_main/summary.md` has the tables and mechanical H3
checks. Things to know when reading them:

- **The sweep did not separate settings on stability.** On the tuning
  sequence (MOT17-09, static camera) the per-frame signal is already
  steady, so every setting gives the same transitions per minute and the
  same missed events. Only onset delay differs, and it grows with `n_on`.
  Under the declared rule 2/3/3 therefore dominates 3/5/5 (same stability,
  shorter delay) and was adopted. This is the rule working as written, but
  the tuning sequence carried little information about stability; a
  tuning sequence with more flicker would be a better test.
- **Fault run transitions.** Every UNKNOWN frame inside a CLEAR stretch adds
  two transitions (CLEAR → UNKNOWN → CLEAR) and, because UNKNOWN counts as
  STOP, one false-STOP episode. The jump in those columns in the fault
  table is this accounting, not the detector getting worse.
