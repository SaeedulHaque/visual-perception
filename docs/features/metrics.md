# Metrics

Modules: `metrics.py` (images), `sequence_metrics.py` (streams)
Tests: `tests/test_metrics.py`, `tests/test_sequence_metrics.py` (every metric
has a hand-computed example)

## Rules that apply to every metric

- A ratio with a zero denominator is **NaN**, never 0. "No data" must not
  look like "perfect" or "zero" performance. Tables show NaN as `–`.
- Thresholds are fixed before any run: confidence 0.5, IoU 0.5.
- Metrics are computed from saved outputs (`per_frame.csv`, stored
  detections), never during inference, so they can be recomputed and checked.

## Image metrics

| Metric | Definition | Function |
|---|---|---|
| mAP@[.5:.95], mAP@.5 | COCO evaluation over all 80 classes, restricted to the subset image ids | `coco_map` (pycocotools) |
| person AP, AP50 | same, `person` category only | `coco_map` |
| person precision / recall | person boxes with confidence ≥ 0.5, matched one-to-one to non-crowd GT persons at IoU ≥ 0.5 | `person_pr` |
| mean matched confidence | mean confidence of true-positive person boxes | `person_pr` |
| detection retention | share of confident (≥ 0.5) clean-image person boxes that a confident person box in the degraded image matches at IoU ≥ 0.5 | `retention` |
| image STOP precision / recall / false-STOP rate | image-level decision vs "image contains a non-crowd person"; false-STOP rate is measured on the 250 negatives | `decision_metrics` |
| latency | n, mean, median, p95 (only when n ≥ 100), throughput = 1000 / mean | `latency_stats` |

### Matching

`greedy_match` sorts predictions by confidence and lets each take the
unmatched GT box with the highest IoU, if that IoU ≥ 0.5. One GT box can be
matched once. This is the same greedy scheme COCO uses per threshold.

### Crowd regions

COCO marks dense groups as one `iscrowd=1` region instead of individual
boxes. A prediction that falls at least half inside such a region is
ignored (not a false positive), because there may well be a person there.
Crowd regions are never counted as missed persons.

### AP vs precision/recall

AP integrates over all confidence thresholds, so it measures how well the
model *ranks* detections. Precision/recall at 0.5 measure one operating
point, the one the decision rule actually uses. A degradation can leave AP
almost unchanged while recall at 0.5 drops, if confidences shift down
uniformly. Reporting both shows this.

### Retention is not recall

Retention compares the degraded output with the clean output of the same
model, not with ground truth. It answers "how much of what the model saw
does it still see", a stability measure. A clean false positive that
disappears under blur lowers retention but is not a loss of accuracy.

## Stream metrics (sequences)

Input: one state per frame (`CLEAR`, `STOP`, `UNKNOWN`), the GT-STOP mask
and the frame rate.

| Metric | Definition |
|---|---|
| precision, recall, accuracy | frame-level, STOP is positive, UNKNOWN counts as STOP |
| transitions per minute | number of frames whose state differs from the previous frame, per minute of video |
| change fraction | transitions / (frames − 1) |
| events, detected, missed | GT events = GT-STOP runs after merging gaps < 5 frames; detected = at least one STOP (or UNKNOWN) inside the event |
| onset delay | frames (and seconds) from event start to the first STOP inside it, detected events only |
| release delay | frames from event end to the first non-STOP output, counted only if it happens before the next event starts |
| false-STOP episodes | runs of STOP output that overlap no GT-STOP frame |
| UNKNOWN frames | count, reported separately |

Correctness and stability sit side by side on purpose: a filter that
never changes state has zero transitions and zero recall. "Stable but
wrong" is visible.

### Worked example (from the tests)

```
frame   0 1 2 3 4 5 6 7 8 9 10 11      fps = 6
output  S C C S S S C S S S C  C
GT      . . X X X . . . X X .  .       events (2,5) and (8,10)
```

- tp 4 (frames 3, 4, 8, 9), fp 3 (0, 5, 7), fn 1 (2): precision 4/7, recall 4/5
- transitions 5, i.e. 150 per minute over 2 s of video
- event 1: onset 1 frame (first STOP at 3), release 1 frame (first CLEAR at 6)
- event 2: onset 0, release 0
- false-STOP episodes: 1 (frame 0)

## Hand-check on real data

See the "Hand-check" section in [image-experiment.md](image-experiment.md)
and [sequence-experiment.md](sequence-experiment.md).
