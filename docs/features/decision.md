# Baseline decision

Module: `src/robust_visual_perception/decision.py`
Tests: `tests/test_decision.py`

## What it decides

For every frame, one of three states:

| State | Meaning |
|---|---|
| `STOP` | a `person` detection with confidence ≥ 0.5 is in view (and inside the ROI, if one is given) |
| `CLEAR` | the detector ran and found no such detection |
| `UNKNOWN` | there is no usable observation: the frame is missing (`missing_frame`) or the detector failed (`detector_error`) |

`UNKNOWN` matters. Treating "no data" as `CLEAR` is the classic way a
perception system fails silently. Here it is a separate state with a reason,
so a consumer can treat it as stop-with-reason.

## What it is not

This is an image-space heuristic. A large box does not mean the person is
close: it could be a tall person further away, or a poster. The rule is a
stand-in for a perception-triggered safety signal, used to study how
degradation and temporal filtering change decisions. It is not collision
avoidance and it does not estimate distance.

## The corridor ROI (used for sequences)

```
          0.30 W     0.70 W
     ┌──────┬──────────┬──────┐  0
     │      │          │      │
     │      ├──────────┤      │  0.30 H   box centre must be
     │      │ corridor │      │           inside this band
     │      │          │      │
     └──────┴──────────┴──────┘  1.00 H
     and box height ≥ 0.25 H
```

`CORRIDOR_ROI = Roi(0.30, 0.70, 0.30, 1.00, 0.25)`. The idea: a person
whose box centre is in the middle of the image, low enough to be on the
ground ahead, and tall enough to be relatively near, is "in the path". The
COCO image experiment uses no ROI (any confident person means STOP), because
COCO photos have no notion of a robot path.

The same geometry is applied to ground-truth boxes in MOT17 to define when
the ground truth says STOP, so prediction and ground truth are compared
under one rule.

## API

```python
decision = baseline_decision(
    result, target_class="person", conf_threshold=0.5, roi=None
)
decision.state  # State.STOP / CLEAR / UNKNOWN
decision.reason  # None, "missing_frame" or "detector_error"
```

Pass `result=None` for a frame that never arrived.

## Why 0.5

The threshold was fixed before running anything. It is the common default
operating point for YOLO-style detectors. Picking the threshold that looks
best on the evaluation data would make the results optimistic, so it is
not tuned.
