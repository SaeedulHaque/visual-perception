# Annotated video of a run

Module: `sim_video.py` (test: `test_sim_video.py`)
Script: `scripts/render_sim_video.py`
Export: `rvp_ros/export_bag.py` writes `detections.csv`

## What it does

The numbers in `live_frames.csv` say what the node decided. This turns a
run back into pictures, so I can see why. Each camera frame from the bag
gets:

```
┌───────────────────────────────────────────────┐
│        │                       │              │   cyan lines: the sim ROI band
│        │  ┌──────┐ green: GT   │              │   (x 0.30 and 0.70)
│        │  │┌────┐│ orange: person ≥ 0.5       │   grey: every other published
│        │  ││    ││   (counts for the decision) │   detection (score ≥ 0.25)
│        │  │└────┘│             │              │
│        │  └──────┘             │              │
├───────────┬───────────┬───────────┬───────────┤
│ t 61.25 s │ frame STOP│filtered   │ GT CLEAR  │   red STOP, green CLEAR,
│           │           │ STOP      │           │   grey UNKNOWN
└───────────┴───────────┴───────────┴───────────┘
```

- **frame**: the per-frame (baseline) state; **filtered**: the state
  after the 2/3/3 filter; **GT**: the ground truth for that frame.
- A frame the node did not process (dropped under load) keeps the last
  published states, shown in dark grey with "(held)", since that is what
  the robot's subscribers saw at that moment.
- The overlays are drawn on a copy, and the banner sits below the image,
  so no camera pixels are covered except by the box outlines.

## Where the data comes from

| Overlay | Source |
|---|---|
| camera frame | `img1/` from the bag export |
| GT box and GT state | `frames.csv` from `make_sim_gt.py` |
| detections | `detections.csv`: the `/perception/detections` messages in the bag, keyed by the image stamp in their header |
| states | the node's `live_frames.csv`, joined by image stamp |
| ROI, class, threshold | the node's `node_params.json` |

Nothing is re-detected: the boxes are the ones the node published during
the run. Published detections stop at score 0.25, so detections below
that are not shown even though the decision saw them (they cannot make a
STOP, since the STOP threshold is 0.5).

## Use

```
uv run python scripts/render_sim_video.py --run teleop_r1 \
    --sheet results/sim/summary/figures/frames_teleop_r1.png
```

- The video goes to `data/raw/sim/videos/<run>.mp4` (MPEG-4, the export's
  frame rate, 640×520). It is git-ignored: the teleop video is about 17 MB.
- `--sheet` also writes a grid of six frames at half size, picked evenly
  from frames the node processed, half of them GT-STOP. That one is small
  enough to commit.
- The sequence needs `detections.csv`. Bags exported before the export
  wrote it have to be exported again (`export_bag`, same command as in
  the run script).
