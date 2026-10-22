# MOT17 data, GT-STOP and sequence selection

Modules: `mot17.py`, `remote_zip.py`
Script: `scripts/fetch_mot17.py`
Tests: `tests/test_mot17.py`, `tests/test_remote_zip.py`
Manifests (committed): `data/manifests/mot17_sequences.csv`,
`data/manifests/mot17_roi_search.csv`

## Why MOT17

The temporal filter needs video with per-frame ground truth for people.
MOT17's seven training sequences have a box for every pedestrian in every
frame, with a visibility value and a "considered" flag. Several are filmed
from a moving camera (handheld or vehicle), the closest public match to a
robot camera.

## Downloading only what is needed

`MOT17.zip` is 5.9 GB: train and test, each sequence three times (one copy
per public detector, with identical images). This project needs GT for the
seven training sequences and images for three of them.

The server supports HTTP range requests, so:

1. `RangeFile` is a seekable file object whose reads become
   `Range: bytes=a-b` requests. Python's `zipfile` can open it and read the
   archive's directory (at the end of the file) with a few small requests.
2. `seqinfo.ini` and `gt/gt.txt` of each training sequence are read as single
   members.
3. All images of one sequence are stored next to each other in the archive,
   so `extract_prefix` downloads that byte span in **one** request to a local
   cache file and extracts from there. Without this, each image would cost
   two to three round trips.

Result: 267 MB on disk instead of 5.9 GB. Extraction goes to `img1.part/`
and is renamed when complete, so an interrupted fetch never leaves a
directory that looks complete.

## GT-STOP rule

A frame is GT-STOP when any GT box satisfies all of:

- class 1 (pedestrian), `considered=1`, visibility ≥ 0.5;
- after clipping to the image (exactly as detections are clipped), its
  centre lies inside the corridor ROI;
- its height is at least the ROI's minimum fraction of the image height.

The prediction side uses the same geometry on person detections with
confidence ≥ 0.5 (`baseline_decision(..., roi=...)`), so prediction and GT
are judged by one rule.

**GT event:** a maximal run of GT-STOP frames, after merging runs separated
by fewer than 5 frames. Without merging, a person whose visibility hovers
around 0.5 would split one approach into many tiny "events".

## Sequence selection (GT only)

Design rule: occupancy (share of GT-STOP frames) between 15 % and 70 %, at
least 3 events, prefer moving camera. Roles:

- **evaluation:** the two qualifying moving-camera sequences with most events;
- **tuning:** the qualifying sequence with most events among the rest
  (used for the filter sweep only).

Camera motion comes from the MOT16 paper (Milan et al., 2016, Table 2).

### ROI adjustment

With the design ROI (`x 0.30–0.70`, `y ≥ 0.30`, height ≥ 0.25) only one
sequence qualified: most MOT17 sequences have someone in the central region
almost always (crowded scenes) or almost never. The design allows adjusting
the ROI "using GT statistics only" in that case. To keep that from becoming
a free choice, the procedure is fixed in code (`roi_grid`, `choose_roi`):

1. Grid: band `x_min ∈ {0.30, 0.35, 0.40}` (symmetric), `y_min ∈ {0.30, 0.50}`,
   minimum height `∈ {0.25, 0.30, …, 0.50}`; 36 settings.
2. Keep settings where the selection rule succeeds (3 qualifying sequences,
   2 with a moving camera).
3. Choose the fewest changed parameters, then the smallest total change from
   the design ROI, then the most events in the selected sequences.

Chosen: **x 0.35–0.65, y ≥ 0.30, height ≥ 0.50** (two parameters changed,
total change 0.30). Every grid row is saved in `mot17_roi_search.csv`. No
model output on MOT17 existed when this was run.

Selected sequences, from `mot17_sequences.csv`:

| Sequence | Camera | fps | Frames | Occupancy | Events | Role |
|---|---|---|---|---|---|---|
| MOT17-09 | static | 30 | 525 | 0.32 | 6 | tuning |
| MOT17-05 | moving | 14 | 837 | 0.55 | 9 | evaluation |
| MOT17-11 | moving | 30 | 900 | 0.64 | 6 | evaluation |

The taller minimum box height (half the image) makes "STOP" mean "a person
large in the frame, near the camera centre line". It is still an image-space
proxy, not a distance.

## Files on disk

```
data/raw/mot17/MOT17-05/seqinfo.ini
data/raw/mot17/MOT17-05/gt/gt.txt
data/raw/mot17/MOT17-05/img1/000001.jpg ...
```

GT for all seven sequences is kept (small); images only for the three
selected.
