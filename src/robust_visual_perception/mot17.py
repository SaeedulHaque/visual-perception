"""MOT17 sequences: metadata, ground truth, GT-STOP states, events, selection.

GT-STOP rule (design section 6.3): a frame is STOP when a pedestrian box
(class 1, ``considered=1``, visibility >= 0.5) has its centre in the corridor
ROI and its height is >= 25 % of the image height. GT boxes are clipped to
the image first, exactly as detections are, so both sides use one geometry.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass

import numpy as np

from robust_visual_perception.decision import CORRIDOR_ROI, Roi

PEDESTRIAN_CLASS = 1
MIN_VISIBILITY = 0.5
EVENT_MERGE_GAP = 5
OCCUPANCY_RANGE = (0.15, 0.70)
MIN_EVENTS = 3

# Camera motion of the MOT17 training sequences (Milan et al., "MOT16: A
# Benchmark for Multi-Object Tracking", 2016, Table 2).
CAMERA_MOTION = {
    "MOT17-02": "static",
    "MOT17-04": "static",
    "MOT17-05": "moving",
    "MOT17-09": "static",
    "MOT17-10": "moving",
    "MOT17-11": "moving",
    "MOT17-13": "moving",
}
TRAIN_SEQUENCES = tuple(CAMERA_MOTION)


@dataclass(frozen=True)
class SeqInfo:
    name: str  # e.g. "MOT17-05" (detector suffix removed)
    fps: float
    length: int
    width: int
    height: int
    img_dir: str
    img_ext: str


def parse_seqinfo(text: str) -> SeqInfo:
    parser = configparser.ConfigParser()
    parser.read_string(text)
    s = parser["Sequence"]
    name = "-".join(s["name"].split("-")[:2])
    return SeqInfo(
        name=name,
        fps=float(s["frameRate"]),
        length=int(s["seqLength"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        img_dir=s["imDir"],
        img_ext=s["imExt"],
    )


def parse_gt(text: str) -> np.ndarray:
    """gt.txt -> N x 9 array: frame, id, x, y, w, h, considered, class, visibility."""
    rows = [line.split(",") for line in text.strip().splitlines() if line.strip()]
    return np.array(rows, dtype=float).reshape(-1, 9)


def gt_boxes_by_frame(
    gt: np.ndarray, info: SeqInfo, min_visibility: float = MIN_VISIBILITY
) -> dict[int, np.ndarray]:
    """Clipped xyxy boxes of usable pedestrians, keyed by frame (1-based)."""
    keep = (
        (gt[:, 7] == PEDESTRIAN_CLASS) & (gt[:, 6] == 1) & (gt[:, 8] >= min_visibility)
    )
    g = gt[keep]
    xyxy = np.stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]], axis=1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, info.width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, info.height)
    frames = g[:, 0].astype(int)
    return {int(f): xyxy[frames == f] for f in np.unique(frames)}


def gt_stop_states(
    gt: np.ndarray,
    info: SeqInfo,
    roi: Roi = CORRIDOR_ROI,
    min_visibility: float = MIN_VISIBILITY,
) -> np.ndarray:
    """Boolean GT-STOP per frame; index 0 is frame 1."""
    states = np.zeros(info.length, bool)
    hw = (info.height, info.width)
    for frame, boxes in gt_boxes_by_frame(gt, info, min_visibility).items():
        if 1 <= frame <= info.length:
            states[frame - 1] = bool(roi.mask(boxes, hw).any())
    return states


def find_events(
    states: np.ndarray, merge_gap: int = EVENT_MERGE_GAP
) -> list[tuple[int, int]]:
    """Maximal runs of True as ``(start, end)`` (end exclusive). Runs separated
    by fewer than ``merge_gap`` False frames are merged into one event."""
    s = np.asarray(states, bool)
    padded = np.concatenate([[False], s, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    runs = list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True)
class SequenceStats:
    name: str
    fps: float
    length: int
    camera: str
    occupancy: float  # fraction of frames that are GT-STOP
    n_events: int


def sequence_stats(info: SeqInfo, states: np.ndarray) -> SequenceStats:
    return SequenceStats(
        name=info.name,
        fps=info.fps,
        length=info.length,
        camera=CAMERA_MOTION.get(info.name, "unknown"),
        occupancy=float(np.mean(states)),
        n_events=len(find_events(states)),
    )


def _qualifies(s: SequenceStats) -> bool:
    low, high = OCCUPANCY_RANGE
    return low <= s.occupancy <= high and s.n_events >= MIN_EVENTS


def select_sequences(stats: list[SequenceStats]) -> dict[str, str]:
    """Assign ``evaluation`` (2), ``tuning`` (1) or ``unused`` from GT only.

    Evaluation: the two qualifying moving-camera sequences with most events.
    Tuning: the qualifying sequence with most events among the rest.
    Ties are broken by name.
    """
    qualifying = sorted(
        (s for s in stats if _qualifies(s)), key=lambda s: (-s.n_events, s.name)
    )
    moving = [s for s in qualifying if s.camera == "moving"]
    if len(qualifying) < 3 or len(moving) < 2:
        raise ValueError(
            "need 3 qualifying sequences (2 moving-camera); adjust the ROI using "
            "GT statistics only and document it"
        )
    evaluation = moving[:2]
    tuning = next(s for s in qualifying if s not in evaluation)
    roles = {s.name: "unused" for s in stats}
    roles.update({s.name: "evaluation" for s in evaluation})
    roles[tuning.name] = "tuning"
    return roles


# ROI search (design 6.3: "if the ROI gives no qualifying sequences, adjust it
# using GT statistics only and document it"). The grid and the choice rule are
# fixed before any model output on MOT17 exists.
ROI_GRID_X_MIN = (0.30, 0.35, 0.40)  # symmetric band [x_min, 1 - x_min]
ROI_GRID_Y_MIN = (0.30, 0.50)
ROI_GRID_MIN_HEIGHT = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def roi_grid() -> list[Roi]:
    return [
        Roi(x, round(1 - x, 2), y, 1.0, h)
        for h in ROI_GRID_MIN_HEIGHT
        for x in ROI_GRID_X_MIN
        for y in ROI_GRID_Y_MIN
    ]


def roi_search(
    sequences: dict[str, tuple[SeqInfo, np.ndarray]], grid: list[Roi]
) -> list[dict]:
    """For each ROI: per-sequence stats, whether selection succeeds, and the
    smallest event count among the three selected sequences."""
    rows = []
    for roi in grid:
        stats = [
            sequence_stats(info, gt_stop_states(gt, info, roi))
            for info, gt in sequences.values()
        ]
        try:
            roles = select_sequences(stats)
        except ValueError:
            roles = None
        chosen = [s for s in stats if roles and roles[s.name] != "unused"]
        rows.append(
            {
                "roi": roi,
                "stats": stats,
                "roles": roles,
                "qualifies": roles is not None,
                "min_events": min((s.n_events for s in chosen), default=0),
            }
        )
    return rows


def _roi_change(roi: Roi) -> tuple[int, float]:
    deltas = (
        abs(roi.x_min - CORRIDOR_ROI.x_min),
        abs(roi.y_min - CORRIDOR_ROI.y_min),
        abs(roi.min_height - CORRIDOR_ROI.min_height),
    )
    return sum(d > 1e-9 for d in deltas), round(sum(deltas), 6)


def choose_roi(rows: list[dict]) -> Roi:
    """Qualifying ROI with the fewest changed parameters, then the smallest
    total change from the design ROI, then the most events (min over the
    three selected sequences)."""
    candidates = [r for r in rows if r["qualifies"]]
    if not candidates:
        raise ValueError("no ROI in the grid gives a valid sequence selection")
    best = min(candidates, key=lambda r: (*_roi_change(r["roi"]), -r["min_events"]))
    return best["roi"]
