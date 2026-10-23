"""Score simulation runs: live node output and offline replay against GT.

For each run (one repeat of one mode) and each method (baseline, filtered):
- **live**: the states the node logged, on the frames it processed, against
  the GT of those frames. The node can drop frames under load, so fps is
  taken from the processed frames' own stamps.
- **replay**: the bag's frames re-run through the offline sequence runner
  (every recorded frame, deterministic).
Plus latency, the share of recorded frames the node processed, the run's
real-time factor, the final robot pose, and live-vs-replay agreement.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robust_visual_perception.metrics import latency_stats
from robust_visual_perception.sequence_metrics import stream_metrics
from robust_visual_perception.sim_export import frame_rate, read_frames_csv

METHODS = ("baseline", "filtered")
STREAM_COLUMNS = [
    "n_frames",
    "fps",
    "precision",
    "recall",
    "accuracy",
    "transitions",
    "transitions_per_min",
    "n_events",
    "events_detected",
    "events_missed",
    "onset_mean_frames",
    "onset_mean_s",
    "release_mean_s",
    "false_stop_episodes",
    "n_unknown",
]
RUN_COLUMNS = [
    "run",
    "mode",
    "repeat",
    "source",
    "method",
    *STREAM_COLUMNS,
    "bag_frames",
    "processed_fraction",
    "agreement",
    "agreement_n",
    "latency_mean_ms",
    "latency_p95_ms",
    "age_sim_mean_ms",
    "age_sim_p95_ms",
    "callback_mean_ms",
    "callback_p95_ms",
    "real_time_factor",
    "stale_episodes",
    "final_x",
    "final_y",
    "final_yaw",
]


def parse_run_name(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_r(\d+)", name)
    if not match:
        raise ValueError(f"run name must look like <mode>_r<k>, got {name!r}")
    return match.group(1), int(match.group(2))


def join_live(rows: Sequence[dict[str, str]], frames: dict[str, np.ndarray]) -> dict:
    """Node rows matched to recorded frames by image stamp."""
    index = {int(s): i for i, s in enumerate(frames["stamp_ns"])}
    keep = [(index[int(r["stamp_ns"])], r) for r in rows if int(r["stamp_ns"]) in index]
    keep.sort(key=lambda item: item[0])
    picked = [i for i, _ in keep]
    return {
        "stamp_ns": frames["stamp_ns"][picked],
        "gt_stop": frames["gt_stop"][picked],
        "baseline": [r["baseline_state"] for _, r in keep],
        "filtered": [r["filtered_state"] for _, r in keep],
        "rows": [r for _, r in keep],
        "unmatched": len(rows) - len(keep),
    }


def stream_row(
    states: Sequence[str], gt_stop: np.ndarray, stamps_ns: np.ndarray
) -> dict[str, Any]:
    fps = frame_rate(stamps_ns)
    metrics = stream_metrics(states, np.asarray(gt_stop, bool), fps)
    return {"fps": fps, **{k: metrics[k] for k in STREAM_COLUMNS if k in metrics}}


def agreement(live: dict, replay_baseline: dict[int, str]) -> tuple[float, int]:
    """Share of node-processed frames where live and replayed per-frame
    states are equal, and the number of frames compared."""
    pairs = [
        (state, replay_baseline[int(stamp)])
        for stamp, state in zip(live["stamp_ns"], live["baseline"], strict=True)
        if int(stamp) in replay_baseline
    ]
    if not pairs:
        return math.nan, 0
    return sum(a == b for a, b in pairs) / len(pairs), len(pairs)


def final_pose(poses: np.ndarray) -> tuple[float, float, float]:
    """(x, y, yaw) of the last robot pose sample (rows: t, x, y, z, qx..qw)."""
    _, x, y, _, qx, qy, qz, qw = (float(v) for v in poses[-1])
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, yaw


def spread(
    rows: Sequence[dict[str, Any]], group_keys: Sequence[str], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    """Mean, min and max of each metric over the repeats of each group."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, members in groups.items():
        result = dict(zip(group_keys, key, strict=True))
        result["n_repeats"] = len(members)
        for metric in metrics:
            values = np.array([float(m.get(metric, math.nan)) for m in members])
            values = values[~np.isnan(values)]
            result[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            result[f"{metric}_min"] = float(values.min()) if len(values) else math.nan
            result[f"{metric}_max"] = float(values.max()) if len(values) else math.nan
        out.append(result)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _floats(rows: Sequence[dict[str, str]], column: str) -> list[float]:
    return [float(r[column]) for r in rows if r.get(column)]


def evaluate_run(
    name: str,
    results_root: Path,
    sequences_root: Path,
    replay_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    """Rows for one run: (live, replay) x (baseline, filtered)."""
    mode, repeat = parse_run_name(name)
    run_dir, seq_dir = Path(results_root) / name, Path(sequences_root) / name
    frames = read_frames_csv(seq_dir / "frames.csv")
    live = join_live(_read_csv(run_dir / "live_frames.csv"), frames)
    run_info = json.loads((run_dir / "run_info.json").read_text())
    node = json.loads((run_dir / "node_summary.json").read_text())
    poses = np.loadtxt(seq_dir / "poses.csv", delimiter=",", skiprows=1, ndmin=2)
    x, y, yaw = final_pose(poses)

    # Replay rows of this run, in frame order (frame_id = MOT17 file stem).
    replay = sorted(
        (int(r["frame_id"]), r) for r in replay_rows if r["sequence"] == name
    )
    replay_stamps = frames["stamp_ns"][[f - 1 for f, _ in replay]]
    replay_states = {m: [r[f"{m}_state"] for _, r in replay] for m in METHODS}
    replay_baseline = dict(
        zip(replay_stamps.tolist(), replay_states["baseline"], strict=True)
    )
    agree, agree_n = agreement(live, replay_baseline)

    latency = latency_stats(_floats(live["rows"], "latency_ms"))
    age = latency_stats(_floats(live["rows"], "age_sim_ms"))
    callback = latency_stats(_floats(live["rows"], "callback_ms"))
    common = {
        "run": name,
        "mode": mode,
        "repeat": repeat,
        "bag_frames": len(frames["stamp_ns"]),
        "real_time_factor": run_info.get("real_time_factor", math.nan),
        "final_x": x,
        "final_y": y,
        "final_yaw": yaw,
    }
    rows = []
    for method in METHODS:
        rows.append(
            {
                **common,
                "source": "live",
                "method": method,
                **stream_row(live[method], live["gt_stop"], live["stamp_ns"]),
                "processed_fraction": len(live["stamp_ns"]) / len(frames["stamp_ns"]),
                "agreement": agree,
                "agreement_n": agree_n,
                "latency_mean_ms": latency["mean_ms"],
                "latency_p95_ms": latency["p95_ms"],
                "age_sim_mean_ms": age["mean_ms"],
                "age_sim_p95_ms": age["p95_ms"],
                "callback_mean_ms": callback["mean_ms"],
                "callback_p95_ms": callback["p95_ms"],
                "stale_episodes": node.get("stale_episodes", math.nan),
            }
        )
        if replay:
            gt = frames["gt_stop"][[f - 1 for f, _ in replay]]
            rows.append(
                {
                    **common,
                    "source": "replay",
                    "method": method,
                    **stream_row(replay_states[method], gt, replay_stamps),
                    "processed_fraction": len(replay) / len(frames["stamp_ns"]),
                }
            )
    return rows
