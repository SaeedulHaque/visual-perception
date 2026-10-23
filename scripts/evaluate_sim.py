"""Evaluate the simulation runs: live node output and offline replay vs GT.

    uv run python scripts/evaluate_sim.py

Reads results/sim/<run>/ (node log, run info), data/raw/sim/sequences/<run>/
(frames.csv with GT, poses.csv) and the replay runs results/sim/sim_replay/
and results/sim/sim_replay_check/. Writes results/sim/summary/:
repeats.csv (one row per run, source and method), spread.csv (mean, min and
max over repeats), figures and summary.md. The GT used for each run
(frames.csv) and its export summary are copied next to the node log as
gt_frames.csv and export.json, so the committed results hold everything the
metrics were computed from.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from robust_visual_perception.image_eval import write_aggregate
from robust_visual_perception.sim_eval import (
    METHODS,
    RUN_COLUMNS,
    evaluate_run,
    join_live,
    parse_run_name,
    spread,
)
from robust_visual_perception.sim_export import read_frames_csv
from robust_visual_perception.sim_report import plot_repeats, plot_timeline, sim_summary

SPREAD_METRICS = [
    "recall",
    "precision",
    "accuracy",
    "transitions_per_min",
    "events_detected",
    "events_missed",
    "onset_mean_s",
    "false_stop_episodes",
    "processed_fraction",
    "agreement",
    "latency_mean_ms",
    "age_sim_mean_ms",
    "callback_mean_ms",
    "real_time_factor",
    "final_x",
    "final_y",
    "final_yaw",
]
COMPARE_COLUMNS = [
    "status",
    "n_detections",
    "n_target",
    "max_target_conf",
    "baseline_state",
    "filtered_state",
    "gt_state",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def replay_differences(a: list[dict], b: list[dict]) -> dict:
    key = lambda r: (r["sequence"], r["frame_id"])  # noqa: E731
    left, right = {key(r): r for r in a}, {key(r): r for r in b}
    common = set(left) & set(right)
    differ = sum(
        any(left[k][c] != right[k][c] for c in COMPARE_COLUMNS) for k in common
    )
    return {"n_rows": len(common), "n_different": differ + len(set(left) ^ set(right))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/sim"))
    parser.add_argument(
        "--sequences", type=Path, default=Path("data/raw/sim/sequences")
    )
    parser.add_argument("--replay", default="sim_replay")
    parser.add_argument("--replay-check", default="sim_replay_check")
    parser.add_argument(
        "--gt-check", type=Path, default=Path("results/sim/summary/gt_check.json")
    )
    args = parser.parse_args()

    runs = sorted(
        p.name for p in args.results.iterdir() if (p / "live_frames.csv").is_file()
    )
    for name in runs:
        seq = args.sequences / name
        shutil.copyfile(seq / "frames.csv", args.results / name / "gt_frames.csv")
        shutil.copyfile(seq / "export.json", args.results / name / "export.json")
    replay = read_csv(args.results / args.replay / "per_frame.csv")
    rows = [
        row
        for name in runs
        for row in evaluate_run(name, args.results, args.sequences, replay)
    ]
    out = args.results / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_aggregate(out / "repeats.csv", rows, RUN_COLUMNS)
    spread_rows = spread(rows, ["mode", "source", "method"], SPREAD_METRICS)
    write_aggregate(out / "spread.csv", spread_rows, list(spread_rows[0]))

    check = None
    check_path = args.results / args.replay_check / "per_frame.csv"
    if check_path.is_file():
        check = replay_differences(replay, read_csv(check_path))
        (out / "replay_check.json").write_text(json.dumps(check, indent=2))
    gt_check = (
        json.loads(args.gt_check.read_text()) if args.gt_check.is_file() else None
    )

    for mode in sorted({parse_run_name(n)[0] for n in runs}):
        name = f"{mode}_r1"
        frames = read_frames_csv(args.sequences / name / "frames.csv")
        live = join_live(read_csv(args.results / name / "live_frames.csv"), frames)
        streams = {"GT-STOP": ["STOP" if g else "CLEAR" for g in live["gt_stop"]]}
        streams.update({f"live {m}": live[m] for m in METHODS})
        plot_timeline(
            np.asarray(live["stamp_ns"]) / 1e9,
            streams,
            out / "figures" / f"timeline_{mode}.png",
            f"{mode} run, repeat 1: GT-STOP and the node's states",
        )
    plot_repeats(
        rows,
        "transitions_per_min",
        "transitions per minute",
        out / "figures" / "repeats_transitions.png",
    )
    plot_repeats(rows, "recall", "STOP recall", out / "figures" / "repeats_recall.png")
    (out / "summary.md").write_text(sim_summary(spread_rows, check, gt_check))
    print(f"wrote {out} ({len(rows)} rows from {len(runs)} runs)")


if __name__ == "__main__":
    main()
