"""Figures and the Markdown summary for the simulation runs.

Every number in the summary comes from the CSVs written by
scripts/evaluate_sim.py; nothing is typed by hand.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robust_visual_perception.plots import (
    GRID,
    INK,
    MARKERS,
    SERIES,
    SURFACE,
    _style,
    plt,
)
from robust_visual_perception.summary import _num

UNKNOWN_GREY = "#9a9891"


def _runs(states: Sequence[str], value: str) -> list[tuple[int, int]]:
    """(start, end) index runs where ``states == value`` (end exclusive)."""
    flags = np.array([s == value for s in states], bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(int), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def plot_timeline(
    times_s: np.ndarray,
    streams: dict[str, Sequence[str]],
    path: Path,
    title: str,
) -> Path:
    """One row per stream (GT first); STOP drawn as filled spans in the row's
    colour, UNKNOWN in grey. Rows are labelled directly on the y-axis."""
    times_s = np.asarray(times_s, float)
    step = float(np.median(np.diff(times_s))) if len(times_s) > 1 else 0.033
    fig, ax = plt.subplots(figsize=(9.0, 0.6 + 0.55 * len(streams)))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    labels = list(streams)
    for row, label in enumerate(labels):
        y = len(labels) - 1 - row
        colour = SERIES[row % len(SERIES)]
        for value, face in (("STOP", colour), ("UNKNOWN", UNKNOWN_GREY)):
            spans = [
                (times_s[a], times_s[b - 1] - times_s[a] + step)
                for a, b in _runs(streams[label], value)
            ]
            if spans:
                ax.broken_barh(
                    spans,
                    (y - 0.32, 0.64),
                    facecolors=face,
                    edgecolor=SURFACE,
                    linewidth=0.5,
                )
    ax.set_yticks(range(len(labels)), labels[::-1])
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlim(times_s[0], times_s[-1] + step)
    ax.set_xlabel("sim time (s); filled = STOP", color=INK, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_repeats(
    rows: Sequence[dict[str, Any]], metric: str, ylabel: str, path: Path
) -> Path:
    """Per-repeat values of one live metric: one panel per mode, baseline and
    filtered side by side, one marker per repeat."""
    modes = sorted({r["mode"] for r in rows}, key=lambda m: (m != "static", m))
    fig, axes = plt.subplots(
        1, len(modes), figsize=(3.6 * len(modes), 3.2), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, mode in zip(axes[0], modes, strict=True):
        _style(ax)
        for i, method in enumerate(("baseline", "filtered")):
            vals = [
                float(r[metric])
                for r in rows
                if r["mode"] == mode and r["method"] == method and r["source"] == "live"
            ]
            xs = np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals))
            ax.plot(
                xs,
                vals,
                MARKERS[i],
                color=SERIES[i],
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=method,
                linestyle="none",
            )
        ax.set_xticks([0, 1], ["baseline", "filtered"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(mode, color=INK, fontsize=10, loc="left")
    axes[0][0].set_ylabel(ylabel, color=INK, fontsize=9)
    fig.suptitle(
        f"{ylabel}, live node, each marker = one repeat",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _range(row: dict[str, Any], metric: str, fmt: str = "{:.3f}") -> str:
    mean, lo, hi = (row[f"{metric}_{k}"] for k in ("mean", "min", "max"))
    if any(isinstance(v, float) and math.isnan(v) for v in (mean, lo, hi)):
        return "–"
    return f"{_num(mean, fmt)} ({_num(lo, fmt)}–{_num(hi, fmt)})"


def sim_summary(
    spread_rows: Sequence[dict[str, Any]],
    replay_check: dict[str, Any] | None,
    gt_check: dict[str, Any] | None,
) -> str:
    def rows_for(source: str) -> list[dict[str, Any]]:
        out = [r for r in spread_rows if r["source"] == source]
        return sorted(
            out, key=lambda r: (r["mode"] != "static", r["mode"], r["method"])
        )

    def cell(metric: str, fmt: str = "{:.3f}"):
        return (metric, lambda row: _range(row, metric, fmt))

    def table(rows, cells) -> str:
        head = ["mode", "method", "repeats", *[title for title, _ in cells]]
        lines = [
            "| " + " | ".join(head) + " |",
            "|" + "|".join("---" for _ in head) + "|",
        ]
        for r in rows:
            values = [r["mode"], r["method"], str(r["n_repeats"])]
            values += [fn(r) for _, (_, fn) in cells]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    correctness = [
        ("recall", cell("recall")),
        ("precision", cell("precision")),
        ("accuracy", cell("accuracy")),
        ("transitions / min", cell("transitions_per_min", "{:.1f}")),
        ("events detected", cell("events_detected", "{:.1f}")),
        ("events missed", cell("events_missed", "{:.1f}")),
        ("onset delay (s)", cell("onset_mean_s", "{:.2f}")),
        ("false-STOP episodes", cell("false_stop_episodes", "{:.1f}")),
    ]
    live_only = [
        ("frames processed", cell("processed_fraction")),
        ("live = replay", cell("agreement")),
        ("inference (ms)", cell("latency_mean_ms", "{:.1f}")),
        ("callback, wall (ms)", cell("callback_mean_ms", "{:.1f}")),
        ("image age at start, sim (ms)", cell("age_sim_mean_ms", "{:.1f}")),
        ("real-time factor", cell("real_time_factor")),
    ]
    pose = [
        ("final x (m)", cell("final_x", "{:.3f}")),
        ("final y (m)", cell("final_y", "{:.3f}")),
        ("final yaw (rad)", cell("final_yaw", "{:.3f}")),
    ]
    live = rows_for("live")
    parts = [
        "<!-- Generated by scripts/evaluate_sim.py from repeats.csv and "
        "spread.csv. Do not edit by hand. -->",
        "# Simulation runs: summary",
        "",
        "Values are the mean over repeats, with the min–max range in "
        "brackets. GT-STOP: the projected person box in the sim ROI "
        "(x 0.30–0.70, height ≥ 0.25 of the image). UNKNOWN counts as STOP.",
        "",
        "## Live node (frames the node processed)",
        "",
        table(live, correctness),
        "",
        table([r for r in live if r["method"] == "filtered"], live_only),
        "",
        "## Offline replay (every recorded frame)",
        "",
        table(rows_for("replay"), correctness),
        "",
        "## Final robot pose (repeatability of the motion)",
        "",
        table([r for r in live if r["method"] == "filtered"], pose),
        "",
    ]
    if replay_check is not None:
        parts += [
            "## Replay determinism",
            "",
            f"Second replay of the same bags: {replay_check['n_rows']} rows "
            f"compared, {replay_check['n_different']} differ in any "
            f"non-timing column.",
            "",
        ]
    if gt_check is not None:
        parts += [
            "## Ground-truth check (not used for tuning)",
            "",
            f"On {gt_check['n_frames']} frames with a GT box and a person "
            f"detection (conf ≥ 0.5), the IoU between the GT box and the best "
            f"detection: median {_num(gt_check['iou_median'])}, 10th percentile "
            f"{_num(gt_check['iou_p10'])}. GT-STOP and detection-STOP agree "
            f"on {_num(gt_check['state_agreement'])} of all checked frames.",
            "",
        ]
    return "\n".join(parts)
