"""Figures from aggregate CSVs (static PNGs for the repository).

Style: one y-axis per panel, thin 2 px lines with >= 8 px markers, recessive
grid, a legend for every multi-series panel, and series colours in a fixed
order (validated categorical palette). Each series also has its own marker
shape, so identity never relies on colour alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua (fixed order)
MARKERS = ("o", "s", "^")
SURFACE = "#fcfcfb"
INK = "#2b2b29"
INK_MUTED = "#6b6a63"
GRID = "#e4e3dc"

KIND_LABELS = {"blur": "Gaussian blur", "low_light": "Low light", "noise": "Noise"}
PARAM_LABELS = {"blur": "kernel {:g}", "low_light": "gain {:.2f}", "noise": "σ {:g}"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=9)


def _figure(n_panels: int, width: float = 4.2, height: float = 3.4):
    fig, axes = plt.subplots(
        1, n_panels, figsize=(width * n_panels, height), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        _style(ax)
    return fig, axes[0]


def _save(fig, path: Path, title: str) -> Path:
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_severity(rows: list[dict[str, str]], metric: str, ylabel: str, path: Path):
    """One panel per degradation kind: metric vs severity, one line per imgsz.
    Severity 0 is the clean condition, shared by all panels."""
    sizes = sorted({int(r["imgsz"]) for r in rows})
    fig, axes = _figure(3)
    for ax, kind in zip(axes, ("blur", "low_light", "noise"), strict=True):
        levels = sorted(
            {float(r["param"]) for r in rows if r["kind"] == kind},
            reverse=kind == "low_light",  # lower gain = more severe
        )
        for i, size in enumerate(sizes):
            clean = next(
                float(r[metric])
                for r in rows
                if r["kind"] == "clean" and int(r["imgsz"]) == size
            )
            values = [clean] + [
                next(
                    float(r[metric])
                    for r in rows
                    if r["kind"] == kind
                    and int(r["imgsz"]) == size
                    and float(r["param"]) == level
                )
                for level in levels
            ]
            ax.plot(
                range(len(values)),
                values,
                color=SERIES[i],
                marker=MARKERS[i],
                markersize=6,
                linewidth=2,
                label=f"imgsz {size}",
            )
        ax.set_xticks(
            range(len(levels) + 1),
            ["clean", *(PARAM_LABELS[kind].format(v) for v in levels)],
        )
        ax.set_title(KIND_LABELS[kind], color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel(ylabel, color=INK)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path, f"{ylabel} vs degradation severity (COCO val2017 subset)")


def plot_latency(rows: list[dict[str, str]], path: Path):
    """Mean detector latency per condition, one series per imgsz."""
    sizes = sorted({int(r["imgsz"]) for r in rows})
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    fig, (ax,) = _figure(1, width=9, height=3.6)
    for i, size in enumerate(sizes):
        values = [
            float(r["latency_mean_ms"])
            for c in conditions
            for r in rows
            if r["condition"] == c and int(r["imgsz"]) == size
        ]
        ax.plot(
            range(len(conditions)),
            values,
            color=SERIES[i],
            marker=MARKERS[i],
            markersize=6,
            linewidth=0,
            label=f"imgsz {size}",
        )
    ax.set_xticks(range(len(conditions)), conditions, rotation=30, ha="right")
    ax.set_ylabel("mean latency (ms)", color=INK)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=3)
    return _save(fig, path, "Detector latency per condition (CPU, batch 1)")


def plot_sequence_metric(
    rows: list[dict[str, str]], metric: str, ylabel: str, path: Path
):
    """Per evaluation sequence: metric per condition, baseline vs filtered."""
    eval_rows = [r for r in rows if r["role"] == "evaluation"]
    sequences = list(dict.fromkeys(r["sequence"] for r in eval_rows))
    methods = [m for m in ("baseline", "filtered", "filtered_adopted")
               if any(r["method"] == m for r in eval_rows)]  # fmt: skip
    fig, axes = _figure(len(sequences), width=5.2, height=3.6)
    for ax, seq in zip(axes, sequences, strict=True):
        conditions = list(
            dict.fromkeys(r["condition"] for r in eval_rows if r["sequence"] == seq)
        )
        for i, method in enumerate(methods):
            values = [
                float(r[metric]) if r[metric] not in ("", "nan") else float("nan")
                for c in conditions
                for r in eval_rows
                if r["sequence"] == seq
                and r["condition"] == c
                and r["method"] == method
            ]
            ax.plot(
                range(len(conditions)),
                values,
                color=SERIES[i],
                marker=MARKERS[i],
                markersize=6,
                linewidth=0,
                label=method,
            )
        ax.set_xticks(range(len(conditions)), conditions, rotation=40, ha="right")
        ax.set_title(seq, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel(ylabel, color=INK)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path, f"{ylabel}: per-frame baseline vs temporal filter")


def plot_sweep(summary: list[dict[str, str]], path: Path):
    """Filter sweep on the tuning sequence: stability vs delay, one point per
    distinct result. Settings with identical results share one point and one
    label (n_on/k_window/m_off). The default 3/5/5 is a square on top."""
    points: dict[tuple[float, float], list[str]] = {}
    default_xy = None
    for r in summary:
        xy = (float(r["onset_mean_s"]), float(r["transitions_per_min"]))
        points.setdefault(xy, []).append(f"{r['n_on']}/{r['k_window']}/{r['m_off']}")
        if (r["n_on"], r["k_window"], r["m_off"]) == ("3", "5", "5"):
            default_xy = xy
    fig, (ax,) = _figure(1, width=7, height=4.2)
    for i, ((x, y), labels) in enumerate(sorted(points.items())):
        ax.plot(x, y, marker="o", markersize=8, color=SERIES[0], linestyle="none")
        ax.annotate(
            ", ".join(labels),
            (x, y),
            textcoords="offset points",
            xytext=(6, -14 if i % 2 else 8),
            fontsize=8,
            color=INK,
        )
    if default_xy is not None:
        ax.plot(*default_xy, marker="s", markersize=10, color=SERIES[1],
                linestyle="none", label="default 3/5/5")  # fmt: skip
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.set_xlabel("mean onset delay (s)", color=INK)
    ax.set_ylabel("state transitions per minute", color=INK)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    return _save(fig, path, "Filter sweep, tuning sequence (mean over conditions)")
