"""Make figures and summary.md from the aggregate CSVs of finished runs.

    uv run python scripts/plot_results.py --image-run results/image_main
    uv run python scripts/plot_results.py --sequence-run results/sequence_main \
        --fault-run results/sequence_faults
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_visual_perception.config import load_sequence_config
from robust_visual_perception.plots import (
    plot_latency,
    plot_sequence_metric,
    plot_severity,
    plot_sweep,
    read_csv,
)
from robust_visual_perception.summary import image_summary, sequence_summary


def image_outputs(run: Path) -> None:
    rows = read_csv(run / "aggregate.csv")
    figures = run / "figures"
    for metric, label in (
        ("person_ap", "person AP"),
        ("map", "mAP@[.5:.95]"),
        ("person_recall", "person recall @0.5"),
        ("retention", "detection retention"),
        ("stop_recall", "image-level STOP recall"),
    ):
        plot_severity(rows, metric, label, figures / f"{metric}.png")
    plot_latency(rows, figures / "latency.png")
    (run / "summary.md").write_text(image_summary(rows))


def sequence_outputs(run: Path, fault_run: Path | None) -> None:
    rows = read_csv(run / "aggregate.csv")
    figures = run / "figures"
    plot_sequence_metric(
        rows, "transitions_per_min", "transitions per minute",
        figures / "sequence_transitions.png",
    )  # fmt: skip
    plot_sequence_metric(
        rows, "onset_mean_s", "mean onset delay (s)", figures / "sequence_onset.png"
    )
    plot_sequence_metric(
        rows, "recall", "frame-level recall", figures / "sequence_recall.png"
    )
    sweep = read_csv(run / "sweep_summary.csv")
    plot_sweep(sweep, figures / "sweep.png")
    adopted = json.loads((run / "adopted_filter.json").read_text())
    faults, rates = None, None
    if fault_run:
        faults = read_csv(fault_run / "aggregate.csv")
        fault = load_sequence_config(fault_run / "config.yaml").fault
        rates = (fault.missing_rate, fault.error_rate)
    summary = sequence_summary(rows, sweep, adopted, faults, fault_rates=rates)
    (run / "summary.md").write_text(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures and summaries.")
    parser.add_argument("--image-run", type=Path)
    parser.add_argument("--sequence-run", type=Path)
    parser.add_argument("--fault-run", type=Path)
    args = parser.parse_args()
    if args.image_run:
        image_outputs(args.image_run)
        print(f"wrote {args.image_run}/summary.md and figures")
    if args.sequence_run:
        sequence_outputs(args.sequence_run, args.fault_run)
        print(f"wrote {args.sequence_run}/summary.md and figures")


if __name__ == "__main__":
    main()
