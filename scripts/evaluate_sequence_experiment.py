"""Compute metrics for a finished sequence run.

    uv run python scripts/evaluate_sequence_experiment.py --run results/sequence_main

Writes aggregate.csv (per sequence, condition, method), and when the run
contains the tuning sequence: sweep.csv, sweep_summary.csv and
adopted_filter.json (setting chosen by the rule declared in the plan).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from robust_visual_perception.config import FilterConfig
from robust_visual_perception.image_eval import write_aggregate
from robust_visual_perception.sequence_eval import (
    AGGREGATE_COLUMNS,
    SWEEP_COLUMNS,
    evaluate_sequence_run,
    read_roles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a sequence run.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/mot17_sequences.csv")
    )
    parser.add_argument(
        "--adopted",
        type=Path,
        help="adopted_filter.json of the main run, applied to this run as well",
    )
    args = parser.parse_args()
    override = None
    if args.adopted:
        override = FilterConfig(**json.loads(args.adopted.read_text())["adopted"])
    result = evaluate_sequence_run(args.run, read_roles(args.manifest), override)
    write_aggregate(args.run / "aggregate.csv", result.aggregate, AGGREGATE_COLUMNS)
    print(f"wrote {args.run / 'aggregate.csv'} ({len(result.aggregate)} rows)")
    if result.sweep:
        write_aggregate(args.run / "sweep.csv", result.sweep, SWEEP_COLUMNS)
        summary = [
            {**asdict(s["setting"]), **{k: v for k, v in s.items() if k != "setting"}}
            for s in result.sweep_summary
        ]
        write_aggregate(args.run / "sweep_summary.csv", summary, list(summary[0]))
        adopted = {
            "default": asdict(result.run_filter),
            "adopted": asdict(result.adopted),
            "changed": result.adopted != result.run_filter,
        }
        (args.run / "adopted_filter.json").write_text(json.dumps(adopted, indent=2))
        print(f"adopted filter: {result.adopted}")


if __name__ == "__main__":
    main()
