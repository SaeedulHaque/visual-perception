"""Compute metrics for a finished image run and write aggregate.csv.

uv run python scripts/evaluate_image_experiment.py --run results/image_main
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robust_visual_perception.image_eval import (
    AGGREGATE_COLUMNS,
    evaluate_image_run,
    write_aggregate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an image run.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("data/raw/coco/instances_val2017.json"),
    )
    args = parser.parse_args()
    rows = evaluate_image_run(args.run, args.instances)
    out = args.run / "aggregate.csv"
    write_aggregate(out, rows, AGGREGATE_COLUMNS)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
