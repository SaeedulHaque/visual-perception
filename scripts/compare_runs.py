"""Compare your run with a committed reference run.

    uv run python scripts/compare_runs.py \
        --reference results/image_main --candidate results/my_image_main

Accuracy columns must agree within --tolerance (exit code 1 if not).
Timing columns are printed side by side: that is your device's benchmark
against the reference machine (see results/image_main/env.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_visual_perception.compare import compare_aggregates
from robust_visual_perception.plots import read_csv


def _machine(run: Path) -> str:
    env_path = run / "env.json"
    if not env_path.exists():
        return "unknown machine"
    env = json.loads(env_path.read_text())
    return f"{env.get('cpu_model')}, torch {env['packages'].get('torch')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two runs.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    ref = read_csv(args.reference / "aggregate.csv")
    cand = read_csv(args.candidate / "aggregate.csv")
    keys = (
        ["sequence", "condition", "method"]
        if "sequence" in ref[0]
        else ["condition", "imgsz"]
    )
    result = compare_aggregates(ref, cand, keys, args.tolerance)

    print(f"reference: {args.reference} ({_machine(args.reference)})")
    print(f"candidate: {args.candidate} ({_machine(args.candidate)})")
    print(f"rows compared: {result.n_compared}; largest accuracy difference: "
          f"{result.max_abs_diff:.6g}")  # fmt: skip
    if result.only_in_reference:
        print(f"rows only in reference (not compared): {len(result.only_in_reference)}")
    for d in result.differences[:15]:
        print(f"  DIFF {'/'.join(d['key'])} {d['column']}: "
              f"{d['reference']} vs {d['candidate']}")  # fmt: skip
    if len(result.differences) > 15:
        print(f"  ... {len(result.differences) - 15} more")

    latency = [t for t in result.timing if t["column"] == "latency_mean_ms"]
    if latency:
        print("\nmean detector latency (ms)      reference  candidate  ratio")
        for t in latency:
            print(f"  {'/'.join(t['key']):30s} {t['reference']:9.1f} "
                  f"{t['candidate']:10.1f} {t['ratio']:6.2f}")  # fmt: skip

    verdict = "MATCH" if result.matches else "DIFFERENT"
    print(f"\naccuracy: {verdict} (tolerance {args.tolerance})")
    sys.exit(0 if result.matches else 1)


if __name__ == "__main__":
    main()
