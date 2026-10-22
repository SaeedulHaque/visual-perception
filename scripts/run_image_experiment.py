"""Run an image experiment (inference only) from a config file.

    uv run python scripts/run_image_experiment.py --config configs/smoke.yaml
    uv run python scripts/run_image_experiment.py --config configs/image_experiment.yaml

Writes results/<run_name>/ with per_frame.csv and stored detections.
Metrics come from scripts/evaluate_image_experiment.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import shlex
import sys
from pathlib import Path

from robust_visual_perception.config import load_config
from robust_visual_perception.detector import UltralyticsDetector
from robust_visual_perception.experiment_runner import run_image_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an image experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing run directory"
    )
    parser.add_argument(
        "--run-name",
        help="override run_name from the config (keeps committed results intact)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    if args.run_name:
        cfg = dataclasses.replace(cfg, run_name=args.run_name)

    def make_detector(imgsz: int) -> UltralyticsDetector:
        return UltralyticsDetector(
            cfg.model.weights,
            imgsz=imgsz,
            conf=cfg.model.conf,
            max_det=cfg.model.max_det,
            device=cfg.model.device,
        )

    command = shlex.join([Path(sys.executable).name, *sys.argv])
    run_dir = run_image_experiment(cfg, make_detector, command, args.overwrite)
    print(f"results written to {run_dir}")


if __name__ == "__main__":
    main()
