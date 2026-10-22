"""Run the offline pipeline end to end on a small image set.

    uv run python scripts/run_smoke_test.py --config configs/smoke.yaml

Needs the smoke subset first:

    uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from robust_visual_perception.config import load_config
from robust_visual_perception.detector import UltralyticsDetector
from robust_visual_perception.experiment_runner import run_image_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline pipeline smoke test.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing run directory"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)

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
