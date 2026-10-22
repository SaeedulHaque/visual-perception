"""Run outputs: run directory, metadata files and the per-frame CSV."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml

from robust_visual_perception.config import ExperimentConfig, config_to_dict
from robust_visual_perception.environment import collect_environment

# One row per (frame, condition, imgsz). Documented in
# docs/features/config-and-outputs.md.
PER_FRAME_COLUMNS = [
    "frame_id",
    "condition",
    "imgsz",
    "status",
    "n_detections",
    "n_target",
    "max_target_conf",
    "latency_ms",
    "baseline_state",
    "state_reason",
    "filtered_state",
    "gt_state",
]


class PerFrameWriter:
    """CSV writer with a fixed header. Unknown columns raise ``KeyError``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> PerFrameWriter:
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=PER_FRAME_COLUMNS)
        self._writer.writeheader()
        return self

    def write(self, row: dict[str, Any]) -> None:
        unknown = set(row) - set(PER_FRAME_COLUMNS)
        if unknown:
            raise KeyError(f"unknown per-frame column(s): {sorted(unknown)}")
        self._writer.writerow(row)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._file.close()


def prepare_run_dir(output_root: str | Path, run_name: str, overwrite: bool) -> Path:
    """Create ``output_root/run_name``. Existing results are never replaced
    unless ``overwrite`` is set."""
    run_dir = Path(output_root) / run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{run_dir} already exists; choose another run_name or pass "
                f"--overwrite to replace it"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def write_run_metadata(run_dir: Path, cfg: ExperimentConfig, command: str) -> None:
    run_dir = Path(run_dir)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config_to_dict(cfg), sort_keys=False)
    )
    (run_dir / "env.json").write_text(json.dumps(collect_environment(), indent=2))
    (run_dir / "command.txt").write_text(command + "\n")
