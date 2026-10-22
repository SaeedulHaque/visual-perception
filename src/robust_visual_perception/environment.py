"""Record the software and hardware a run used (saved as env.json)."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

PACKAGES = ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_environment() -> dict[str, Any]:
    import torch

    status = _git("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "packages": _package_versions(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }
