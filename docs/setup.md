# Setup

## Requirements

- Linux (developed on Ubuntu 26.04, x86_64)
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management
- About 2 GB of disk for the environment, 300 MB for COCO annotations

No system packages, no `sudo`, and no change to the system Python.

## Create the environment

```
uv sync
```

This creates `.venv/` with **Python 3.12** and installs the locked
dependencies from `uv.lock`, including the dev tools (pytest, ruff).

Why 3.12: the system Python on this machine is 3.14, which is newer than
what PyTorch, OpenCV and Ultralytics reliably ship wheels for. `uv`
downloads a standalone 3.12 just for this project.

Why CPU wheels: `pyproject.toml` points `torch` and `torchvision` at the
PyTorch CPU index. The default wheels bundle CUDA libraries (several GB) that
are useless on a machine without an NVIDIA GPU. The GPU comparison
(Phase 4) uses a separate environment on another machine.

## Check it works

```
uv run python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
uv run pytest
uv run ruff check .
```

The first line should print a `+cpu` torch version and `False`.

`uv run` runs a command inside `.venv` without activating it. Activating
works too: `source .venv/bin/activate`.

## ROS 2 simulation (Phase 3)

The simulation needs Docker with Compose v2 and about 7 GB of disk for the
image. Nothing ROS-related is installed on the host:

```
docker compose -f docker/compose.yaml build
HOST_UID=$(id -u) HOST_GID=$(id -g) RENDER_GID=$(getent group render | cut -d: -f3) \
    docker compose -f docker/compose.yaml run --rm -T sim docker/check_container.sh
```

What the image contains and why: [features/ros2-container.md](features/ros2-container.md).
Running the simulation: [features/sim-runs.md](features/sim-runs.md).

## Notes

- `uv` installed from snap: `uv run` exits with code 120 when its output is
  redirected to a file under `/tmp` (snap confinement). Calling
  `.venv/bin/python` directly, or piping through `cat`, avoids it.
- Ultralytics writes a small settings file to `~/.config/Ultralytics/` on
  first import.
