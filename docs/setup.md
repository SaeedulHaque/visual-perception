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

## Notes

- `uv` installed from snap: `uv run` exits with code 120 when its output is
  redirected to a file under `/tmp` (snap confinement). Calling
  `.venv/bin/python` directly, or piping through `cat`, avoids it.
- Ultralytics writes a small settings file to `~/.config/Ultralytics/` on
  first import.
