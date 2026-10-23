"""Run repeated simulation runs, each in a fresh container.

    uv run python scripts/run_sim.py --mode static --repeats 3
    uv run python scripts/run_sim.py --mode scripted --repeats 3

Repeat k is named <mode>_r<k> (or <prefix>_r<k>). Each repeat starts a new
container (docker compose run --rm), so no Gazebo or ROS state carries over.
Per run: results/sim/<name>/ (node log, run info), data/raw/sim/bags/<name>/
(bag) and data/raw/sim/sequences/<name>/ (extracted frames). The container
log is saved to data/raw/sim/logs/<name>.log.
"""

from __future__ import annotations

import argparse
import grp
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

COMPOSE = ["docker", "compose", "-f", "docker/compose.yaml"]


def _gid(name: str, default: int) -> str:
    try:
        return str(grp.getgrnam(name).gr_gid)
    except KeyError:
        return str(default)


def write_env(run_dir: Path, env: dict[str, str]) -> None:
    """env.json: software inside the container (where the node ran), the
    image id, and the host's git state (the container has no git)."""
    code = (
        "import json; from robust_visual_perception.environment import "
        "collect_environment; print(json.dumps(collect_environment()))"
    )
    out = subprocess.run(
        [*COMPOSE, "run", "--rm", "-T", "sim", "python", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout.strip().splitlines()[-1])
    image = subprocess.run(
        ["docker", "image", "inspect", "rvp-sim:jazzy", "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    data.update(
        container_image=image.stdout.strip(),
        git_commit=commit.stdout.strip(),
        git_dirty=bool(status.stdout.strip()),
    )
    (run_dir / "env.json").write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["static", "scripted"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefix", help="run name prefix (default: the mode)")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = {
        **os.environ,
        "HOST_UID": str(os.getuid()),
        "HOST_GID": str(os.getgid()),
        "VIDEO_GID": _gid("video", 44),
        "RENDER_GID": _gid("render", 991),
    }
    logs = Path("data/raw/sim/logs")
    logs.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.mode
    failed = []
    for k in range(1, args.repeats + 1):
        name = f"{prefix}_r{k}"
        cmd = [
            *COMPOSE,
            "run",
            "--rm",
            "-T",
            "sim",
            "docker/sim_run.sh",
            args.mode,
            name,
            str(args.duration),
            str(args.seed),
        ]
        print(f"[{name}] {shlex.join(cmd)}", flush=True)
        with (logs / f"{name}.log").open("w") as log:
            code = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode
        print(f"[{name}] exit {code}", flush=True)
        if code == 0:
            run_dir = Path("results/sim") / name
            (run_dir / "command.txt").write_text(
                shlex.join(["python", *sys.argv]) + "\n"
            )
            write_env(run_dir, env)
        else:
            failed.append(name)
    if failed:
        sys.exit(f"failed runs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
