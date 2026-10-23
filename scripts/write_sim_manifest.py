"""List the recorded simulation bags with sizes and checksums.

    uv run python scripts/write_sim_manifest.py

Bags are not committed (data/raw/sim/bags/ is git-ignored); this manifest
is, so a copy of a bag can be checked against the one the results came from.
Writes data/manifests/sim_bags.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bags", type=Path, default=Path("data/raw/sim/bags"))
    parser.add_argument(
        "--sequences", type=Path, default=Path("data/raw/sim/sequences")
    )
    parser.add_argument("--out", type=Path, default=Path("data/manifests/sim_bags.csv"))
    args = parser.parse_args()
    rows = []
    for bag in sorted(p for p in args.bags.iterdir() if p.is_dir()):
        export = args.sequences / bag.name / "export.json"
        frames = json.loads(export.read_text())["frames"] if export.is_file() else ""
        for file in sorted(bag.iterdir()):
            rows.append(
                {
                    "run": bag.name,
                    "file": file.name,
                    "bytes": file.stat().st_size,
                    "sha256": sha256(file),
                    "camera_frames": frames if file.suffix == ".mcap" else "",
                }
            )
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} files)")


if __name__ == "__main__":
    main()
