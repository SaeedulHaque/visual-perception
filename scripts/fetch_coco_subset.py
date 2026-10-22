"""Fetch a reproducible COCO val2017 subset.

Downloads the official annotation archive once (about 240 MB, cached in
data/raw/coco/), selects the subset, writes the manifest to
data/manifests/coco_val2017_<name>.csv and downloads only the selected
images (not the 1 GB image archive).

    uv run python scripts/fetch_coco_subset.py \
        --n-person 14 --n-negative 6 --name smoke
    uv run python scripts/fetch_coco_subset.py \
        --n-person 500 --n-negative 250 --name main
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from robust_visual_perception.coco_subset import (
    download_file,
    select_subset,
    write_manifest,
)

ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
INSTANCES_MEMBER = "annotations/instances_val2017.json"

logger = logging.getLogger("fetch_coco_subset")


def load_instances(raw_dir: Path) -> dict:
    json_path = raw_dir / "instances_val2017.json"
    if not json_path.exists():
        archive = download_file(
            ANNOTATIONS_URL, raw_dir / "annotations_trainval2017.zip"
        )
        logger.info("extracting %s", INSTANCES_MEMBER)
        part = json_path.with_name(json_path.name + ".part")
        with zipfile.ZipFile(archive) as zf, zf.open(INSTANCES_MEMBER) as src:
            part.write_bytes(src.read())
        part.replace(json_path)
    with json_path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-person", type=int, required=True)
    parser.add_argument("--n-negative", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", required=True, help="manifest suffix, e.g. smoke")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/coco"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    instances = load_instances(args.raw_dir)
    subset = select_subset(instances, args.n_person, args.n_negative, args.seed)
    manifest = args.manifest_dir / f"coco_val2017_{args.name}.csv"
    write_manifest(manifest, subset)
    logger.info("wrote %s (%d images)", manifest, len(subset))

    image_dir = args.raw_dir / "val2017"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                lambda img: download_file(img.coco_url, image_dir / img.file_name),
                subset,
            )
        )
    logger.info("images in %s", image_dir)


if __name__ == "__main__":
    main()
