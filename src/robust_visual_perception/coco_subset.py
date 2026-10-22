"""Select and fetch a reproducible COCO val2017 subset.

Person images have at least one non-crowd ``person`` annotation. Negative
images have no ``person`` annotation at all. Images whose only people are
crowd regions are left out of both pools, because their ground truth is
ambiguous for an image-level STOP decision.
"""

from __future__ import annotations

import csv
import shutil
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_COLUMNS = ["image_id", "file_name", "coco_url", "n_person"]


@dataclass(frozen=True)
class SubsetImage:
    image_id: int
    file_name: str
    coco_url: str
    n_person: int  # non-crowd person annotations


def _person_category_id(instances: dict[str, Any]) -> int:
    for cat in instances["categories"]:
        if cat["name"] == "person":
            return int(cat["id"])
    raise ValueError("no 'person' category in annotations")


def _permuted_prefix(ids: list[int], n: int, seed: int, pool: str) -> list[int]:
    if n > len(ids):
        raise ValueError(f"requested {n} {pool} images, only {len(ids)} available")
    order = np.random.default_rng(seed).permutation(len(ids))
    return [ids[i] for i in order[:n]]


def select_subset(
    instances: dict[str, Any], n_person: int, n_negative: int, seed: int = 0
) -> list[SubsetImage]:
    """Pick ``n_person`` person images then ``n_negative`` negatives.

    Each pool is sorted by image id and permuted with ``seed``; the first n
    are taken. A smaller request is therefore a prefix of a larger one, so
    the smoke subset is contained in the full subset.
    """
    person_id = _person_category_id(instances)
    non_crowd: Counter[int] = Counter()
    any_person: set[int] = set()
    for ann in instances["annotations"]:
        if ann["category_id"] != person_id:
            continue
        any_person.add(ann["image_id"])
        if not ann.get("iscrowd", 0):
            non_crowd[ann["image_id"]] += 1

    images = {img["id"]: img for img in instances["images"]}
    person_pool = sorted(non_crowd)
    negative_pool = sorted(i for i in images if i not in any_person)

    chosen = _permuted_prefix(person_pool, n_person, seed, "person") + (
        _permuted_prefix(negative_pool, n_negative, seed, "negative")
    )
    return [
        SubsetImage(
            image_id=i,
            file_name=images[i]["file_name"],
            coco_url=images[i]["coco_url"],
            n_person=non_crowd.get(i, 0),
        )
        for i in chosen
    ]


def write_manifest(path: str | Path, images: list[SubsetImage]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLUMNS)
        for img in images:
            writer.writerow([img.image_id, img.file_name, img.coco_url, img.n_person])


def read_manifest(path: str | Path) -> list[SubsetImage]:
    with Path(path).open(newline="") as f:
        return [
            SubsetImage(
                int(row["image_id"]),
                row["file_name"],
                row["coco_url"],
                int(row["n_person"]),
            )
            for row in csv.DictReader(f)
        ]


def download_file(
    url: str,
    dest: str | Path,
    retries: int = 3,
    timeout_s: float = 60.0,
    backoff_s: float = 2.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Download ``url`` to ``dest`` unless it already exists.

    Data goes to ``dest.part`` first and is renamed only when complete, so
    an interrupted download never leaves a file that looks valid.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with opener(url, timeout=timeout_s) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out)
            part.replace(dest)
            return dest
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(backoff_s * attempt)
    raise AssertionError("unreachable")
