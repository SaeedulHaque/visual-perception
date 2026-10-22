"""Fetch MOT17 ground truth, select sequences from GT only, fetch their images.

    uv run python scripts/fetch_mot17.py

1. Reads the directory of the remote MOT17.zip (5.9 GB) with HTTP range
   requests; nothing large is downloaded.
2. Fetches seqinfo.ini and gt/gt.txt of the seven training sequences.
3. Searches the ROI grid on GT only (data/manifests/mot17_roi_search.csv),
   chooses the ROI by the documented rule, assigns sequence roles, and
   writes data/manifests/mot17_sequences.csv.
4. Downloads the images (one detector copy, FRCNN) of the selected
   sequences only, one contiguous byte range per sequence.
"""

from __future__ import annotations

import argparse
import csv
import logging
import zipfile
from dataclasses import asdict
from pathlib import Path

from robust_visual_perception.decision import Roi
from robust_visual_perception.mot17 import (
    TRAIN_SEQUENCES,
    SequenceStats,
    choose_roi,
    gt_stop_states,
    parse_gt,
    parse_seqinfo,
    roi_grid,
    roi_search,
    select_sequences,
    sequence_stats,
)
from robust_visual_perception.remote_zip import RangeFile, extract_prefix

URL = "https://motchallenge.net/data/MOT17.zip"
COPY = "FRCNN"  # images are identical across the DPM/FRCNN/SDP copies

logger = logging.getLogger("fetch_mot17")


def fetch_small(zf: zipfile.ZipFile, member: str, dest: Path) -> str:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(zf.read(member))
        part.replace(dest)
    return dest.read_text()


def _roi_fields(roi: Roi) -> dict[str, float]:
    return {f"roi_{k}": v for k, v in asdict(roi).items()}


def write_roi_search(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [s.name for s in rows[0]["stats"]]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [*_roi_fields(rows[0]["roi"]), "qualifies", "min_events"]
            + [f"{n}_{m}" for n in names for m in ("occupancy", "events")]
        )
        for r in rows:
            per_seq = [
                v for s in r["stats"] for v in (f"{s.occupancy:.4f}", s.n_events)
            ]
            writer.writerow(
                [*_roi_fields(r["roi"]).values(), r["qualifies"], r["min_events"]]
                + per_seq
            )


def write_manifest(
    path: Path, stats: list[SequenceStats], roles: dict[str, str], roi: Roi
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [*asdict(stats[0]), "role", *_roi_fields(roi)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in stats:
            row = {**asdict(s), "occupancy": f"{s.occupancy:.4f}"}
            writer.writerow({**row, "role": roles[s.name], **_roi_fields(roi)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MOT17 sequences.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mot17"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/mot17_sequences.csv"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rf = RangeFile(URL)
    zf = zipfile.ZipFile(rf)
    logger.info(
        "remote archive: %.1f GB, %d entries", rf.size / 1e9, len(zf.infolist())
    )

    sequences = {}
    for name in TRAIN_SEQUENCES:
        prefix = f"MOT17/train/{name}-{COPY}/"
        seq_dir = args.raw_dir / name
        info_text = fetch_small(zf, prefix + "seqinfo.ini", seq_dir / "seqinfo.ini")
        gt_text = fetch_small(zf, prefix + "gt/gt.txt", seq_dir / "gt" / "gt.txt")
        sequences[name] = (parse_seqinfo(info_text), parse_gt(gt_text))

    # GT only: search the ROI grid, keep the search table, choose by the rule.
    search = roi_search(sequences, roi_grid())
    write_roi_search(args.manifest.with_name("mot17_roi_search.csv"), search)
    roi = choose_roi(search)
    logger.info("ROI: %s", roi)
    stats = [
        sequence_stats(info, gt_stop_states(gt, info, roi))
        for info, gt in sequences.values()
    ]
    roles = select_sequences(stats)
    write_manifest(args.manifest, stats, roles, roi)
    for s in stats:
        logger.info("%-10s %s", roles[s.name], s)

    for name, role in roles.items():
        if role == "unused":
            continue
        prefix = f"MOT17/train/{name}-{COPY}/img1/"
        dest = args.raw_dir / name / "img1"
        n = extract_prefix(zf, rf, prefix, dest, cache_dir=args.raw_dir)
        logger.info(
            "%s (%s): %s", name, role, f"{n} images" if n else "already present"
        )


if __name__ == "__main__":
    main()
