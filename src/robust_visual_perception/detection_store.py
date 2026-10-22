"""Store raw detections per (condition, imgsz) for later COCO evaluation.

Each file ``detections_<condition>_<imgsz>.npz`` holds one N x 7 float array
``[image_id, x, y, w, h, score, coco_category_id]``. This is the array layout
``pycocotools.COCO.loadRes`` accepts, and much smaller than COCO JSON.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from robust_visual_perception.detector import Detection

# YOLO's 80 contiguous class indices -> COCO's 91-id category ids (with gaps).
# Same table as ultralytics.data.converter.coco80_to_coco91_class().
COCO80_TO_91 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)  # fmt: skip


def _path(run_dir: Path, condition: str, imgsz: int) -> Path:
    return Path(run_dir) / f"detections_{condition}_{imgsz}.npz"


class DetectionStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    def add(
        self, condition: str, imgsz: int, image_id: int, detections: list[Detection]
    ) -> None:
        key = (condition, imgsz)
        self._rows.setdefault(key, [])
        if not detections:
            return
        self._rows[key].append(
            np.array(
                [
                    [
                        image_id,
                        d.x1,
                        d.y1,
                        d.width,
                        d.height,
                        d.confidence,
                        COCO80_TO_91[d.class_id],
                    ]
                    for d in detections
                ],
                dtype=np.float64,
            )
        )

    def save(self, run_dir: Path) -> None:
        for (condition, imgsz), chunks in self._rows.items():
            array = np.concatenate(chunks) if chunks else np.zeros((0, 7))
            np.savez_compressed(_path(run_dir, condition, imgsz), detections=array)


def load_detections(run_dir: Path, condition: str, imgsz: int) -> np.ndarray:
    path = _path(run_dir, condition, imgsz)
    if not path.is_file():
        raise FileNotFoundError(f"no stored detections: {path}")
    with np.load(path) as data:
        return data["detections"]
