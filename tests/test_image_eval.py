import json
import math

import cv2
import numpy as np
import pytest

from robust_visual_perception.coco_subset import SubsetImage, write_manifest
from robust_visual_perception.config import ExperimentConfig, ModelConfig
from robust_visual_perception.degradation import Condition
from robust_visual_perception.experiment_runner import run_image_experiment
from robust_visual_perception.image_eval import AGGREGATE_COLUMNS, evaluate_image_run
from tests.fakes import FakeDetector, person

# Image 1 has one person at xywh (10, 10, 20, 30); image 2 has none.
GT_BOX_XYXY = (10.0, 10.0, 30.0, 40.0)


def _make_run(tmp_path, detections_per_call):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    entries = []
    for image_id, n_person in [(1, 1), (2, 0)]:
        name = f"{image_id:012d}.jpg"
        cv2.imwrite(str(image_dir / name), np.full((60, 80, 3), 120, np.uint8))
        entries.append(SubsetImage(image_id, name, "http://x", n_person))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, entries)
    instances = {
        "images": [{"id": i, "width": 80, "height": 60} for i in (1, 2)],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 30],
                "area": 600,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(json.dumps(instances))
    cfg = ExperimentConfig(
        run_name="r",
        image_dir=str(image_dir),
        manifest=str(manifest),
        output_root=str(tmp_path / "results"),
        model=ModelConfig(imgsz=(320,)),
        conditions=(Condition("clean", "clean", None), Condition("blur_k7", "blur", 7)),
        warmup=0,
    )
    run_dir = run_image_experiment(
        cfg, lambda s: FakeDetector(detections_per_call), "t"
    )
    return run_dir, instances_path


def test_perfect_detector_scores_one(tmp_path):
    # calls: image 1 clean, image 1 blur, image 2 clean, image 2 blur
    script = [[person(0.9, GT_BOX_XYXY)], [person(0.9, GT_BOX_XYXY)], [], []]
    run_dir, instances = _make_run(tmp_path, script)
    rows = {r["condition"]: r for r in evaluate_image_run(run_dir, instances)}
    clean = rows["clean"]
    assert clean["person_ap"] == pytest.approx(1.0)
    assert clean["person_precision"] == pytest.approx(1.0)
    assert clean["person_recall"] == pytest.approx(1.0)
    assert clean["stop_precision"] == pytest.approx(1.0)
    assert clean["stop_recall"] == pytest.approx(1.0)
    assert clean["false_stop_rate"] == pytest.approx(0.0)
    assert clean["retention"] == pytest.approx(1.0)
    assert clean["mean_matched_conf"] == pytest.approx(0.9)
    assert clean["n_images"] == 2
    assert clean["n_person_images"] == 1


def test_degraded_condition_losing_the_person(tmp_path):
    # blur condition finds the person only at 0.3 (below threshold)
    script = [[person(0.9, GT_BOX_XYXY)], [person(0.3, GT_BOX_XYXY)], [], []]
    run_dir, instances = _make_run(tmp_path, script)
    rows = {r["condition"]: r for r in evaluate_image_run(run_dir, instances)}
    blur = rows["blur_k7"]
    assert blur["person_recall"] == pytest.approx(0.0)
    assert math.isnan(blur["person_precision"])  # no prediction >= 0.5
    assert blur["retention"] == pytest.approx(0.0)
    assert blur["stop_recall"] == pytest.approx(0.0)
    assert blur["person_ap"] == pytest.approx(1.0)  # ranking still correct


def test_false_stop_on_negative_image(tmp_path):
    fp_box = (50.0, 5.0, 70.0, 55.0)
    script = [
        [person(0.9, GT_BOX_XYXY)],
        [person(0.9, GT_BOX_XYXY)],
        [person(0.8, fp_box)],
        [],
    ]
    run_dir, instances = _make_run(tmp_path, script)
    rows = {r["condition"]: r for r in evaluate_image_run(run_dir, instances)}
    clean = rows["clean"]
    assert clean["false_stop_rate"] == pytest.approx(1.0)
    assert clean["stop_precision"] == pytest.approx(0.5)
    assert clean["person_fp"] == 1


def test_aggregate_rows_have_all_columns(tmp_path):
    script = [[person(0.9, GT_BOX_XYXY)], [], [], []]
    run_dir, instances = _make_run(tmp_path, script)
    for row in evaluate_image_run(run_dir, instances):
        assert list(row) == AGGREGATE_COLUMNS
