import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from robust_visual_perception.coco_subset import SubsetImage, write_manifest
from robust_visual_perception.config import ExperimentConfig, ModelConfig
from robust_visual_perception.degradation import Condition
from robust_visual_perception.experiment_runner import run_image_experiment
from tests.fakes import FakeDetector, person

CONDITIONS = (Condition("clean", "clean", None), Condition("blur_k7", "blur", 7))


def _setup(tmp_path: Path, corrupt: bool = False) -> ExperimentConfig:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i, n_person in [(1, 1), (2, 0), (3, 2)]:
        name = f"{i:012d}.jpg"
        cv2.imwrite(str(image_dir / name), np.full((40, 60, 3), 90, np.uint8))
        rows.append(SubsetImage(i, name, f"http://x/{name}", n_person))
    if corrupt:
        (image_dir / rows[1].file_name).write_bytes(b"not a jpeg")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    return ExperimentConfig(
        run_name="t",
        image_dir=str(image_dir),
        manifest=str(manifest),
        output_root=str(tmp_path / "results"),
        model=ModelConfig(imgsz=(320,)),
        conditions=CONDITIONS,
        warmup=2,
    )


def _rows(run_dir: Path) -> list[dict]:
    with (run_dir / "per_frame.csv").open() as f:
        return list(csv.DictReader(f))


def test_writes_one_row_per_frame_and_condition(tmp_path):
    cfg = _setup(tmp_path)
    run_dir = run_image_experiment(
        cfg, lambda imgsz: FakeDetector([[person(0.9)]]), command="test"
    )
    rows = _rows(run_dir)
    assert len(rows) == 3 * 2
    assert {r["condition"] for r in rows} == {"clean", "blur_k7"}
    assert all(r["imgsz"] == "320" for r in rows)
    assert all(r["baseline_state"] == "STOP" for r in rows)
    assert all(r["n_target"] == "1" for r in rows)
    assert all(r["filtered_state"] == "" for r in rows)
    gt = {r["frame_id"]: r["gt_state"] for r in rows}
    assert gt == {
        "000000000001": "STOP",
        "000000000002": "CLEAR",
        "000000000003": "STOP",
    }
    for name in ["config.yaml", "env.json", "command.txt"]:
        assert (run_dir / name).exists()


def test_warmup_calls_are_not_written(tmp_path):
    cfg = _setup(tmp_path)
    detectors = []

    def factory(imgsz):
        detectors.append(FakeDetector([[]]))
        return detectors[-1]

    run_dir = run_image_experiment(cfg, factory, command="test")
    assert detectors[0].calls == 2 + 3 * 2
    assert len(_rows(run_dir)) == 6


def test_low_confidence_target_is_clear_but_reported(tmp_path):
    cfg = _setup(tmp_path)
    run_dir = run_image_experiment(
        cfg, lambda imgsz: FakeDetector([[person(0.3)]]), command="test"
    )
    row = _rows(run_dir)[0]
    assert row["baseline_state"] == "CLEAR"
    assert row["n_target"] == "0"
    assert float(row["max_target_conf"]) == pytest.approx(0.3)
    assert row["status"] == "OK"


def test_corrupt_image_gives_missing_rows(tmp_path):
    cfg = _setup(tmp_path, corrupt=True)
    run_dir = run_image_experiment(
        cfg, lambda imgsz: FakeDetector([[person(0.9)]]), command="test"
    )
    missing = [r for r in _rows(run_dir) if r["frame_id"] == "000000000002"]
    assert len(missing) == 2
    for row in missing:
        assert row["status"] == "MISSING"
        assert row["baseline_state"] == "UNKNOWN"
        assert row["state_reason"] == "missing_frame"
        assert row["latency_ms"] == ""


def test_detector_error_gives_unknown_and_run_continues(tmp_path):
    cfg = _setup(tmp_path)
    script = [[], [], RuntimeError("boom"), [person(0.9)]]  # 2 warm-up calls first
    run_dir = run_image_experiment(
        cfg, lambda imgsz: FakeDetector(script), command="test"
    )
    rows = _rows(run_dir)
    assert rows[0]["status"] == "ERROR"
    assert rows[0]["baseline_state"] == "UNKNOWN"
    assert rows[0]["state_reason"] == "detector_error"
    assert len(rows) == 6
    assert rows[-1]["baseline_state"] == "STOP"


class _RaisingDetector:
    def detect(self, image):
        raise RuntimeError("bug inside a detector")


def test_detector_that_raises_is_contained(tmp_path):
    cfg = _setup(tmp_path)
    cfg = ExperimentConfig(**{**cfg.__dict__, "warmup": 0})
    run_dir = run_image_experiment(cfg, lambda imgsz: _RaisingDetector(), "test")
    assert {r["status"] for r in _rows(run_dir)} == {"ERROR"}


def test_existing_run_is_not_overwritten(tmp_path):
    cfg = _setup(tmp_path)
    factory = lambda imgsz: FakeDetector([[]])  # noqa: E731
    run_image_experiment(cfg, factory, command="test")
    with pytest.raises(FileExistsError):
        run_image_experiment(cfg, factory, command="test")
    run_image_experiment(cfg, factory, command="test", overwrite=True)


def test_without_manifest_uses_folder_and_empty_gt(tmp_path):
    cfg = _setup(tmp_path)
    cfg = ExperimentConfig(**{**cfg.__dict__, "manifest": None})
    run_dir = run_image_experiment(cfg, lambda imgsz: FakeDetector([[]]), "test")
    rows = _rows(run_dir)
    assert len(rows) == 6
    assert {r["gt_state"] for r in rows} == {""}
