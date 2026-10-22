import csv
from pathlib import Path

import cv2
import numpy as np

from robust_visual_perception.config import (
    FaultConfig,
    FilterConfig,
    SequenceExperimentConfig,
)
from robust_visual_perception.degradation import Condition
from robust_visual_perception.sequence_runner import fault_for, run_sequence_experiment
from tests.fakes import FakeDetector, person

CLEAN = Condition("clean", "clean", None)
BLUR = Condition("blur_k7", "blur", 7)
IN_PATH = (80.0, 40.0, 120.0, 90.0)  # centre (0.5 W, 0.65 H), height 0.5 H
OFF_PATH = (0.0, 40.0, 30.0, 90.0)  # centre x 0.075 W


def _make_sequence(root: Path, name: str = "MOT17-05", length: int = 6) -> None:
    seq = root / name
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}-FRCNN\nimDir=img1\nframeRate=10\n"
        f"seqLength={length}\nimWidth=200\nimHeight=100\nimExt=.jpg\n"
    )
    # GT: pedestrian in the path on frames 3-5
    rows = [f"{f},1,80,40,40,50,1,1,1.0" for f in (3, 4, 5)]
    (seq / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    for f in range(1, length + 1):
        cv2.imwrite(
            str(seq / "img1" / f"{f:06d}.jpg"), np.full((100, 200, 3), 90, np.uint8)
        )


def _cfg(tmp_path, conditions=(CLEAN,), fault=None, **kw):
    return SequenceExperimentConfig(
        run_name="seq",
        data_root=str(tmp_path / "mot"),
        sequences=("MOT17-05",),
        output_root=str(tmp_path / "results"),
        conditions=conditions,
        fault=fault or FaultConfig(),
        warmup=0,
        filter=FilterConfig(n_on=2, k_window=3, m_off=2),
        **kw,
    )


def _rows(run_dir: Path) -> list[dict]:
    with (run_dir / "per_frame.csv").open() as f:
        return list(csv.DictReader(f))


def _script(*per_frame):
    return [
        [person(0.9, IN_PATH)]
        if s == "S"
        else [person(0.9, OFF_PATH)]
        if s == "O"
        else []
        for s in per_frame
    ]


def test_rows_carry_gt_baseline_and_filtered_states(tmp_path):
    _make_sequence(tmp_path / "mot")
    det = FakeDetector(_script("C", "C", "S", "S", "S", "C"))
    run_dir = run_sequence_experiment(_cfg(tmp_path), lambda s: det, "t")
    rows = _rows(run_dir)
    assert [r["frame_id"] for r in rows] == [f"{i:06d}" for i in range(1, 7)]
    assert {r["sequence"] for r in rows} == {"MOT17-05"}
    assert [r["gt_state"] for r in rows] == ["CLEAR"] * 2 + ["STOP"] * 3 + ["CLEAR"]
    assert [r["baseline_state"] for r in rows] == ["CLEAR"] * 2 + ["STOP"] * 3 + [
        "CLEAR"
    ]
    # n_on=2 of k=3, m_off=2
    assert [r["filtered_state"] for r in rows] == ["CLEAR"] * 3 + ["STOP"] * 3
    assert all(r["imgsz"] == "416" for r in rows)


def test_roi_is_applied_to_detections(tmp_path):
    _make_sequence(tmp_path / "mot")
    det = FakeDetector(_script("O", "O", "O", "O", "O", "O"))
    run_dir = run_sequence_experiment(_cfg(tmp_path), lambda s: det, "t")
    assert {r["baseline_state"] for r in _rows(run_dir)} == {"CLEAR"}


def test_missing_image_keeps_frame_alignment(tmp_path):
    _make_sequence(tmp_path / "mot")
    (tmp_path / "mot" / "MOT17-05" / "img1" / "000003.jpg").unlink()
    det = FakeDetector(_script("C", "C", "S", "S", "C"))
    run_dir = run_sequence_experiment(_cfg(tmp_path), lambda s: det, "t")
    rows = _rows(run_dir)
    assert len(rows) == 6
    assert rows[2]["frame_id"] == "000003"
    assert rows[2]["status"] == "MISSING"
    assert rows[2]["baseline_state"] == "UNKNOWN"
    assert rows[2]["filtered_state"] == "UNKNOWN"
    assert rows[3]["baseline_state"] == "STOP"


def test_full_fault_rates(tmp_path):
    _make_sequence(tmp_path / "mot")
    det = FakeDetector(_script("S"))
    cfg = _cfg(tmp_path, fault=FaultConfig(missing_rate=1.0))
    rows = _rows(run_sequence_experiment(cfg, lambda s: det, "t"))
    assert {r["status"] for r in rows} == {"MISSING"}
    assert det.calls == 0
    cfg = _cfg(tmp_path, fault=FaultConfig(error_rate=1.0))
    rows = _rows(run_sequence_experiment(cfg, lambda s: det, "t", overwrite=True))
    assert {r["status"] for r in rows} == {"ERROR"}
    assert {r["state_reason"] for r in rows} == {"detector_error"}


def test_faults_hit_the_same_frames_under_every_condition(tmp_path):
    _make_sequence(tmp_path / "mot", length=40)
    cfg = _cfg(tmp_path, conditions=(CLEAN, BLUR), fault=FaultConfig(0.3, 0.2, 7))
    rows = _rows(run_sequence_experiment(cfg, lambda s: FakeDetector([[]]), "t"))
    by_cond = {
        c: [r["status"] for r in rows if r["condition"] == c]
        for c in ("clean", "blur_k7")
    }
    assert by_cond["clean"] == by_cond["blur_k7"]
    assert {"MISSING", "ERROR", "EMPTY"} <= set(by_cond["clean"])


def test_fault_for_is_deterministic():
    fault = FaultConfig(0.02, 0.01, 5)
    assert fault_for("MOT17-05/000001", fault) == fault_for("MOT17-05/000001", fault)
    assert fault_for("x", FaultConfig()) is None


def test_filter_is_reset_between_sequences(tmp_path):
    _make_sequence(tmp_path / "mot", "MOT17-05", length=3)
    _make_sequence(tmp_path / "mot", "MOT17-10", length=3)
    cfg = SequenceExperimentConfig(
        **{**_cfg(tmp_path).__dict__, "sequences": ("MOT17-05", "MOT17-10")}
    )
    det = FakeDetector(_script("S", "S", "S", "C", "C", "C"))
    rows = _rows(run_sequence_experiment(cfg, lambda s: det, "t"))
    second = [r["filtered_state"] for r in rows if r["sequence"] == "MOT17-10"]
    assert second == ["CLEAR"] * 3  # STOP from the first sequence is not carried over
