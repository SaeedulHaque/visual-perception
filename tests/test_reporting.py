import csv
import json

import pytest
import yaml

from robust_visual_perception.config import ExperimentConfig
from robust_visual_perception.environment import collect_environment
from robust_visual_perception.reporting import (
    PER_FRAME_COLUMNS,
    PerFrameWriter,
    prepare_run_dir,
    write_resources,
    write_run_metadata,
)


def test_per_frame_writer_header_and_rows(tmp_path):
    path = tmp_path / "per_frame.csv"
    with PerFrameWriter(path) as writer:
        writer.write({"frame_id": "1", "condition": "clean", "imgsz": 320})
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == PER_FRAME_COLUMNS
    assert rows[0]["frame_id"] == "1"
    assert rows[0]["gt_state"] == ""


def test_per_frame_writer_rejects_unknown_column(tmp_path):
    with PerFrameWriter(tmp_path / "p.csv") as writer, pytest.raises(KeyError):
        writer.write({"frame_id": "1", "bogus": 2})


def test_per_frame_columns_match_design():
    assert PER_FRAME_COLUMNS == [
        "sequence",
        "frame_id",
        "condition",
        "imgsz",
        "status",
        "n_detections",
        "n_target",
        "max_target_conf",
        "latency_ms",
        "load_ms",
        "degrade_ms",
        "baseline_state",
        "state_reason",
        "filtered_state",
        "gt_state",
    ]


def test_prepare_run_dir_refuses_existing_without_overwrite(tmp_path):
    run_dir = prepare_run_dir(tmp_path, "run", overwrite=False)
    assert run_dir.is_dir()
    (run_dir / "old.txt").write_text("x")
    with pytest.raises(FileExistsError):
        prepare_run_dir(tmp_path, "run", overwrite=False)
    again = prepare_run_dir(tmp_path, "run", overwrite=True)
    assert again == run_dir
    assert not (run_dir / "old.txt").exists()


def test_write_run_metadata(tmp_path):
    cfg = ExperimentConfig(run_name="r", image_dir="imgs")
    write_run_metadata(tmp_path, cfg, "python scripts/run.py --config x.yaml")
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["run_name"] == "r"
    env = json.loads((tmp_path / "env.json").read_text())
    assert "python" in env
    assert (tmp_path / "command.txt").read_text().strip().endswith("x.yaml")


def test_collect_environment_has_core_fields():
    env = collect_environment()
    for key in [
        "python",
        "platform",
        "cpu_model",
        "logical_cores",
        "ram_gib",
        "packages",
        "torch_threads",
        "git_commit",
        "git_dirty",
    ]:
        assert key in env
    assert "torch" in env["packages"]
    json.dumps(env)


def test_write_resources(tmp_path):
    write_resources(tmp_path, peak_rss_bytes=3 * 2**20, wall_time_s=12.5, n_rows=7)
    data = json.loads((tmp_path / "resources.json").read_text())
    assert data == {"peak_rss_mib": 3.0, "wall_time_s": 12.5, "n_rows": 7}
