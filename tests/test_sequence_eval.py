import pytest

from robust_visual_perception.config import SequenceExperimentConfig
from robust_visual_perception.sequence_eval import evaluate_sequence_run
from robust_visual_perception.sequence_metrics import SWEEP_SETTINGS
from robust_visual_perception.sequence_runner import run_sequence_experiment
from tests.fakes import FakeDetector
from tests.test_sequence_runner import CLEAN, _cfg, _make_sequence, _script


def _run(tmp_path, sequences):
    for name in sequences:
        _make_sequence(tmp_path / "mot", name, length=6)
    cfg = SequenceExperimentConfig(
        **{
            **_cfg(tmp_path).__dict__,
            "sequences": tuple(sequences),
            "conditions": (CLEAN,),
        }
    )
    det = FakeDetector(_script(*(["C", "C", "S", "S", "S", "C"] * len(sequences))))
    return run_sequence_experiment(cfg, lambda s: det, "t")


def test_aggregate_has_baseline_and_filtered_rows(tmp_path):
    run_dir = _run(tmp_path, ["MOT17-05"])
    result = evaluate_sequence_run(run_dir, {"MOT17-05": "evaluation"})
    methods = {(r["sequence"], r["method"]) for r in result.aggregate}
    assert methods == {("MOT17-05", "baseline"), ("MOT17-05", "filtered")}
    base = next(r for r in result.aggregate if r["method"] == "baseline")
    assert base["recall"] == pytest.approx(1.0)
    assert base["events_detected"] == 1
    assert base["role"] == "evaluation"
    assert result.sweep == []
    assert result.adopted == result.run_filter


def test_sweep_runs_on_tuning_sequence_only(tmp_path):
    run_dir = _run(tmp_path, ["MOT17-05", "MOT17-09"])
    roles = {"MOT17-05": "evaluation", "MOT17-09": "tuning"}
    result = evaluate_sequence_run(run_dir, roles)
    assert {r["sequence"] for r in result.sweep} == {"MOT17-09"}
    assert len(result.sweep) == len(SWEEP_SETTINGS)  # one condition
    assert result.adopted in SWEEP_SETTINGS


def test_adopted_setting_adds_rows_when_it_differs(tmp_path):
    run_dir = _run(tmp_path, ["MOT17-05", "MOT17-09"])
    roles = {"MOT17-05": "evaluation", "MOT17-09": "tuning"}
    result = evaluate_sequence_run(run_dir, roles)
    methods = {r["method"] for r in result.aggregate}
    if result.adopted != result.run_filter:
        assert "filtered_adopted" in methods
    else:
        assert "filtered_adopted" not in methods
