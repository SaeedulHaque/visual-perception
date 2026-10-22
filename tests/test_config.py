import pytest
import yaml

from robust_visual_perception.config import (
    ExperimentConfig,
    config_to_dict,
    load_config,
)
from robust_visual_perception.degradation import DEFAULT_CONDITIONS, Condition

MINIMAL = {"run_name": "demo", "image_dir": "data/raw/images"}


def _write(tmp_path, data) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_minimal_config_uses_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.conditions == DEFAULT_CONDITIONS
    assert cfg.model.weights == "models/yolo26n.pt"
    assert cfg.model.imgsz == (640,)
    assert cfg.model.conf == 0.001
    assert cfg.model.max_det == 300
    assert cfg.decision.conf_threshold == 0.5
    assert (cfg.filter.n_on, cfg.filter.k_window, cfg.filter.m_off) == (3, 5, 5)
    assert cfg.noise_seed == 1234
    assert cfg.warmup == 10
    assert cfg.manifest is None


def test_nested_sections_and_conditions_are_parsed(tmp_path):
    data = {
        **MINIMAL,
        "model": {"imgsz": [320, 416]},
        "conditions": [
            {"name": "clean", "kind": "clean"},
            {"name": "blur_k7", "kind": "blur", "param": 7},
        ],
    }
    cfg = load_config(_write(tmp_path, data))
    assert cfg.model.imgsz == (320, 416)
    assert cfg.conditions == (
        Condition("clean", "clean", None),
        Condition("blur_k7", "blur", 7),
    )


def test_round_trip_through_dict(tmp_path):
    data = {**MINIMAL, "model": {"imgsz": [320]}, "warmup": 2}
    cfg = load_config(_write(tmp_path, data))
    again = load_config(_write(tmp_path, config_to_dict(cfg)))
    assert again == cfg


@pytest.mark.parametrize(
    "data",
    [
        {**MINIMAL, "typo_key": 1},
        {**MINIMAL, "model": {"imgsize": [320]}},
        {"image_dir": "x"},
    ],
)
def test_unknown_or_missing_keys_raise(tmp_path, data):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, data))


def test_duplicate_condition_names_raise(tmp_path):
    data = {
        **MINIMAL,
        "conditions": [
            {"name": "a", "kind": "clean"},
            {"name": "a", "kind": "blur", "param": 7},
        ],
    }
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, data))


def test_sequence_config_defaults_and_round_trip(tmp_path):
    from robust_visual_perception.config import (
        config_to_dict,
        load_sequence_config,
    )
    from robust_visual_perception.decision import CORRIDOR_ROI

    data = {"run_name": "s", "data_root": "data/raw/mot17", "sequences": ["MOT17-05"]}
    cfg = load_sequence_config(_write(tmp_path, data))
    assert cfg.model.imgsz == (416,)
    assert cfg.roi == CORRIDOR_ROI
    assert cfg.fault.missing_rate == 0.0
    assert cfg.sequences == ("MOT17-05",)
    again = load_sequence_config(_write(tmp_path, config_to_dict(cfg)))
    assert again == cfg


def test_sequence_config_rejects_several_sizes_and_bad_fault_key(tmp_path):
    from robust_visual_perception.config import load_sequence_config

    base = {"run_name": "s", "data_root": "d", "sequences": ["a"]}
    with pytest.raises(ValueError):
        load_sequence_config(_write(tmp_path, {**base, "model": {"imgsz": [320, 416]}}))
    with pytest.raises(ValueError):
        load_sequence_config(_write(tmp_path, {**base, "fault": {"drop": 0.1}}))
