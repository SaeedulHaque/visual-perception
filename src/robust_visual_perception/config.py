"""Experiment configuration loaded from YAML.

Unknown keys are rejected so that a typo (``imgsize`` for ``imgsz``) fails
loudly instead of silently running with a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from robust_visual_perception.decision import CORRIDOR_ROI, Roi
from robust_visual_perception.degradation import (
    DEFAULT_CONDITIONS,
    DEFAULT_NOISE_SEED,
    Condition,
)


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "models/yolo26n.pt"
    imgsz: tuple[int, ...] = (640,)
    conf: float = 0.001
    max_det: int = 300
    device: str = "cpu"


@dataclass(frozen=True)
class DecisionConfig:
    target_class: str = "person"
    conf_threshold: float = 0.5


@dataclass(frozen=True)
class FilterConfig:
    n_on: int = 3
    k_window: int = 5
    m_off: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    image_dir: str
    manifest: str | None = None
    output_root: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    conditions: tuple[Condition, ...] = DEFAULT_CONDITIONS
    noise_seed: int = DEFAULT_NOISE_SEED
    warmup: int = 10


SEQUENCE_IMGSZ = (416,)


@dataclass(frozen=True)
class FaultConfig:
    """Seeded fault injection: share of frames dropped (MISSING) or failed
    (ERROR). The same frames are affected under every condition."""

    missing_rate: float = 0.0
    error_rate: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class SequenceExperimentConfig:
    run_name: str
    data_root: str
    sequences: tuple[str, ...]
    output_root: str = "results"
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(imgsz=SEQUENCE_IMGSZ)
    )
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    conditions: tuple[Condition, ...] = DEFAULT_CONDITIONS
    roi: Roi = CORRIDOR_ROI
    fault: FaultConfig = field(default_factory=FaultConfig)
    noise_seed: int = DEFAULT_NOISE_SEED
    warmup: int = 10


def _check_keys(section: str, data: dict[str, Any], cls: type) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in {section}: {sorted(unknown)}")


def _parse_conditions(items: list[dict[str, Any]]) -> tuple[Condition, ...]:
    conditions = []
    for item in items:
        _check_keys("conditions", item, Condition)
        conditions.append(Condition(item["name"], item["kind"], item.get("param")))
    names = [c.name for c in conditions]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate condition names: {names}")
    return tuple(conditions)


def _common_sections(
    data: dict[str, Any], model_defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Parse the sections shared by image and sequence configs, in place."""
    model = {**(model_defaults or {}), **(data.pop("model", None) or {})}
    _check_keys("model", model, ModelConfig)
    if "imgsz" in model:
        model["imgsz"] = tuple(int(s) for s in model["imgsz"])
    decision = dict(data.pop("decision", None) or {})
    _check_keys("decision", decision, DecisionConfig)
    filt = dict(data.pop("filter", None) or {})
    _check_keys("filter", filt, FilterConfig)
    parsed = {
        "model": ModelConfig(**model),
        "decision": DecisionConfig(**decision),
        "filter": FilterConfig(**filt),
    }
    conditions = data.pop("conditions", None)
    if conditions is not None:
        parsed["conditions"] = _parse_conditions(conditions)
    return parsed


def _require(data: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if key not in data:
            raise ValueError(f"config is missing required key {key!r}")


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    _check_keys("config", data, ExperimentConfig)
    _require(data, "run_name", "image_dir")
    data = dict(data)
    return ExperimentConfig(**_common_sections(data), **data)


def sequence_config_from_dict(data: dict[str, Any]) -> SequenceExperimentConfig:
    _check_keys("config", data, SequenceExperimentConfig)
    _require(data, "run_name", "data_root", "sequences")
    data = dict(data)
    parsed = _common_sections(data, {"imgsz": SEQUENCE_IMGSZ})
    if len(parsed["model"].imgsz) != 1:
        raise ValueError("a sequence experiment uses exactly one imgsz")
    data["sequences"] = tuple(data["sequences"])
    if "roi" in data:
        roi = dict(data.pop("roi") or {})
        _check_keys("roi", roi, Roi)
        parsed["roi"] = Roi(**roi)
    if "fault" in data:
        fault = dict(data.pop("fault") or {})
        _check_keys("fault", fault, FaultConfig)
        parsed["fault"] = FaultConfig(**fault)
    return SequenceExperimentConfig(**parsed, **data)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> ExperimentConfig:
    return config_from_dict(_read_yaml(path))


def load_sequence_config(path: str | Path) -> SequenceExperimentConfig:
    return sequence_config_from_dict(_read_yaml(path))


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def config_to_dict(cfg: ExperimentConfig | SequenceExperimentConfig) -> dict[str, Any]:
    """Plain dict (lists, no tuples) that the matching loader reads back."""
    return _plain(asdict(cfg))
