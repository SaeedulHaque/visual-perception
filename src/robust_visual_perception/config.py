"""Experiment configuration loaded from YAML.

Unknown keys are rejected so that a typo (``imgsize`` for ``imgsz``) fails
loudly instead of silently running with a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

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


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    _check_keys("config", data, ExperimentConfig)
    for required in ("run_name", "image_dir"):
        if required not in data:
            raise ValueError(f"config is missing required key {required!r}")
    data = dict(data)

    model = dict(data.pop("model", None) or {})
    _check_keys("model", model, ModelConfig)
    if "imgsz" in model:
        model["imgsz"] = tuple(int(s) for s in model["imgsz"])

    decision = dict(data.pop("decision", None) or {})
    _check_keys("decision", decision, DecisionConfig)
    filt = dict(data.pop("filter", None) or {})
    _check_keys("filter", filt, FilterConfig)

    conditions = data.pop("conditions", None)
    if conditions is not None:
        data["conditions"] = _parse_conditions(conditions)

    return ExperimentConfig(
        model=ModelConfig(**model),
        decision=DecisionConfig(**decision),
        filter=FilterConfig(**filt),
        **data,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Plain dict (lists, no tuples) that ``config_from_dict`` reads back."""
    return {
        "run_name": cfg.run_name,
        "image_dir": cfg.image_dir,
        "manifest": cfg.manifest,
        "output_root": cfg.output_root,
        "model": {
            "weights": cfg.model.weights,
            "imgsz": list(cfg.model.imgsz),
            "conf": cfg.model.conf,
            "max_det": cfg.model.max_det,
            "device": cfg.model.device,
        },
        "decision": {
            "target_class": cfg.decision.target_class,
            "conf_threshold": cfg.decision.conf_threshold,
        },
        "filter": {
            "n_on": cfg.filter.n_on,
            "k_window": cfg.filter.k_window,
            "m_off": cfg.filter.m_off,
        },
        "conditions": [
            {"name": c.name, "kind": c.kind, "param": c.param} for c in cfg.conditions
        ],
        "noise_seed": cfg.noise_seed,
        "warmup": cfg.warmup,
    }
