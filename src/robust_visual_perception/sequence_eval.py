"""Evaluate a finished sequence run: per (sequence, condition, method) metrics,
the filter sweep on the tuning sequence, and the adopted filter setting."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robust_visual_perception.config import FilterConfig, load_sequence_config
from robust_visual_perception.metrics import latency_stats
from robust_visual_perception.mot17 import parse_seqinfo
from robust_visual_perception.sequence_metrics import (
    DEFAULT_FILTER,
    SWEEP_SETTINGS,
    adopt_setting,
    refilter,
    stream_metrics,
)

METRIC_COLUMNS = [
    "n_frames",
    "precision",
    "recall",
    "accuracy",
    "transitions",
    "transitions_per_min",
    "change_fraction",
    "n_events",
    "events_detected",
    "events_missed",
    "onset_mean_frames",
    "onset_mean_s",
    "onset_median_frames",
    "release_mean_frames",
    "release_mean_s",
    "release_n",
    "false_stop_episodes",
    "false_stop_per_min",
    "n_unknown",
    "tp",
    "fp",
    "fn",
    "tn",
]
KEY_COLUMNS = ["sequence", "role", "condition", "kind", "param", "method"]
FILTER_COLUMNS = ["n_on", "k_window", "m_off"]
AGGREGATE_COLUMNS = [
    *KEY_COLUMNS,
    *FILTER_COLUMNS,
    *METRIC_COLUMNS,
    "latency_mean_ms",
    "latency_p95_ms",
]
SWEEP_COLUMNS = [*KEY_COLUMNS, *FILTER_COLUMNS, *METRIC_COLUMNS]


@dataclass(frozen=True)
class SequenceEvaluation:
    aggregate: list[dict[str, Any]]
    sweep: list[dict[str, Any]]
    sweep_summary: list[dict[str, Any]]
    run_filter: FilterConfig
    adopted: FilterConfig


def _read_streams(run_dir: Path) -> dict[tuple[str, str], dict[str, list[str]]]:
    streams: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with (run_dir / "per_frame.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            stream = streams[(row["sequence"], row["condition"])]
            for column in (
                "baseline_state",
                "filtered_state",
                "gt_state",
                "latency_ms",
            ):
                stream[column].append(row[column])
    return streams


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _filter_fields(setting: FilterConfig | None) -> dict[str, Any]:
    if setting is None:
        return {"n_on": "", "k_window": "", "m_off": ""}
    return {"n_on": setting.n_on, "k_window": setting.k_window, "m_off": setting.m_off}


def evaluate_sequence_run(
    run_dir: Path,
    roles: dict[str, str],
    adopted_override: FilterConfig | None = None,
) -> SequenceEvaluation:
    """``adopted_override`` applies a setting adopted on another run (the
    main run's sweep) to a run without a tuning sequence, e.g. the fault run."""
    run_dir = Path(run_dir)
    cfg = load_sequence_config(run_dir / "config.yaml")
    streams = _read_streams(run_dir)
    fps = {
        name: parse_seqinfo(
            (Path(cfg.data_root) / name / "seqinfo.ini").read_text()
        ).fps
        for name in cfg.sequences
    }
    conditions = {c.name: c for c in cfg.conditions}

    def key_fields(sequence: str, condition: str, method: str) -> dict[str, Any]:
        c = conditions[condition]
        return {
            "sequence": sequence,
            "role": roles.get(sequence, ""),
            "condition": condition,
            "kind": c.kind,
            "param": "" if c.param is None else c.param,
            "method": method,
        }

    # Filter sweep on the tuning sequence only (design 6.3).
    sweep, summaries = [], []
    tuning = [s for s in cfg.sequences if roles.get(s) == "tuning"]
    for setting in SWEEP_SETTINGS if tuning else ():
        per_setting = []
        for sequence in tuning:
            for condition in conditions:
                stream = streams[(sequence, condition)]
                gt = np.array([g == "STOP" for g in stream["gt_state"]])
                states = refilter(stream["baseline_state"], setting)
                m = stream_metrics(states, gt, fps[sequence])
                per_setting.append(m)
                sweep.append(
                    {
                        **key_fields(sequence, condition, "sweep"),
                        **_filter_fields(setting),
                        **{k: m[k] for k in METRIC_COLUMNS},
                    }
                )
        summaries.append(
            {
                "setting": setting,
                **{
                    k: _nanmean([m[k] for m in per_setting])
                    for k in ("events_missed", "transitions_per_min", "onset_mean_s")
                },
            }
        )
    if summaries:
        adopted = adopt_setting(summaries, DEFAULT_FILTER)
    else:
        adopted = adopted_override or cfg.filter

    aggregate = []
    for (sequence, condition), stream in streams.items():
        gt = np.array([g == "STOP" for g in stream["gt_state"]])
        latency = latency_stats([float(v) for v in stream["latency_ms"] if v])
        methods: list[tuple[str, list[str], FilterConfig | None]] = [
            ("baseline", stream["baseline_state"], None),
            ("filtered", stream["filtered_state"], cfg.filter),
        ]
        if adopted != cfg.filter:
            methods.append(
                (
                    "filtered_adopted",
                    refilter(stream["baseline_state"], adopted),
                    adopted,
                )
            )
        for method, states, setting in methods:
            m = stream_metrics(states, gt, fps[sequence])
            aggregate.append(
                {
                    **key_fields(sequence, condition, method),
                    **_filter_fields(setting),
                    **{k: m[k] for k in METRIC_COLUMNS},
                    "latency_mean_ms": latency["mean_ms"],
                    "latency_p95_ms": latency["p95_ms"],
                }
            )
    return SequenceEvaluation(aggregate, sweep, summaries, cfg.filter, adopted)


def read_roles(manifest: Path) -> dict[str, str]:
    with Path(manifest).open(newline="") as f:
        return {row["name"]: row["role"] for row in csv.DictReader(f)}
