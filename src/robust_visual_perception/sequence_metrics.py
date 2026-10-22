"""Temporal metrics for a stream of safety states against GT-STOP.

Correctness (precision, recall, accuracy) and stability (transitions) are
reported side by side, so "stable but wrong" is visible. UNKNOWN counts as
STOP for correctness, because a consumer treats it as stop-with-reason, and
is also reported as a count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from robust_visual_perception.config import FilterConfig
from robust_visual_perception.decision import State
from robust_visual_perception.metrics import decision_metrics
from robust_visual_perception.mot17 import EVENT_MERGE_GAP, find_events
from robust_visual_perception.temporal_filter import PersistenceFilter

DEFAULT_FILTER = FilterConfig(3, 5, 5)
SWEEP_SETTINGS = tuple(
    FilterConfig(n_on, k, m_off)
    for n_on, k in ((2, 3), (3, 5), (4, 6), (5, 8))
    for m_off in (3, 5, 8)
)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def stream_metrics(
    states: Sequence[str],
    gt_stop: np.ndarray,
    fps: float,
    merge_gap: int = EVENT_MERGE_GAP,
) -> dict[str, Any]:
    states = list(states)
    n = len(states)
    gt = np.asarray(gt_stop, bool)
    stop = np.array([s != State.CLEAR.value for s in states], bool)
    decision = decision_metrics(stop, gt)

    transitions = sum(states[i] != states[i - 1] for i in range(1, n))
    minutes = n / fps / 60 if fps else math.nan

    events = find_events(gt, merge_gap)
    onsets, releases = [], []
    for k, (start, end) in enumerate(events):
        hits = np.flatnonzero(stop[start:end])
        if not len(hits):
            continue
        onsets.append(int(hits[0]))
        next_start = events[k + 1][0] if k + 1 < len(events) else n
        cleared = np.flatnonzero(~stop[end:next_start])
        if len(cleared):
            releases.append(int(cleared[0]))

    stop_runs = find_events(stop, merge_gap=1)
    false_episodes = sum(not gt[s:e].any() for s, e in stop_runs)

    return {
        "n_frames": n,
        "tp": decision["tp"],
        "fp": decision["fp"],
        "fn": decision["fn"],
        "tn": decision["tn"],
        "precision": decision["precision"],
        "recall": decision["recall"],
        "accuracy": decision["accuracy"],
        "transitions": transitions,
        "transitions_per_min": transitions / minutes if minutes else math.nan,
        "change_fraction": transitions / (n - 1) if n > 1 else math.nan,
        "n_events": len(events),
        "events_detected": len(onsets),
        "events_missed": len(events) - len(onsets),
        "onset_mean_frames": _mean(onsets),
        "onset_mean_s": _mean(onsets) / fps,
        "onset_median_frames": float(np.median(onsets)) if onsets else math.nan,
        "release_mean_frames": _mean(releases),
        "release_mean_s": _mean(releases) / fps,
        "release_n": len(releases),
        "false_stop_episodes": false_episodes,
        "false_stop_per_min": false_episodes / minutes if minutes else math.nan,
        "n_unknown": sum(s == State.UNKNOWN.value for s in states),
    }


def refilter(baseline: Sequence[str], setting: FilterConfig) -> list[str]:
    """Apply a filter setting to stored per-frame states. Identical to online
    filtering, because the filter only sees the per-frame state."""
    filt = PersistenceFilter(setting.n_on, setting.k_window, setting.m_off)
    return [filt.update(State(s)).value for s in baseline]


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("events_missed", "transitions_per_min", "onset_mean_s")
    no_worse = all(a[k] <= b[k] for k in keys)
    better = any(a[k] < b[k] for k in keys)
    return no_worse and better


def adopt_setting(
    summaries: list[dict[str, Any]], default: FilterConfig = DEFAULT_FILTER
) -> FilterConfig:
    """Keep ``default`` unless another setting dominates it (no more missed
    events, no more transitions per minute, no larger onset delay, strictly
    better on one). Several dominating settings: fewest transitions wins."""
    base = next(s for s in summaries if s["setting"] == default)
    dominating = [s for s in summaries if _dominates(s, base)]
    if not dominating:
        return default
    return min(dominating, key=lambda s: s["transitions_per_min"])["setting"]
