"""Compare a re-run's aggregate CSV with a reference run.

Accuracy columns (AP, precision, recall, delays, counts) should agree: the
data, seeds and model are fixed. Exact equality is expected on the same
software and CPU family; other hardware or library builds can change float
results slightly and flip a few borderline detections, hence a tolerance.
Timing columns are hardware-dependent by nature and are reported side by
side, never counted as a mismatch: they are the device benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

TIMING_MARKERS = ("latency", "throughput", "degrade", "load_ms")


def is_timing_column(column: str) -> bool:
    return any(marker in column for marker in TIMING_MARKERS)


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Comparison:
    n_compared: int = 0
    max_abs_diff: float = 0.0
    differences: list[dict[str, Any]] = field(default_factory=list)
    timing: list[dict[str, Any]] = field(default_factory=list)
    only_in_reference: list[tuple[str, ...]] = field(default_factory=list)
    only_in_candidate: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def matches(self) -> bool:
        return self.n_compared > 0 and not self.differences


def compare_aggregates(
    reference: list[dict[str, str]],
    candidate: list[dict[str, str]],
    key_columns: list[str],
    tolerance: float = 0.01,
) -> Comparison:
    """Compare rows that share the same key. Numeric accuracy columns differ
    when ``|a - b| > tolerance`` (NaN equals NaN); text columns must be equal."""

    def keyed(rows):
        return {tuple(r[k] for k in key_columns): r for r in rows}

    ref, cand = keyed(reference), keyed(candidate)
    result = Comparison(
        only_in_reference=sorted(set(ref) - set(cand)),
        only_in_candidate=sorted(set(cand) - set(ref)),
    )
    for key in sorted(set(ref) & set(cand)):
        result.n_compared += 1
        a_row, b_row = ref[key], cand[key]
        for column in a_row:
            if column in key_columns or column not in b_row:
                continue
            a, b = _number(a_row[column]), _number(b_row[column])
            if is_timing_column(column):
                if a is not None and b is not None and not math.isnan(a):
                    ratio = b / a if a else math.nan
                    result.timing.append(
                        {
                            "key": key,
                            "column": column,
                            "reference": a,
                            "candidate": b,
                            "ratio": ratio,
                        }
                    )
                continue
            if a is None or b is None:
                if a_row[column] != b_row[column]:
                    result.differences.append(
                        {
                            "key": key,
                            "column": column,
                            "reference": a_row[column],
                            "candidate": b_row[column],
                            "abs_diff": math.nan,
                        }
                    )
                continue
            if math.isnan(a) and math.isnan(b):
                continue
            diff = abs(a - b) if not (math.isnan(a) or math.isnan(b)) else math.inf
            result.max_abs_diff = max(result.max_abs_diff, diff)
            if diff > tolerance:
                result.differences.append(
                    {
                        "key": key,
                        "column": column,
                        "reference": a,
                        "candidate": b,
                        "abs_diff": diff,
                    }
                )
    return result
