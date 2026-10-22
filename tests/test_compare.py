import pytest

from robust_visual_perception.compare import compare_aggregates, is_timing_column

KEY = ["condition", "imgsz"]


def _row(cond, size, ap, latency):
    return {
        "condition": cond,
        "imgsz": size,
        "person_ap": ap,
        "latency_mean_ms": latency,
        "kind": "clean",
    }


def test_timing_columns_are_recognised():
    assert is_timing_column("latency_mean_ms")
    assert is_timing_column("throughput_ips")
    assert is_timing_column("degrade_mean_ms")
    assert not is_timing_column("person_ap")


def test_identical_results_match_and_timing_is_reported_separately():
    ref = [_row("clean", "320", "0.5", "14.0")]
    cand = [_row("clean", "320", "0.5", "28.0")]
    result = compare_aggregates(ref, cand, KEY, tolerance=0.01)
    assert result.matches
    assert result.max_abs_diff == pytest.approx(0.0)
    (timing,) = result.timing
    assert timing["column"] == "latency_mean_ms"
    assert timing["ratio"] == pytest.approx(2.0)


def test_accuracy_difference_above_tolerance_is_reported():
    ref = [_row("clean", "320", "0.50", "14")]
    cand = [_row("clean", "320", "0.47", "14")]
    result = compare_aggregates(ref, cand, KEY, tolerance=0.01)
    assert not result.matches
    (diff,) = result.differences
    assert diff["column"] == "person_ap"
    assert diff["abs_diff"] == pytest.approx(0.03)


def test_small_difference_within_tolerance_matches():
    ref = [_row("clean", "320", "0.500", "14")]
    cand = [_row("clean", "320", "0.505", "14")]
    assert compare_aggregates(ref, cand, KEY, tolerance=0.01).matches


def test_only_shared_rows_are_compared_and_missing_ones_listed():
    ref = [_row("clean", "320", "0.5", "14"), _row("blur_k7", "320", "0.4", "14")]
    cand = [_row("clean", "320", "0.5", "14")]
    result = compare_aggregates(ref, cand, KEY)
    assert result.n_compared == 1
    assert result.only_in_reference == [("blur_k7", "320")]
    assert result.matches


def test_nan_equals_nan():
    ref = [_row("clean", "320", "nan", "14")]
    cand = [_row("clean", "320", "nan", "14")]
    assert compare_aggregates(ref, cand, KEY).matches


def test_nothing_in_common_does_not_match():
    result = compare_aggregates(
        [_row("a", "1", "0.5", "1")], [_row("b", "1", "0.5", "1")], KEY
    )
    assert result.n_compared == 0
    assert not result.matches
