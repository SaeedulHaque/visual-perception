from robust_visual_perception.plots import plot_severity, plot_sweep
from robust_visual_perception.summary import image_summary, sequence_summary

LEVELS = {"blur": (7, 15, 31), "low_light": (0.6, 0.45, 0.25), "noise": (10, 20, 35)}


def _image_rows(decreasing: bool = True):
    rows = []
    for size in (320, 640):
        rows.append(_row("clean", "clean", "", size, 0.5))
        for kind, levels in LEVELS.items():
            for i, level in enumerate(levels, start=1):
                ap = 0.5 - 0.1 * i if decreasing else 0.5 + 0.01 * i
                rows.append(_row(f"{kind}_{level}", kind, str(level), size, ap))
    return rows


def _row(cond, kind, param, size, ap):
    return {
        "condition": cond,
        "kind": kind,
        "param": param,
        "imgsz": str(size),
        "n_images": "20",
        "n_person_images": "14",
        "map": "0.4",
        "person_ap": f"{ap}",
        "person_precision": "0.9",
        "person_recall": f"{ap}",
        "retention": "1",
        "stop_recall": "0.8",
        "false_stop_rate": "0.01",
        "latency_n": "750",
        "latency_mean_ms": "10",
        "latency_median_ms": "9",
        "latency_p95_ms": "nan",
        "throughput_ips": "100",
    }


def test_image_summary_reports_monotonic_check():
    text = image_summary(_image_rows())
    assert "20 images (14 with people, 6 without)" in text
    assert "decreases monotonically with severity at 2 of 2 input sizes" in text
    assert "| clean | 0.400 | 0.500 |" in text
    assert "–" in text  # NaN p95 rendered as a dash, not "nan"
    text = image_summary(_image_rows(decreasing=False))
    assert "at 0 of 2 input sizes" in text


def _seq_row(seq, role, cond, method, tpm, onset, missed):
    return {
        "sequence": seq,
        "role": role,
        "condition": cond,
        "method": method,
        "precision": "0.9",
        "recall": "0.8",
        "transitions_per_min": str(tpm),
        "events_detected": str(3 - missed),
        "n_events": "3",
        "onset_mean_s": str(onset),
        "release_mean_s": "0.2",
        "events_missed": str(missed),
        "false_stop_episodes": "1",
        "n_unknown": "0",
    }


def test_sequence_summary_h3_check():
    rows = [
        _seq_row("MOT17-05", "evaluation", "clean", "baseline", 20, 0.1, 0),
        _seq_row("MOT17-05", "evaluation", "clean", "filtered", 5, 0.3, 1),
    ]
    sweep = [
        {
            "n_on": "3",
            "k_window": "5",
            "m_off": "5",
            "events_missed": "1",
            "transitions_per_min": "5",
            "onset_mean_s": "0.3",
        }
    ]
    adopted = {"adopted": {"n_on": 3, "k_window": 5, "m_off": 5}, "changed": False}
    text = sequence_summary(rows, sweep, adopted, rows, fault_rates=(0.05, 0.04))
    assert "Fault injection (5 % MISSING, 4 % ERROR" in text
    assert "fewer transitions/min in 1 of 1 conditions" in text
    assert "+0.20 to +0.20 s" in text
    assert "more missed events in 1 of 1 conditions" in text
    assert "(the default)" in text


def test_plots_write_files(tmp_path):
    rows = _image_rows()
    assert plot_severity(rows, "person_ap", "person AP", tmp_path / "a.png").exists()
    sweep = [
        {
            "n_on": "3",
            "k_window": "5",
            "m_off": "5",
            "onset_mean_s": "0.3",
            "transitions_per_min": "5",
        }
    ]
    assert plot_sweep(sweep, tmp_path / "s.png").exists()
