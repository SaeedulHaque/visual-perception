import numpy as np
import pytest

from robust_visual_perception.sim_geometry import SIM_ROI
from robust_visual_perception.sim_video import (
    BANNER_H,
    COLORS,
    DetectionRow,
    draw_overlay,
    read_detections_csv,
    read_gt_boxes,
    write_detections_csv,
)

H, W = 480, 640


def overlay(**kwargs):
    args = {
        "image": np.zeros((H, W, 3), np.uint8),
        "t_s": 1.0,
        "gt_box": None,
        "detections": [],
        "roi": SIM_ROI,
        "target_class": "person",
        "conf_threshold": 0.5,
        "baseline": "CLEAR",
        "filtered": "CLEAR",
        "gt_stop": False,
        "processed": True,
    }
    args.update(kwargs)
    return draw_overlay(**args)


def cell_color(canvas, index):
    """Colour in the middle of banner cell ``index`` (0..3), left edge."""
    return tuple(canvas[H + BANNER_H // 2, index * W // 4 + 3])


def test_canvas_adds_banner_and_leaves_input_unchanged():
    image = np.full((H, W, 3), 7, np.uint8)
    out = overlay(image=image)
    assert out.shape == (H + BANNER_H, W, 3)
    assert (image == 7).all()


def test_roi_band_lines():
    out = overlay()
    assert tuple(out[10, round(0.30 * W)]) == COLORS["roi"]
    assert tuple(out[10, round(0.70 * W)]) == COLORS["roi"]


def test_gt_box_drawn():
    out = overlay(gt_box=(100, 50, 200, 300))
    assert tuple(out[175, 100]) == COLORS["gt"]


def test_target_detection_colour_depends_on_threshold():
    strong = DetectionRow("person", 0.9, 400, 50, 450, 300)
    weak = DetectionRow("person", 0.3, 400, 50, 450, 300)
    other = DetectionRow("chair", 0.9, 400, 50, 450, 300)
    assert tuple(overlay(detections=[strong])[175, 400]) == COLORS["target"]
    assert tuple(overlay(detections=[weak])[175, 400]) == COLORS["weak"]
    assert tuple(overlay(detections=[other])[175, 400]) == COLORS["weak"]


@pytest.mark.parametrize(
    ("baseline", "filtered", "gt_stop", "expected"),
    [
        ("STOP", "CLEAR", True, ("STOP", "CLEAR", "STOP")),
        ("CLEAR", "STOP", False, ("CLEAR", "STOP", "CLEAR")),
        ("UNKNOWN:stale", "UNKNOWN:stale", False, ("UNKNOWN", "UNKNOWN", "CLEAR")),
    ],
)
def test_banner_cells_follow_states(baseline, filtered, gt_stop, expected):
    out = overlay(baseline=baseline, filtered=filtered, gt_stop=gt_stop)
    for index, state in enumerate(expected, 1):
        assert cell_color(out, index) == COLORS[state]


def test_unprocessed_frame_greys_out_node_cells():
    out = overlay(baseline="STOP", filtered="STOP", processed=False)
    assert cell_color(out, 1) == COLORS["held"]
    assert cell_color(out, 2) == COLORS["held"]


def test_detections_csv_round_trip(tmp_path):
    rows = {
        5: [DetectionRow("person", 0.75, 1.0, 2.0, 3.0, 4.0)],
        9: [
            DetectionRow("person", 0.5, 10.0, 20.0, 30.0, 40.0),
            DetectionRow("chair", 0.25, 0.0, 0.0, 1.0, 1.0),
        ],
    }
    path = tmp_path / "detections.csv"
    write_detections_csv(path, [(s, d) for s, ds in rows.items() for d in ds])
    assert read_detections_csv(path) == rows


def test_gt_boxes_from_frames_csv(tmp_path):
    path = tmp_path / "frames.csv"
    path.write_text(
        "frame,stamp_ns,x1,y1,x2,y2,gt_stop\n1,0,,,,,0\n2,33000000,10.5,20,30,40,1\n"
    )
    assert read_gt_boxes(path) == {0: None, 33000000: (10.5, 20.0, 30.0, 40.0)}
