import numpy as np
import pytest

from robust_visual_perception.decision import Roi
from robust_visual_perception.mot17 import (
    SequenceStats,
    find_events,
    gt_stop_states,
    parse_gt,
    parse_seqinfo,
    select_sequences,
)

SEQINFO = """[Sequence]
name=MOT17-05-FRCNN
imDir=img1
frameRate=14
seqLength=4
imWidth=200
imHeight=100
imExt=.jpg
"""


def test_parse_seqinfo():
    info = parse_seqinfo(SEQINFO)
    assert info.name == "MOT17-05"
    assert (info.fps, info.length, info.width, info.height) == (14.0, 4, 200, 100)
    assert (info.img_dir, info.img_ext) == ("img1", ".jpg")


def _gt_text(rows):
    return "\n".join(",".join(str(v) for v in r) for r in rows) + "\n"


# frame, id, x, y, w, h, considered, class, visibility
IN_PATH = (80, 40, 40, 50)  # centre (100, 65) = (0.5 W, 0.65 H), height 0.5 H


def test_gt_stop_states_apply_all_filters():
    rows = [
        (1, 1, *IN_PATH, 1, 1, 1.0),  # frame 1: counts
        (2, 1, *IN_PATH, 0, 1, 1.0),  # frame 2: not considered
        (3, 1, *IN_PATH, 1, 2, 1.0),  # frame 3: wrong class (not pedestrian)
        (4, 1, *IN_PATH, 1, 1, 0.4),  # frame 4: visibility below 0.5
    ]
    info = parse_seqinfo(SEQINFO)
    states = gt_stop_states(parse_gt(_gt_text(rows)), info)
    assert states.tolist() == [True, False, False, False]


def test_gt_stop_states_use_roi_geometry():
    rows = [
        (1, 1, 0, 40, 30, 50, 1, 1, 1.0),  # centre x 0.075 W: outside corridor
        (2, 1, 90, 60, 20, 20, 1, 1, 1.0),  # height 0.2 H: too short
        (3, 1, 80, 40, 40, 50, 1, 1, 1.0),  # inside
    ]
    info = parse_seqinfo(SEQINFO)
    states = gt_stop_states(parse_gt(_gt_text(rows)), info)
    assert states.tolist() == [False, False, True, False]


def test_gt_boxes_are_clipped_to_the_image_like_detections():
    # box extends below the image: clipped height is 40 -> centre y 0.8 H
    rows = [(1, 1, 80, 60, 40, 80, 1, 1, 1.0)]
    info = parse_seqinfo(SEQINFO)
    roi = Roi(0.3, 0.7, 0.3, 0.85, 0.25)
    states = gt_stop_states(parse_gt(_gt_text(rows)), info, roi=roi)
    assert states[0]


def test_find_events_merges_short_gaps():
    s = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], bool)
    assert find_events(s, merge_gap=3) == [(1, 6), (11, 13)]
    assert find_events(s, merge_gap=1) == [(1, 3), (5, 6), (11, 13)]


def test_find_events_empty():
    assert find_events(np.zeros(5, bool)) == []


def _stats(name, camera, occupancy, n_events):
    return SequenceStats(name, 30.0, 600, camera, occupancy, n_events)


def test_select_sequences_rule():
    stats = [
        _stats("MOT17-02", "static", 0.40, 6),
        _stats("MOT17-04", "static", 0.90, 9),  # occupancy too high
        _stats("MOT17-05", "moving", 0.30, 4),
        _stats("MOT17-09", "static", 0.50, 8),
        _stats("MOT17-10", "moving", 0.20, 7),
        _stats("MOT17-11", "moving", 0.25, 2),  # too few events
        _stats("MOT17-13", "moving", 0.60, 5),
    ]
    roles = select_sequences(stats)
    assert roles["MOT17-10"] == "evaluation"
    assert roles["MOT17-13"] == "evaluation"
    assert roles["MOT17-09"] == "tuning"
    assert roles["MOT17-04"] == "unused"
    assert sum(r == "evaluation" for r in roles.values()) == 2


def test_select_sequences_needs_three_qualifying():
    stats = [_stats("MOT17-05", "moving", 0.3, 4), _stats("MOT17-10", "moving", 0.3, 4)]
    with pytest.raises(ValueError):
        select_sequences(stats)


def _row(x_min, y_min, min_height, roles_ok, min_events):
    return {
        "roi": Roi(x_min, 1 - x_min, y_min, 1.0, min_height),
        "qualifies": roles_ok,
        "min_events": min_events,
    }


def test_choose_roi_prefers_design_roi_when_it_qualifies():
    from robust_visual_perception.mot17 import choose_roi

    rows = [_row(0.30, 0.30, 0.25, True, 3), _row(0.35, 0.30, 0.25, True, 9)]
    assert choose_roi(rows) == Roi(0.30, 0.70, 0.30, 1.0, 0.25)


def test_choose_roi_fewest_changes_then_smallest_change_then_events():
    from robust_visual_perception.mot17 import choose_roi

    rows = [
        _row(0.30, 0.30, 0.25, False, 0),  # design ROI, does not qualify
        _row(0.40, 0.50, 0.25, True, 3),  # 2 changes, total 0.30
        _row(0.35, 0.30, 0.50, True, 6),  # 2 changes, total 0.30, more events
        _row(0.30, 0.50, 0.30, True, 9),  # 2 changes, total 0.25 -> wins
        _row(0.35, 0.50, 0.50, True, 9),  # 3 changes
    ]
    assert choose_roi(rows) == Roi(0.30, 0.70, 0.50, 1.0, 0.30)
    rows.pop(3)
    assert choose_roi(rows) == Roi(0.35, 0.65, 0.30, 1.0, 0.50)


def test_choose_roi_without_qualifying_setting_raises():
    from robust_visual_perception.mot17 import choose_roi

    with pytest.raises(ValueError):
        choose_roi([_row(0.30, 0.30, 0.25, False, 0)])
