import numpy as np
import pytest

from robust_visual_perception.mot17 import gt_stop_states, parse_gt, parse_seqinfo
from robust_visual_perception.sim_export import (
    frame_rate,
    gt_boxes,
    read_frames_csv,
    stamp_gaps,
    write_sequence,
)
from robust_visual_perception.sim_geometry import (
    SIM_ROI,
    ActorPath,
    CameraModel,
    PersonVolume,
)

CAMERA = CameraModel(530.47, 530.47, 320.0, 240.0, 640, 480)
PERSON = PersonVolume(0.0, 1.8, 0.25)
# Person walks along x = 3 from y = -3 to y = 3 (6 s), turns, walks back.
PATH = ActorPath(((3.0, -3.0), (3.0, 3.0)), speed=1.0, turn_s=0.5)
# Robot parked at the origin, facing +x (camera offset 0: camera at z = 0.1).
POSE_STAMPS = np.array([0.0, 100.0])
POSES = np.array([[0, 0, 0.1, 0, 0, 0, 1], [0, 0, 0.1, 0, 0, 0, 1]], float)


def boxes_at(times):
    return gt_boxes(
        np.asarray(times), POSE_STAMPS, POSES, CAMERA, PATH, PERSON, (0, 0, 0)
    )


def test_gt_box_follows_the_person():
    # t = 3 s: person at (3, 0), straight ahead.
    (box,) = boxes_at([3.0])
    assert (box[0] + box[2]) / 2 == pytest.approx(320.0, abs=1e-6)
    # t = 0: person at (3, -3), 45 degrees to the right: out of view.
    assert boxes_at([0.0]) == [None]


def test_gt_uses_each_frame_stamp_so_a_dropped_frame_does_not_shift_gt():
    full = boxes_at([2.9, 2.933, 2.966, 3.0])
    dropped = boxes_at([2.9, 2.966, 3.0])
    assert np.array_equal(dropped[1], full[2])
    assert np.array_equal(dropped[2], full[3])


def test_frame_rate_and_gaps():
    stamps = np.array([0, 33, 67, 100, 167, 200]) * 1_000_000  # one frame missing
    fps = frame_rate(stamps)
    assert fps == pytest.approx(5 / 0.2)
    assert stamp_gaps(stamps) == 1


def test_written_sequence_reads_back_with_the_mot17_parsers(tmp_path):
    stamps = (np.array([2.9, 3.0, 3.1, 0.0]) * 1e9).astype(np.int64)
    boxes = boxes_at(stamps / 1e9)
    info = write_sequence(tmp_path / "static_r1", "static_r1", stamps, boxes, CAMERA)
    seq = tmp_path / "static_r1"
    parsed = parse_seqinfo((seq / "seqinfo.ini").read_text())
    assert parsed == info
    assert (parsed.name, parsed.length, parsed.width, parsed.height) == (
        "static_r1",
        4,
        640,
        480,
    )
    assert parsed.img_dir == "img1" and parsed.img_ext == ".png"
    gt = parse_gt((seq / "gt" / "gt.txt").read_text())
    assert gt[:, 0].tolist() == [1, 2, 3]  # frame 4 has no box
    states = gt_stop_states(gt, parsed, SIM_ROI)
    assert states.tolist() == [True, True, True, False]
    frames = read_frames_csv(seq / "frames.csv")
    assert frames["stamp_ns"].tolist() == stamps.tolist()
    assert frames["gt_stop"].tolist() == [1, 1, 1, 0]
    x, y, w, h = gt[0, 2:6]
    assert [x, y, x + w, y + h] == pytest.approx(boxes[0].tolist(), abs=0.01)


def test_sequence_name_is_kept(tmp_path):
    # The MOT17 parser keeps the part before a second hyphen; run names use
    # underscores, so they come back unchanged.
    stamps = np.array([0, 33_000_000])
    info = write_sequence(tmp_path / "s", "scripted_r2", stamps, [None, None], CAMERA)
    assert info.name == "scripted_r2"


def test_build_sequence_from_exported_raw_data(tmp_path):
    import json

    from robust_visual_perception.sim_export import build_sequence

    seq = tmp_path / "static_r1"
    seq.mkdir()
    (seq / "stamps.csv").write_text("frame,stamp_ns\n1,3000000000\n2,3033000000\n")
    (seq / "poses.csv").write_text(
        "stamp_s,x,y,z,qx,qy,qz,qw\n0.0,0,0,0.1,0,0,0,1\n100.0,0,0,0.1,0,0,0,1\n"
    )
    (seq / "camera.json").write_text(json.dumps(CAMERA.__dict__))
    info = build_sequence(seq, PATH, PERSON, camera_offset=(0, 0, 0))
    assert info.length == 2
    assert read_frames_csv(seq / "frames.csv")["gt_stop"].tolist() == [1, 1]
