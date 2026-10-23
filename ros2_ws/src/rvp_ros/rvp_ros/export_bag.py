"""Extract the raw data of a recorded run from its bag.

  ros2 run rvp_ros export_bag --bag /work/data/raw/sim/bags/static_r1 \
      --out /work/data/raw/sim/sequences/static_r1

Writes PNG frames in image-stamp order (img1/000001.png ...), stamps.csv
(frame number and image stamp), poses.csv (true robot pose samples),
camera.json (intrinsics from camera_info), detections.csv (the node's
published detections, keyed by image stamp) and export.json. Ground truth is
computed afterwards without ROS (scripts/make_sim_gt.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from robust_visual_perception.ros_image import image_to_bgr
from robust_visual_perception.sim_export import frame_rate, stamp_gaps
from robust_visual_perception.sim_video import DetectionRow, write_detections_csv
from robust_visual_perception.sim_world import ROBOT_NAME

IMAGE_TOPIC = "/camera/image_raw"
INFO_TOPIC = "/camera/camera_info"
POSE_TOPIC = "/rvp/robot_pose"
DETECTIONS_TOPIC = "/perception/detections"


def _stamp_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def read_bag(bag: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    reader.set_filter(
        rosbag2_py.StorageFilter(
            topics=[IMAGE_TOPIC, INFO_TOPIC, POSE_TOPIC, DETECTIONS_TOPIC]
        )
    )
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, types[topic])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a simulation bag.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    img_dir = args.out / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[int, Path]] = []
    pose_rows: list[list[float]] = []
    detections: list[tuple[int, DetectionRow]] = []
    camera = None
    for topic, msg in read_bag(args.bag):
        if topic == IMAGE_TOPIC:
            stamp = _stamp_ns(msg.header)
            tmp = img_dir / f"{stamp}.png"
            bgr = image_to_bgr(msg.data, msg.width, msg.height, msg.step, msg.encoding)
            cv2.imwrite(str(tmp), bgr)
            images.append((stamp, tmp))
        elif topic == INFO_TOPIC and camera is None:
            k = msg.k
            camera = {
                "fx": float(k[0]),
                "fy": float(k[4]),
                "cx": float(k[2]),
                "cy": float(k[5]),
                "width": int(msg.width),
                "height": int(msg.height),
            }
        elif topic == POSE_TOPIC and msg.child_frame_id == ROBOT_NAME:
            t, r = msg.transform.translation, msg.transform.rotation
            pose_rows.append(
                [_stamp_ns(msg.header) / 1e9, t.x, t.y, t.z, r.x, r.y, r.z, r.w]
            )
        elif topic == DETECTIONS_TOPIC:
            stamp = _stamp_ns(msg.header)
            for d in msg.detections:
                cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
                hypothesis = d.results[0].hypothesis
                detections.append(
                    (
                        stamp,
                        DetectionRow(
                            hypothesis.class_id,
                            round(float(hypothesis.score), 4),
                            round(cx - d.bbox.size_x / 2, 2),
                            round(cy - d.bbox.size_y / 2, 2),
                            round(cx + d.bbox.size_x / 2, 2),
                            round(cy + d.bbox.size_y / 2, 2),
                        ),
                    )
                )
    if camera is None or not images or not pose_rows:
        raise SystemExit(
            f"bag is missing data: camera_info={camera is not None}, "
            f"images={len(images)}, robot poses={len(pose_rows)}"
        )

    # Frames in image-stamp order, renamed to MOT17 numbering.
    images.sort()
    stamps = np.array([s for s, _ in images], np.int64)
    for i, (_, tmp) in enumerate(images, 1):
        tmp.rename(img_dir / f"{i:06d}.png")
    poses = np.array(sorted(pose_rows))
    with (args.out / "stamps.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "stamp_ns"])
        writer.writerows([i, int(s)] for i, s in enumerate(stamps, 1))
    with (args.out / "poses.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stamp_s", "x", "y", "z", "qx", "qy", "qz", "qw"])
        writer.writerows(poses.tolist())
    (args.out / "camera.json").write_text(json.dumps(camera, indent=2))
    write_detections_csv(args.out / "detections.csv", sorted(detections))
    summary = {
        "frames": len(stamps),
        "fps": round(frame_rate(stamps), 3),
        "missing_frames": stamp_gaps(stamps),
        "first_stamp_s": stamps[0] / 1e9,
        "last_stamp_s": stamps[-1] / 1e9,
        "robot_pose_samples": len(poses),
        "detections": len(detections),
    }
    (args.out / "export.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
