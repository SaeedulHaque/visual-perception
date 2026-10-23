"""Perception node: camera image -> detections + safety state.

Thin adapter around the core package. Each frame goes through the same
``process_frame`` step as the offline experiments (clean condition), then
the temporal filter. The node never publishes velocity commands.

Publishes:
  /perception/detections    vision_msgs/Detection2DArray (image header kept)
  /perception/safety_state  std_msgs/String: CLEAR | STOP | UNKNOWN:<reason>
A watchdog on the steady clock publishes UNKNOWN:stale when no frame has
arrived for ``stale_timeout_s``. Every processed frame is logged to a CSV.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from robust_visual_perception.decision import Roi, State
from robust_visual_perception.degradation import Condition
from robust_visual_perception.detector import UltralyticsDetector
from robust_visual_perception.experiment_runner import process_frame, warm_up
from robust_visual_perception.image_source import Frame, FrameStatus
from robust_visual_perception.ros_image import image_to_bgr
from robust_visual_perception.temporal_filter import PersistenceFilter

CLEAN = Condition("clean", "clean", None)
LIVE_COLUMNS = [
    "stamp_ns",
    "stamp_s",
    "status",
    "n_detections",
    "n_target",
    "max_target_conf",
    "latency_ms",
    "callback_ms",
    "age_sim_ms",
    "baseline_state",
    "state_reason",
    "filtered_state",
]


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        p = self.declare_parameters(
            "",
            [
                ("weights", "/work/models/yolo26n.pt"),
                ("imgsz", 416),
                ("conf", 0.001),
                ("max_det", 300),
                ("target_class", "person"),
                ("conf_threshold", 0.5),
                ("publish_threshold", 0.25),
                ("roi", [0.30, 0.70, 0.0, 1.0, 0.25]),
                ("n_on", 2),
                ("k_window", 3),
                ("m_off", 3),
                ("image_topic", "/camera/image_raw"),
                ("stale_timeout_s", 0.5),
                ("warmup", 10),
                ("output_dir", ""),
            ],
        )
        cfg = {param.name: param.value for param in p}
        self.target_class = cfg["target_class"]
        self.conf_threshold = cfg["conf_threshold"]
        self.publish_threshold = cfg["publish_threshold"]
        self.roi = Roi(*cfg["roi"])
        self.filter = PersistenceFilter(cfg["n_on"], cfg["k_window"], cfg["m_off"])
        self.stale_timeout_s = cfg["stale_timeout_s"]

        self.detector = UltralyticsDetector(
            cfg["weights"], cfg["imgsz"], cfg["conf"], cfg["max_det"]
        )
        # Warm up before subscribing, so the first frames are not slow and
        # the run monitor can use the subscription as the "ready" signal.
        blank = np.zeros((480, 640, 3), np.uint8)
        warm_up({cfg["imgsz"]: self.detector}, blank, cfg["warmup"])

        self.out_dir = Path(cfg["output_dir"]) if cfg["output_dir"] else None
        self._csv_file = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._csv_file = (self.out_dir / "live_frames.csv").open("w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=LIVE_COLUMNS)
            self._csv.writeheader()
            (self.out_dir / "node_params.json").write_text(json.dumps(cfg, indent=2))

        self.pub_detections = self.create_publisher(
            Detection2DArray, "/perception/detections", 10
        )
        self.pub_state = self.create_publisher(String, "/perception/safety_state", 10)

        self.steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.last_frame_steady: float | None = None
        self.stale = False
        self.n_frames = 0
        self.n_stale_episodes = 0
        self.create_timer(0.1, self._watchdog, clock=self.steady)

        # Sensor-data QoS with depth 1: a frame that arrives while the
        # previous one is processed replaces the queued one (stale frames are
        # dropped, not queued).
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, cfg["image_topic"], self._on_image, qos)
        self.get_logger().info(
            f"ready: imgsz {cfg['imgsz']}, filter {cfg['n_on']}/{cfg['k_window']}/"
            f"{cfg['m_off']}, roi {cfg['roi']}"
        )

    def _steady_s(self) -> float:
        return self.steady.now().nanoseconds / 1e9

    def _on_image(self, msg: Image) -> None:
        start = time.perf_counter()
        self.last_frame_steady = self._steady_s()
        self.stale = False
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        # Age of the image when processing starts, in sim time: rendering,
        # bridge and queueing. The node's sim clock only advances between
        # callbacks (from /clock), so processing time is measured separately,
        # on the wall clock (callback_ms).
        age_ns = self.get_clock().now().nanoseconds - stamp_ns
        try:
            image = image_to_bgr(
                msg.data, msg.width, msg.height, msg.step, msg.encoding
            )
            frame = Frame(str(stamp_ns), image, stamp_ns / 1e9, FrameStatus.OK)
        except ValueError as exc:
            self.get_logger().error(f"bad image: {exc}")
            frame = Frame(
                str(stamp_ns), None, stamp_ns / 1e9, FrameStatus.MISSING, str(exc)
            )
        out = process_frame(
            frame,
            CLEAN,
            self.detector,
            target_class=self.target_class,
            conf_threshold=self.conf_threshold,
            noise_key=frame.frame_id,
            noise_seed=0,
            roi=self.roi,
        )
        filtered = self.filter.update(out.decision.state)

        if out.result is not None:
            self.pub_detections.publish(self._detections_msg(msg, out.result))
        reason = out.decision.reason
        text = filtered.value
        if filtered is State.UNKNOWN:
            text = f"UNKNOWN:{reason or 'no_reason'}"
        self.pub_state.publish(String(data=text))

        self.n_frames += 1
        if self._csv_file is not None:
            self._csv.writerow(
                {
                    "stamp_ns": stamp_ns,
                    "stamp_s": f"{stamp_ns / 1e9:.6f}",
                    "status": out.row.get("status", ""),
                    "n_detections": out.row.get("n_detections", ""),
                    "n_target": out.row.get("n_target", ""),
                    "max_target_conf": out.row.get("max_target_conf", ""),
                    "latency_ms": out.row.get("latency_ms", ""),
                    "callback_ms": f"{(time.perf_counter() - start) * 1e3:.3f}",
                    "age_sim_ms": f"{age_ns / 1e6:.3f}",
                    "baseline_state": out.decision.state.value,
                    "state_reason": reason or "",
                    "filtered_state": filtered.value,
                }
            )
            if self.n_frames % 30 == 0:
                self._csv_file.flush()

    def _detections_msg(self, image: Image, result) -> Detection2DArray:
        msg = Detection2DArray(header=image.header)
        for det in result.detections:
            if det.confidence < self.publish_threshold:
                continue
            d = Detection2D(header=image.header)
            d.bbox.center.position.x = (det.x1 + det.x2) / 2
            d.bbox.center.position.y = (det.y1 + det.y2) / 2
            d.bbox.size_x = det.x2 - det.x1
            d.bbox.size_y = det.y2 - det.y1
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det.class_name
            hypothesis.hypothesis.score = float(det.confidence)
            d.results.append(hypothesis)
            msg.detections.append(d)
        return msg

    def _watchdog(self) -> None:
        if self.last_frame_steady is None:
            # Nothing received yet (e.g. simulation still paused).
            self.pub_state.publish(String(data="UNKNOWN:stale"))
            return
        if self._steady_s() - self.last_frame_steady > self.stale_timeout_s:
            if not self.stale:
                self.n_stale_episodes += 1
            self.stale = True
            self.pub_state.publish(String(data="UNKNOWN:stale"))

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            summary = {
                "frames_processed": self.n_frames,
                "stale_episodes": self.n_stale_episodes,
            }
            (self.out_dir / "node_summary.json").write_text(
                json.dumps(summary, indent=2)
            )


def main() -> None:
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
