#!/bin/bash
# Sanity checks inside the container:
#   docker compose -f docker/compose.yaml run --rm sim docker/check_container.sh
# 1. Python imports (ROS messages under NumPy 1.26, core package, ultralytics)
# 2. the core test suite
# 3. a headless camera frame from the inspection world
set -euo pipefail

echo "== imports"
python - <<'PY'
import numpy, cv2, torch, ultralytics, rclpy
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
import robust_visual_perception
print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| torch", torch.__version__,
      "| ultralytics", ultralytics.__version__)
PY

echo "== core tests"
python -m pytest -q -p no:cacheprovider /work/tests 2>&1 | tail -3

echo "== headless camera"
world=$(ros2 pkg prefix rvp_ros)/share/rvp_ros/worlds/inspection.sdf
export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models
gz sim -s -r --headless-rendering -v 1 "$world" > /tmp/gz_check.log 2>&1 &
gz_pid=$!
trap 'kill $gz_pid 2>/dev/null || true' EXIT
# Save one message first: grep -q closing the pipe early would count as a
# failure under pipefail.
timeout 60 gz topic -e -n 1 -t /camera/image_raw > /tmp/frame.txt || true
if grep -q "width: 640" /tmp/frame.txt; then
    echo "camera frame received (640 px wide)"
else
    echo "no camera frame within 60 s; see /tmp/gz_check.log" >&2
    exit 1
fi
