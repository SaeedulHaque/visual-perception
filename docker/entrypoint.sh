#!/bin/bash
# Source ROS and the project workspace, then run the given command.
set -e
mkdir -p "$HOME" "$ROS_LOG_DIR" "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
# Fallback when the GPU cannot be used for rendering (slower, CPU only).
if [ "${RVP_SOFTWARE_RENDERING:-0}" = "1" ]; then
    export LIBGL_ALWAYS_SOFTWARE=1
fi
source /opt/ros/jazzy/setup.bash
source /opt/rvp_ws/install/setup.bash
exec "$@"
