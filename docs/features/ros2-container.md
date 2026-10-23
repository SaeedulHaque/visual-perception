# ROS 2 container

Files: `docker/Dockerfile`, `docker/compose.yaml`, `docker/requirements.txt`,
`docker/entrypoint.sh`, `docker/check_container.sh`, `.dockerignore`

## Why a container

The host runs Ubuntu 26.04. ROS 2 Jazzy ships binary packages only for
Ubuntu 24.04, so ROS 2, Gazebo Harmonic and the TurtleBot3 simulation live
in one Docker image based on `ros:jazzy-ros-base` (Ubuntu 24.04). The host
stays clean, and the core package still runs without ROS.

## What is in the image

| Layer | Contents |
|---|---|
| apt | `ros-jazzy-ros-gz` (Gazebo Harmonic and the bridge), `ros-jazzy-turtlebot3-gazebo` (binary, 2.3.7), `ros-jazzy-vision-msgs`, `ros-jazzy-teleop-twist-keyboard`, `ros-jazzy-rosbag2-storage-mcap` |
| Fuel cache | the walking person model (`Mingfei/actor`), fetched at build time into `/opt/gz-fuel`, so runs do not need the network |
| venv | `/opt/rvp-venv`, created with `--system-site-packages`: torch 2.14.1 CPU, ultralytics 8.4.183, OpenCV, pytest (pinned in `docker/requirements.txt`) |
| workspace | `ros2_ws/src/rvp_ros` built with colcon into `/opt/rvp_ws` |

The core package is **not** copied into the image. The repository is mounted
at `/work` and `PYTHONPATH` starts with `/work/src`, so the container always
runs the checked-out code. Only a change to `ros2_ws/` or `docker/` needs a
rebuild.

## NumPy 1.26 inside, NumPy 2 outside

The ROS Jazzy Python message packages (`sensor_msgs`, etc.) are compiled
against NumPy 1.x and fail to load under NumPy 2. The venv therefore keeps
the Ubuntu 24.04 system NumPy (1.26.4), which in turn limits OpenCV to a
NumPy-1-compatible release (4.11). The core package's pin is `numpy>=1.26`,
and `check_container.sh` runs the whole core test suite inside the container
to show that the code behaves the same on both.

The node also avoids `cv_bridge`: images are converted with plain numpy
([perception-node.md](perception-node.md)), so nothing depends on
cv_bridge's own NumPy and OpenCV build.

## Runtime settings (compose)

- The container runs as the host user (`HOST_UID`, `HOST_GID`), so files
  written to the mounted repository are not owned by root. `HOME`, the ROS
  log directory and the Ultralytics settings point to `/tmp`.
- `/dev/dri` is passed through with the `video` and `render` groups, so
  Gazebo can render on the Intel iGPU. Gazebo runs headless
  (`--headless-rendering`); the camera sensor needs no display.
  `RVP_SOFTWARE_RENDERING=1` switches to Mesa's software renderer if the
  GPU cannot be used.
- `/tmp/.X11-unix` and `DISPLAY` are passed in for the optional Gazebo
  window (`RVP_GUI=1`, teleop runs only). On a Wayland desktop it opens on
  XWayland. The host has to allow local processes of my user once per
  login: `xhost +SI:localuser:$(whoami)`. This does not open the display
  to other users.
- `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`: ROS discovery stays inside
  the container.
- colcon writes the node scripts with `/usr/bin/python3` (the same Python
  3.12 as the venv), so the venv's `site-packages` is on `PYTHONPATH` too.

## Build and check

```
docker compose -f docker/compose.yaml build
HOST_UID=$(id -u) HOST_GID=$(id -g) RENDER_GID=$(getent group render | cut -d: -f3) \
    docker compose -f docker/compose.yaml run --rm -T sim docker/check_container.sh
```

The check prints the package versions, runs the core tests and waits for
one 640 px camera frame from the inspection world. The first build is
slow (ROS, Gazebo and PyTorch downloads); later builds reuse those layers.
