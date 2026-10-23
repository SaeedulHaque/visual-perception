# ROS 2 Jazzy + Gazebo Harmonic + TurtleBot3 simulation, with the core
# perception package's Python dependencies in a venv that can also see the
# ROS system packages.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz \
        ros-jazzy-turtlebot3-gazebo \
        ros-jazzy-vision-msgs \
        ros-jazzy-teleop-twist-keyboard \
        ros-jazzy-rosbag2-storage-mcap \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The walking person comes from Gazebo Fuel; fetching it at build time means
# runs never depend on the network or on Fuel being up.
ENV GZ_FUEL_CACHE_PATH=/opt/gz-fuel
RUN mkdir -p /opt/gz-fuel \
    && . /opt/ros/jazzy/setup.sh \
    && gz fuel download -v 3 -u https://fuel.gazebosim.org/1.0/Mingfei/models/actor \
    && chmod -R a+rX /opt/gz-fuel

ENV VIRTUAL_ENV=/opt/rvp-venv
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
COPY docker/requirements.txt /tmp/requirements.txt
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PATH=$VIRTUAL_ENV/bin:$PATH

# ROS package with the nodes, launch file and world.
COPY ros2_ws/src /opt/rvp_ws/src
RUN . /opt/ros/jazzy/setup.sh \
    && cd /opt/rvp_ws \
    && colcon build --merge-install \
    && rm -rf build log

# The repository is mounted at /work; the core package is used from there,
# so the container always runs the checked-out code. colcon writes the node
# scripts with /usr/bin/python3 (same Python 3.12), so the venv's packages
# are put on the path explicitly.
ENV PYTHONPATH=/work/src:/opt/rvp-venv/lib/python3.12/site-packages \
    TURTLEBOT3_MODEL=waffle_pi \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    HOME=/tmp/home \
    ROS_LOG_DIR=/tmp/ros_logs \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib
COPY docker/entrypoint.sh /entrypoint.sh
WORKDIR /work
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
