#!/bin/bash
# One simulation run inside the container: launch, then extract the bag.
#   docker/sim_run.sh <mode> <run_name> [duration_s] [seed]
# mode: static | scripted | teleop. RVP_GUI=1 opens the Gazebo window
# (teleop only; the measured runs stay headless). Results go to
# results/sim/<run_name>/, the bag to data/raw/sim/bags/<run_name>/, the
# extracted frames to data/raw/sim/sequences/<run_name>/.
set -euo pipefail
mode=$1
run_name=$2
duration=${3:-120.0}
seed=${4:-7}
bag=/work/data/raw/sim/bags/$run_name
results=/work/results/sim/$run_name
if [ -e "$bag" ] || [ -e "$results" ]; then
    echo "run $run_name exists ($bag or $results); choose another name" >&2
    exit 1
fi

gui=false
if [ "${RVP_GUI:-0}" = "1" ]; then
    if [ "$mode" != teleop ]; then
        echo "RVP_GUI=1 is for teleop runs only" >&2
        exit 1
    fi
    gui=true
fi

launch=(ros2 launch rvp_ros sim.launch.py mode:="$mode" run_name:="$run_name"
        duration:="$duration" seed:="$seed" gui:="$gui")
if [ "$mode" = teleop ]; then
    # The simulation runs in the background; the keyboard teleop needs this
    # terminal. It publishes to /rvp/cmd_vel, so the command guard applies.
    # The log goes to the mounted repository, so it outlives the container.
    log_dir=/work/data/raw/sim/logs
    mkdir -p "$log_dir"
    "${launch[@]}" > "$log_dir/launch_$run_name.log" 2>&1 &
    launch_pid=$!
    # End the teleop when the run monitor ends the launch.
    (wait "$launch_pid" 2>/dev/null; pkill -f teleop_twist_keyboard) &
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
        -p stamped:=true -p speed:=0.1 -p turn:=0.4 -r cmd_vel:=/rvp/cmd_vel || true
    wait "$launch_pid" || true
else
    "${launch[@]}"
fi

if [ ! -f "$results/run_info.json" ] || ! grep -q '"completed": true' "$results/run_info.json"; then
    if [ "$mode" = teleop ]; then
        echo "run $run_name did not complete; see $log_dir/launch_$run_name.log" >&2
    else
        echo "run $run_name did not complete; see the launch output" >&2
    fi
    exit 1
fi
ros2 run rvp_ros export_bag --bag "$bag" --out "/work/data/raw/sim/sequences/$run_name"
