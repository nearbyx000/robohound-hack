#!/bin/bash
# Launch contract (T1): /entrypoint.sh ping | T2 | T3
#   ping -> exactly one JSON object on stdout, exit 0 (diagnostics only on stderr)
#   T2   -> Safety Block only (/lowcmd_bare -> /lowcmd), running until the container is stopped
#   T3   -> Safety Block + RL policy ROS 2 nodes, running until the container is stopped
set -u
MODE="${1:-ping}"

if [ "$MODE" = "ping" ]; then
    cat /opt/go1_solution/ping.json
    exit 0
fi

# ROS environment; keep ROS_DOMAIN_ID, RMW_IMPLEMENTATION and CYCLONEDDS_URI given by the launcher
set +u
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source /opt/unitree_ws/install/setup.bash >/dev/null 2>&1
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

case "$MODE" in
    T2)
        exec python3 -m go1_solution.safety_node --ros-args -p use_sim_time:=true 1>&2
        ;;
    T3)
        python3 -m go1_solution.safety_node --ros-args -p use_sim_time:=true 1>&2 &
        SAFETY=$!
        python3 -m go1_solution.policy_node --ros-args -p use_sim_time:=true 1>&2 &
        POLICY=$!
        trap 'kill -TERM $SAFETY $POLICY 2>/dev/null; wait' TERM INT
        wait -n $SAFETY $POLICY
        echo "a node exited, stopping" >&2
        kill -TERM $SAFETY $POLICY 2>/dev/null
        wait
        exit 1
        ;;
    *)
        echo "unsupported mode: $MODE (supported: ping, T2, T3)" >&2
        exit 2
        ;;
esac
