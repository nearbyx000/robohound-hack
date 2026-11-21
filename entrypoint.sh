#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    printf 'Usage: %s {ping|T2}\n' "$0" >&2
    exit 64
fi

case "$1" in
    ping)
        printf '%s\n' '{"protocol_version":"1.0","supported_tasks":["T2"],"self_trained_rl":false,"preferred_speed_mps":null}'
        ;;
    T2)
        source /opt/ros/humble/setup.bash
        source /opt/go1/local_setup.bash
        exec python3 /opt/t2/safety_node.py
        ;;
    *)
        printf 'Unsupported mode: %s\n' "$1" >&2
        exit 64
        ;;
esac
