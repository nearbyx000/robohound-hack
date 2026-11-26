#!/bin/bash
set -e

case "${1:-ping}" in
    ping)
        printf '%s\n' '{"protocol_version":"1.0","supported_tasks":["T4"],"self_trained_rl":false,"preferred_speed_mps":null}'
        ;;
    T4)
        source /opt/ros/humble/setup.bash
        exec python3 /opt/go1-t4/node.py 1>&2
        ;;
    *)
        printf 'unsupported mode: %s\n' "$1" >&2
        exit 64
        ;;
esac
