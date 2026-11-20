#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    printf 'Usage: %s ping\n' "$0" >&2
    exit 64
fi

case "$1" in
    ping)
        printf '%s\n' '{"protocol_version":"1.0","supported_tasks":[],"self_trained_rl":false,"preferred_speed_mps":null}'
        ;;
    T2|T3|T4|T5)
        printf 'Task %s is not implemented.\n' "$1" >&2
        exit 64
        ;;
    *)
        printf 'Unknown mode: %s\n' "$1" >&2
        exit 64
        ;;
esac
