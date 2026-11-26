#!/bin/bash
# Starts Simulator + Participant (T2) with the public launcher and runs tests/t2_demo_judge.py.
# Usage: tests/run_t2_demo.sh IMAGE [--gui]
set -u
IMAGE=${1:-go1-solution:dev}
GUIFLAG=${2:---headless}
SIM_IMAGE=${SIM_IMAGE:-registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0}
SIMDIR=${SIMDIR:-/root/robohack/go1-simulator-public}
HERE=$(cd "$(dirname "$0")" && pwd)
export DISPLAY=${DISPLAY:-:0} HOST_UID=1000 HOST_GID=1000
LOG=$(mktemp)
python3 "$SIMDIR/run_local.py" --task T2 $GUIFLAG --sim-image "$SIM_IMAGE" --participant-image "$IMAGE" > "$LOG" 2>&1 &
LP=$!
SIM=""
for _ in $(seq 1 240); do
    grep -q 'Pair started' "$LOG" && SIM=$(grep -o 'Simulator=[^ ]*' "$LOG" | cut -d= -f2) && break
    kill -0 $LP 2>/dev/null || break
    sleep 0.5
done
if [ -z "$SIM" ]; then echo "launcher failed"; cat "$LOG"; kill -INT $LP 2>/dev/null; wait $LP; exit 1; fi
sleep 2
docker cp "$HERE/t2_demo_judge.py" "$SIM:/tmp/t2_demo_judge.py" >/dev/null
docker exec "$SIM" bash -c "source /etc/go1_sim_env.sh; python3 /tmp/t2_demo_judge.py" 2>/dev/null | tail -1
sleep ${HOLD_AFTER:-3}
kill -INT $LP 2>/dev/null
wait $LP 2>/dev/null
rm -f "$LOG"
