#!/bin/bash
# Runs N independent T3 episodes like the Judge: for every episode BOTH containers are created anew
# (public launcher run_local.py), then tests/local_judge.py drives /cmd_vel and checks the rules.
# Usage: tests/run_episodes.sh IMAGE [N] [SPEED]
set -u
IMAGE=${1:-go1-solution:dev}
N=${2:-10}
SPEED=${3:-0.3}
SIM_IMAGE=${SIM_IMAGE:-registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0}
SIMDIR=${SIMDIR:-/root/robohack/go1-simulator-public}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/../episode_results.jsonl}
: > "$OUT"
for i in $(seq 1 "$N"); do
    LOG=$(mktemp)
    python3 "$SIMDIR/run_local.py" --task T3 ${GUIFLAG:---headless} --sim-image "$SIM_IMAGE" --participant-image "$IMAGE" > "$LOG" 2>&1 &
    LP=$!
    SIM=""
    for _ in $(seq 1 240); do
        grep -q 'Pair started' "$LOG" && SIM=$(grep -o 'Simulator=[^ ]*' "$LOG" | cut -d= -f2) && break
        kill -0 $LP 2>/dev/null || break
        sleep 0.5
    done
    if [ -z "$SIM" ]; then echo "episode $i: launcher failed"; cat "$LOG"; kill -INT $LP 2>/dev/null; wait $LP; continue; fi
    docker cp "$HERE/local_judge.py" "$SIM:/tmp/local_judge.py" >/dev/null
    RES=$(docker exec "$SIM" bash -c "source /etc/go1_sim_env.sh; python3 /tmp/local_judge.py $SPEED" 2>/dev/null | tail -1)
    echo "episode $i: $RES"
    echo "$RES" >> "$OUT"
    kill -INT $LP 2>/dev/null
    wait $LP 2>/dev/null
    rm -f "$LOG"
done
python3 - "$OUT" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith('{')]
ok = sum(r['success'] for r in rows)
print(f'SUCCESS {ok}/{len(rows)}')
EOF
