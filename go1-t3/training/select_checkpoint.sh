#!/bin/bash
# Exports checkpoints and evaluates each on 20 domain-randomised episodes (T3 rules) -> selection.jsonl
# Usage: training/select_checkpoint.sh RUN_DIR STEP1 STEP2 ...
set -e
RUN=$1; shift
OUT=$RUN/selection.jsonl
: > "$OUT"
for S in "$@"; do
    CK=$RUN/checkpoints/ppo_${S}_steps.zip
    NPZ=$RUN/exported/policy_${S}.npz
    mkdir -p "$RUN/exported"
    python training/export_policy.py "$CK" "$NPZ" >/dev/null
    python training/eval_policy.py "$NPZ" --episodes 20 | python3 -c "
import json,sys
d=json.load(sys.stdin); e=d['episodes']
ts=[x['t'] for x in e if x['success']]
print(json.dumps(dict(step=$S, success=d['success'], mean_t=round(sum(ts)/max(1,len(ts)),2), max_lateral=max(abs(x.get('lateral',0)) for x in e))))" | tee -a "$OUT"
done
