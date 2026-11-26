#!/usr/bin/env python3
"""Prints a compact table of the training log (progress.csv)."""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'training/runs/go1_walk_ppo/progress.csv'
rows = list(csv.DictReader(open(path)))
keys = ['time/total_timesteps', 'rollout/ep_rew_mean', 'rollout/ep_len_mean', 'episode_terms/forward_distance',
        'episode_terms/tracking_lin_vel', 'episode_terms/termination', 'train/std', 'time/fps']
print(' | '.join(k.split('/')[-1] for k in keys))
step = max(1, len(rows) // 15)
for r in rows[::step] + [rows[-1]]:
    print(' | '.join(f"{float(r[k]):.3g}" if r.get(k) else '-' for k in keys))
