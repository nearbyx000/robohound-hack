#!/usr/bin/env python3
"""Evaluates an exported policy (policy.npz, NumPy inference = the deployed code path) in the training
environment with the T3 rules: 1.5 s warm-up with zero command, then the fixed forward command for at most
90 s; success = 3 m forward without a fall (height < 0.16 m or |roll|/|pitch| > 50 deg for > 0.5 s).
Usage: python training/eval_policy.py policy/policy.npz [--episodes 10] [--speed 0.3] [--no-dr]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from go1_env import Go1WalkEnv  # noqa: E402
from go1_solution import common as C  # noqa: E402
from go1_solution.policy import Policy  # noqa: E402


def run_episode(env, pol, speed, seed):
    env.reset(seed=seed)
    env.push_interval = 0
    env.cmd_resample = 10 ** 9
    env.cmd = np.zeros(3)
    warm = int(1.5 / C.POLICY_DT)
    obs = env.hist.push(env._frame())
    x0 = y0 = None
    viol = 0.0
    for k in range(warm + int(90.0 / C.POLICY_DT)):
        if k == warm:
            env.cmd = np.array([speed, 0.0, 0.0])
            x0, y0 = env.data.qpos[0], env.data.qpos[1]
            w, x, y, z = env.data.qpos[3:7]
            yaw0 = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        obs, _, _, _, _ = env.step(pol(obs))
        g = C.projected_gravity(env.data.qpos[3:7])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        bad = env.data.qpos[2] < 0.16 or tilt > 50.0
        viol = viol + C.POLICY_DT if bad else 0.0
        if viol > 0.5:
            return dict(success=False, fell=True, t=k * C.POLICY_DT)
        if k >= warm:
            dx, dy = env.data.qpos[0] - x0, env.data.qpos[1] - y0
            fwd = dx * math.cos(yaw0) + dy * math.sin(yaw0)
            if fwd >= 3.0:
                lat = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
                return dict(success=True, fell=False, t=round((k - warm) * C.POLICY_DT, 2), lateral=round(float(lat), 3))
    return dict(success=False, fell=False, t=90.0, forward=float(fwd))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('policy')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--speed', type=float, default=0.3)
    p.add_argument('--no-dr', action='store_true', help='disable domain randomisation (nominal simulator)')
    a = p.parse_args()
    cfg = yaml.safe_load(open(os.path.join(HERE, 'config.yaml')))
    if a.no_dr:
        cfg['domain_rand'] = {}
    env = Go1WalkEnv(cfg, seed=123)
    pol = Policy(a.policy)
    res = [run_episode(env, pol, a.speed, 1000 + i) for i in range(a.episodes)]
    ok = sum(r['success'] for r in res)
    print(json.dumps(dict(policy=a.policy, speed=a.speed, dr=not a.no_dr, success=f'{ok}/{a.episodes}', episodes=res)))


if __name__ == '__main__':
    main()
