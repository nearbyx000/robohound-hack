"""Smoke test: env speed, stand pose stability, safety checker on stand pose and a colliding pose."""
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
import yaml  # noqa: E402
from go1_env import Go1WalkEnv  # noqa: E402
from go1_solution import common as C  # noqa: E402
from go1_solution.safety_core import SelfCollisionChecker  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, 'training', 'config.yaml')))
env = Go1WalkEnv(cfg, seed=0)
obs, _ = env.reset()
print('obs', obs.shape)
t = time.time()
n = 500
for i in range(n):
    obs, r, term, trunc, info = env.step(np.zeros(12))
    if term:
        print('terminated at', i)
        break
dt = time.time() - t
print(f'steps/s single env: {n / dt:.0f}; height after zero-action stand {env.data.qpos[2]:.3f}')

chk = SelfCollisionChecker(os.path.join(ROOT, 'models', 'unitree_go1'))
print('checked pairs', len(chk.pairs))
print('penetration at Q_DEFAULT', chk.penetration(C.Q_DEFAULT))
bad = C.Q_DEFAULT.copy()
bad[0] = -0.8   # FR hip inward
bad[3] = 0.8    # FL hip inward -> legs cross
bad[1] = bad[4] = 0.0
print('penetration at crossed legs', chk.penetration(bad))
t = time.time()
for _ in range(1000):
    chk.penetration(C.Q_DEFAULT)
print(f'collision check: {(time.time() - t) * 1000:.3f} us/check')
