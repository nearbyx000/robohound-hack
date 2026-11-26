import cProfile
import os
import pstats
import sys
import time

import mujoco
import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'training'))
from go1_env import Go1WalkEnv  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, 'training', 'config.yaml')))
env = Go1WalkEnv(cfg, seed=0)
env.reset()
rng = np.random.default_rng(0)
t = time.time()
for _ in range(300):
    env.step(rng.normal(0, 0.5, 12))
print('env steps/s', 300 / (time.time() - t))
t = time.time()
for _ in range(3000):
    mujoco.mj_step(env.model, env.data)
print('mj_step us', (time.time() - t) / 3000 * 1e6)
pr = cProfile.Profile()
pr.enable()
for _ in range(300):
    o, r, te, tr, _ = env.step(rng.normal(0, 0.5, 12))
    if te or tr:
        env.reset()
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(12)
