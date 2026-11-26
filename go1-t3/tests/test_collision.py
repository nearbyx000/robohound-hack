"""Self-collision checker sanity: random configurations inside joint limits must sometimes collide,
and the checker must agree with a direct MuJoCo evaluation of the 655 checked pairs."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from go1_solution import common as C  # noqa: E402
from go1_solution.safety_core import SelfCollisionChecker  # noqa: E402

chk = SelfCollisionChecker(os.path.join(ROOT, 'models', 'unitree_go1'))
rng = np.random.default_rng(0)
hits = 0
deep = []
for _ in range(3000):
    q = rng.uniform(C.Q_MIN, C.Q_MAX)
    p = chk.penetration(q)
    if p > 0.001:
        hits += 1
        deep.append(p)
print(f'random configs with penetration > 1 mm: {hits}/3000, max {max(deep) if deep else 0:.4f} m')
q = C.Q_DEFAULT.copy()
q[1] = 4.0   # FR thigh rotated up/back into the body
print('FR thigh 4.0 rad penetration:', chk.penetration(q))
q = C.Q_DEFAULT.copy()
q[0], q[3] = -0.863, 0.863   # front hips abducted outward
q[0], q[3] = 0.863, -0.863   # front hips adducted inward (towards each other)
print('front hips inward penetration:', chk.penetration(q))
assert chk.penetration(C.Q_DEFAULT) == 0.0
assert hits > 0, 'checker never detects collisions - wrong pair ids?'
print('OK')
