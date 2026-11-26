"""Unit tests of the Safety Block logic against the T2 rules (run: python tests/test_safety.py)."""
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from go1_solution import common as C  # noqa: E402
from go1_solution.safety_core import SafetyBlock, SelfCollisionChecker, PENETRATION_LIMIT  # noqa: E402

chk = SelfCollisionChecker(os.path.join(ROOT, 'models', 'unitree_go1'))


def cmd(q, kp=C.KP, kd=C.KD):
    return dict(q=np.array(q, float), dq=np.zeros(12), tau=np.zeros(12), kp=np.full(12, kp), kd=np.full(12, kd))


def new_block():
    b = SafetyBlock(chk)
    b.init_from_state(C.Q_DEFAULT)
    return b


def run(b, t0, t1, dt=C.SAFETY_DT):
    outs, t = [], t0
    while t < t1:
        outs.append(b.step(t))
        t += dt
    return outs, t


def check_stream(outs, prev):
    for q, dq, kp, kd, tau in outs:
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(kp)) and np.all(np.isfinite(kd)) and np.all(np.isfinite(tau))
        assert np.all(q >= C.Q_MIN - 1e-12) and np.all(q <= C.Q_MAX + 1e-12), 'joint limits'
        assert np.max(np.abs(q - prev)) < math.radians(5.0), 'step > 5 deg'
        assert chk.transition_ok(prev, q, limit=PENETRATION_LIMIT), 'self-collision in transition'
        prev = q
    return prev


# 1. safe command is followed (not just holding one pose)
b = new_block()
goal = C.Q_DEFAULT + np.array([0.1, 0.3, -0.3] * 4)
b.on_command(cmd(goal), 0.0)
outs, t = run(b, 0.0, 0.2)
prev = check_stream(outs, C.Q_DEFAULT)
assert np.allclose(prev, goal), 'safe goal must be reached'
print('1 safe command followed: OK')

# 2. big jump is ramped with < 5 deg per message
b.on_command(cmd(C.Q_DEFAULT - np.array([0.0, 0.6, -0.6] * 4)), t)
outs, t = run(b, t, t + 0.3)
prev = check_stream(outs, prev)
print('2 big jump ramped: OK')

# 3. out-of-range target is clipped
b.on_command(cmd([5.0, 9.0, 3.0] * 4), t)
outs, t = run(b, t, t + 0.25)
prev = check_stream(outs, prev)
print('3 out of range clipped: OK, final', np.round(prev[:3], 3))

# 4. NaN / Inf message rejected entirely; last accepted target is kept
b = new_block()
b.on_command(cmd(goal), 0.0)
outs, t = run(b, 0.0, 0.2)
prev = check_stream(outs, C.Q_DEFAULT)
for bad in (float('nan'), float('inf'), -float('inf')):
    q = goal.copy(); q[4] = bad
    assert not b.on_command(cmd(q), t)
    c = cmd(goal + 0.2); c['kd'][7] = bad
    assert not b.on_command(c, t)
outs, t = run(b, t, t + 0.1)
prev = check_stream(outs, prev)
assert np.allclose(prev, goal)
print('4 NaN/Inf rejected: OK')

# 5. self-colliding target (front legs crossing) is not executed
b = new_block()
bad = C.Q_DEFAULT.copy(); bad[0], bad[3] = 0.863, -0.863
assert chk.penetration(bad) > PENETRATION_LIMIT
b.on_command(cmd(bad), 0.0)
outs, t = run(b, 0.0, 0.5)
prev = check_stream(outs, C.Q_DEFAULT)
assert chk.penetration(prev) <= PENETRATION_LIMIT
print('5 self-collision prevented: OK, max penetration', chk.penetration(prev))

# 6. input loss: within 0.5 s hold the last ACCEPTED target, dq = 0, tau = 0, valid kp/kd, stream continues
b = new_block()
c = cmd(goal); c['tau'][:] = 1.0
b.on_command(c, 0.0)
outs, t = run(b, 0.0, 0.3)
outs, t = run(b, t, 1.5)          # no input any more
q, dq, kp, kd, tau = outs[-1]
assert np.allclose(q, goal) and np.all(dq == 0) and np.all(tau == 0) and np.all(kp > 0) and np.all(kd > 0)
print('6 input loss hold: OK')
print('ALL SAFETY TESTS PASSED')
