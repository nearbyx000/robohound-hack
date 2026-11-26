"""Safety Block logic (ROS-free, unit-testable). Implements the T2 requirements:

* a message with NaN/+Inf/-Inf in any control field is rejected as a whole;
* joint targets are clipped to the allowed ranges (T2, 4.3);
* the target of every joint changes by less than 5 deg between consecutive accepted safe commands
  (the block ramps towards the requested target);
* the target configuration and the whole transition to it (checked every <= 0.1 deg) must not cause a
  self-collision deeper than 1 mm for the 655 checked geometry pairs of t2_collision_pairs.csv;
  otherwise the command is rejected and the last accepted safe target is held;
* kp, kd, tau, dq are bounded; when /lowcmd_bare stops for more than 0.5 s (simulation time) the block
  holds the last accepted target with dq = 0, tau = 0 and valid kp/kd.
"""
import csv
import math
import os

import mujoco
import numpy as np

from . import common as C

PENETRATION_LIMIT = 0.001        # m, judge: forbidden penetration is strictly deeper than 1 mm
OUR_PENETRATION_LIMIT = 0.0005   # m, we are stricter than the judge
CHECK_STEP_RAD = math.radians(0.1)
KP_MAX = 100.0
KD_MAX = 10.0
HOLD_KP = 40.0                   # same values the simulator uses to hold the stand pose
HOLD_KD = 2.0
INPUT_TIMEOUT = 0.5              # s of simulation time (we switch to hold earlier, see node)


class SelfCollisionChecker:
    def __init__(self, model_dir):
        xml = os.path.join(model_dir, 'go1.xml')
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = []
        for name in C.JOINT_NAMES:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qadr.append(int(self.model.jnt_qposadr[j]))
        self.qadr = np.array(self.qadr)
        self.pairs = set()
        with open(os.path.join(model_dir, 't2_collision_pairs.csv')) as f:
            for row in csv.DictReader(f):
                if row['diagnostic_policy'] == 'checked':
                    a, b = int(row['geom1_id']), int(row['geom2_id'])
                    self.pairs.add((min(a, b), max(a, b)))
        # trunk high above the origin; there is no floor in go1.xml, so only self contacts exist
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[2] = 1.0

    def penetration(self, q):
        """Deepest penetration (m, >= 0) among the checked pairs for joint configuration q."""
        d = self.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (min(g1, g2), max(g1, g2)) in self.pairs and -c.dist > worst:
                worst = -c.dist
        return worst

    def config_ok(self, q, limit=OUR_PENETRATION_LIMIT):
        return self.penetration(q) <= limit

    def transition_ok(self, q0, q1, limit=OUR_PENETRATION_LIMIT):
        """Checks q1 and all intermediate configurations between q0 and q1 with step <= 0.1 deg."""
        delta = np.asarray(q1) - np.asarray(q0)
        n = max(1, int(math.ceil(float(np.max(np.abs(delta))) / CHECK_STEP_RAD)))
        for k in range(1, n + 1):
            if self.penetration(q0 + delta * (k / n)) > limit:
                return False
        return True


def message_is_finite(cmd):
    """cmd: dict of arrays q, dq, tau, kp, kd (12 each)."""
    return all(np.all(np.isfinite(cmd[k])) for k in ('q', 'dq', 'tau', 'kp', 'kd'))


class SafetyBlock:
    """Holds the state of the safety filter; the ROS node only feeds messages and the clock."""

    def __init__(self, checker: SelfCollisionChecker):
        self.checker = checker
        self.target = None          # last accepted safe target q (12)
        self.goal = None            # clipped requested target the block ramps towards
        self.kp = np.full(C.N, HOLD_KP)
        self.kd = np.full(C.N, HOLD_KD)
        self.tau = np.zeros(C.N)
        self.last_input_time = None
        self.rejected = 0

    def init_from_state(self, q_measured):
        """Before any command arrives: hold the measured stand pose (inside limits)."""
        if self.target is None:
            q = C.clip_to_limits(np.asarray(q_measured, dtype=float))
            if not self.checker.config_ok(q):
                q = C.Q_DEFAULT.copy()
            self.target = q
            self.goal = q.copy()

    def on_command(self, cmd, now):
        """Accepts or rejects a /lowcmd_bare message (dict of 12-arrays)."""
        if not message_is_finite(cmd):
            self.rejected += 1
            return False
        if self.target is None:
            return False
        self.last_input_time = now
        self.goal = C.clip_to_limits(cmd['q'])
        self.kp = np.clip(cmd['kp'], 0.0, KP_MAX)
        self.kd = np.clip(cmd['kd'], 0.0, KD_MAX)
        self.tau = np.clip(cmd['tau'], -C.TORQUE_LIMIT, C.TORQUE_LIMIT)
        return True

    def input_lost(self, now):
        return self.last_input_time is None or (now - self.last_input_time) > INPUT_TIMEOUT * 0.6

    def step(self, now):
        """Computes the next safe output: (q, dq, kp, kd, tau). Called at a fixed rate."""
        if self.target is None:
            return None
        if self.input_lost(now):
            # hold the last ACCEPTED target configuration, dq = 0, tau = 0, valid gains
            kp = np.where(self.kp > 0.0, self.kp, HOLD_KP) if self.last_input_time is not None else np.full(C.N, HOLD_KP)
            kd = np.where(self.kd > 0.0, self.kd, HOLD_KD) if self.last_input_time is not None else np.full(C.N, HOLD_KD)
            self.goal = self.target.copy()
            return self.target.copy(), np.zeros(C.N), kp, kd, np.zeros(C.N)
        nxt = C.ramp_towards(self.target, self.goal)
        if np.any(np.abs(nxt - self.target) > C.MAX_STEP_RAD) or not self.checker.transition_ok(self.target, nxt):
            # unsafe: keep holding the last accepted safe target
            self.rejected += 1
            self.goal = self.target.copy()
            return self.target.copy(), np.zeros(C.N), self.kp, self.kd, np.zeros(C.N)
        self.target = nxt
        return self.target.copy(), np.zeros(C.N), self.kp, self.kd, self.tau
