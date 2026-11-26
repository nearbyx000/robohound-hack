"""Shared, ROS-free definitions used by BOTH the training environment and the ROS 2 nodes.

Keeping one copy of the observation builder, action mapping and safety limits guarantees that the
policy sees exactly the same signals in training and in the simulator container.

Motor order (Unitree): FR, FL, RR, RL; for each leg hip, thigh, calf.
"""
import math

import numpy as np

N = 12
JOINT_NAMES = [f'{leg}_{kind}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for kind in ('hip', 'thigh', 'calf')]

# Joint target limits from the technical specification (T2, section 4.3), rad.
Q_MIN = np.array([-0.863, -0.686, -2.818] * 4)
Q_MAX = np.array([0.863, 4.501, -0.888] * 4)

# Go1 torque limits (same as the simulator: hip/thigh 23.7 N*m, calf 35.55 N*m).
TORQUE_LIMIT = np.array([23.7, 23.7, 35.55] * 4)

# Default ("home") pose of menagerie go1.xml keyframe 'home': the robot starts in it.
Q_DEFAULT = np.array([0.0, 0.9, -1.8] * 4)

MODE_SERVO = 0x0A

# ---- Control constants (identical in training and deployment) ----
SIM_DT = 0.002                     # MuJoCo timestep of go1.xml (default option)
POLICY_DT = 0.02                   # policy runs at 50 Hz of simulation time
SAFETY_DT = 0.004                  # Safety Block publishes /lowcmd at 250 Hz
TICKS_PER_POLICY = int(round(POLICY_DT / SAFETY_DT))      # 5
SUBSTEPS_PER_TICK = int(round(SAFETY_DT / SIM_DT))        # 2
MAX_STEP_RAD = math.radians(5.0)   # T2: max change of a target angle between consecutive safe commands
RAMP_STEP_RAD = math.radians(4.5)  # we use a little less than 5 deg to stay strictly inside the limit
KP = 30.0
KD = 0.8
ACTION_SCALE = 0.25
ACTION_CLIP = 4.0

# ---- Observation ----
OBS_SCALES = dict(ang_vel=0.25, dof_pos=1.0, dof_vel=0.05, cmd_lin=2.0, cmd_ang=0.25)
FRAME_DIM = 3 + 3 + 3 + N + N + N   # 45
HISTORY = 3
OBS_DIM = FRAME_DIM * HISTORY       # 135


def projected_gravity(quat_wxyz):
    """Gravity unit vector expressed in the body frame (from the IMU quaternion w, x, y, z)."""
    w, x, y, z = quat_wxyz
    # R^T * [0, 0, -1]
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.array([gx, gy, gz])


def build_frame(gyro, quat_wxyz, cmd, q, dq, last_action):
    """One observation frame (45 values)."""
    s = OBS_SCALES
    return np.concatenate([
        np.asarray(gyro, dtype=np.float64) * s['ang_vel'],
        projected_gravity(quat_wxyz),
        np.array([cmd[0] * s['cmd_lin'], cmd[1] * s['cmd_lin'], cmd[2] * s['cmd_ang']]),
        (np.asarray(q, dtype=np.float64) - Q_DEFAULT) * s['dof_pos'],
        np.asarray(dq, dtype=np.float64) * s['dof_vel'],
        np.asarray(last_action, dtype=np.float64),
    ])


class ObsHistory:
    """Stacks the last HISTORY frames, newest first."""

    def __init__(self):
        self.buf = None

    def reset(self, frame):
        self.buf = np.tile(frame, (HISTORY, 1))

    def push(self, frame):
        if self.buf is None:
            self.reset(frame)
        else:
            self.buf = np.roll(self.buf, 1, axis=0)
            self.buf[0] = frame
        return self.buf.reshape(-1).astype(np.float32)


def action_to_target(action):
    a = np.clip(np.asarray(action, dtype=np.float64), -ACTION_CLIP, ACTION_CLIP)
    return Q_DEFAULT + ACTION_SCALE * a


def pd_torque(q_des, q, dq, kp=KP, kd=KD, tau_ff=0.0, dq_des=0.0):
    """Same law as the simulator: tau = tau_ff + kp(q_des - q) + kd(dq_des - dq), clipped to motor limits."""
    tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


def clip_to_limits(q):
    return np.clip(q, Q_MIN, Q_MAX)


def ramp_towards(current, desired, step=RAMP_STEP_RAD):
    """Move every joint target towards `desired` by at most `step` rad."""
    return current + np.clip(desired - current, -step, step)
