"""Command validation and state machine, independent of ROS and MuJoCo."""
from dataclasses import dataclass
import math
import struct

LEGS = ("FR", "FL", "RR", "RL")
JOINTS = ("hip", "thigh", "calf")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in JOINTS)
JOINT_COUNT = len(JOINT_NAMES)
MOTOR_COUNT = 20
ACTIVE_MODE = 0x0A
LOWER = (-0.863, -0.686, -2.818) * 4
UPPER = (0.863, 4.501, -0.888) * 4
MAX_STEP = math.radians(5) - 1e-6
SAMPLE_STEP = math.radians(0.1)
WATCHDOG_NS = 450_000_000
FIELDS = ("q", "dq", "kp", "kd", "tau")


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def clamp_pose(angles):
    # Leave room for ROS float32 rounding at the inclusive joint limits.
    return tuple(float32(max(lo + 1e-6, min(hi - 1e-6, q)))
                 for q, lo, hi in zip(angles, LOWER, UPPER))


def limit_step(start, target):
    change = max(abs(b - a) for a, b in zip(start, target))
    if change <= MAX_STEP:
        return target
    factor = MAX_STEP / change
    return tuple(float32(a + (b - a) * factor) for a, b in zip(start, target))


def segment_samples(start, end):
    change = max(abs(b - a) for a, b in zip(start, end))
    steps = math.ceil(change / SAMPLE_STEP)
    yield start
    for index in range(1, steps):
        fraction = index / steps
        yield tuple(a + (b - a) * fraction for a, b in zip(start, end))
    if steps:
        yield end


@dataclass(frozen=True)
class Motor:
    q: float
    dq: float
    kp: float
    kd: float
    tau: float


def hold(pose):
    return tuple(Motor(angle, 0.0, 35.0, 1.5, 0.0) for angle in pose)


def read_command(motors):
    if len(motors) != MOTOR_COUNT:
        return None
    if any(not math.isfinite(getattr(motor, field)) for motor in motors for field in FIELDS):
        return None
    active = motors[:JOINT_COUNT]
    if any(motor.mode != ACTIVE_MODE or motor.kp < 0 or motor.kd < 0 for motor in active):
        return None
    return tuple(Motor(m.q, m.dq, m.kp, m.kd, m.tau) for m in active)


class Safety:
    def __init__(self, clear_pose, home):
        self.clear_pose = clear_pose
        self.home = clamp_pose(home)
        if not self.clear_pose(self.home):
            raise ValueError("MuJoCo home pose is in self-collision")
        self.pose = None
        self.output = None
        self.accepted_at = None

    def initialize(self, measured=None):
        # The first finite, collision-free measurement minimizes initial motion.
        self.pose = self.home
        if measured is not None and all(math.isfinite(q) for q in measured):
            measured = clamp_pose(measured)
            if self.clear_pose(measured):
                self.pose = measured
        self.output = hold(self.pose)

    def accept(self, request, sim_ns):
        candidate = limit_step(self.pose, clamp_pose(m.q for m in request))
        if not all(self.clear_pose(pose) for pose in segment_samples(self.pose, candidate)):
            return False
        self.pose = candidate
        self.output = tuple(
            Motor(q, max(-30.0, min(30.0, m.dq)), min(200.0, m.kp), min(20.0, m.kd),
                  max(-limit, min(limit, m.tau)))
            for q, m, limit in zip(candidate, request, (23.7, 23.7, 35.55) * 4)
        )
        self.accepted_at = sim_ns
        return True

    def tick(self, sim_ns):
        if self.accepted_at is not None and (
            sim_ns < self.accepted_at or sim_ns - self.accepted_at >= WATCHDOG_NS
        ):
            self.output = hold(self.pose)
            self.accepted_at = None
        return self.output
