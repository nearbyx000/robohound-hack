"""ROS 2 adapter for MuJoCo Lowlevel T2."""
from pathlib import Path
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock as ClockMessage
from unitree_go.msg import LowCmd, LowState

from collision import Collision
from safety import ACTIVE_MODE, JOINT_COUNT, Safety, read_command

MODEL_DIR = Path(__file__).resolve().parent / "model"
PERIOD_SECONDS = 0.008


class SafetyNode(Node):
    def __init__(self, collision):
        super().__init__("go1_safety_block")
        self.safety = Safety(collision.clear, collision.home)
        self.sim_ns = None
        self.pending = None
        self.measured = None
        self.startup_deadline = time.monotonic() + 0.2

        # The simulator requests a reliable command stream. Best-effort inputs
        # are compatible with both reliable and best-effort publishers.
        self.output = self.create_publisher(LowCmd, "/lowcmd", 1)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(ClockMessage, "/clock", self.on_clock, sensor_qos)
        self.create_subscription(LowState, "/lowstate", self.on_state, sensor_qos)
        self.create_subscription(LowCmd, "/lowcmd_bare", self.on_command, sensor_qos)
        # /clock carries simulated time for the watchdog, but its publication
        # may be slower than the required actuator stream.
        self.publish_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(PERIOD_SECONDS, self.publish_command, clock=self.publish_clock)

    def on_clock(self, message):
        stamp = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if self.sim_ns is not None and stamp < self.sim_ns:
            self.pending = None
            self.safety.tick(stamp)
        self.sim_ns = stamp

    def on_state(self, message):
        if self.safety.pose is None:
            self.measured = tuple(motor.q for motor in message.motor_state[:JOINT_COUNT])

    def on_command(self, message):
        self.pending = read_command(message.motor_cmd)

    def publish_command(self):
        if self.safety.pose is None:
            if self.measured is None and time.monotonic() < self.startup_deadline:
                return
            self.safety.initialize(self.measured)

        if self.sim_ns is not None and self.pending is not None:
            request, self.pending = self.pending, None
            self.safety.accept(request, self.sim_ns)

        motors = self.safety.tick(self.sim_ns)
        message = LowCmd()
        for slot, motor in zip(message.motor_cmd, motors):
            slot.mode = ACTIVE_MODE
            slot.q = motor.q
            slot.dq = motor.dq
            slot.kp = motor.kp
            slot.kd = motor.kd
            slot.tau = motor.tau
        self.output.publish(message)


def main():
    collision = Collision(MODEL_DIR)
    rclpy.init()
    node = None
    try:
        node = SafetyNode(collision)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
