#!/usr/bin/env python3
"""ROS 2 Safety Block: /lowcmd_bare (input) -> checks -> /lowcmd (safe output to the actuators).

Output rate: one /lowcmd every 2nd /lowstate message (the simulator publishes /lowstate at 500 Hz of
simulation time) = 250 Hz of simulation time; a wall-clock fallback timer keeps the stream alive if
/lowstate stalls. Simulation time is taken from LowState.tick (ms of simulation time).
"""
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from unitree_go.msg import LowCmd, LowState

from . import common as C
from .safety_core import SafetyBlock, SelfCollisionChecker

MODEL_DIR = os.environ.get('GO1_SOLUTION_MODEL_DIR', os.path.join(os.path.dirname(__file__), '..', 'models', 'unitree_go1'))

SUB_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
PUB_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)


def read_cmd(msg):
    m = msg.motor_cmd[:C.N]
    return {k: np.array([getattr(c, k) for c in m], dtype=np.float64) for k in ('q', 'dq', 'tau', 'kp', 'kd')}


class SafetyNode(Node):
    def __init__(self):
        super().__init__('go1_safety_block')
        self.block = SafetyBlock(SelfCollisionChecker(MODEL_DIR))
        self.sim_time = None            # s, from LowState.tick
        self.state_count = 0
        self.last_pub_wall = time.monotonic()
        self.pub = self.create_publisher(LowCmd, '/lowcmd', PUB_QOS)
        self.create_subscription(LowState, '/lowstate', self.on_state, SUB_QOS)
        self.create_subscription(LowCmd, '/lowcmd_bare', self.on_cmd, SUB_QOS)
        self.create_timer(0.01, self.on_fallback_timer)
        self.get_logger().info(f'Safety Block ready, model {MODEL_DIR}, {len(self.block.checker.pairs)} checked pairs')

    def now(self):
        if self.sim_time is not None:
            return self.sim_time
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg):
        self.sim_time = msg.tick * 1e-3
        if self.block.target is None:
            self.block.init_from_state([msg.motor_state[i].q for i in range(C.N)])
        self.state_count += 1
        if self.state_count % 2 == 0:
            self.publish_safe()

    def on_cmd(self, msg):
        if not self.block.on_command(read_cmd(msg), self.now()):
            if self.block.target is not None:
                self.get_logger().warn('rejected /lowcmd_bare message (NaN/Inf)', throttle_duration_sec=1.0)

    def on_fallback_timer(self):
        # keep /lowcmd alive if /lowstate stalls (never faster than the normal stream)
        if time.monotonic() - self.last_pub_wall > 0.008:
            self.publish_safe()

    def publish_safe(self):
        out = self.block.step(self.now())
        if out is None:
            return
        q, dq, kp, kd, tau = out
        if not all(np.all(np.isfinite(v)) for v in out):   # never let NaN/Inf reach /lowcmd
            return
        msg = LowCmd()
        msg.head = [0xFE, 0xEF]
        msg.level_flag = 0xFF
        for i in range(C.N):
            m = msg.motor_cmd[i]
            m.mode = C.MODE_SERVO
            m.q = float(q[i])
            m.dq = float(dq[i])
            m.kp = float(kp[i])
            m.kd = float(kd[i])
            m.tau = float(tau[i])
        self.pub.publish(msg)
        self.last_pub_wall = time.monotonic()


def main():
    rclpy.init(args=sys.argv)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
