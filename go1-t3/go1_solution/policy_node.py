#!/usr/bin/env python3
"""ROS 2 RL policy node: /cmd_vel + /lowstate -> policy -> /lowcmd_bare (goes through the Safety Block).

The policy runs every 20 ms of simulation time (LowState.tick), exactly like in training.
"""
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from unitree_go.msg import LowCmd, LowState

from . import common as C
from .policy import Policy

POLICY_PATH = os.environ.get('GO1_POLICY_PATH', os.path.join(os.path.dirname(__file__), '..', 'policy', 'policy.npz'))
VX_RANGE = (0.0, 0.6)

SUB_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
PUB_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)


class PolicyNode(Node):
    def __init__(self):
        super().__init__('go1_rl_policy')
        self.policy = Policy(POLICY_PATH)
        self.hist = C.ObsHistory()
        self.cmd = np.zeros(3)
        self.last_action = np.zeros(C.N)
        self.next_tick_ms = None
        self.pub = self.create_publisher(LowCmd, '/lowcmd_bare', PUB_QOS)
        self.create_subscription(LowState, '/lowstate', self.on_state, SUB_QOS)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, SUB_QOS)
        self.get_logger().info(f'RL policy loaded from {POLICY_PATH}')

    def on_cmd_vel(self, msg):
        vx = float(np.clip(msg.linear.x, *VX_RANGE)) if np.isfinite(msg.linear.x) else 0.0
        vy = float(msg.linear.y) if np.isfinite(msg.linear.y) else 0.0
        wz = float(msg.angular.z) if np.isfinite(msg.angular.z) else 0.0
        self.cmd = np.array([vx, vy, wz])

    def on_state(self, msg):
        tick = int(msg.tick)
        if self.next_tick_ms is None:
            self.next_tick_ms = tick
        if tick < self.next_tick_ms:
            return
        self.next_tick_ms = max(self.next_tick_ms + int(C.POLICY_DT * 1000), tick)
        q = np.array([msg.motor_state[i].q for i in range(C.N)])
        dq = np.array([msg.motor_state[i].dq for i in range(C.N)])
        imu = msg.imu_state
        frame = C.build_frame(imu.gyroscope, imu.quaternion, self.cmd, q, dq, self.last_action)
        obs = self.hist.push(frame)
        action = self.policy(obs)
        self.last_action = action
        target = C.clip_to_limits(C.action_to_target(action))
        out = LowCmd()
        out.head = [0xFE, 0xEF]
        out.level_flag = 0xFF
        for i in range(C.N):
            m = out.motor_cmd[i]
            m.mode = C.MODE_SERVO
            m.q = float(target[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = C.KP
            m.kd = C.KD
        self.pub.publish(out)


def main():
    rclpy.init(args=sys.argv)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
