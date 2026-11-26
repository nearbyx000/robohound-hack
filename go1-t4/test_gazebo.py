"""External test harness; run in the public Simulator container, never in Participant."""

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def xyz(value):
    return [value.x, value.y, value.z]


def pose_sample(position, orientation):
    return np.array([*xyz(position), *xyz(orientation), orientation.w])


def yaw(sample):
    x, y, z, w = sample[3:]
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


class Trial(Node):
    def __init__(self):
        super().__init__('t4_public_trial')
        self.odom = {}
        self.truth = {}
        self.tf = {}
        self.clock = None
        self.ready = False
        self.command = self.create_publisher(Twist, '/cmd_vel', 10)
        for topic, data in [('/odom', self.odom), ('/odom/ground_truth', self.truth)]:
            self.create_subscription(Odometry, topic, lambda msg, out=data: self.record(msg, out),
                                     qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/tf', self.record_tf, qos_profile_sensor_data)
        self.create_subscription(Clock, '/clock', self.record_clock, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/go1_ready', self.record_ready, latched)

    def record(self, msg, output):
        pose = msg.pose.pose
        if output is self.odom:
            assert (msg.header.frame_id, msg.child_frame_id) == ('odom', 'trunk')
            twist = msg.twist.twist
            assert np.isfinite([*xyz(twist.linear), *xyz(twist.angular)]).all()
        sample = pose_sample(pose.position, pose.orientation)
        assert np.isfinite(sample).all() and abs(np.linalg.norm(sample[3:]) - 1.0) < 1e-5
        output[seconds(msg.header.stamp)] = sample

    def record_tf(self, msg):
        for transform in msg.transforms:
            if (transform.header.frame_id, transform.child_frame_id) == ('odom', 'trunk'):
                stamp = seconds(transform.header.stamp)
                self.tf[stamp] = pose_sample(transform.transform.translation, transform.transform.rotation)

    def record_clock(self, msg):
        self.clock = seconds(msg.clock)

    def record_ready(self, msg):
        self.ready = msg.data

    def move(self, linear, angular):
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self.command.publish(command)

    def run(self):
        deadline = time.monotonic() + 480
        while not (self.ready and self.clock is not None and self.odom and self.truth and self.tf):
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.monotonic() > deadline - 420:
                raise TimeoutError(f'Waiting for interfaces: ready={self.ready}, clock={self.clock}, '
                                   f'odom={len(self.odom)}, truth={len(self.truth)}, tf={len(self.tf)}')
        start = self.clock
        end = start
        last_command = -math.inf
        phases = [(2.0, 0.0, 0.0)] + [(6.0, 0.2, 0.0), (math.pi / 0.6, 0.0, 0.3)] * 4
        phases += [(13.0, 0.0, 0.0)]
        for duration, linear, angular in phases:
            end += duration
            while self.clock < end:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Simulation stopped or too slow at t={self.clock:.2f}')
                if self.clock - last_command >= 0.045:
                    self.move(linear, angular)
                    last_command = self.clock
                rclpy.spin_once(self, timeout_sec=0.1)
        self.move(0.0, 0.0)
        self.measure(start + 1.0, end - 10.2, end - 0.2)

    def measure(self, start, stop_start, stop_end):
        stamps = np.array(sorted(t for t in self.odom if start <= t <= stop_end))
        predicted = np.array([self.odom[t] for t in stamps])
        common = [t for t in stamps if t in self.tf]
        assert all(np.array_equal(self.odom[t], self.tf[t]) for t in common)
        truth_times = sorted(self.truth)
        first = min(truth_times, key=lambda t: abs(t - stamps[0]))
        angle = yaw(predicted[0]) - yaw(self.truth[first])
        c, s = math.cos(angle), math.sin(angle)
        xy = np.array([self.truth[t][:3] for t in truth_times])
        reference = np.column_stack([np.interp(stamps, truth_times, axis) for axis in xy.T])
        reference[:, :2] = reference[:, :2] @ [[c, s], [-s, c]]
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]
        indices = np.unique(np.searchsorted(distance, np.arange(0, distance[-1], 1.0)))
        differences = np.diff(predicted[indices, :3], axis=0) - np.diff(reference[indices], axis=0)
        errors = np.linalg.norm(differences, axis=1) / np.diff(distance[indices])
        first_stop, last_stop = np.searchsorted(stamps, [stop_start, stop_end], side='right') - 1
        initial, final = predicted[first_stop], predicted[last_stop]
        drift = np.linalg.norm(final[:3] - initial[:3])
        rotation_drift = math.degrees(2 * math.acos(min(1.0, abs(initial[3:] @ final[3:]))))
        duration = stamps[-1] - stamps[0]
        result = {'odom_hz': (len(stamps) - 1) / duration, 'tf_hz': (len(common) - 1) / duration,
                  'meters': distance[-1], 'segments': len(errors),
                  'mean_rpe': float(errors.mean()) if len(errors) else None,
                  'position_drift_m': drift, 'orientation_drift_deg': rotation_drift}
        print(json.dumps(result), flush=True)
        assert result['odom_hz'] >= 20 and result['tf_hz'] >= 20
        assert len(errors) >= 2 and result['mean_rpe'] <= 0.20
        assert drift <= 0.10 and rotation_drift <= 5.0


def main():
    rclpy.init()
    trial = Trial()
    try:
        trial.run()
    finally:
        trial.move(0.0, 0.0)
        trial.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
