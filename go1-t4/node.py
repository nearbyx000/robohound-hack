"""ROS 2 adapter: estimate on sensor stamps and publish on simulation time."""

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import TransformBroadcaster

from odometry import LidarOdometry, scan_points


def nanoseconds(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def heading(rotation):
    matrix = rotation.as_matrix()
    return np.arctan2(matrix[1, 0], matrix[0, 0])


class OdometryNode(Node):
    def __init__(self):
        super().__init__('go1_t4_odometry')
        self.reset()
        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10)
        self.tf_publisher = TransformBroadcaster(self)
        self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)

    def reset(self):
        self.filter = LidarOdometry()
        self.imu = deque(maxlen=256)
        self.scans = deque(maxlen=4)
        self.initial_yaw = None
        self.clock_stamp = None
        self.publish_stamp = None

    def on_imu(self, message):
        stamp = nanoseconds(message.header.stamp)
        if self.imu and stamp <= self.imu[-1][0]:
            return
        q = message.orientation
        quaternion = np.array([q.x, q.y, q.z, q.w])
        if (message.orientation_covariance[0] < 0 or not np.isfinite(quaternion).all()
                or not 0.5 < np.linalg.norm(quaternion) < 1.5):
            return
        angular = message.angular_velocity
        rate = np.array([angular.x, angular.y, angular.z])
        rate = np.where(np.isfinite(rate), rate, 0.0)
        rotation = Rotation.from_quat(quaternion)
        if self.initial_yaw is None:
            self.initial_yaw = heading(rotation)
        self.imu.append((stamp, rotation, rate))
        self.process_scans()

    def on_scan(self, message):
        stamp = nanoseconds(message.header.stamp)
        if self.scans and stamp <= nanoseconds(self.scans[-1].header.stamp):
            return
        self.scans.append(message)
        self.process_scans()

    def process_scans(self):
        while self.scans and self.imu:
            message = self.scans[0]
            stamp = nanoseconds(message.header.stamp)
            if self.imu[-1][0] < stamp:
                break
            self.scans.popleft()
            closest = min(self.imu, key=lambda sample: abs(sample[0] - stamp))
            if abs(closest[0] - stamp) > 30_000_000:
                continue
            rotation = closest[1]
            yaw = heading(rotation)
            tilt = (Rotation.from_euler('z', -yaw) * rotation).as_matrix()
            points = scan_points(np.asarray(message.ranges), message.angle_min,
                                 message.angle_increment, message.range_min, message.range_max, tilt)
            self.filter.update(points, yaw - self.initial_yaw, stamp / 1e9)

    def on_clock(self, message):
        stamp = nanoseconds(message.clock)
        if self.clock_stamp is not None and stamp < self.clock_stamp:
            self.reset()
        self.clock_stamp = stamp
        if not self.imu or (self.publish_stamp is not None and stamp - self.publish_stamp < 20_000_000):
            return
        # DDS delivers /clock and /imu independently. Do not stamp future IMU
        # data as an older pose.
        for imu_stamp, orientation, angular in reversed(self.imu):
            if imu_stamp <= stamp:
                break
        else:
            return
        if self.filter.last_match is not None and self.filter.last_match > stamp / 1e9:
            return

        self.publish_stamp = stamp
        attitude = Rotation.from_euler('z', -self.initial_yaw) * orientation
        x, y, z, w = attitude.as_quat().tolist()
        position = self.filter.position.tolist()
        velocity = np.zeros(3)
        if self.filter.last_match is not None and stamp / 1e9 - self.filter.last_match < 0.25:
            velocity = attitude.inv().apply([*self.filter.velocity, 0.0])

        message_odom = Odometry()
        message_odom.header.stamp = message.clock
        message_odom.header.frame_id = 'odom'
        message_odom.child_frame_id = 'trunk'
        pose = message_odom.pose.pose
        pose.position.x, pose.position.y = position
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w
        twist = message_odom.twist.twist
        twist.linear.x, twist.linear.y, twist.linear.z = velocity.tolist()
        twist.angular.x, twist.angular.y, twist.angular.z = angular.tolist()

        transform = TransformStamped()
        transform.header = message_odom.header
        transform.child_frame_id = message_odom.child_frame_id
        transform.transform.translation.x, transform.transform.translation.y = position
        transform.transform.rotation = pose.orientation
        self.odom_publisher.publish(message_odom)
        self.tf_publisher.sendTransform(transform)


def main():
    rclpy.init()
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
