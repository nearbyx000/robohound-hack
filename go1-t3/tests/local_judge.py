#!/usr/bin/env python3
"""Local re-implementation of the public T3 rules (the official Judge is private).

Runs inside the simulator container. Warm-up 1.5 s of sim time, then /cmd_vel = (v, 0, 0) for at most
90 s of sim time. Success: >= 3 m forward along the initial heading without a fall.
Fall: trunk height < 0.16 m, |roll| > 50 deg or |pitch| > 50 deg continuously for more than 0.5 s.
Also measures the /lowcmd stream (rate and max gap in sim time, NaN/Inf).
Prints one JSON line with the result.
"""
import json
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from unitree_go.msg import LowCmd

SPEED = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
WARMUP, LIMIT, DIST = 1.5, 90.0, 3.0


def rpy(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Judge:
    def __init__(self):
        self.n = rclpy.create_node('local_t3_judge')
        self.t = None
        self.t0 = None
        self.pose = None
        self.start = None
        self.viol_since = None
        self.fell = False
        self.cmd_times = []
        self.bad_numbers = 0
        self.max_dist = 0.0
        self.min_height = 9.0
        self.max_tilt = 0.0
        self.pub = self.n.create_publisher(Twist, '/cmd_vel', 10)
        self.n.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.n.create_subscription(Odometry, '/odom/ground_truth', self.on_odom, qos_profile_sensor_data)
        self.n.create_subscription(LowCmd, '/lowcmd', self.on_lowcmd, qos_profile_sensor_data)

    def on_clock(self, m):
        self.t = m.clock.sec + m.clock.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = self.t

    def on_lowcmd(self, m):
        if self.t is not None:
            self.cmd_times.append(self.t)
        for c in m.motor_cmd[:12]:
            if not all(math.isfinite(v) for v in (c.q, c.dq, c.kp, c.kd, c.tau)):
                self.bad_numbers += 1

    def on_odom(self, m):
        self.pose = m.pose.pose

    def run(self):
        while rclpy.ok() and (self.t is None or self.pose is None):
            rclpy.spin_once(self.n, timeout_sec=0.05)
        t_begin = self.t
        while rclpy.ok() and self.t - t_begin < WARMUP:
            rclpy.spin_once(self.n, timeout_sec=0.01)
        p = self.pose
        _, _, yaw0 = rpy(p.orientation)
        self.start = (p.position.x, p.position.y, yaw0)
        t_start = self.t
        last_pub = -1.0
        success_time = None
        while rclpy.ok() and self.t - t_start < LIMIT:
            rclpy.spin_once(self.n, timeout_sec=0.005)
            if self.t - last_pub >= 0.05:
                tw = Twist()
                tw.linear.x = SPEED
                self.pub.publish(tw)
                last_pub = self.t
            p = self.pose
            roll, pitch, _ = rpy(p.orientation)
            h = p.position.z
            self.min_height = min(self.min_height, h)
            self.max_tilt = max(self.max_tilt, abs(math.degrees(roll)), abs(math.degrees(pitch)))
            violation = h < 0.16 or abs(roll) > math.radians(50) or abs(pitch) > math.radians(50)
            if violation:
                if self.viol_since is None:
                    self.viol_since = self.t
                elif self.t - self.viol_since > 0.5:
                    self.fell = True
                    break
            else:
                self.viol_since = None
            dx, dy = p.position.x - self.start[0], p.position.y - self.start[1]
            fwd = dx * math.cos(self.start[2]) + dy * math.sin(self.start[2])
            self.max_dist = max(self.max_dist, fwd)
            if fwd >= DIST and success_time is None:
                success_time = self.t - t_start
                break
        # /lowcmd stream statistics over the run (sim time)
        ts = [x for x in self.cmd_times if x >= t_begin]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        p = self.pose
        lateral = -(p.position.x - self.start[0]) * math.sin(self.start[2]) + (p.position.y - self.start[1]) * math.cos(self.start[2])
        res = dict(success=bool(success_time is not None and not self.fell), fell=self.fell,
                   time_to_3m=success_time, max_forward=round(self.max_dist, 3), lateral=round(lateral, 3),
                   min_height=round(self.min_height, 3), max_tilt_deg=round(self.max_tilt, 1),
                   lowcmd_rate_hz=round(len(ts) / span, 1) if span else 0.0,
                   lowcmd_max_gap_ms=round(1000 * max(gaps), 1) if gaps else None,
                   lowcmd_nonfinite=self.bad_numbers, speed=SPEED)
        print(json.dumps(res), flush=True)


def main():
    rclpy.init()
    Judge().run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
