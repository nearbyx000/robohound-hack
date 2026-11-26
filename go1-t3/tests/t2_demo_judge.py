#!/usr/bin/env python3
"""Local T2 test (the official Judge is private). Runs inside the simulator container and publishes
test commands to /lowcmd_bare, while checking the /lowcmd output of the Safety Block:
  1. safe slow squat (sinusoid)          -> output must follow it
  2. big jump of the target              -> output must ramp, < 5 deg per message
  3. out-of-range targets                -> output inside joint limits
  4. NaN / +Inf / -Inf messages          -> never in output, last safe target kept
  5. self-colliding target (legs cross)  -> must not be executed
  6. input stream stops                  -> within 0.5 s hold last accepted target, dq=0, tau=0, stream continues
Rate: >= 50 Hz average over 5 s and max gap <= 100 ms (simulation time).
"""
import json
import math

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from unitree_go.msg import LowCmd

Q_MIN = np.array([-0.863, -0.686, -2.818] * 4)
Q_MAX = np.array([0.863, 4.501, -0.888] * 4)
STAND = np.array([0.0, 0.9, -1.8] * 4)


class T2Demo:
    def __init__(self):
        self.n = rclpy.create_node('local_t2_demo')
        self.t = None
        self.out = []      # (t, q, dq, tau, kp, kd)
        self.pub = self.n.create_publisher(LowCmd, '/lowcmd_bare', 10)
        self.n.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.n.create_subscription(LowCmd, '/lowcmd', self.on_out, qos_profile_sensor_data)

    def on_clock(self, m):
        self.t = m.clock.sec + m.clock.nanosec * 1e-9

    def on_out(self, m):
        if self.t is None:
            return
        c = m.motor_cmd[:12]
        self.out.append((self.t, np.array([x.q for x in c]), np.array([x.dq for x in c]),
                         np.array([x.tau for x in c]), np.array([x.kp for x in c]), np.array([x.kd for x in c])))

    def send(self, q, kp=40.0, kd=2.0):
        m = LowCmd()
        for i in range(12):
            m.motor_cmd[i].mode = 0x0A
            m.motor_cmd[i].q = float(q[i])
            m.motor_cmd[i].kp = float(kp)
            m.motor_cmd[i].kd = float(kd)
        self.pub.publish(m)

    def phase(self, duration, qfun, rate=100.0, publish=True):
        t0 = self.t
        last = -1.0
        while rclpy.ok() and self.t - t0 < duration:
            rclpy.spin_once(self.n, timeout_sec=0.002)
            if publish and self.t - last >= 1.0 / rate:
                self.send(qfun(self.t - t0))
                last = self.t
        return t0, self.t

    def run(self):
        while rclpy.ok() and (self.t is None or not self.out):
            rclpy.spin_once(self.n, timeout_sec=0.05)
        marks = {}
        squat = lambda s: STAND + np.array([0.0, 0.25, -0.5] * 4) * math.sin(2 * math.pi * 0.25 * s)  # noqa: E731
        marks['1_safe_squat'] = self.phase(8.0, squat)
        marks['2_big_jump'] = self.phase(2.0, lambda s: STAND + np.array([0.0, -0.8, 0.8] * 4))
        marks['3_out_of_range'] = self.phase(2.0, lambda s: np.array([2.0, 6.0, 1.0] * 4))
        marks['back_to_stand'] = self.phase(3.0, lambda s: STAND)
        nan_q = STAND.copy(); nan_q[1] = float('nan')
        inf_q = STAND.copy(); inf_q[5] = float('inf')
        marks['4_nan_inf'] = self.phase(2.0, lambda s: [nan_q, inf_q, -inf_q][int(s * 10) % 3])
        cross = STAND.copy(); cross[0], cross[3] = 0.863, -0.863
        marks['5_self_collision'] = self.phase(2.0, lambda s: cross)
        marks['back_to_stand2'] = self.phase(2.0, lambda s: STAND + 0.05)
        last_accepted_t = self.t
        marks['6_input_loss'] = self.phase(3.0, None, publish=False)
        self.report(marks, last_accepted_t)

    def report(self, marks, loss_start):
        out = self.out
        ts = np.array([o[0] for o in out])
        qs = np.array([o[1] for o in out])
        res = {}
        finite = all(np.all(np.isfinite(np.concatenate(o[1:]))) for o in out)
        res['no_nan_inf_in_output'] = bool(finite)
        res['within_joint_limits'] = bool(np.all(qs >= Q_MIN - 1e-9) and np.all(qs <= Q_MAX + 1e-9))
        steps = np.max(np.abs(np.diff(qs, axis=0)), axis=1) if len(qs) > 1 else np.zeros(1)
        res['max_step_deg'] = round(float(np.degrees(steps.max())), 3)
        res['step_below_5deg'] = bool(res['max_step_deg'] < 5.0)
        # rate: worst 5 s window
        worst_rate, max_gap = 1e9, 0.0
        for i in range(len(ts)):
            j = np.searchsorted(ts, ts[i] + 5.0)
            if j >= len(ts):
                break
            worst_rate = min(worst_rate, (j - i) / 5.0)
        if len(ts) > 1:
            max_gap = float(np.max(np.diff(ts)))
        res['min_rate_5s_hz'] = round(worst_rate, 1)
        res['max_gap_ms'] = round(1000 * max_gap, 1)
        # squat followed (output range during phase 1)
        a, b = marks['1_safe_squat']
        sel = (ts >= a) & (ts <= b)
        res['squat_thigh_range_rad'] = round(float(np.ptp(qs[sel, 1])), 3)
        a, b = marks['5_self_collision']
        sel = (ts >= a) & (ts <= b)
        res['collision_target_reached'] = bool(np.any(np.all(np.abs(qs[sel][:, [0, 3]] - [0.863, -0.863]) < 0.05, axis=1)))
        sel = ts >= loss_start + 0.5
        held = qs[sel]
        o_sel = [o for o in out if o[0] >= loss_start + 0.5]
        res['loss_hold_constant'] = bool(len(held) > 0 and np.max(np.ptp(held, axis=0)) < 1e-9)
        res['loss_hold_dq_tau_zero'] = bool(all(np.all(o[2] == 0) and np.all(o[3] == 0) for o in o_sel))
        res['loss_hold_kp_kd_valid'] = bool(all(np.all(o[4] > 0) and np.all(o[5] > 0) for o in o_sel))
        res['loss_stream_continues'] = bool(len(o_sel) > 100)
        print(json.dumps(res), flush=True)


def main():
    rclpy.init()
    T2Demo().run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
