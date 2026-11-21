"""Regression checks for the T2 command contract."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

from safety import (ACTIVE_MODE, LOWER, UPPER, WATCHDOG_NS, Safety,
                    read_command, segment_samples)

HOME = (0.0, 0.9, -1.8) * 4


def commands(**changes):
    result = []
    for q in HOME + (0.0,) * 8:
        fields = dict(mode=ACTIVE_MODE, q=q, dq=0.5, kp=20.0, kd=1.0, tau=0.25)
        fields.update(changes)
        result.append(SimpleNamespace(**fields))
    return result


class TestSafety(unittest.TestCase):
    def test_safe_target_is_applied_then_held_after_timeout(self):
        state = Safety(lambda _: True, HOME)
        state.initialize()
        request = commands()
        request[0].q = 0.02
        self.assertTrue(state.accept(read_command(request), 100))
        self.assertAlmostEqual(state.output[0].q, 0.02)
        self.assertEqual(state.tick(100 + WATCHDOG_NS - 1)[0].tau, 0.25)
        output = state.tick(100 + WATCHDOG_NS)
        self.assertEqual(output[0].q, state.pose[0])
        self.assertTrue(all(m.dq == m.tau == 0 for m in output))

    def test_checks_whole_message_for_nonfinite_controls(self):
        for index in (0, 11, 12, 19):
            for field in ("q", "dq", "kp", "kd", "tau"):
                for bad in (math.nan, math.inf, -math.inf):
                    request = commands()
                    setattr(request[index], field, bad)
                    self.assertIsNone(read_command(request))

    def test_clamping_preserves_angle_and_step_after_float32(self):
        state = Safety(lambda _: True, HOME)
        state.initialize()
        request = read_command(commands(q=1e30))
        for tick in range(70):
            previous = state.pose
            state.accept(request, tick)
            for before, after, low, high in zip(previous, state.pose, LOWER, UPPER):
                self.assertLessEqual(abs(after - before), math.radians(5))
                self.assertLessEqual(low, after)
                self.assertLessEqual(after, high)
        self.assertTrue(all(abs(q - high) < 2e-6 for q, high in zip(state.pose, UPPER)))

    def test_rejects_internal_collision_with_clear_endpoints(self):
        state = Safety(lambda p: not 0.018 < p[0] < 0.022, HOME)
        state.initialize()
        request = commands()
        request[0].q = 0.04
        previous = state.output
        self.assertFalse(state.accept(read_command(request), 300))
        self.assertIs(state.output, previous)
        samples = list(segment_samples(HOME, (0.04,) + HOME[1:]))
        for a, b in zip(samples, samples[1:]):
            self.assertLessEqual(abs(b[0] - a[0]), math.radians(0.1))

    def test_backward_clock_enters_hold(self):
        state = Safety(lambda _: True, HOME)
        state.initialize()
        state.accept(read_command(commands()), 1000)
        self.assertEqual(state.tick(0)[0].tau, 0.0)


def test_native_model():
    import random

    import mujoco
    from collision import Collision
    from unitree_go.msg import LowCmd, LowState

    assert mujoco.__version__ == "3.3.7"
    assert len(LowCmd().motor_cmd) == len(LowState().motor_state) == 20
    collision = Collision(Path(__file__).with_name("model"))
    assert collision.clear(collision.home)
    rng = random.Random(42)
    obstructed = 0
    for _ in range(40):
        pose = tuple(rng.uniform(lo, hi) for lo, hi in zip(LOWER, UPPER))
        optimized_clear = collision.clear(pose)
        direct_clear = all(
            mujoco.mj_geomDistance(
                collision.model, collision.data, first, second, 0.0, collision.witness
            )
            >= -0.00095
            for first, second in collision.pairs
        )
        assert optimized_clear == direct_clear
        obstructed += not direct_clear
    assert obstructed > 0
    print("Verified MuJoCo model, 655 pairs, sample collisions and message layouts")


if __name__ == "__main__":
    if "--native" in sys.argv:
        sys.argv.remove("--native")
        test_native_model()
    unittest.main()
