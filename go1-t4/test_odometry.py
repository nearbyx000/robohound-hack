"""Regression: noisy 3D rays from a tilted, off-center lidar along a metric path."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from odometry import LidarOdometry, SCAN_OFFSET, scan_points


ANGLES = np.linspace(-np.pi, np.pi, 720)
RAYS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES), np.zeros(len(ANGLES))))


def render(position, yaw, roll, pitch, rng):
    attitude = Rotation.from_euler('zyx', [0, pitch, roll])
    tilt = attitude.as_matrix()
    world_rotation = Rotation.from_euler('z', yaw).as_matrix() @ tilt
    origin = world_rotation @ SCAN_OFFSET + [*position, 0.32]
    rays = RAYS @ world_rotation.T
    with np.errstate(divide='ignore', invalid='ignore'):
        intersections = np.column_stack((
            (-4.0 - origin[0]) / rays[:, 0], (5.0 - origin[0]) / rays[:, 0],
            (-3.0 - origin[1]) / rays[:, 1], (4.0 - origin[1]) / rays[:, 1],
            -origin[2] / rays[:, 2],
        ))
    intersections[~(intersections > 0)] = np.inf
    ranges = intersections.min(axis=1) + rng.normal(0, 0.01, len(ANGLES))
    ranges[rng.random(len(ANGLES)) < 0.03] = np.inf
    return scan_points(ranges, ANGLES[0], ANGLES[1] - ANGLES[0], 0.12, 20.0, tilt)


class OdometryTest(unittest.TestCase):
    def test_square_and_stationary_drift(self):
        rng = np.random.default_rng(7)
        estimator = LidarOdometry()
        position = np.zeros(2)
        segment_errors = []
        sample = 0

        def update(yaw, moving):
            nonlocal sample
            phase = sample * 0.4
            roll = 0.08 * np.sin(phase) if moving else 0.0
            pitch = 0.10 * np.cos(phase) if moving else 0.0
            estimator.update(render(position, yaw, roll, pitch, rng), yaw, sample / 20)
            sample += 1

        for side in range(4):
            yaw = side * np.pi / 2
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            start = estimator.position.copy()
            for _ in range(100):
                update(yaw, True)
                position += 0.01 * direction
            update(yaw, True)
            segment_errors.append(np.linalg.norm(estimator.position - start - direction))
            for turn in np.linspace(0, np.pi / 2, 105):
                update(yaw + turn, True)

        update(2 * np.pi, False)
        stop = estimator.position.copy()
        for _ in range(200):
            update(2 * np.pi, False)
        drift = np.linalg.norm(estimator.position - stop)
        rpe = np.mean(segment_errors)
        print(f'Synthetic mean 1 m RPE: {rpe:.2%}; 10 s position drift: {drift:.3f} m')
        self.assertLess(rpe, 0.20)
        self.assertLess(drift, 0.10)


if __name__ == '__main__':
    unittest.main()
