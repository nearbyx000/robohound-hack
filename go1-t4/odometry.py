"""Planar lidar odometry with IMU attitude and the public Go1 sensor geometry."""

import numpy as np
from scipy.spatial import cKDTree


SCAN_OFFSET = np.array([0.22, 0.0, 0.11])


def rotation_2d(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def scan_points(ranges, angle_min, angle_step, range_min, range_max, tilt):
    """Project hits into the level trunk frame; preserve gaps between beams."""
    angles = angle_min + angle_step * np.arange(len(ranges))
    valid = np.isfinite(ranges) & (ranges > max(0.34, range_min))
    valid &= ranges < min(12.0, range_max - 0.01)
    distances = np.where(valid, ranges, 0.0)
    hits = np.column_stack((distances * np.cos(angles), distances * np.sin(angles),
                            np.zeros(len(ranges))))
    leveled = (hits + SCAN_OFFSET) @ tilt.T
    # Downward rays can hit the floor instead of vertical surfaces.
    valid &= leveled[:, 2] > -0.20
    leveled[~valid] = np.nan
    return leveled[:, :2]


class ScanReference:
    """Local line segments from smooth, contiguous parts of a key scan."""

    def __init__(self, scan, position, yaw):
        self.position = position.copy()
        self.yaw = yaw
        neighborhoods = np.lib.stride_tricks.sliding_window_view(scan, 5, axis=0)
        neighborhoods = neighborhoods.transpose(0, 2, 1)
        neighborhoods = neighborhoods[np.isfinite(neighborhoods).all(axis=(1, 2))]
        gaps = np.linalg.norm(np.diff(neighborhoods, axis=1), axis=2)
        neighborhoods = neighborhoods[np.max(gaps, axis=1) < 0.20]
        centers = neighborhoods.mean(axis=1)
        offsets = neighborhoods - centers[:, None, :]
        covariance = offsets.transpose(0, 2, 1) @ offsets
        values, vectors = np.linalg.eigh(covariance)
        linear = (values[:, 1] > 0.001) & (values[:, 0] < 0.12 * values[:, 1])
        self.centers = centers[linear]
        self.normals = vectors[linear, :, 0]
        self.tree = cKDTree(self.centers)

    @property
    def usable(self):
        return len(self.centers) >= 30

    def match(self, points, yaw, position):
        to_world = rotation_2d(self.yaw)
        aligned = points @ rotation_2d(yaw - self.yaw).T
        translation = to_world.T @ (position - self.position)
        initial = translation.copy()
        for _ in range(12):
            shifted = aligned + translation
            distance, indices = self.tree.query(shifted)
            near = distance < 0.28
            normals = self.normals[indices[near]]
            residual = np.einsum('ij,ij->i', normals, shifted[near] - self.centers[indices[near]])
            if len(residual) < max(30, 0.22 * len(points)):
                return None

            # Cauchy weights attenuate edges, moving legs and mismatched surfaces.
            weights = 1.0 / (1.0 + (residual / 0.035) ** 2)
            weighted = normals.T * weights
            information = weighted @ normals
            eigenvalues, axes = np.linalg.eigh(information)
            observed = eigenvalues > max(2.0, 0.035 * eigenvalues[-1])
            if not np.any(observed):
                return None
            # A single visible wall constrains only motion normal to that wall.
            basis = axes[:, observed]
            correction = -basis @ ((basis.T @ (weighted @ residual)) / eigenvalues[observed])
            translation += correction
            if np.linalg.norm(correction) < 0.0005:
                break

        if np.median(np.abs(residual)) > 0.065 or np.linalg.norm(translation - initial) > 0.28:
            return None
        return self.position + to_world @ translation


class LidarOdometry:
    def __init__(self):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.reference = None
        self.last_match = None
        self.failures = 0

    def update(self, scan, yaw, stamp):
        if len(scan) < 5 or (self.last_match is not None and stamp <= self.last_match):
            return
        points = scan[np.isfinite(scan).all(axis=1)]
        if len(points) < 30:
            return
        if self.reference is None:
            self.set_reference(scan, yaw, stamp)
            return

        position = self.reference.match(points, yaw, self.position)
        if position is None:
            self.failures += 1
            if self.failures >= 8:
                # Recover from occlusion without inserting an unobserved pose jump.
                self.set_reference(scan, yaw, stamp)
            return

        self.failures = 0
        dt = stamp - self.last_match
        movement = position - self.position
        alpha = dt / (0.15 + dt)
        self.velocity += alpha * (movement / dt - self.velocity)
        self.position = position
        self.last_match = stamp
        if (np.linalg.norm(position - self.reference.position) > 0.30
                or abs(wrap(yaw - self.reference.yaw)) > 0.40):
            self.set_reference(scan, yaw, stamp)

    def set_reference(self, scan, yaw, stamp):
        reference = ScanReference(scan, self.position, yaw)
        if reference.usable:
            if self.failures:
                self.velocity.fill(0.0)
            self.reference = reference
            self.last_match = stamp
            self.failures = 0
