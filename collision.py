"""Evaluate exactly the pairs selected by the organizer's collision table."""
import csv
from functools import lru_cache

import mujoco
import numpy as np

from safety import JOINT_NAMES


class Collision:
    def __init__(self, directory):
        self.model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.witness = np.zeros(6)
        self.addresses = np.array([self.model.joint(name).qposadr[0] for name in JOINT_NAMES])
        self.home = tuple(float(q) for q in self.model.key("home").qpos[self.addresses])
        self.data.qpos[:] = self.model.key("home").qpos

        pair_ids = []
        seen = set()
        with (directory / "t2_collision_pairs.csv").open(newline="") as source:
            for row in csv.DictReader(source):
                first, second = int(row["geom1_id"]), int(row["geom2_id"])
                pair = (first, second)
                if not 0 <= first < second < self.model.ngeom or pair in seen:
                    raise ValueError(f"Invalid collision pair {pair}")
                seen.add(pair)
                for suffix, geom_id in (("1", first), ("2", second)):
                    body_id = int(self.model.geom_bodyid[geom_id])
                    if body_id != int(row[f"body{suffix}_id"]):
                        raise ValueError(f"Body ID mismatch for geom {geom_id}")
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    if (name or f"UNNAMED_geom_{geom_id}") != row[f"geom{suffix}"]:
                        raise ValueError(f"Name mismatch for geom {geom_id}")
                if row["diagnostic_policy"] == "checked":
                    if not ((self.model.geom_contype[first] & self.model.geom_conaffinity[second])
                            or (self.model.geom_contype[second] & self.model.geom_conaffinity[first])):
                        raise ValueError(f"Checked geom pair has incompatible masks: {pair}")
                    pair_ids.append(pair)
        if self.model.ngeom != 56 or len(pair_ids) != 655 or len(seen) != 1540:
            raise ValueError("Expected 56 geoms and 655 checked pairs out of 1540")

        self.pairs = pair_ids
        self.first, self.second = np.asarray(pair_ids).T
        radii = self.model.geom_rbound
        self.overlap_radius_squared = (radii[self.first] + radii[self.second] + 1e-9) ** 2

    @lru_cache(maxsize=256)
    def clear(self, pose):
        self.data.qpos[self.addresses] = pose
        mujoco.mj_kinematics(self.model, self.data)
        centers = self.data.geom_xpos
        delta = centers[self.first] - centers[self.second]
        close = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= self.overlap_radius_squared)
        for index in close:
            first, second = self.pairs[index]
            distance = mujoco.mj_geomDistance(self.model, self.data, first, second, 0.0, self.witness)
            # Conservative 0.05 mm margin around the judge's 1 mm threshold.
            if not np.isfinite(distance) or distance < -0.00095:
                return False
        return True
