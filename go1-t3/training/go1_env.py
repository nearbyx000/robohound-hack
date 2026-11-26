"""MuJoCo training environment for Go1 forward walking (gymnasium API).

It reproduces the hackathon simulator (go1_mujoco.py) as closely as possible:
* the same menagerie go1.xml/scene.xml, timestep 0.002 s, start from keyframe 'home';
* built-in position actuators disabled, torques applied through qfrc_applied with the Unitree motor law
  tau = kp(q_des - q) + kd(0 - dq), clipped to the Go1 torque limits;
* the policy acts every 20 ms; between policy steps the Safety Block ramp (<= 4.5 deg per 4 ms safety
  tick, joint-limit clipping) is applied exactly as in the deployed Safety Block;
* observations are built with go1_solution.common from the same signals the /lowstate message carries
  (IMU quaternion, gyro = body angular velocity, joint q/dq); no base linear velocity is observed.
Domain randomisation (friction, mass, gains, latency, pushes, sensor noise) makes the policy robust.
"""
import os
import sys

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from go1_solution import common as C  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'unitree_go1')


class Go1WalkEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg=None, seed=None):
        super().__init__()
        self.cfg = cfg or {}
        self.rcfg = self.cfg.get('rewards', {})
        self.dr = self.cfg.get('domain_rand', {})
        self.cmd_cfg = self.cfg.get('commands', {})
        self.episode_steps = int(self.cfg.get('episode_length_s', 20.0) / C.POLICY_DT)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(MODEL_DIR, 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        assert abs(self.model.opt.timestep - C.SIM_DT) < 1e-9, self.model.opt.timestep
        self.model.actuator_gainprm[:, 0] = 0.0
        self.model.actuator_biasprm[:, :] = 0.0
        self.key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.JOINT_NAMES]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in jid])
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'trunk')
        self.floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        # foot geoms = geoms of class 'foot' (spheres with priority 1); thigh/calf/trunk geoms = bad contacts
        self.foot_geoms = []
        for leg in ('FR', 'FL', 'RR', 'RL'):
            calf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_calf')
            g = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == calf
                 and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
            self.foot_geoms.append(g[0])
        self.foot_set = set(self.foot_geoms)
        self.nominal_friction = self.model.geom_friction.copy()
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (C.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-C.ACTION_CLIP, C.ACTION_CLIP, (C.N,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.hist = C.ObsHistory()

    # ------------------------------------------------------------------ helpers
    def _q(self):
        return self.data.qpos[self.qadr].copy()

    def _dq(self):
        return self.data.qvel[self.dadr].copy()

    def _base_lin_vel_body(self):
        q = self.data.qpos[3:7]
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, q)
        v = np.zeros(3)
        mujoco.mju_rotVecQuat(v, self.data.qvel[0:3], inv)
        return v

    def _foot_contacts(self):
        contact = np.zeros(4, dtype=bool)
        bad = False
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.floor or g2 == self.floor:
                other = g2 if g1 == self.floor else g1
                if other in self.foot_set:
                    contact[self.foot_geoms.index(other)] = True
                elif self.model.geom_contype[other] or self.model.geom_conaffinity[other]:
                    bad = True
        return contact, bad

    def _sample_command(self):
        c = self.cmd_cfg
        if self.rng.random() < c.get('zero_prob', 0.1):
            return np.zeros(3)
        if self.rng.random() < c.get('focus_prob', 0.5):
            vx = float(c.get('focus_speed', 0.3)) + self.rng.uniform(-0.05, 0.05)
        else:
            vx = self.rng.uniform(*c.get('vx', [0.1, 0.6]))
        return np.array([vx, 0.0, 0.0])

    def _randomize(self):
        m, dr = self.model, self.dr
        m.geom_friction[:] = self.nominal_friction
        m.body_mass[:] = self.nominal_mass
        m.body_ipos[:] = self.nominal_ipos
        if dr.get('friction'):
            m.geom_friction[:, 0] = self.nominal_friction[:, 0] * self.rng.uniform(*dr['friction'])
        if dr.get('added_mass'):
            m.body_mass[self.trunk] = self.nominal_mass[self.trunk] + self.rng.uniform(*dr['added_mass'])
        if dr.get('com_shift'):
            m.body_ipos[self.trunk] = self.nominal_ipos[self.trunk] + self.rng.uniform(-1, 1, 3) * dr['com_shift']
        self.kp_scale = self.rng.uniform(*dr.get('kp_scale', [1.0, 1.0]), C.N)
        self.kd_scale = self.rng.uniform(*dr.get('kd_scale', [1.0, 1.0]), C.N)
        self.latency_steps = int(self.rng.integers(0, dr.get('max_latency_substeps', 0) + 1))
        self.push_interval = int(dr.get('push_interval_s', 0) / C.POLICY_DT) if dr.get('push_interval_s') else 0
        self.push_vel = dr.get('push_vel', 0.0)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        noise = self.dr.get('init_joint_noise', 0.0)
        if noise:
            self.data.qpos[self.qadr] += self.rng.uniform(-noise, noise, C.N)
        mujoco.mj_forward(self.model, self.data)
        # Before the first policy command the simulator holds the stand pose; we let the robot settle
        # for a random time like in the 1.5 s warm-up of the judge.
        settle = int(self.rng.uniform(0.0, 0.5) / C.SIM_DT)
        hold_q = self._q()
        for _ in range(settle):
            self.data.qfrc_applied[self.dadr] = C.pd_torque(hold_q, self._q(), self._dq(), kp=40.0, kd=2.0)
            mujoco.mj_step(self.model, self.data)
        self.safe_target = C.clip_to_limits(self._q())
        self.applied_target = self.safe_target.copy()
        self.pending = []      # latency queue of (substep_index, target)
        self.cmd = self._sample_command()
        self.cmd_resample = int(self.rng.uniform(4.0, 10.0) / C.POLICY_DT)
        self.last_action = np.zeros(C.N)
        self.last_last_action = np.zeros(C.N)
        self.last_dq = self._dq()
        self.air_time = np.zeros(4)
        self.last_contact = np.ones(4, dtype=bool)
        self.step_count = 0
        self.ep_info = {}
        frame = self._frame()
        self.hist.reset(frame)
        return self.hist.push(frame), {}

    def _frame(self):
        d = self.data
        n = self.dr.get('obs_noise', 0.0)
        gyro = d.qvel[3:6].copy()
        quat = d.qpos[3:7].copy()
        q, dq = self._q(), self._dq()
        if n:
            gyro += self.rng.normal(0, 0.1 * n, 3)
            q += self.rng.normal(0, 0.01 * n, C.N)
            dq += self.rng.normal(0, 0.5 * n, C.N)
        return C.build_frame(gyro, quat, self.cmd, q, dq, self.last_action)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -C.ACTION_CLIP, C.ACTION_CLIP)
        desired = C.clip_to_limits(C.action_to_target(action))
        d, m = self.data, self.model
        tau_sq = 0.0
        sub = 0
        for _tick in range(C.TICKS_PER_POLICY):
            # Safety Block tick: ramp towards the requested target (<= 4.5 deg), clipped to limits
            self.safe_target = C.ramp_towards(self.safe_target, desired)
            self.pending.append((sub + self.latency_steps, self.safe_target.copy()))
            for _s in range(C.SUBSTEPS_PER_TICK):
                while self.pending and self.pending[0][0] <= sub:
                    self.applied_target = self.pending.pop(0)[1]
                tau = C.pd_torque(self.applied_target, self._q(), self._dq(),
                                  kp=C.KP * self.kp_scale, kd=C.KD * self.kd_scale)
                d.qfrc_applied[:] = 0.0
                d.qfrc_applied[self.dadr] = tau
                mujoco.mj_step(m, d)
                tau_sq += float(np.sum(np.square(tau)))
                sub += 1
        self.pending = [(s - sub, t) for s, t in self.pending]
        self.step_count += 1

        if self.push_interval and self.step_count % self.push_interval == 0:
            d.qvel[0:2] += self.rng.uniform(-self.push_vel, self.push_vel, 2)
        if self.step_count % self.cmd_resample == 0:
            self.cmd = self._sample_command()

        reward, terminated = self._reward(action, tau_sq / (C.TICKS_PER_POLICY * C.SUBSTEPS_PER_TICK))
        self.last_last_action = self.last_action
        self.last_action = action
        self.last_dq = self._dq()
        truncated = self.step_count >= self.episode_steps
        obs = self.hist.push(self._frame())
        info = {}
        if terminated or truncated:
            info['episode_metrics'] = dict(self.ep_info)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ reward
    def _reward(self, action, tau_sq_mean):
        r = self.rcfg
        d = self.data
        dt = C.POLICY_DT
        v = self._base_lin_vel_body()
        w = d.qvel[3:6]
        g = C.projected_gravity(d.qpos[3:7])
        q, dq = self._q(), self._dq()
        contact, bad_contact = self._foot_contacts()
        height = d.qpos[2]

        terms = {}
        lin_err = (self.cmd[0] - v[0]) ** 2 + (self.cmd[1] - v[1]) ** 2
        terms['tracking_lin_vel'] = np.exp(-lin_err / r.get('tracking_sigma', 0.25))
        terms['tracking_ang_vel'] = np.exp(-(self.cmd[2] - w[2]) ** 2 / r.get('tracking_sigma', 0.25))
        terms['lin_vel_z'] = v[2] ** 2
        terms['ang_vel_xy'] = w[0] ** 2 + w[1] ** 2
        terms['orientation'] = g[0] ** 2 + g[1] ** 2
        terms['base_height'] = (height - r.get('base_height_target', 0.30)) ** 2
        terms['torques'] = tau_sq_mean
        terms['dof_acc'] = float(np.sum(np.square((dq - self.last_dq) / dt)))
        terms['action_rate'] = float(np.sum(np.square(action - self.last_action)))
        terms['action_smoothness'] = float(np.sum(np.square(action - 2 * self.last_action + self.last_last_action)))
        terms['hip_pos'] = float(np.sum(np.square(q[[0, 3, 6, 9]] - C.Q_DEFAULT[[0, 3, 6, 9]])))
        margin = 0.05
        terms['dof_pos_limits'] = float(np.sum(np.clip(C.Q_MIN + margin - q, 0, None) + np.clip(q - (C.Q_MAX - margin), 0, None)))
        terms['collision'] = float(bad_contact)
        # feet air time: reward steps of reasonable length when moving
        first_contact = (self.air_time > 0.0) & contact
        self.air_time += dt
        air = float(np.sum((self.air_time - r.get('air_time_target', 0.3)) * first_contact))
        terms['feet_air_time'] = air if np.linalg.norm(self.cmd[:2]) > 0.05 else 0.0
        self.air_time *= ~contact
        terms['stand_still'] = float(np.sum(np.abs(q - C.Q_DEFAULT))) if np.linalg.norm(self.cmd[:2]) < 0.05 else 0.0
        # symmetric trot: diagonal pairs (FR,RL) and (FL,RR) should share contact state
        terms['gait_symmetry'] = float((contact[0] != contact[3]) + (contact[1] != contact[2]))

        fallen = (height < r.get('term_height', 0.18)) or (g[2] > -np.cos(np.radians(r.get('term_tilt_deg', 45.0))))
        terms['termination'] = float(fallen)

        scales = r.get('scales', {})
        total = 0.0
        for k, val in terms.items():
            s = scales.get(k, 0.0)
            if s:
                contrib = s * val * (dt if k != 'termination' else 1.0)
                total += contrib
                self.ep_info[k] = self.ep_info.get(k, 0.0) + contrib
        self.ep_info['forward_distance'] = float(d.qpos[0])
        self.ep_info['mean_vx_cmd'] = float(self.cmd[0])
        if r.get('only_positive_rewards', True):
            total = max(total, 0.0) if not fallen else total
        return total, bool(fallen)
