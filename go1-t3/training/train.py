#!/usr/bin/env python3
"""Trains the Go1 walking policy from scratch with PPO (stable-baselines3, CPU).

Outputs (in --out, default training/runs/<run_name>):
  config.yaml          exact copy of the configuration used
  tb/                  TensorBoard logs (rewards, every reward term, losses)
  progress.csv         the same scalars as CSV
  checkpoints/         intermediate model checkpoints
  model_final.zip      final SB3 model
  vecnormalize.pkl     reward-normalisation statistics
Usage: python training/train.py --config training/config.yaml
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_vec_env import BatchedSubprocVecEnv  # noqa: E402
from go1_env import Go1WalkEnv  # noqa: E402


class EpisodeMetrics(BaseCallback):
    """Logs mean reward terms and forward distance of finished episodes."""

    def __init__(self):
        super().__init__()
        self.buf = {}

    def _on_step(self):
        for info in self.locals['infos']:
            m = info.get('episode_metrics')
            if m:
                for k, v in m.items():
                    self.buf.setdefault(k, []).append(v)
        return True

    def _on_rollout_end(self):
        for k, v in self.buf.items():
            self.logger.record(f'episode_terms/{k}', float(np.mean(v)))
        self.buf = {}


def make_env(cfg, rank):
    def _f():
        return Go1WalkEnv(cfg, seed=cfg['seed'] * 1000 + rank)
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    p.add_argument('--out', default=None)
    p.add_argument('--timesteps', type=int, default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = args.out or os.path.join(os.path.dirname(__file__), 'runs', cfg['run_name'])
    os.makedirs(out, exist_ok=True)
    shutil.copy(args.config, os.path.join(out, 'config.yaml'))
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    venv = BatchedSubprocVecEnv([make_env(cfg, i) for i in range(cfg['n_envs'])], n_workers=cfg['n_workers'])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg['ppo']['gamma'])
    pc = cfg['ppo']
    model = PPO(
        'MlpPolicy', venv, seed=cfg['seed'], verbose=1, device='cpu',
        learning_rate=pc['learning_rate'], n_steps=pc['n_steps'], batch_size=pc['batch_size'],
        n_epochs=pc['n_epochs'], gamma=pc['gamma'], gae_lambda=pc['gae_lambda'], clip_range=pc['clip_range'],
        ent_coef=pc['ent_coef'], vf_coef=pc['vf_coef'], max_grad_norm=pc['max_grad_norm'],
        target_kl=pc['target_kl'],
        policy_kwargs=dict(net_arch=dict(pi=pc['net_arch'], vf=pc['net_arch']),
                           activation_fn=torch.nn.ELU, log_std_init=pc['log_std_init']),
    )
    model.set_logger(configure(out, ['stdout', 'csv', 'tensorboard']))
    ckpt = CheckpointCallback(save_freq=max(1, 2_000_000 // cfg['n_envs']), save_path=os.path.join(out, 'checkpoints'),
                              name_prefix='ppo', save_vecnormalize=True)
    model.learn(total_timesteps=args.timesteps or cfg['total_timesteps'], callback=[ckpt, EpisodeMetrics()])
    model.save(os.path.join(out, 'model_final'))
    venv.save(os.path.join(out, 'vecnormalize.pkl'))
    venv.close()


if __name__ == '__main__':
    main()
