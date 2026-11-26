#!/usr/bin/env python3
"""Exports the actor of an SB3 PPO checkpoint to policy/policy.npz (NumPy weights) and checks that
NumPy inference matches PyTorch.
Usage: python training/export_policy.py runs/go1_walk_ppo/model_final.zip policy/policy.npz
"""
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from go1_solution import common as C  # noqa: E402
from go1_solution.policy import Policy  # noqa: E402


def main(src, dst):
    model = PPO.load(src, device='cpu')
    pol = model.policy
    linears = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    arrays = {'n_hidden': np.array(len(linears))}
    for i, lin in enumerate(linears):
        arrays[f'W{i}'] = lin.weight.detach().numpy().astype(np.float32)
        arrays[f'b{i}'] = lin.bias.detach().numpy().astype(np.float32)
    arrays['W_out'] = pol.action_net.weight.detach().numpy().astype(np.float32)
    arrays['b_out'] = pol.action_net.bias.detach().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    np.savez(dst, **arrays)

    np_pol = Policy(dst)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(0, 1, C.OBS_DIM).astype(np.float32)
        with torch.no_grad():
            ref, _, _ = pol(torch.as_tensor(obs)[None], deterministic=True)
        ref = np.clip(ref.numpy()[0], -C.ACTION_CLIP, C.ACTION_CLIP)
        assert np.allclose(ref, np_pol(obs), atol=1e-4), (ref, np_pol(obs))
    print(f'exported {src} -> {dst}; numpy == torch OK')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
