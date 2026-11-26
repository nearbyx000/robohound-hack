"""NumPy inference of the trained actor network (no PyTorch needed inside the Docker image).

The weights file (policy/policy.npz) is produced by training/export_policy.py from the SB3 checkpoint:
W0, b0, W1, b1, ... for the hidden ELU layers, then W_out, b_out for the action mean.
"""
import numpy as np

from . import common as C


def _elu(x):
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class Policy:
    def __init__(self, path):
        data = np.load(path)
        n_hidden = int(data['n_hidden'])
        self.hidden = [(data[f'W{i}'].astype(np.float64), data[f'b{i}'].astype(np.float64)) for i in range(n_hidden)]
        self.out = (data['W_out'].astype(np.float64), data['b_out'].astype(np.float64))
        assert self.hidden[0][0].shape[1] == C.OBS_DIM, 'observation size mismatch'
        assert self.out[0].shape[0] == C.N

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        for W, b in self.hidden:
            x = _elu(W @ x + b)
        W, b = self.out
        return np.clip(W @ x + b, -C.ACTION_CLIP, C.ACTION_CLIP)
