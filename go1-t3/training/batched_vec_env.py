"""VecEnv where every worker process steps several environments (fewer IPC round trips than SubprocVecEnv)."""
import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def _worker(remote, parent, env_fns):
    parent.close()
    envs = [fn() for fn in env_fns]
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, rews, dones, infos = [], [], [], []
            for env, a in zip(envs, data):
                o, r, term, trunc, info = env.step(a)
                done = term or trunc
                if done:
                    info['terminal_observation'] = o
                    info['TimeLimit.truncated'] = trunc and not term
                    o, _ = env.reset()
                obs.append(o)
                rews.append(r)
                dones.append(done)
                infos.append(info)
            remote.send((np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones), infos))
        elif cmd == 'reset':
            remote.send(np.stack([env.reset()[0] for env in envs]))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_attr':
            remote.send([getattr(env, data) for env in envs])
        else:
            raise NotImplementedError(cmd)


class BatchedSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, n_workers):
        assert len(env_fns) % n_workers == 0
        self.k = len(env_fns) // n_workers
        ctx = mp.get_context('fork')
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            remote, work_remote = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(work_remote, remote, env_fns[w * self.k:(w + 1) * self.k]), daemon=True)
            p.start()
            work_remote.close()
            self.remotes.append(remote)
            self.procs.append(p)
        probe = env_fns[0]()
        super().__init__(len(env_fns), probe.observation_space, probe.action_space)
        probe.close()
        self._actions = None

    def reset(self):
        for r in self.remotes:
            r.send(('reset', None))
        return np.concatenate([r.recv() for r in self.remotes])

    def step_async(self, actions):
        for i, r in enumerate(self.remotes):
            r.send(('step', actions[i * self.k:(i + 1) * self.k]))

    def step_wait(self):
        res = [r.recv() for r in self.remotes]
        obs = np.concatenate([x[0] for x in res])
        rews = np.concatenate([x[1] for x in res])
        dones = np.concatenate([x[2] for x in res])
        infos = [i for x in res for i in x[3]]
        return obs, rews, dones, infos

    def close(self):
        for r in self.remotes:
            r.send(('close', None))
        for p in self.procs:
            p.join(timeout=5)

    def get_attr(self, attr_name, indices=None):
        for r in self.remotes:
            r.send(('get_attr', attr_name))
        vals = [v for r in self.remotes for v in r.recv()]
        return vals if indices is None else [vals[i] for i in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
