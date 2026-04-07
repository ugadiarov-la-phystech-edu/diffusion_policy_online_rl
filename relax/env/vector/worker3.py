import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json

import gymnasium
import numpy as np

import setproctitle
from relax.prctl import set_client_pdeathsig
from relax.futex import futex_client_wait, futex_client_notify
import relax.env.register_env

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^gymnasium\.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--image_obs", action="store_true", default=False)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--num_stack", type=int, default=1)
    return parser.parse_args()

def initialize_shm(descr, mode):
    arr = np.memmap(f"/dev/shm/{descr['name']}", dtype=descr["dtype"], mode=mode, shape=tuple(descr["shape"]))
    return arr

def main():
    args = parse_args()

    indices = [int(i) for i in args.index.split(",")]
    seeds = [int(i) for i in args.seed.split(",")]
    assert len(indices) == len(seeds)

    render_mode = "rgb_array" if args.image_obs else None
    envs = []
    for seed in seeds:
        try:
            env = gymnasium.make(args.env, render_mode=render_mode, seed=seed)
        except TypeError as e:
            env = gymnasium.make(args.env, render_mode=render_mode)

        if args.image_obs:
            from relax.env import _apply_image_wrappers
            env = _apply_image_wrappers(env, args.image_size, args.num_stack)
        env.reset(seed=seed)
        envs.append(env)

    setproctitle.setproctitle(f"FutexWorker:{indices[0]}")
    set_client_pdeathsig()

    descr = json.loads(args.descr)
    action = initialize_shm(descr["action"], "r")
    obs = initialize_shm(descr["obs"], "r+")
    obs2 = initialize_shm(descr["obs2"], "r+")
    reward = initialize_shm(descr["reward"], "r+")
    terminated = initialize_shm(descr["terminated"], "r+")
    truncated = initialize_shm(descr["truncated"], "r+")
    success = initialize_shm(descr["success"], "r+")
    signal = initialize_shm(descr["signal"], "r+")
    signal_pointer = signal.ctypes.data

    for i, env in zip(indices, envs):
        obs[i], _ = env.reset()
    futex_client_notify(signal_pointer)

    last = 0

    try:
        while True:
            last = futex_client_wait(signal_pointer, last)
            command = last & 0b11
            if command == 0b01:
                for i, env in zip(indices, envs):
                    _obs2, _reward, _terminated, _truncated, info = env.step(action[i])
                    obs2[i] = _obs2
                    reward[i] = _reward
                    terminated[i] = _terminated
                    truncated[i] = _truncated
                    success[i] = bool(info.get("success", False))
                    if _terminated or _truncated:
                        obs[i], _ = env.reset()
                    else:
                        obs[i] = _obs2
                futex_client_notify(signal_pointer)
            elif command == 0b10:
                for env in envs:
                    env.close()
                break
            else:
                raise ValueError(f"Unknown command: {command}")
    except KeyboardInterrupt:
        pass
    finally:
        futex_client_notify(signal_pointer)

if __name__ == "__main__":
    main()
