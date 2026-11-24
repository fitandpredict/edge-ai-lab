"""Train the maze brain on random 5x5 mazes.

  python sim/train.py --brain lstm --steps 3000000      # RecurrentPPO, 128-unit LSTM (the real one)
  python sim/train.py --brain mlp  --steps 3000000      # same inputs, no memory (the ablation)

Writes results/train_<brain>.csv (steps, solve rate, mean decisions, bumps) for the reward-curve shot,
and checkpoints to models/<brain>_<steps>.zip so the video can show early vs late brains.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from maze import MazeEnv  # noqa: E402

ROOT = HERE.parent


class Log(BaseCallback):
    def __init__(self, path, every):
        super().__init__()
        self.path, self.every, self.next = path, every, every
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["steps", "episodes", "solve_rate", "decisions", "bumps", "secs"])
        self.buf, self.t0 = [], time.time()

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                ep = info.get("episode")
                self.buf.append((not info.get("TimeLimit.truncated", False), info["bumps"], ep["l"] if ep else 0))
        if self.num_timesteps >= self.next and self.buf:
            b = np.array(self.buf, dtype=float)
            row = [self.num_timesteps, len(b), b[:, 0].mean(), b[:, 2].mean(), b[:, 1].mean(), round(time.time() - self.t0)]
            self.w.writerow(row)
            self.f.flush()
            print(*[f"{v:.3f}" if isinstance(v, float) else v for v in row], flush=True)
            self.buf, self.next = [], self.next + self.every
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["lstm", "mlp"], default="lstm")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-visits", action="store_true", help="hide the visited-cell memory from the brain")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    name = a.brain + ("_novis" if a.no_visits else "")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    from stable_baselines3.common.monitor import Monitor
    env = make_vec_env(lambda: Monitor(MazeEnv(sizes=(5,), visits=not a.no_visits), info_keywords=("bumps",)), n_envs=a.envs, seed=a.seed)
    common = dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, n_epochs=4, verbose=0, seed=a.seed)
    if a.brain == "lstm":
        model = RecurrentPPO("MlpLstmPolicy", env, n_steps=128, batch_size=512,
                             policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, net_arch=dict(pi=[64], vf=[64])),
                             **common)
    else:
        model = PPO("MlpPolicy", env, n_steps=128, batch_size=512,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64])), **common)
    every = 50_000
    cb = Log(ROOT / "results" / f"train_{name}.csv", every)

    class Ckpt(BaseCallback):
        marks = [100_000, 300_000, 1_000_000]

        def _on_step(self):
            while self.marks and self.num_timesteps >= self.marks[0]:
                self.model.save(ROOT / "models" / f"{name}_{self.marks.pop(0)}")
            return True

    model.learn(a.steps, callback=[cb, Ckpt()])
    model.save(ROOT / "models" / f"{name}_final")


if __name__ == "__main__":
    main()
