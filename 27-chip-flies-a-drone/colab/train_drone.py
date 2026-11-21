"""Train the drone's brain with Brax PPO on the Colab GPU (DroneHover in drone_env.py).

usage (on the VM, in /content/venv-mjp, with drone_env.py and drone_model.py next to it):
  python train_drone.py --name main --hidden 64 64                 # the brain for Chip, 100 Hz
  python train_drone.py --name hz25 --hidden 64 64 --hz 25         # the loop-rate experiment
  python train_drone.py --name smoke --steps 2e6 --envs 1024 --evals 3

Writes /content/runs/<name>/: progress.jsonl, snap_<k>.npz (one per eval, for the generations montage),
final.npz (our plain-MLP format: obs_mean/std, W0,b0,..., swish hidden, tanh head), summary.json.
"""

import argparse
import functools
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _device_put_replicated(x, devices):
    """JAX 0.12 removed this; brax 0.14.2 still calls it (same shim as episode 26)."""
    return jax.tree.map(lambda a: jax.device_put(jnp.broadcast_to(a, (len(devices),) + jnp.shape(a))), x)


jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

import drone_env


def flat_policy(params):
    norm, pol = params[0], params[1]
    mean, std = norm.mean, norm.std
    if isinstance(mean, dict):
        mean, std = mean["state"], std["state"]
    layers = pol["params"]
    names = sorted(layers, key=lambda n: int(n.split("_")[-1]))
    out = {"obs_mean": np.asarray(mean, np.float32), "obs_std": np.asarray(std, np.float32),
           "n_layers": len(names), "activation": "swish", "head": "tanh"}
    for i, n in enumerate(names):
        W = np.asarray(layers[n]["kernel"], np.float32)
        b = np.asarray(layers[n]["bias"], np.float32)
        if i == len(names) - 1:
            W, b = W[:, : W.shape[1] // 2], b[: b.shape[0] // 2]
        out[f"W{i}"], out[f"b{i}"] = W, b
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--name", required=True)
    a.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    a.add_argument("--hz", type=float, default=100.0)
    a.add_argument("--steps", type=float, default=150e6)
    a.add_argument("--envs", type=int, default=8192)
    a.add_argument("--evals", type=int, default=21)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--min-delay-ms", type=float, default=0.0)
    a.add_argument("--max-delay-ms", type=float, default=4.0)
    a.add_argument("--hist", type=int, default=3, help="past actions in the observation")
    a.add_argument("--no-gust", action="store_true")
    args = a.parse_args()

    out = Path("/content/runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    overrides = {"ctrl_dt": 1.0 / args.hz, "min_delay_ms": args.min_delay_ms,
                 "max_delay_ms": args.max_delay_ms, "hist": args.hist}
    if args.no_gust:
        overrides["gust_n"] = 0.0
    cfg = drone_env.default_config()
    env = drone_env.DroneHover(cfg, overrides)
    eval_env = drone_env.DroneHover(drone_env.default_config(), overrides)
    episode_length = int(round(cfg.episode_s * args.hz))
    # same wall-clock horizon at every rate: gamma per step from a ~1 s half-life... keep 0.99 at 100 Hz
    gamma = float(0.99 ** (100.0 / args.hz))

    t0 = time.monotonic()
    log = (out / "progress.jsonl").open("w")
    marks, last = [], {"steps": 0}

    def progress(num_steps, metrics):
        now = time.monotonic() - t0
        marks.append(now)
        last["steps"] = int(num_steps)
        row = {"steps": int(num_steps), "wall_s": round(now, 1),
               "reward": float(metrics.get("eval/episode_reward", float("nan")))}
        for k in ("dist", "tilt", "crash"):
            key = f"eval/episode_{k}"
            if key in metrics:
                row[k] = float(metrics[key])
        row["ep_len"] = float(metrics.get("eval/avg_episode_length", float("nan")))
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    snap = {"k": 0}

    def policy_params_fn(current_step, make_policy, params):
        d = flat_policy(params)
        d["env_steps"] = int(current_step)
        np.savez(out / f"snap_{snap['k']:02d}.npz", **d)
        snap["k"] += 1

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=int(args.steps), num_evals=args.evals, episode_length=episode_length,
        reward_scaling=0.1, normalize_observations=True, action_repeat=1,
        unroll_length=20, num_minibatches=32, num_updates_per_batch=4, discounting=gamma,
        learning_rate=3e-4, entropy_cost=1e-3, num_envs=args.envs, batch_size=min(1024, args.envs),
        max_grad_norm=1.0, num_eval_envs=256, clipping_epsilon=0.2,
        network_factory=functools.partial(ppo_networks.make_ppo_networks,
                                          policy_hidden_layer_sizes=tuple(args.hidden),
                                          value_hidden_layer_sizes=(256, 256)),
        seed=args.seed,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
    )
    _, params, _ = train_fn(environment=env, eval_env=eval_env, progress_fn=progress,
                             policy_params_fn=policy_params_fn)
    total = time.monotonic() - t0
    fp = flat_policy(params)
    np.savez(out / "final.npz", **fp)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    jit = marks[0] if marks else None
    train_s = total - (jit or 0)
    summary = {"name": args.name, "hidden": args.hidden, "hz": args.hz, "num_envs": args.envs,
               "num_timesteps": int(args.steps), "episode_length": episode_length, "gamma": gamma,
               "min_delay_ms": args.min_delay_ms, "max_delay_ms": args.max_delay_ms, "hist": args.hist, "gusts": not args.no_gust,
               "first_eval_s (mostly JIT)": jit, "train_s_after_first_eval": round(train_s, 1),
               "total_s": round(total, 1), "env_steps_done": last["steps"],
               "env_steps_per_s": round(last["steps"] / max(train_s, 1e-6)),
               "gpu": gpu, "jax": jax.__version__,
               "params_policy": int(sum(v.size for k, v in fp.items() if k[0] in "Wb" and k[1:].isdigit()))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
