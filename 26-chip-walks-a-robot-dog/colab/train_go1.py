"""Train a Go1 walking policy with Brax PPO in MuJoCo Playground, on the Colab GPU.

usage (on the VM, inside /content/venv-mjp):
  python train_go1.py --name big   --hidden 512 256 128          # the published size
  python train_go1.py --name small --hidden 128 128              # sized for Chip
  python train_go1.py --name tiny  --hidden 32 32 --steps 100e6

Writes /content/runs/<name>/:
  progress.jsonl          one line per eval: env steps, wall seconds, eval reward
  snap_<k>.npz            the policy at every eval (k = 0..N), our plain-MLP format for go1_loop.py
  final.npz               the last one
  summary.json            JIT time, train time, steps/s, GPU name
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
    """JAX 0.12 removed this; brax 0.14.2 still calls it. Same result: a leading axis of len(devices)."""
    return jax.tree.map(lambda a: jax.device_put(jnp.broadcast_to(a, (len(devices),) + jnp.shape(a))), x)


jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

ENV = "Go1JoystickFlatTerrain"


def flat_policy(params):
    """Brax params (normaliser, policy, value) -> our npz dict."""
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
        if i == len(names) - 1:  # mean and log-std: keep the mean
            W, b = W[:, : W.shape[1] // 2], b[: b.shape[0] // 2]
        out[f"W{i}"], out[f"b{i}"] = W, b
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--name", required=True)
    a.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    a.add_argument("--steps", type=float, default=None, help="env steps (default: the tuned config, 200M)")
    a.add_argument("--evals", type=int, default=None)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--no-dr", action="store_true", help="turn off domain randomisation")
    args = a.parse_args()

    out = Path("/content/runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = registry.get_default_config(ENV)
    env = registry.load(ENV, config=env_cfg)
    eval_env = registry.load(ENV, config=registry.get_default_config(ENV))
    ppo_params = locomotion_params.brax_ppo_config(ENV)
    if args.steps:
        ppo_params.num_timesteps = int(args.steps)
    if args.evals:
        ppo_params.num_evals = args.evals
    ppo_params.network_factory.policy_hidden_layer_sizes = tuple(args.hidden)
    training_params = dict(ppo_params)
    nf = training_params.pop("network_factory")
    num_eval_envs = training_params.pop("num_eval_envs", 128)
    if not args.no_dr:
        training_params["randomization_fn"] = registry.get_domain_randomizer(ENV)

    t0 = time.monotonic()
    log = (out / "progress.jsonl").open("w")
    marks = []
    last = {"steps": 0}

    def progress(num_steps, metrics):
        now = time.monotonic() - t0
        marks.append(now)
        last["steps"] = int(num_steps)
        row = {"steps": int(num_steps), "wall_s": round(now, 1),
               "reward": float(metrics.get("eval/episode_reward", float("nan"))),
               "reward_std": float(metrics.get("eval/episode_reward_std", float("nan")))}
        for k in ("eval/episode_tracking_lin_vel", "eval/episode_termination"):
            if k in metrics:
                row[k.split("/")[-1]] = float(metrics[k])
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
        ppo.train, **training_params,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **nf),
        seed=args.seed, wrap_env_fn=wrapper.wrap_for_brax_training, num_eval_envs=num_eval_envs,
    )
    _, params, _ = train_fn(environment=env, eval_env=eval_env, progress_fn=progress,
                             policy_params_fn=policy_params_fn)
    total = time.monotonic() - t0
    np.savez(out / "final.npz", **flat_policy(params))

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    jit = marks[0] if marks else None
    train_s = total - (jit or 0)
    summary = {"name": args.name, "hidden": args.hidden, "env": ENV, "num_envs": training_params["num_envs"],
               "num_timesteps": training_params["num_timesteps"], "domain_randomization": not args.no_dr,
               "first_eval_s (mostly JIT)": jit, "train_s_after_first_eval": round(train_s, 1),
               "total_s": round(total, 1), "env_steps_done": last["steps"],
               "env_steps_per_s": round(last["steps"] / train_s),
               "gpu": gpu, "jax": jax.__version__, "params_policy": int(sum(
                   v.size for k, v in flat_policy(params).items() if k[0] in "Wb" and k[1:].isdigit()))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
