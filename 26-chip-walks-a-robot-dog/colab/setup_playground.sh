#!/usr/bin/env bash
# Set up MuJoCo Playground (RSS 2025 locomotion envs) with CUDA JAX on a Colab VM (run there, from /content):
#   colab upload -s ep26 setup_playground.sh /content/setup_playground.sh
#   colab ssh -s ep26 -- bash /content/setup_playground.sh
# Own venv (Python 3.12), never Colab's Python. Safe to re-run.
set -euo pipefail
cd /content

MJP_COMMIT=${MJP_COMMIT:-8bac7b2c703ff61846ced7b00693cedbd5dbd7d8}   # the commit trained with (20 Nov 2026)

command -v uv >/dev/null || pip install -q uv
if [ ! -d mujoco_playground ]; then
  git clone -q https://github.com/google-deepmind/mujoco_playground mujoco_playground
fi
git -C mujoco_playground checkout -q "$MJP_COMMIT"
echo "mujoco_playground commit: $(git -C mujoco_playground rev-parse HEAD)"

if [ ! -x venv-mjp/bin/python ]; then
  uv venv -q --python 3.12 venv-mjp
fi
export VIRTUAL_ENV=/content/venv-mjp PATH=/content/venv-mjp/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
uv pip install -q -e "mujoco_playground/rss2025"
# Colab's driver 580 is CUDA 13.0: the cuda12 plugin fails its version check and JAX falls back to CPU
uv pip uninstall -q jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true
# versions from the repo's uv.lock at that commit (jax 0.12.1 drops device_put_replicated, which brax 0.14.2 calls)
uv pip install -q "jax[cuda13]==0.12.0.1" "jaxlib==0.12.0.1" "brax==0.14.2" \
  "mujoco==3.16.0" "mujoco-mjx==3.16.0" "warp-lang==1.18.0"

python - <<'PY'
import jax
print("jax", jax.__version__, "backend", jax.default_backend(), jax.devices())
from mujoco_playground import registry
env = registry.load("Go1JoystickFlatTerrain")   # also fetches the Menagerie Go1 meshes
print("obs", env.observation_size, "act", env.action_size)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "playground ready"
