#!/usr/bin/env bash
# All episode-27 training runs back to back, detached from ssh:
#   colab ssh -s ep27 -- bash -lc 'nohup bash /content/run_all.sh > /content/runs/run_all.log 2>&1 &'
set -uo pipefail
cd /content
export LD_LIBRARY_PATH=/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/content/venv-mjp/bin/python
$PY train_drone.py --name main  --hidden 64 64 --hz 100 --steps 150e6 --evals 21
$PY train_drone.py --name tiny  --hidden 16 16 --hz 100 --steps 150e6 --evals 11
$PY train_drone.py --name hz50  --hidden 64 64 --hz 50  --steps 100e6 --evals 11
$PY train_drone.py --name hz25  --hidden 64 64 --hz 25  --steps 100e6 --evals 11
$PY train_drone.py --name hz200 --hidden 64 64 --hz 200 --steps 200e6 --evals 11
echo ALL_DONE
