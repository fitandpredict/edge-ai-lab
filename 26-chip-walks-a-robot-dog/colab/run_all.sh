#!/usr/bin/env bash
# Both episode-26 training runs back to back, detached from the ssh session:
#   colab ssh -s ep26 -- bash -lc 'nohup bash /content/run_all.sh > /content/runs/run_all.log 2>&1 &'
set -uo pipefail
cd /content
export LD_LIBRARY_PATH=/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/content/venv-mjp/bin/python
$PY train_go1.py --name small --hidden 128 128 --evals 20
$PY train_go1.py --name big --hidden 512 256 128 --evals 20
echo ALL_DONE
