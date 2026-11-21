#!/usr/bin/env bash
# The "practise with lag" brains: random 0-40 ms control delay every episode.
set -uo pipefail
cd /content
export LD_LIBRARY_PATH=/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/content/venv-mjp/bin/python
$PY train_drone.py --name lag40    --hidden 64 64 --max-delay-ms 40 --hist 6 --steps 200e6 --evals 21
$PY train_drone.py --name lag40_h3 --hidden 64 64 --max-delay-ms 40 --hist 3 --steps 200e6 --evals 11
echo ALL_DONE
