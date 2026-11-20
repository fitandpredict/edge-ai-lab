"""Replay every training snapshot of a run on the Mac: how far each 'generation' walks, plus a clip each.

usage: python sim/generations.py results/colab/small [--secs 6] [--video]
writes <run>/generations.json (+ <run>/clips/snap_NN.mp4 with --video)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    a = argparse.ArgumentParser()
    a.add_argument("run")
    a.add_argument("--secs", type=float, default=6.0)
    a.add_argument("--video", action="store_true")
    args = a.parse_args()
    run = Path(args.run)
    progress = [json.loads(l) for l in (run / "progress.jsonl").read_text().splitlines()]
    rows = []
    for snap in sorted(run.glob("snap_*.npz")):
        k = int(snap.stem.split("_")[1])
        steps = int(np.load(snap)["env_steps"])
        cmd = [sys.executable, str(HERE / "go1_loop.py"), "--brain", "npz", "--policy", str(snap),
               "--secs", str(args.secs), "--json", str(run / "clips" / f"{snap.stem}.json")]
        if args.video:
            cmd += ["--video", str(run / "clips" / f"{snap.stem}.mp4")]
        subprocess.run(cmd, check=True, capture_output=True)
        r = json.loads((run / "clips" / f"{snap.stem}.json").read_text())
        wall = next((p["wall_s"] for p in progress if p["steps"] == steps), None)
        rows.append({"snap": k, "env_steps": steps, "wall_s": wall, "distance_m": round(r["distance_m"], 2),
                     "falls": r["falls"], "vx_rmse": round(r["vx_rmse"], 3)})
        print(rows[-1], flush=True)
    (run / "generations.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
