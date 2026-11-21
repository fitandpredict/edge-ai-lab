"""The episode's experiments, each a set of drone_loop.py runs on one fixed course. Writes results/<exp>.json.

usage (from code/):  PY tools/sweep.py delay [--brain chip --port /dev/cu.usbmodem101]
                     PY tools/sweep.py rate
                     PY tools/sweep.py size
The course: take off from 0.3 m, hover at (0,0,1), fly to 3 waypoints, a 0.25 N side gust at 10 s, 14 s total.
Every run is repeated with 3 seeds (the brain is deterministic; the seed only matters for the 'random'
brain) - so one run per setting is enough, and the course is identical every time.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
PY = sys.executable
COURSE = ["--secs", "14", "--start", "0,0,0.3",
          "--targets", "0,0,1;1,0,1.5@3;1,1,1@6;0,0,1@9", "--gust", "0.25@10@90"]


def fly(extra):
    out = subprocess.run([PY, str(CODE / "sim" / "drone_loop.py"), *COURSE, *extra],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out[out.index("{"):])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("exp", choices=["delay", "rate", "size"])
    a.add_argument("--brain", default="npz")
    a.add_argument("--port")
    a.add_argument("--policy", default=str(CODE / "sim" / "brains" / "main.npz"))
    a.add_argument("--tag", help="results/<exp>[_<tag>].json (default: the policy's run name unless it's main)")
    args = a.parse_args()
    brain = ["--brain", args.brain] + (["--port", args.port] if args.port else [])
    if args.brain == "chip":   # Chip can't tell the Mac its input size: read it from the npz of the same brain
        import numpy as np
        brain += ["--hist", str((np.load(args.policy)["W0"].shape[0] - 15) // 4)]
    rows = []
    if args.exp == "delay":
        for ms in (0, 5, 10, 15, 20, 25, 30, 40, 60, 80):
            r = fly(brain + ["--policy", args.policy, "--delay-ms", str(ms)])
            rows.append(r)
            print(f"{ms:3d} ms extra: crashed={r['crashed']} at {r['crash_t']}, err {r['pos_err_mean_m']} m", flush=True)
    elif args.exp == "rate":
        for hz, name in ((200, "hz200"), (100, "main"), (50, "hz50"), (25, "hz25")):
            pol = CODE / "sim" / "brains" / f"{name}.npz"
            r = fly(["--policy", str(pol), "--hz", str(hz)])
            r["trained_at_hz"] = hz
            rows.append(r)
            print(f"{hz:3d} Hz: crashed={r['crashed']} at {r['crash_t']}, err {r['pos_err_mean_m']} m", flush=True)
            # and the main (100 Hz) brain forced to run at this rate, no retraining
            if hz != 100:
                r2 = fly(["--policy", args.policy, "--hz", str(hz)])
                r2["trained_at_hz"] = 100
                rows.append(r2)
                print(f"     100 Hz brain run at {hz} Hz: crashed={r2['crashed']} at {r2['crash_t']}, "
                      f"err {r2['pos_err_mean_m']} m", flush=True)
    else:
        for name in ("main", "tiny"):
            for int8 in (False, True):
                r = fly(["--policy", str(CODE / "sim" / "brains" / f"{name}.npz")] + (["--int8"] if int8 else []))
                r["name"] = name
                rows.append(r)
                print(f"{name} int8={int8}: crashed={r['crashed']}, err {r['pos_err_mean_m']} m", flush=True)
    run = Path(args.policy).parent.name
    tag = args.exp + (f"_{args.tag}" if args.tag else (f"_{run}" if run != "main" and args.exp == "delay" else "")) \
        + ("_chip" if args.brain == "chip" else "")
    (CODE / "results").mkdir(exist_ok=True)
    (CODE / "results" / f"{tag}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
