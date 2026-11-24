"""Talk to Chip's maze brain over USB serial.

  python sim/chip.py info                     # params, sizes, free RAM
  python sim/chip.py bench                    # microseconds per decision in every mode
  python sim/chip.py check --mazes 20         # Chip vs the numpy twin on unseen 7x7 mazes (same actions?)
"""

import argparse
import glob
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
OUT, IN = b"\xaa\x55", b"\x55\xaa"


class ChipBrain:
    def __init__(self, port=None, mode=None):
        port = port or sorted(glob.glob("/dev/cu.usbmodem*"))[0]
        self.s = serial.Serial(port, 921600, timeout=2)
        time.sleep(0.3)
        self.s.reset_input_buffer()
        if mode is not None:
            self.command(str(mode))
        self.us = []

    def command(self, c, lines=1):
        self.s.reset_input_buffer()
        self.s.write(c.encode())
        self.s.timeout = 20
        try:
            return [json.loads(self.s.readline()) for _ in range(lines)]
        finally:
            self.s.timeout = 2

    def reset(self):
        self.command("R")

    def __call__(self, obs):
        self.s.write(OUT + np.asarray(obs, "<f4").tobytes())
        buf = self.s.read(2 + 16 + 1 + 4)
        if len(buf) != 23 or buf[:2] != IN:
            raise IOError(f"bad reply from Chip: {buf!r}")
        logits = np.frombuffer(buf[2:18], "<f4")
        action, us = buf[18], struct.unpack("<I", buf[19:23])[0]
        self.us.append(us)
        return logits, action


def check(a):
    from maze import Maze, MazeEnv
    from export_brain import NumpyBrain, load
    layers, _ = load(a.model)
    chip = ChipBrain(a.port, a.mode)
    twin = NumpyBrain(layers, int8=(a.mode == 4))
    same = n = solved = 0
    worst = 0.0
    rtt = []
    for k in range(a.mazes):
        env = MazeEnv(sizes=(7,), seed=10_000 + k, noise=0.0, visits="novis" not in Path(a.model).name)
        obs, _ = env.reset()
        chip.reset()
        twin.reset()
        done = trunc = False
        while not (done or trunc):
            t = time.perf_counter()
            logits, act = chip(obs)
            rtt.append((time.perf_counter() - t) * 1e3)
            ref = twin(obs)
            worst = max(worst, float(np.abs(logits - ref).max()))
            same += int(act == int(np.argmax(ref)))
            n += 1
            obs, _, done, trunc, _ = env.step(act)   # Chip drives
        solved += int(done)
        if a.verbose:
            print(f"maze {k + 1:2d}/{a.mazes}  {'solved' if done else 'FAILED'}  same as Mac: {same}/{n}  Chip {np.mean(chip.us) / 1000:.2f} ms", flush=True)
    res = dict(mode=a.mode, mazes=a.mazes, solved=solved, decisions=n, same_action_as_mac=same,
               max_logit_diff=round(worst, 6), chip_us_mean=round(float(np.mean(chip.us)), 1),
               round_trip_ms_mean=round(float(np.mean(rtt)), 2))
    print(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["info", "bench", "check"])
    ap.add_argument("--port")
    ap.add_argument("--mode", type=int, default=None)
    ap.add_argument("--model", default=str(ROOT / "models" / "lstm_final.zip"))
    ap.add_argument("--mazes", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--verbose", action="store_true", help="one line per maze")
    a = ap.parse_args()
    if a.what == "info":
        res = ChipBrain(a.port).command("I")
    elif a.what == "bench":
        c = ChipBrain(a.port)   # stream each mode's line as Chip finishes it
        c.s.reset_input_buffer(); c.s.write(b"B"); c.s.timeout = 20
        res = []
        for _ in range(5):
            r = json.loads(c.s.readline())
            res.append(r)
            ms = f"{r['mean_us'] / 1000:6.2f} ms" if "mean_us" in r else "  doesn't fit"
            print(f"{r['name']:<52s}{ms}", flush=True)
    else:
        res = check(a)
    for r in (res if isinstance(res, list) else []):
        if a.what != "bench":
            print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
