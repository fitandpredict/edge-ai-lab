"""Check Chip's brain against the Mac's, then benchmark every mode.

usage: python sim/chip_check.py --port /dev/cu.usbmodem101 --npz sim/big.npz
  1. sends 200 recorded observations, compares Chip's 12 actions with numpy's (max abs error per mode)
  2. asks Chip to benchmark every mode ('B') and prints the JSON lines
writes results/chip_check_<name>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go1_loop import ChipBrain, NpzBrain  # noqa: E402

CODE = Path(__file__).resolve().parent.parent


def lines(ser, until_s=8.0):
    out, t = [], time.time()
    while time.time() - t < until_s:
        l = ser.readline().decode(errors="ignore").strip()
        if l.startswith("{"):
            out.append(json.loads(l))
            t = time.time()
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--port", default="/dev/cu.usbmodem101")
    a.add_argument("--npz", default=str(CODE / "sim" / "big.npz"))
    args = a.parse_args()
    ref = NpzBrain(args.npz)
    chip = ChipBrain(args.port)
    rng = np.random.default_rng(0)
    obs = (rng.normal(size=(200, 48)) * 0.5).astype(np.float32)
    want = np.stack([ref(o)[0] for o in obs])

    chip.ser.write(b"I")
    info = lines(chip.ser, 2)
    print("info:", info)
    errs = {}
    for m in range(5):
        chip.ser.write(str(m).encode())
        ack = lines(chip.ser, 1.5)
        if ack and "error" in ack[0]:
            errs[m] = ack[0]["error"]
            continue
        got = np.stack([chip(o)[0] for o in obs])
        errs[m] = float(np.abs(got - want).max())
        print(f"mode {m}: max |chip - mac| = {errs[m]:.2e}, chip µs median {np.median(chip.chip_us[-200:]):.0f}")
    chip.ser.write(b"2")
    lines(chip.ser, 1)
    chip.ser.write(b"B")
    bench = lines(chip.ser, 12)
    for b in bench:
        print(b)
    out = {"npz": args.npz, "info": info, "max_abs_err_by_mode": errs, "bench": bench}
    name = Path(args.npz).stem
    (CODE / "results").mkdir(exist_ok=True)
    (CODE / "results" / f"chip_check_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
