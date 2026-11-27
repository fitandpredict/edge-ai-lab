"""Play the stock test frames into Chip over USB (inject mode) and log what each model finds.

  .venv-quant/bin/python tools/feed_clips.py --model N [--set stock/test60.txt | --set all] [--run NAME]

Chip decodes each 640x480 JPEG and runs the same timed detector as on the camera. Output: results/<run>.jsonl,
one line per frame with the frame path, timings (capture_ms here = JPEG decode on Chip) and boxes.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parent.parent
STOCK = ROOT / "stock"


def wait_for(ser, token, timeout):
    buf, end = b"", time.time() + timeout
    while time.time() < end:
        buf += ser.read(4096)
        if token in buf:
            return buf
    raise TimeoutError(f"no {token!r} from Chip in {timeout}s; got {buf[-300:]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem1101")
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--set", default="stock/test60.txt")
    ap.add_argument("--run")
    ap.add_argument("--keep-cam", action="store_true", help="leave the camera running while injecting")
    a = ap.parse_args()
    if a.set == "all":
        frames = sorted(str(p.relative_to(STOCK)) for p in (STOCK / "frames").glob("*/*.jpg"))
    else:
        frames = (ROOT / a.set).read_text().split()
    ser = serial.Serial(a.port, 115200, timeout=0.2)
    ser.dtr, ser.rts = False, True
    time.sleep(0.1)
    ser.rts = False
    boot = wait_for(ser, b"MODEL id=", 15)
    models = dict(kv.split("=", 1) for kv in boot.split(b"MODELS ")[1].split(b"\n")[0].decode().split())
    name = models[str(a.model)]
    if a.model != 0:
        ser.write(str(a.model).encode())
        wait_for(ser, f"MODEL id={a.model}".encode(), 60)
    if a.keep_cam:
        ser.write(b"k")
    ser.write(b"i")
    wait_for(ser, b"INJECT on", 30)  # the camera frame in flight finishes first (26 s for YOLO11n 640)
    run = a.run or f"stock_{name}_{Path(a.set).stem}"
    (ROOT / "results").mkdir(exist_ok=True)
    out = open(ROOT / "results" / f"{run}.jsonl", "w")
    print(f"model {a.model} = {name}, {len(frames)} frames -> {out.name}", flush=True)
    for i, rel in enumerate(frames):
        jpg = (STOCK / rel).read_bytes()
        ser.write(b"J" + struct.pack("<I", len(jpg)) + jpg)
        buf = b""
        while b"RUNG" not in buf or not buf.split(b"RUNG", 1)[1].count(b"\n"):
            chunk = ser.read(4096)
            buf += chunk
            if b"INJECT error" in buf:
                sys.exit(f"Chip rejected {rel}")
        line = b"RUNG" + buf.split(b"RUNG", 1)[1].split(b"\n")[0]
        kv = dict(p.split("=", 1) for p in line.decode().split()[1:])
        boxes = []
        for part in filter(None, kv.get("boxes", "").split(";")):
            cat, score, xy = part.split(":")
            boxes.append({"cat": int(cat), "score": float(score), "box": [int(v) for v in xy.split(",")]})
        rec = {"frame": rel, "model": a.model, "model_name": name, "decode_ms": float(kv["capture_ms"]),
               **{k: float(kv[k]) for k in ("detect_ms", "pre_ms", "model_ms", "post_ms")},
               "boxes": boxes}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{i + 1}/{len(frames)}] {rel}: {rec['detect_ms']:.0f} ms, "
              f"{sum(b['cat'] == 0 for b in boxes)} person box(es)", flush=True)


if __name__ == "__main__":
    main()
