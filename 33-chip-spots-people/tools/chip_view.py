"""Read Chip's detect_cam serial stream: log every frame's timings, and show the live view in a browser.

  .venv-quant/bin/python tools/chip_view.py [--port /dev/cu.usbmodem101] [--model N] [--run NAME]

- results/<run>.jsonl   one line per frame: model, capture/pre/model/post/detect ms, people, boxes
- results/<run>/NNNN.jpg  each frame Chip sent, with its boxes drawn (for the edit); raw/NNNN.jpg without
- http://localhost:8765  the latest frame + boxes + a big fps / seconds-per-frame counter (screen-record this)
Keys typed in this terminal are forwarded to Chip ('0', '1', '2'.. models, 'p' preview).
"""
import argparse
import http.server
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

import serial
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]

state = {"jpg": None, "last": None, "models": {}, "lock": threading.Lock()}

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Chip sees</title><style>
body{margin:0;background:#0b0f14;color:#e8eef5;font:600 22px/1.3 -apple-system,Helvetica,sans-serif;
display:flex;gap:28px;padding:28px;align-items:flex-start}
img{width:960px;height:720px;object-fit:contain;background:#000;border-radius:10px}
.k{font-size:15px;color:#8aa0b5;text-transform:uppercase;letter-spacing:.08em;margin-top:18px}
.big{font-size:84px;font-weight:800;color:#ffd23f;font-variant-numeric:tabular-nums}
.v{font-variant-numeric:tabular-nums}
</style></head><body><img id=f src=/frame.jpg><div style="min-width:330px">
<div class=k>model</div><div id=m class=v>-</div>
<div class=k id=bl>per frame</div><div id=b class="big">-</div>
<div class=k>camera &middot; prepare &middot; network &middot; boxes (ms)</div><div id=s class=v>-</div>
<div class=k>people found</div><div id=p class=v>-</div></div>
<script>
async function tick(){try{const r=await fetch('/state.json');const j=await r.json();
if(j.frame!==window.lf){document.getElementById('f').src='/frame.jpg?'+j.frame;window.lf=j.frame}
document.getElementById('m').textContent=j.model_name||'-';
const ms=j.detect_ms; const b=document.getElementById('b');
if(ms>=1000){b.textContent=(ms/1000).toFixed(1)+' s';document.getElementById('bl').textContent='seconds per frame'}
else{b.textContent=(1000/(ms+j.capture_ms)).toFixed(1)+' fps';document.getElementById('bl').textContent='frames per second'}
document.getElementById('s').textContent=[j.capture_ms,j.pre_ms,j.model_ms,j.post_ms].map(x=>x==null?'-':Math.round(x)).join(' \\u00b7 ');
document.getElementById('p').textContent=j.people;}catch(e){}setTimeout(tick,150)}tick();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with state["lock"]:
            jpg, last = state["jpg"], state["last"]
        if self.path.startswith("/frame.jpg") and jpg:
            body, ctype = jpg, "image/jpeg"
        elif self.path.startswith("/state.json"):
            body, ctype = json.dumps(last or {}).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_boxes(s):
    out = []
    for part in filter(None, s.split(";")):
        cat, score, xy = part.split(":")
        out.append({"cat": int(cat), "score": float(score), "box": [int(v) for v in xy.split(",")]})
    return out


def draw(jpg, rec):
    im = Image.open(io.BytesIO(jpg)).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    one_class = rec["model"] >= 2
    for b in rec["boxes"]:
        x1, y1, x2, y2 = b["box"]
        name = rec.get("class_name", "person") if one_class else COCO_NAMES[b["cat"]] if b["cat"] < 80 else str(b["cat"])
        col = (255, 210, 63) if name == "person" else (120, 200, 255)
        d.rectangle([x1, y1, x2, y2], outline=col, width=4)
        label = f"{name} {b['score']:.2f}"
        tw = d.textlength(label, font=font)
        d.rectangle([x1, max(0, y1 - 24), x1 + tw + 10, max(24, y1)], fill=col)
        d.text((x1 + 5, max(0, y1 - 23)), label, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem1101")
    ap.add_argument("--model", type=int, default=None, help="switch Chip to this model id at start")
    ap.add_argument("--run", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = run until Ctrl-C)")
    ap.add_argument("--http", type=int, default=8765)
    a = ap.parse_args()

    res = ROOT / "results"
    (res / a.run).mkdir(parents=True, exist_ok=True)
    log = open(res / f"{a.run}.jsonl", "a")
    raw = open(res / f"{a.run}.serial.log", "ab")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.http), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"live view: http://localhost:{a.http}   log: {log.name}", flush=True)

    ser = serial.Serial(a.port, 115200, timeout=0.2)
    ser.dtr, ser.rts = False, True  # reset Chip so we see its MODELS line from boot
    time.sleep(0.1)
    ser.rts = False
    if a.model is not None:
        t_end = time.time() + 8  # wait for boot + the default model to load, then switch
        boot = b""
        while time.time() < t_end and b"MODEL id=" not in boot:
            boot += ser.read(4096)
        buf_boot = boot
        ser.write(str(a.model).encode())
    else:
        buf_boot = b""

    def keys():
        for line in sys.stdin:
            ser.write(line.strip().encode())
    threading.Thread(target=keys, daemon=True).start()

    buf, lat, pending, n = buf_boot, {}, None, 0
    while True:
        buf += ser.read(4096)
        while True:
            if buf.startswith(b"\n@@JPG ") or buf.startswith(b"@@JPG "):
                head_end = buf.find(b"\n", 2)
                if head_end < 0:
                    break
                _, ln, fr = buf[:head_end].strip().split(b" ")
                ln = int(ln)
                if len(buf) < head_end + 1 + ln:
                    break
                jpg, buf = buf[head_end + 1: head_end + 1 + ln], buf[head_end + 1 + ln:]
                (res / a.run / "raw").mkdir(exist_ok=True)
                (res / a.run / "raw" / f"{int(fr):05d}.jpg").write_bytes(jpg)  # what the camera saw, no boxes
                if pending and pending["frame"] == int(fr):
                    drawn = draw(jpg, pending)
                    (res / a.run / f"{pending['frame']:05d}.jpg").write_bytes(drawn)
                    with state["lock"]:
                        state["jpg"] = drawn
                continue
            nl = buf.find(b"\n")
            if nl < 0:
                break
            j = buf.find(b"@@JPG ")
            if 0 <= j < nl:  # binary marker before the newline: drop the text before it
                buf = buf[j:]
                continue
            line, buf = buf[:nl].decode(errors="replace").strip(), buf[nl + 1:]
            raw.write((line + "\n").encode())
            if m := re.search(r"detect: (pre|model|post): (\d+) us", line):
                lat[m.group(1)] = int(m.group(2)) / 1000
            elif line.startswith("MODELS"):
                state["models"] = dict(kv.split("=", 1) for kv in line.split()[1:])
                print(line, flush=True)
            elif line.startswith("MODEL "):
                print(line, flush=True)
            elif line.startswith("RUNG"):
                kv = dict(p.split("=", 1) for p in line.split()[1:])
                rec = {"t": time.time(), "model": int(kv["model"]), "frame": int(kv["frame"]), "img": kv["img"],
                       "capture_ms": float(kv["capture_ms"]), "detect_ms": float(kv["detect_ms"]),
                       "pre_ms": float(kv["pre_ms"]), "model_ms": float(kv["model_ms"]),
                       "post_ms": float(kv["post_ms"]),
                       "people": int(kv["people"]), "boxes": parse_boxes(kv.get("boxes", ""))}
                rec["model_name"] = state["models"].get(str(rec["model"]), str(rec["model"]))
                rec["class_name"] = "cat" if "cat" in rec["model_name"] else "person"
                lat = {}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                with state["lock"]:
                    state["last"] = rec
                pending = rec
                n += 1
                print(f"[{rec['model_name']}] frame {rec['frame']}: detect {rec['detect_ms']:.0f} ms "
                      f"(pre {rec['pre_ms']} / net {rec['model_ms']} / post {rec['post_ms']}), "
                      f"capture {rec['capture_ms']:.0f} ms, people {rec['people']}", flush=True)
                if a.frames and n >= a.frames and rec["model"] == (a.model if a.model is not None else rec["model"]):
                    return
            elif "E (" in line or "Guru" in line or "abort" in line:
                print("CHIP:", line, flush=True)


if __name__ == "__main__":
    main()
