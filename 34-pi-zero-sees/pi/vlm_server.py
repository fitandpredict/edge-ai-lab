#!/usr/bin/env python3
"""The Pi Zero's half of episode 34: Chip POSTs a JPEG, the Pi describes it, fully offline.

    POST /photo          body = JPEG from chip_eye (BOOT button) -> describes it, prints the text live
    GET  /shoot?size=vga pulls a photo from http://chip.local/capture and describes it

Each run is a fresh llama-mtmd-cli process reaped with os.wait4 (the Pi has no GNU time), so every caption
logs its own peak RSS, major page faults, swap and wall time to runs.jsonl next to the saved photo. Standard library only.
"""
import json, os, re, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOME = os.path.expanduser("~/vlm")
BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
MODEL = os.environ.get("VLM_MODEL", f"{HOME}/models/SmolVLM-256M-Instruct-Q8_0.gguf")
MMPROJ = os.environ.get("VLM_MMPROJ", f"{HOME}/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
PROMPT = os.environ.get("VLM_PROMPT", "Describe this image in one sentence.")
CTX, NPRED, THREADS = os.environ.get("VLM_CTX", "512"), os.environ.get("VLM_N", "40"), os.environ.get("VLM_T", "4")
LOCK = threading.Lock()  # one caption at a time: two would not fit in 416 MB


def swap_used_kb():
    m = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    return int(m["SwapTotal"].split()[0]) - int(m["SwapFree"].split()[0])


def describe(jpg_path, source):
    cmd = [BIN, "-m", MODEL, "--mmproj", MMPROJ, "--image", jpg_path, "-p", PROMPT,
           "-c", CTX, "-n", NPRED, "-t", THREADS, "--temp", "0"]
    with LOCK:
        swap0, t0 = swap_used_kb(), time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        first, words = None, []
        for chunk in iter(lambda: p.stdout.read(1), ""):
            if first is None and chunk.strip():
                first = time.time() - t0
            words.append(chunk)
            sys.stdout.write(chunk); sys.stdout.flush()   # the terminal types as it thinks (on camera)
        err = p.stderr.read()
        _, status, ru = os.wait4(p.pid, 0)        # this child's own rusage: ru_maxrss is in kB on Linux
        p.returncode = os.waitstatus_to_exitcode(status)
        wall = time.time() - t0
    enc = [int(x) for x in re.findall(r"encoding done in (\d+) ms", err)]
    run = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "image": os.path.basename(jpg_path),
           "model": os.path.basename(MODEL), "ctx": int(CTX), "threads": int(THREADS), "rc": p.returncode,
           "text": "".join(words).strip(), "wall_s": round(wall, 2),
           "first_token_s": round(first, 2) if first else None, "image_encode_ms": enc,
           "peak_rss_mb": round(ru.ru_maxrss / 1024, 1), "major_faults": ru.ru_majflt,
           "swap_delta_mb": round((swap_used_kb() - swap0) / 1024, 1),
           "killed": p.returncode == -9}
    with open(f"{HOME}/runs.jsonl", "a") as f:
        f.write(json.dumps(run) + "\n")
    with open(f"{HOME}/last_stderr.log", "w") as f:
        f.write(err)
    print(f"\n[{run['wall_s']} s, first word {run['first_token_s']} s, peak {run['peak_rss_mb']} MB, rc {p.returncode}]")
    return run


def save(data, tag):
    os.makedirs(f"{HOME}/img", exist_ok=True)
    path = f"{HOME}/img/{time.strftime('%Y%m%d_%H%M%S')}_{tag}.jpg"
    with open(path, "wb") as f:
        f.write(data)
    return path


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        if self.path != "/photo":
            return self._json(404, {"error": "not found"})
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        path = save(data, "button")
        self._json(202, {"saved": os.path.basename(path), "bytes": len(data)})  # Chip's LED goes green now
        threading.Thread(target=describe, args=(path, "button"), daemon=True).start()

    def do_GET(self):
        if not self.path.startswith("/shoot"):
            return self._json(404, {"error": "not found"})
        size = self.path.split("size=")[1] if "size=" in self.path else "vga"
        data = urllib.request.urlopen(f"http://chip.local/capture?size={size}", timeout=10).read()
        self._json(200, describe(save(data, size), f"pull:{size}"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:                      # one-off: python3 vlm_server.py photo.jpg
        describe(sys.argv[1], "file")
    else:
        print(f"listening on :8090, model {os.path.basename(MODEL)}, ctx {CTX}")
        ThreadingHTTPServer(("", 8090), H).serve_forever()
