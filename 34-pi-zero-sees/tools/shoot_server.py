#!/usr/bin/env python3
"""The hallucination-test booth: Chip's live view + a Take photo button + both models' captions, side by side.

    python3 shoot_server.py            # http://localhost:8766

GET  /                  the page (preview.html)
POST /shoot             {"truth": "...", "trick": "..."} -> Chip takes a VGA photo, both models caption it
                        (one sentence + list every object), saved to photos/test/ and results.jsonl
POST /truth             {"id": "03", "truth": "..."} -> correct the truth of an earlier photo
POST /delete            {"id": "03"} -> drop a photo (moved to photos/test/deleted/, not erased)
GET  /results           every shot so far
GET  /photos/<id>.jpg   a saved photo
"""
import json, pathlib, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import shoot  # noqa: E402  (capture + caption helpers)

HERE = pathlib.Path(__file__).parent
RESULTS = shoot.OUT / "results.jsonl"
LOCK = threading.Lock()


def load():
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def save_all(rows):
    RESULTS.write_text("".join(json.dumps(r) + "\n" for r in rows))


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/preview.html"):
            return self._send(200, (HERE / "preview.html").read_bytes(), "text/html")
        if self.path == "/results":
            return self._send(200, load())
        if self.path.startswith("/photos/"):
            f = shoot.OUT / pathlib.Path(self.path.split("?")[0]).name   # drop the ?v= cache-buster
            if f.exists():
                return self._send(200, f.read_bytes(), "image/jpeg")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
        if self.path == "/shoot":
            if not LOCK.acquire(blocking=False):
                return self._send(409, {"error": "still captioning the last photo"})
            try:
                rows = load()
                nid = f"{max([int(r['id']) for r in rows] or [0]) + 1:02d}"   # never reuses a kept photo's number
                rec = shoot.take(nid, body.get("truth", ""), body.get("trick", ""), body.get("size", "vga"))
                return self._send(200, rec)
            except Exception as e:  # Chip unreachable etc.: show it on the page
                return self._send(500, {"error": str(e)})
            finally:
                LOCK.release()
        if self.path == "/delete":   # moved to photos/test/deleted/, not erased, so a wrong click can be undone
            rows, keep = load(), []
            for r in rows:
                if r["id"] == body.get("id"):
                    d = shoot.OUT / "deleted"; d.mkdir(exist_ok=True)
                    src = shoot.OUT / r["image"]
                    if src.exists():
                        src.rename(d / f"{r['id']}_{r['t'].replace(':', '').replace(' ', '_')}.jpg")
                    with open(d / "deleted.jsonl", "a") as f:
                        f.write(json.dumps(r) + "\n")
                else:
                    keep.append(r)
            save_all(keep)
            return self._send(200, {"ok": True, "left": len(keep)})
        if self.path == "/truth":
            rows = load()
            for r in rows:
                if r["id"] == body.get("id"):
                    r["truth"] = body.get("truth", r["truth"])
            save_all(rows)
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    print(f"booth on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
