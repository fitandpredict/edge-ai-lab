"""Record a real command's output with its real timing, for the terminal replays in the video.

  python tools/rec_term.py results/casts/bench.json -- .venv/bin/python sim/chip.py bench
Writes {"cmd": "...", "lines": [[seconds, "text"], ...]}. video/three/term.html types the command and replays the lines.
"""
import json, subprocess, sys, time
from pathlib import Path

out, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
shown = " ".join(["python" if c.endswith("python") else c for c in cmd])
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
lines = []
for line in p.stdout:
    lines.append([round(time.time() - t0, 3), line.rstrip("\n")])
    print(line, end="")
p.wait()
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(dict(cmd=shown, lines=lines, exit=p.returncode), indent=1))
