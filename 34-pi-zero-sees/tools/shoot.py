#!/usr/bin/env python3
"""One photo of the hallucination test: Chip takes it, both models caption it on the Mac, all logged.

    python3 shoot.py 05 "mug, keys" [--size vga] [--trick "half out of frame"]

Saves code/photos/test/<id>.jpg and appends to code/photos/test/results.jsonl: the truth you typed, and
each model's answer to both prompts (one sentence; list every object). Scoring (right / half / made up)
is done by hand afterwards in score.md, because "toy car" vs "camera board" needs a human.
"""
import argparse, json, pathlib, subprocess, time, urllib.request

CODE = pathlib.Path(__file__).resolve().parents[1]
OUT = CODE / "photos" / "test"
MODELS = {m: (CODE / f"models/SmolVLM-{m}-Instruct-Q8_0.gguf", CODE / f"models/mmproj-SmolVLM-{m}-Instruct-Q8_0.gguf")
          for m in ("256M", "500M")}
PROMPTS = {"sentence": "Describe this image in one sentence.", "list": "List every object you can see."}


def caption(model, prompt, img):
    m, mm = MODELS[model]
    t = time.time()
    r = subprocess.run(["llama-mtmd-cli", "-m", m, "--mmproj", mm, "--image", img, "-p", prompt, "-ngl", "0",
                        "--no-mmproj-offload", "-t", "4", "-c", "512", "-n", "60", "--temp", "0"],
                       capture_output=True, text=True)
    return " ".join(r.stdout.split()), round(time.time() - t, 2)


def take(nid, truth, trick="", size="vga"):
    """Chip takes the photo, both models caption it with both prompts; appended to results.jsonl."""
    OUT.mkdir(parents=True, exist_ok=True)
    img = OUT / f"{nid}.jpg"
    img.write_bytes(urllib.request.urlopen(f"http://chip.local/capture?size={size}", timeout=15).read())
    rec = {"id": nid, "truth": truth, "trick": trick, "size": size, "image": img.name,
           "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    for model in MODELS:
        for key, p in PROMPTS.items():
            rec[f"{model}_{key}"], rec[f"{model}_{key}_s"] = caption(model, p, str(img))
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id"); ap.add_argument("truth")
    ap.add_argument("--size", default="vga"); ap.add_argument("--trick", default="")
    a = ap.parse_args()
    rec = take(a.id, a.truth, a.trick, a.size)
    print(f"[{a.id}] truth: {a.truth}" + (f"  (trick: {a.trick})" if a.trick else ""))
    for k, v in rec.items():
        if k.startswith(("256M", "500M")) and not k.endswith("_s"):
            print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
