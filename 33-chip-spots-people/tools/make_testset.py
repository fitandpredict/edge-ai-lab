"""Turn the free Pexels clips in stock/clips/ into 640x480 frames shaped like Chip's camera, and label them.

  .venv-quant/bin/python tools/make_testset.py

- stock/frames/<clip>/NNNN.jpg  2 frames a second, centre-cropped to 4:3 and scaled to 640x480 (Chip's VGA)
- stock/test60.txt              the fixed 60-frame test set: 5 evenly spaced frames from each of the 12 clips
- stock/labels.json             reference person boxes from YOLO11x (float, 640, on the Mac), conf >= 0.5
The reference is a big model, not a human: spot-check stock/labels_check/*.jpg.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOCK = ROOT / "stock"


def main():
    frames = STOCK / "frames"
    for clip in sorted((STOCK / "clips").glob("*.mp4")):
        out = frames / clip.stem
        if out.exists() and any(out.iterdir()):
            continue
        out.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
                        "fps=2,crop=ih*4/3:ih,scale=640:480", "-q:v", "3", str(out / "%04d.jpg")], check=True)
    test = []
    for d in sorted(p for p in frames.iterdir() if p.is_dir()):
        fs = sorted(d.glob("*.jpg"))
        test += [fs[round(i * (len(fs) - 1) / 4)] for i in range(5)]
    (STOCK / "test60.txt").write_text("\n".join(str(p.relative_to(STOCK)) for p in test) + "\n")

    from ultralytics import YOLO
    from PIL import Image, ImageDraw
    m = YOLO(str(ROOT / "weights" / "yolo11x.pt"))
    allf = sorted(frames.glob("*/*.jpg"))
    labels = {}
    for i in range(0, len(allf), 16):
        batch = allf[i:i + 16]
        for p, r in zip(batch, m.predict([str(x) for x in batch], imgsz=640, conf=0.5, classes=[0], verbose=False)):
            labels[str(p.relative_to(STOCK))] = [[round(v) for v in b] for b in r.boxes.xyxy.tolist()]
    (STOCK / "labels.json").write_text(json.dumps(labels, indent=0))
    chk = STOCK / "labels_check"
    chk.mkdir(exist_ok=True)
    for rel in (STOCK / "test60.txt").read_text().split():
        im = Image.open(STOCK / rel).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in labels[rel]:
            d.rectangle(b, outline=(0, 255, 0), width=3)
        im.save(chk / rel.replace("/", "_"))
    n = sum(1 for k in (STOCK / "test60.txt").read_text().split() if labels[k])
    print(f"{len(allf)} frames, {len(labels)} labelled; test60 has {n} frames with a person, {60 - n} without")


if __name__ == "__main__":
    main()
