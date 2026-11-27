"""Build /content/esp-detection/datasets/coco_person from COCO 2017 (Ultralytics YOLO labels).

Keeps every image with at least one person (class 0, only those boxes) and adds a fixed number of
images with no people as negatives (empty label files), so the model learns "nobody here" too.
Also copies 200 val person images as int8 calibration data (to be swapped for my desk photos later).
"""
import os
import random
import shutil
from pathlib import Path

SRC = Path("/content/coco/coco")
DST = Path("/content/esp-detection/datasets/coco_person")
CALIB = Path("/content/esp-detection/deploy/person_calib")
NEG = {"train2017": 6000, "val2017": 300}
random.seed(0)


def build(split, out):
    imgs, negs = [], []
    for lab in sorted((SRC / "labels" / split).glob("*.txt")):
        lines = [l for l in lab.read_text().splitlines() if l.split() and l.split()[0] == "0"]
        (imgs if lines else negs).append((lab.stem, lines))
    # images with no label file at all are negatives too
    have = {s for s, _ in imgs} | {s for s, _ in negs}
    for p in (SRC / "images" / split).glob("*.jpg"):
        if p.stem not in have:
            negs.append((p.stem, []))
    random.shuffle(negs)
    chosen = imgs + negs[: NEG[split]]
    (DST / "images" / out).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / out).mkdir(parents=True, exist_ok=True)
    for stem, lines in chosen:
        src = SRC / "images" / split / f"{stem}.jpg"
        dst = DST / "images" / out / f"{stem}.jpg"
        if not dst.exists():
            os.link(src, dst)
        (DST / "labels" / out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{split}: {len(imgs)} with people, {min(len(negs), NEG[split])} negatives")
    return imgs


build("train2017", "train")
val = build("val2017", "val")

CALIB.mkdir(parents=True, exist_ok=True)
for stem, _ in random.sample(val, 200):
    shutil.copy(SRC / "images" / "val2017" / f"{stem}.jpg", CALIB / f"{stem}.jpg")

Path("/content/esp-detection/cfg/datasets/coco_person.yaml").write_text(
    f"path: {DST}\ntrain: images/train\nval: images/val\ntest:\n\nnames:\n  0: person\n"
)
print("calib images:", len(list(CALIB.glob("*.jpg"))))
