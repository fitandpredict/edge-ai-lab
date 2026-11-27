"""Score Chip's detections on the stock test frames against the YOLO11x reference labels.

  .venv-quant/bin/python tools/score.py results/stock_*.jsonl [--conf 0.25]

Per model: frame-level "did it notice a person" accuracy (the number for the video), plus box recall and
precision at IoU 0.5, and the median per-frame time on Chip.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0


def score(path, labels, conf):
    recs = [json.loads(l) for l in open(path)]
    right = tp = fp = fn = 0
    fp_frames = miss_frames = 0
    for r in recs:
        gt = labels[r["frame"]]
        pred = [b["box"] for b in r["boxes"] if b["cat"] == 0 and b["score"] >= conf]
        right += bool(gt) == bool(pred)
        fp_frames += bool(pred) and not gt
        miss_frames += bool(gt) and not pred
        used = set()
        for p in sorted(pred, key=lambda b: 0):
            best = max(((iou(p, g), j) for j, g in enumerate(gt) if j not in used), default=(0, -1))
            if best[0] >= 0.5:
                tp += 1
                used.add(best[1])
            else:
                fp += 1
        fn += len(gt) - len(used)
    return {"run": Path(path).stem, "model": recs[0]["model_name"], "frames": len(recs),
            "noticed_right": f"{right}/{len(recs)} ({100 * right / len(recs):.0f}%)",
            "false_alarm_frames": fp_frames, "missed_frames": miss_frames,
            "box_recall": round(tp / (tp + fn), 2) if tp + fn else None,
            "box_precision": round(tp / (tp + fp), 2) if tp + fp else None,
            "median_ms": round(statistics.median(r["detect_ms"] for r in recs), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    labels = json.loads((ROOT / "stock" / "labels.json").read_text())
    for p in a.runs:
        print(json.dumps(score(p, labels, a.conf)))
