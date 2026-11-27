"""Train espdet_pico to find people, then stop: ONNX export and int8 .espdl happen on the Mac
(tools/quantize.py), since Colab's Python 3.13 lacks onnxscript and esp-ppq.

Runs on the Colab VM from /content/esp-detection:
  cd /content/esp-detection && nohup python /content/train_person.py 224 224 25 > /content/train_224.log 2>&1 &
Args: H W EPOCHS. Fine-tunes from Espressif's cat weights at the same size (a detector that already
knows edges and shapes), since training from scratch is 1200 epochs in their defaults.
"""
import json
import sys
import time

sys.path.insert(0, "/content/esp-detection")
from train import Train  # noqa: E402

h, w, epochs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
size = [h, w]
pre = f"examples/cat_detection/espdet_pico_{h}_{w}_cat.pt"
t0 = time.time()
fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0  # e.g. 0.05 for a smoke test
tag = f"person_{h}_{w}" + ("" if fraction == 1.0 else f"_f{fraction}")
kw = dict(epochs=epochs, device=0, batch=256, workers=2, cache="disk", close_mosaic=min(10, epochs),
          fraction=fraction, name=tag, exist_ok=True)
if h != w:
    kw["rect"] = True
res = Train(pre, "cfg/datasets/coco_person.yaml", size if h != w else h, **kw)
train_s = time.time() - t0

best = f"{res.save_dir}/weights/best.pt"  # ONNX export + int8 happen on the Mac (tools/quantize.py)

summary = {"size": size, "epochs": epochs, "train_seconds": round(train_s), "best_pt": best,
           "metrics": {k: float(v) for k, v in (res.results_dict or {}).items()}}
print(json.dumps(summary, indent=2))
open(f"/content/summary_{tag}.json", "w").write(json.dumps(summary, indent=2))
