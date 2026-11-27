"""Turn an espdet_pico .pt (or its exported .onnx) into an int8 .espdl for the ESP32-S3.

  .venv-quant/bin/python tools/quantize.py <model.pt|model.onnx> <H> <W> <calib_dir> <out.espdl>

Uses Espressif's own export + esp-ppq quantizer from vendor/esp-detection, on the CPU.
"""
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "vendor" / "esp-detection"
sys.path.insert(0, str(REPO))

from deploy.export import Export  # noqa: E402
from deploy.quantize import quant_espdet  # noqa: E402

src, h, w, calib, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
size = [h, w]
t0 = time.time()
if src.endswith(".pt"):
    Export(src, size)
    onnx = src[:-3] + ".onnx"
else:
    onnx = src
t1 = time.time()
quant_espdet(onnx_path=onnx, target="esp32s3", num_of_bits=8, device="cpu", batchsz=32, imgsz=size,
             calib_dir=calib, espdl_model_path=out)
print(f"QUANT done: export {t1 - t0:.1f}s, quantize {time.time() - t1:.1f}s -> {out} ({Path(out).stat().st_size} B)")
