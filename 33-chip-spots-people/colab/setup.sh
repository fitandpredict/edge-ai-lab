#!/usr/bin/env bash
# Runs on the Colab VM: esp-detection + a COCO-2017 person-only dataset in YOLO format.
# Usage (from the Mac): colab ssh -s spots -- bash < code/colab/setup.sh > logs/setup.log 2>&1
set -euo pipefail
cd /content

# 1. esp-detection at the commit we read on 26 Nov 2026
if [ ! -d esp-detection ]; then
  git clone https://github.com/espressif/esp-detection.git
  git -C esp-detection checkout 28de590
fi

# 2. Python deps. Colab's own torch is newer than the repo's pin (2.2.0, no wheels for this Python);
#    ultralytics 8.4.0: the repo says >=8.3.112, but its ESPDetect head passes reg_max to Detect, which only 8.4.x accepts (tested 26 Nov).
#    esp-ppq (the int8 quantizer) needs Python < 3.13 and Colab is on 3.13, so quantization runs on the Mac
#    (tools/quantize.py); Colab only trains and exports ONNX.
pip install -q "ultralytics==8.4.0" onnx onnxslim onnxruntime
python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ultralytics', ultralytics.__version__)"

# 3. COCO 2017 images + Ultralytics' YOLO-format labels
mkdir -p /content/coco && cd /content/coco
[ -f train2017.zip ] || wget -q http://images.cocodataset.org/zips/train2017.zip
[ -f val2017.zip ]   || wget -q http://images.cocodataset.org/zips/val2017.zip
[ -f coco2017labels.zip ] || wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
[ -d coco/images/train2017 ] || { unzip -q coco2017labels.zip; mkdir -p coco/images; unzip -q train2017.zip -d coco/images; unzip -q val2017.zip -d coco/images; }
ls coco/images/train2017 | wc -l

# 4. Person-only subset (class 0), plus some images with no people as negatives
python /content/make_person_subset.py
