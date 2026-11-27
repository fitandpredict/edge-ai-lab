# 33 · My $5 Chip Took 26 Seconds to Spot a Person. So I Made It 190× Faster

A speedrun ladder for person detection on an ESP32-S3 (N16R8) with its OV3660 camera, from Espressif's YOLO11n at
25.7 s per frame to a person-trained `espdet_pico` at 7.4 fps live, with every stage timed on the board and a
60-frame accuracy scoreboard at every rung.

## Results (from `RESULTS.md`)
| Rung | Per frame on the chip | Scoreboard (60 frames) |
|---|---|---|
| YOLO11n 640 (ESP-DL `coco_detect`, 80 classes) | 25.7 s (99.6% of it the network) | 60/60 |
| YOLO11n 320 | 6.1 s | 59/60 |
| `espdet_pico` cat model (Espressif's) | 136 ms | 29/60 (it's a cat finder) |
| `espdet_pico` trained on people (ours), 1 core | 5.2 fps live | 57/60 |
| + detector on core 1, camera at 320×240 | **7.4 fps live** | 59/60 |

- int8 calibration on 1 photo (the example folder) put a false CAT box on 22/22 frames of a dark ceiling (up to 0.90);
  on 200 photos, 4/21 (max 0.38). Same model, same speed.
- At 640×480 the OV3660 only hands over a frame every 180 ms, so the detector waited for the camera. An idle camera
  also slowed the network 8% (it keeps writing frames into PSRAM).

## Run it
**Firmware** (ESP-IDF v5.4.x, ESP-DL 3.3.8, an ESP32-S3 with 8 MB PSRAM and an OV2640/OV3660 on the ESP32-S3-EYE pins):
```bash
. ~/esp/esp-idf/export.sh
cd firmware/detect_cam
# put your espdet_pico .espdl files in components/espdet_models/models/ (see "Make the model" below), then:
idf.py reconfigure build flash
```
Keys over USB serial: `0` YOLO11n 640, `1` YOLO11n 320, `2`… your espdet models, `p` preview, `m` two-core mode,
`q` camera 640×480 / 320×240, `i` inject mode (frames from the laptop instead of the camera).

**Laptop tools** (Python 3.12):
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/espressif/esp-detection vendor/esp-detection && git -C vendor/esp-detection checkout 28de590
.venv/bin/python tools/chip_view.py --model 4          # live view at http://localhost:8765, every frame logged
.venv/bin/python tools/feed_clips.py --model 4          # send the 60 test frames to the chip
.venv/bin/python tools/score.py results/stock_*.jsonl   # the scoreboard
```
The test frames come from 12 free Pexels clips (`stock/SOURCES.md`): download them into `stock/clips/<id>.mp4`, put
`yolo11x.pt` in `weights/`, and run `tools/make_testset.py` (writes the frames and the reference labels).

**Make the model**: train on Colab (`colab/setup.sh`, then `colab/train_person.py 224 224 25`; a T4 took 3 h 32 min),
then quantize on the laptop with 200 calibration photos:
```bash
.venv/bin/python tools/quantize.py best.pt 224 224 calib/ firmware/detect_cam/components/espdet_models/models/espdet_pico_224_224_person.espdl
```

## Caveats
- The scoreboard frames are sent over USB, not seen through the lens; the speeds are on the real camera.
- Reference boxes are YOLO11x's (checked by eye), not hand labels. Threshold 0.25 for every model.
- No model files here: the `.espdl`/`.pt` weights build on Espressif's AGPL-3.0 esp-detection weights, so make your
  own with the steps above.

## Files
- `firmware/detect_cam/`: the ladder firmware (camera, all models, per-stage timing, preview, inject mode, two cores)
- `firmware/detect_cam/components/espdet_models/`: embeds every `.espdl` in `models/` and runs it with ESP-DL
- `tools/chip_view.py`: serial reader + live view; `tools/feed_clips.py`: inject frames; `tools/score.py`: scoreboard
- `tools/make_testset.py`: stock clips → 640×480 frames + reference labels; `tools/quantize.py`: .pt → int8 .espdl
- `colab/`: the COCO person subset and training scripts; `results/`: every run's per-frame records

Third party: ESP-DL and the coco_detect models (Espressif, MIT), esp-detection / espdet_pico (Espressif, AGPL-3.0,
called, not copied), Ultralytics YOLO (AGPL-3.0), COCO 2017 (CC BY 4.0 annotations), Pexels clips (Pexels License).
