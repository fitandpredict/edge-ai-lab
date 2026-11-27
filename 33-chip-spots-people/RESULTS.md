# Episode 33 results: from 26 seconds a frame to 7 frames a second

Board: ESP32-S3 (QFN56 rev v0.2), 16 MB flash, 8 MB OPI PSRAM, 240 MHz; **OV3660** camera (ESP32-S3-EYE pins),
on `/dev/cu.usbmodem1101`. ESP-IDF v5.4.4, ESP-DL **3.3.8** (the version Espressif's workshop timed), coco_detect 0.4.0.
All timings are `esp_timer` on the board. Raw logs: `results/*.serial.log`, per-frame records: `results/*.jsonl`,
frames with boxes: `results/<run>/NNNNN.jpg`.

## The ladder so far (26 Nov 2026)

| Rung | Model | Input | Per frame on Chip | Stage split (prepare / network / boxes) | Notes |
|---|---|---|---|---|---|
| 0a | YOLO11n `coco_detect_yolo11n_s8_v1` | `bus.jpg` 405×540 → 640 | **25.70–25.96 s** (3 passes) | n/a (+88 ms JPEG decode) | Reproduces Espressif's "~26,000 ms". Same scores as their workshop log (0.851953). `results/rung0a_bus.serial.log` |
| 0b | YOLO11n 640 | live OV3660 640×480 | **25.67–25.69 s** | 57 / 25,571 / 44 ms | 99.6 % of the time is the network. First capture 569 ms, then 0 (a fresh frame is always waiting). `results/rung0b_yolo640_cam*`, `rung1_yolo320_cam` frame 0 |
| 1 | YOLO11n 320 `coco_detect_yolo11n_320_s8_v1` | live, 640×480 → 320 | **6.10 s** | 17 / 6,075 / 11 ms | 4.2× faster for ¼ of the pixels. Espressif's workshop says ~7,698 ms for this model: ours is 1.6 s faster (their figure likely includes decoding a JPEG and/or an older ESP-DL; not verified). `results/rung1_yolo320_cam*` |
| 2 (stand-in) | Espressif `espdet_pico_224_224_cat` (0.36M params), our int8 | live → 224 | **136 ms** (median of 36) | 8.6 / 127.2 / 0.6 ms | Network 127 ms vs Espressif's table 126.2 ms: matches. **7.3 fps** detection-only. But see "Broken int8" below. `results/rung2_espdet_cat_cam*` |

### Watching it costs more than thinking (a real rung for the video)

With the live preview on (Chip JPEG-encodes each 640×480 frame and pushes it over USB), the cat model runs at
**2.5 fps wall-clock (398 ms per frame)**, though detection itself is 136 ms. Turning the preview off is worth
almost 3×. Without preview: 1000 / (136 + 10 ms pause) ≈ **6.8 fps**.

## Broken int8 (found 26 Nov, being fixed)

The first int8 cat model drew frame-sized boxes at 0.82–0.88 on a blank ceiling. The **float** model on the Mac
finds nothing on the same frame and finds the cat in Espressif's `cat.jpg` (0.85), so the fault is in the int8
conversion: Espressif's `deploy/cat_calib` folder holds **one** image, so the quantizer set every activation range
from a single photo.

**Confirmed:** re-quantized with 200 COCO val images (`calib/person_calib`, 2 greyscale images converted to RGB;
20 min on the Mac, mostly esp-ppq's per-layer error analysis). Both flashed side by side, same blank ceiling:

| int8 calibrated on | Frames with a false box | Highest false score | Network |
|---|---|---|---|
| 1 photo (Espressif's `cat_calib`) | **22 / 22** | **0.90** | 129.6 ms |
| 200 photos | 4 / 21 | 0.38 | 126.3 ms |

Same weights, same speed, same chip: only the 200 numbers-per-layer the quantizer picked changed. Strong beat for
the video ("the model wasn't wrong, its rounding was"). `results/ceiling_cat_model2*`, `results/ceiling_cat_model3*`.

## Problems hit and fixes (for the "making it" beat)

- Camera is an OV3660, not the OV2640 on the listing (ep 03). Frames came out green until brightness +1 /
  saturation −2 (the same fix as ep 03).
- ESP-DL hard-codes `DL_LOG_INFER_LATENCY 0` in `dl_define.hpp`, so its own stage timing can't be switched on
  with a flag; `TimedDetect` in `app_main.cpp` repeats `DetectImpl::run` with a timer per stage.
- esp-detection says `ultralytics>=8.3.112`, but its `ESPDetect` head passes `reg_max` to `Detect`, which only
  the **8.4** series accepts (8.3.112 / 8.3.200 / 8.3.240 fail; 8.4.0 / 60 / 120 / 219 build). Pinned 8.4.0.
- `esp-ppq` (the quantizer) needs Python < 3.13; Colab is on 3.13 and lacks `onnxscript`, so Colab only trains
  and the Mac (Python 3.12, `.venv-quant`) exports ONNX and quantizes (`tools/quantize.py`).
- No A100 free on Colab (503 ×4), H100 not on the plan, L4 503: training on a **T4** (2 vCPU, so the data
  loader, not the GPU, is the limit: ~9 min per epoch on 70k images).

## Training the person model (in progress)

COCO 2017 person subset (`colab/make_person_subset.py`): **64,115 train images with people + 6,000 without**,
val 2,693 + 300. Fine-tuned from Espressif's 224×224 cat weights, 25 epochs, batch 256, T4.
Smoke test (1 epoch, 5 % of data): runs end to end, 26 s for 3.5k images.

## Accuracy test: free stock clips fed to Chip over USB (27 Nov)

No one sits in front of the camera: 12 free Pexels clips (`stock/SOURCES.md`) become 640×480 frames shaped like
Chip's camera, and `tools/feed_clips.py` sends them to Chip's new **inject mode** (`'i'`: Chip decodes the JPEG and
runs the same timed detector). Reference boxes: YOLO11x on the Mac. Score: `tools/score.py` (conf 0.25).
Same 60 frames for every rung (`stock/test60.txt`: 36 with a person, 24 without).

| Model on Chip | Noticed person / no person correctly | False-alarm frames | Missed frames | Box recall @0.5 | Median per frame |
|---|---|---|---|---|---|
| YOLO11n 640 (Espressif, 80 classes) | **60/60 (100 %)** | 0 | 0 | 0.98 | 28.7 s |
| YOLO11n 320 | 59/60 (98 %) | 0 | 1 | 0.95 | 6.8 s |
| espdet_pico cat (stand-in, 200-photo calib) | 29/60 (48 %) | 6 | 25 | 0.00 | 143 ms |
| **espdet_pico person (ours)**, 640×480 frames | 57/60 (95 %) | 2 | 1 | 0.95 | 133 ms |
| **espdet_pico person (ours)**, 320×240 frames | **59/60 (98 %)** | 1 | 0 | 0.98 | 131 ms |

Person model's mistakes at 640×480: false alarms on an empty conference table and an empty sofa (both 0.27, just
over the 0.25 line); the miss is two people seen through a glass door, which YOLO11n 320 also missed. (At conf 0.35
it scores 59/60, but the threshold was not tuned on this test set: 0.25 for every model.)

The cat model is the control: a cat detector shown people is a coin flip, as it should be.
Caveat for the script: these frames skip Chip's lens and sensor, so they test the brain, not the eye.

### The idle camera slows the network (found 27 Nov)

The YOLO rows above were 12 % slower than on the live camera (network 28.6 s vs 25.6 s; 6.8 s vs 6.1 s), and int8
compute doesn't depend on pixel content. Cause: in inject mode the camera sat idle with **both** frame buffers free,
so it kept DMA-writing frames into PSRAM, the same memory the network streams its weights from. Same 60 frames,
cat model: camera running **133.1 ms**, camera switched off (`esp_camera_deinit`) **122.9 ms** (−8 %). Inject mode
now turns the camera off (`'k'` keeps it on). On the live path Chip holds one buffer during detection, which is
why it measured in between (127 ms). Possible rung: grab a frame only on demand.

## Person model trained (27 Nov)

25 epochs on a Colab T4, 3 h 32 min (12,726 s), fine-tuned from Espressif's 224 cat weights on the COCO person
subset. Val (float, on Colab): **mAP50 0.462, mAP50-95 0.264** (epoch 1: 0.292 / 0.150). `trained/`.

## The final ladder, live on Chip's camera (27 Nov)

| Rung | Change | Per frame, live | Live fps | Test60 right |
|---|---|---|---|---|
| 0 | YOLO11n 640, 80 classes | 25.7 s | 0.04 | 60/60 |
| 1 | YOLO11n 320 (¼ of the pixels) | 6.1 s | 0.16 | 59/60 |
| 2 | espdet_pico cat stand-in (0.36M params, 1 class) | 136 ms detect | n/a | 29/60 (it's a cat model) |
| 3 | our person espdet_pico, 1 core, VGA camera | 139 ms detect + 29 ms camera wait + 10 ms pause | **5.2** | 57/60 |
| 4 | detector on core 1, capture task on core 0, 3 frame buffers | 139 ms detect + 39 ms wait | 5.55 | same |
| 5 | camera at 320×240 (QVGA) instead of 640×480 | **135 ms** between frames, wait 0 | **7.4** | 59/60 |

**The camera was the bottleneck at rung 4:** with VGA RGB565 the OV3660 hands over a frame only every **180 ms**
(`CAMGAP`, from the driver's frame timestamps), so a 139 ms brain sat waiting. At QVGA the camera keeps up
(135 ms gap = the detector's own pace) and the wait drops to 0. The model only looks at 224×224 anyway.
Overall: 25.7 s → 135 ms per frame = **190× faster**, and the fast one still gets 59/60 on the test set.
Runs: `results/person_cam_speed*`, `cores_test*`, `cores_q2*`. Keys: `m` = 2-core mode, `q` = camera VGA/QVGA.
