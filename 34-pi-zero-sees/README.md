# 34 · Can a $15 Computer Describe What It Sees?

A vision-language model on a Raspberry Pi Zero 2 W (416 MB usable), fully offline. An ESP32-S3 with an OV3660
camera ("Chip") takes the photo when you press BOOT and POSTs it over Wi-Fi; the Pi runs SmolVLM through
llama.cpp and writes a one-sentence caption. One photo at a time, not live video.

## Results
| | SmolVLM-256M (Q8_0) | SmolVLM-500M (Q8_0) |
|---|---|---|
| Files (model + mmproj) | 175 + 104 = 279 MB | 437 + 109 = 546 MB |
| Fits in the Pi's ~270 MB free? | almost: peak 280 MB, 21 MB into zram | no: llama.cpp prints "failed to fit params … abort", then runs memory-mapped off the SD card |
| Time per caption on the Pi (glasses photo, 640×480) | 111.4 s (load 20.5 s, 2 image passes of ~34 s, ~11 s writing) | 707 s on the desk photo (first word at 311 s, 217,558 page faults) |
| My 20 desk photos, one-sentence prompt (right / half / wrong) | 9 / 6 / 5 | 13 / 6 / 1 |
| Same photos, "list every object" prompt | 9 / 8 / 3 | 10 / 7 / 3 |

- What made the 500M run at all: mmap (llama.cpp's default) keeps the weights on the SD card, zram squashes idle
  pages in RAM, `-c 512` instead of 2048 (−35 MB on a Mac), and a 640×480 photo instead of 800×600 (2 image passes
  instead of 3, ~35 s saved).
- Building llama.cpp on the Pi itself took 1 h 43 min (`-j2`) and went 607 MB into swap.
- The 20-photo scores are from Mac runs with the same files (on the Pi they would take all night); hand-scored in
  `photos/test/score.md`. Pi spot checks gave the same words, except photo 20 ("a box with some text on it" on the
  Pi, "a camera box" on the Mac).

## Run it
**The Pi** (Raspberry Pi OS / Debian 13, 64-bit):
```bash
sudo apt install -y git cmake build-essential libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build
cmake --build ~/llama.cpp/build -j2      # 1 h 43 min on mine; have zram swap on first
mkdir -p ~/vlm/models && cd ~/vlm/models
# from https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF (and -500M-):
#   SmolVLM-256M-Instruct-Q8_0.gguf  mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
python3 pi/vlm_server.py                   # listens on :8090, standard library only
```
Environment knobs: `VLM_MODEL`, `VLM_MMPROJ`, `VLM_PROMPT`, `VLM_CTX` (512), `VLM_N` (40), `VLM_T` (4). Every
caption is logged with its wall time, first-word time, peak RSS, page faults and swap.

**Chip** (Arduino IDE, ESP32 core 3.x, an ESP32-S3 with PSRAM and an OV2640/OV3660 on the ESP32-S3-EYE pins):
copy `firmware/chip_eye/secrets.h.example` to `secrets.h`, fill in your Wi-Fi and the Pi's hostname, flash
`chip_eye.ino`. Press BOOT: the LED flashes and the photo goes to the Pi. `GET http://chip.local/capture?size=vga`
returns one JPEG.

**The 20-photo test** (on a laptop, with llama.cpp's `llama-mtmd-cli` on the PATH and the GGUF files in `models/`):
```bash
python3 tools/shoot_server.py      # http://localhost:8766: Chip's live view, a Take photo button, both models' captions
```
Photos and answers land in `photos/test/` and `results.jsonl`; scoring is by hand.

## Files
- `pi/vlm_server.py`: the Pi side (POST /photo from Chip, GET /shoot to pull one), one llama-mtmd-cli run per photo
- `firmware/chip_eye/`: Chip's camera + BOOT button + HTTP endpoints
- `tools/`: the hallucination-test booth (`shoot_server.py` + `preview.html`) and a command-line `shoot.py`
- `photos/`: the two photos from the video and the 20 test photos with `score.md` and `results.jsonl`
- `results/runs.jsonl`: every run on the Pi (times, memory, page faults, captions)

## Credits
SmolVLM by Hugging Face (Apache-2.0), GGUF files by ggml-org; llama.cpp (MIT).
