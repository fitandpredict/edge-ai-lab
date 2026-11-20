# 26 · My $5 Microcontroller Walks a Robot Dog

A simulated Unitree Go1 learns to walk with reinforcement learning on a rented GPU (about 9 minutes per brain), then
an ESP32-S3 runs the learned brain and drives the dog in MuJoCo over USB, 50 times a second. Includes the training
scripts, the two trained brains, the ESP32-S3 firmware (five ways of running the brain: flash, PSRAM, internal RAM,
plain C vs SIMD, int8) and the sim that puts Chip in charge.

## Run it

**Watch a trained brain walk (Mac/Linux, no board):**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python sim/go1_loop.py --brain npz --policy sim/ours_small.npz --secs 20 --video out/small.mp4
```

**Put the brain on an ESP32-S3** (N16R8, Arduino core 3.3.x, `arduino-cli`): plug the board in, then

```bash
tools/flash.sh ours_small     # builds, flashes, checks Chip against the Mac, benchmarks every mode
.venv/bin/python sim/go1_loop.py --brain chip --chip-mode 3 --secs 20 --video out/chip.mp4
.venv/bin/python sim/go1_loop.py --brain chip --chip-mode 0 --real-latency --secs 20   # act at Chip's real speed
```

`tools/flash.sh ours_big` does the same with the big brain (its float32 weights don't fit in internal RAM, so
modes 0, 1, 2 and 4 only). Make your own header from any policy with `sim/export_policy.py <policy.npz|.onnx> --name x`.

**Train your own brains** (Google Colab A100 via the `colab` CLI; about 9 minutes and ~1.5 compute units each):

```bash
colab new -s go1 --gpu A100
colab upload -s go1 colab/setup_playground.sh /content/setup_playground.sh
colab upload -s go1 colab/train_go1.py /content/train_go1.py
colab ssh -s go1 -- bash /content/setup_playground.sh
colab ssh -s go1 -- bash -lc 'cd /content && LD_LIBRARY_PATH=/usr/lib64-nvidia venv-mjp/bin/python train_go1.py --name small --hidden 128 128 --evals 20'
colab download -s go1 /content/runs/small/final.npz sim/my_small.npz
colab stop -s go1          # an idle A100 still burns units
```

Each eval also saves `snap_NN.npz`; `sim/generations.py <run folder> --video` replays them ("the dog that stood still").
The delay test (`--delay-ms`) and DeepMind's own pretrained brain need `sim/go1_policy.onnx`: copy it from
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) at
`rss2025/mujoco_playground/experimental/sim2sim/onnx/go1_policy.onnx`.

## My results

All measured on 20 Nov 2026 (full table in `RESULTS.md`):

- Training (Brax PPO, 8,192 dogs, 218M steps, Colab A100): small brain (24,332 numbers) 8.7 min, big brain
  (190,860) 9.4 min. Both walk 19.4–19.7 m in 20 s at 1 m/s with no falls.
- The small brain stood still until ~8 minutes in, then walked; the big one walked from ~4.3 minutes (one run each).
- On the ESP32-S3 (240 MHz), small brain: **575 µs** per step with weights in internal RAM + SIMD (2.9% of the
  20 ms budget), 4.1 ms from flash with plain C. Chip's answers match the Mac to ~6e-7.
- Big brain at Chip's real speed: flash/plain C 30.9 ms → falls; PSRAM/plain C 20.5 ms → falls (on the edge,
  somewhere different each run); PSRAM + SIMD 14.4 ms → walks; int8 in internal RAM 12.2 ms → walks.
- Delay test (pretrained brain): fine at 10 ms, sloppy at 20 ms, down at 30 ms.

## Caveats

- It's a simulation: no real robot walked. Real motors need actuator models and more (see UCLA's MiNI-Q,
  arXiv 2610.02728, which does run a [128, 128] policy on an ESP32-S3 on a real 270 g robot).
- The GPU did the learning; Chip only runs it. One training run per brain, so "minute 8" could be minute 3 or never.
- The USB round trip (~1.1 ms) is not counted in `--real-latency` (a brain on the robot wouldn't have one).
- The int8 mode is plain C, no SIMD yet, which is why it's slower than float + SIMD on the small brain.
- Training uses Playground's feet-only Go1; the videos render the full-collision model so a fallen dog doesn't sink
  through the floor. Walking is identical; under stress the full model does a bit worse (see `RESULTS.md`).

## Files

- `sim/go1_loop.py`: MuJoCo Go1 with a brain on the Mac (ONNX, numpy) or on the ESP32-S3 over USB; `--delay-ms`,
  `--real-latency`, video with labels.
- `sim/export_policy.py`: policy (Playground ONNX or our npz) → numpy check → C header (float32 + int8).
- `sim/chip_check.py`: Chip vs Mac for every firmware mode, plus the on-board benchmark.
- `sim/generations.py`, `sim/still.py`, `sim/export_crowd.py`, `sim/look.py`, `sim/hud.py`: replays, stills and
  the video look.
- `sim/ours_small.npz`, `sim/ours_big.npz`: the two brains trained for the video.
- `colab/`: Playground setup (CUDA 13 JAX fixes) and the PPO training script.
- `firmware/dog_brain/`: the ESP32-S3 brain (binary protocol over USB; text commands `0`–`4`, `B`, `I`) with both headers.
- `tools/flash.sh`, `tools/render_clips.sh`: flash + check, and every clip in the video.

Third-party: Go1 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause,
Unitree Robotics, `sim/go1/LICENSE-unitree_go1`); scene files and observation layout from
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Apache 2.0, `sim/LICENSE-mujoco_playground`);
training with [Brax](https://github.com/google/brax) PPO (Apache 2.0); ESP-DSP (Apache 2.0) for the SIMD dot product.
