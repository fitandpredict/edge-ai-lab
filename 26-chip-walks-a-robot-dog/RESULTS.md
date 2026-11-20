# Episode 26 build log and measured numbers

Everything here was measured by us, 20 Nov 2026, unless marked otherwise.

## Pipeline

| Part | Where | Files |
|---|---|---|
| Training (Brax PPO, MuJoCo Playground `Go1JoystickFlatTerrain`, domain randomisation on) | Colab A100-SXM4-40GB | `colab/setup_playground.sh`, `colab/train_go1.py`, `colab/run_all.sh` |
| Physics + rendering (plain MuJoCo, CPU) | Mac | `sim/go1_loop.py`, model files in `sim/go1/` (Playground + Menagerie, Apache 2.0) |
| Export (ONNX/npz → numpy check → C header, float32 + int8) | Mac | `sim/export_policy.py` |
| The brain at 50 Hz over USB | ESP32-S3 N16R8 | `firmware/dog_brain/` |

Colab setup gotchas (all fixed in the scripts): Colab's driver is CUDA 13.0, so `jax[cuda13]`; keep
`LD_LIBRARY_PATH=/usr/lib64-nvidia`; pin jax 0.12.0.1 / brax 0.14.2 / mujoco 3.16 from the repo's lockfile;
brax still calls `jax.device_put_replicated`, which JAX 0.12 removed (shim in `train_go1.py`). Only one
`colab ssh` per runtime: check runs with `colab exec -f`.

## DeepMind's pretrained Go1 brain (sim2sim ONNX from Playground)

- MLP 48 → 512 → 256 → 128 → 12, swish, tanh head: **190,860 parameters, 746 KiB float32, 186 KiB int8**.
- Mac CPU (onnxruntime): 36 µs per step. Walks 10.03 m in 10 s at a 1 m/s command, 0 falls,
  forward-speed RMSE 0.037 m/s (`results/onnx_flat.json`).
- Our numpy copy matches ONNX to 1.1e-6; it walks the same (10.03 m, RMSE 0.036).
- **int8 (per-channel weights, per-layer activations, as the firmware does): still walks**, 20.09 m in 20 s,
  0 falls, RMSE 0.034 (`results/big_int8_flat.json`).

## How late can the brain be? (`--delay-ms`, 20 s at 1 m/s, pretrained brain)

| Extra delay | Falls | Distance | Speed RMSE |
|---|---|---|---|
| 0 ms | 0 | 20.3 m | 0.035 |
| 10 ms | 0 | 20.4 m | 0.046 |
| 20 ms | 0 | 16.2 m | 0.135 |
| 30 ms | 1 | 4.7 m | 0.872 |
| 40 ms | 1 | 1.4 m | 0.960 |
| 100 ms | 1 | 1.2 m | 0.979 |

Fine at 10 ms, stumbling at 20 ms, down at 30 ms. Clips: `results/clips/delay_{0,20,30}ms.mp4`.

## Chip memory (firmware compile, N16R8)

Sketch 1.27 MB flash; ~300 KB internal RAM free → the big brain's 746 KiB float32 weights **do not fit in
internal RAM** (PSRAM or int8 only). The small 128×128 brain (24,332 params, 95 KiB) does.

## Training on the A100 (our own brains)

Brax PPO, 8,192 parallel dogs, 200M-step budget (218M done), domain randomisation on, seed 1.
Whole night's Colab cost: **3.05 compute units** (209.64 → 206.59), session stopped.

| Brain | Params | Wall clock (incl. ~38 s compile) | Env steps/s | Learned to walk at | Final, 20 s at 1 m/s |
|---|---|---|---|---|---|
| small 48→128→128→12 | 24,332 (95 KiB f32) | 520 s (8.7 min) | 452k | ~195M steps, ~8 min (snap 17) | 19.66 m, 0 falls, RMSE 0.042; int8 19.72 m |
| big 48→512→256→128→12 | 190,860 (746 KiB f32) | 565 s (9.4 min) | 414k | ~68M steps, ~4.3 min (snap 6) | 19.43 m, 0 falls, RMSE 0.039; int8 19.58 m |

"Generations" (`sim/generations.py`, each snapshot replayed 6 s on the Mac): before the breakthrough, every
snapshot stands still (moves < 7 cm) while the eval reward plateaus around 20 (balance/posture rewards);
walking (~5.5 m in 6 s) appears at once when the reward jumps (small: 20.6 → 23.6 → 24.8; big: 19.9 → 23.1 →
25.3). The big brain found walking ~3× sooner. One seed each: the small brain's late breakthrough could be
luck, so say "in this run". Clips: `results/colab/small/clips/snap_NN.mp4`.

## Chip runs the brain (ESP32-S3 N16R8, 240 MHz, measured 20 Nov 2026)

Chip's actions match numpy to ≤ 2.1e-6 in every float32 mode (int8 mode: ≤ 0.18 on random inputs, but it
walks). Benchmarks are the firmware's own 'B' command, 200 runs (`results/chip_check_ours_{small,big}.json`).
"Walk" = 20 s at 1 m/s with `--real-latency`: each action lands only after Chip's measured compute time,
and a tick is skipped if Chip is still busy (as if Chip were on the robot; the USB round trip, ~1.1 ms, is
not counted because an onboard brain wouldn't have it).

**Small brain (ours, 24,332 params):**

| Mode | µs per step | % of 20 ms | Walk |
|---|---|---|---|
| float32, weights read from flash, plain C | 4,148 | 20.7 | 19.7 m, 0 falls |
| float32, PSRAM, plain C | 2,862 | 14.3 | |
| float32, PSRAM, SIMD | 2,103 | 10.5 | |
| **float32, internal RAM, SIMD** | **575** | **2.9** | **19.7 m, 0 falls, RMSE 0.042** |
| int8 weights, internal RAM, plain C | 1,889 | 9.4 | 19.8 m, 0 falls |

575 µs is almost exactly UCLA MiNI-Q's 594 µs for a [128,128] policy on the same chip (arXiv 2610.02728),
and plain C from flash (4.1 ms) matches their ~4,000 µs scalar figure.

**Big brain (ours, 190,860 params, 746 KiB float32: does not fit in the ~150 KB internal RAM left):**

| Mode | µs per step | % of 20 ms | Walk (real latency) |
|---|---|---|---|
| float32, weights read from flash, plain C | 30,849 | 154 | **falls** after 2.3 m (every other tick skipped) |
| float32, PSRAM, plain C | 20,541 | 103 | **falls** after 9.8 m |
| float32, PSRAM, SIMD | 14,392 | 72 | walks: 19.0 m, 0 falls, RMSE 0.060 |
| float32, internal RAM | — | — | doesn't fit |
| int8 weights, internal RAM, plain C | 12,226 | 61 | walks: 19.0 m, 0 falls, RMSE 0.064 |

The story: the big brain is 8× the work and runs 25× slower on Chip (memory, not maths). Where its weights
live decides whether the dog walks. The small brain walks in every mode with room to spare.
Lockstep runs (Chip waits for nothing): small 19.66 m (identical to the Mac), USB round trip 1.65 ms.

Clips: `results/clips/chip_ours_small.mp4`, `chip_small_sram_simd_m3.mp4`, `chip_big_flash_m0.mp4` (falls),
`chip_big_psram_simd_m2.mp4` (walks).

Notes for the script: int8 here is plain C (no SIMD), which is why it loses to float SIMD on the small brain;
esp-nn's SIMD int8 kernels are a possible follow-up. One training seed per brain.

## Video clips use the full-collision Go1 (20 Nov)

The training model only gives the **feet** collision shapes, so when the dog falls its body sinks through
the floor (fine for learning, unwatchable on screen). Every clip in `video/assets/footage/D*.mp4` is
rendered with Playground's full-collision Go1 (`--terrain full`, `tools/render_clips.sh`). Same brains,
same command; 20 s results, feet-only vs full:

| Run | Feet-only | Full collisions |
|---|---|---|
| pretrained, 0 ms | 0 falls, 20.3 m | 0 falls, 20.3 m |
| pretrained, 10 ms | 0, 20.4 m | 0, 20.4 m |
| pretrained, 20 ms | 0, 16.2 m | 0, 8.5 m (stumbles more) |
| pretrained, 30 ms | fell, 4.7 m | fell, 2.7 m |
| ours small | 0, 19.7 m | 0, 19.7 m |
| ours big | 0, 19.4 m | 0, 19.4 m |

Walking is identical; under stress the full model does a bit worse. The story (fine at 10 ms, sloppy at
20, down at 30) holds in both. The numbers on screen in each clip come from that clip's own run
(`results/clips/D*.json`).

## Live Chip clips for the video (20 Nov, full-collision Go1, Chip driving at 50 Hz)

| Clip | Chip mode | Chip µs (median) | Result |
|---|---|---|---|
| D01Hook (14 s) | small, RAM + SIMD | 572 | 13.7 m, 0 falls |
| D07ChipWalk (22 s) | small, RAM + SIMD | 573 | 21.6 m, 0 falls |
| D09BigMode0 (10 s, real latency) | big, flash plain C | 30,862 | fell, 1.9 m |
| D09BigMode1 (20 s, real latency) | big, PSRAM plain C | 20,541 | fell at ~9 s, 8.8 m |
| D09BigMode2 (10 s, real latency) | big, PSRAM SIMD | 14,362 | 9.4 m, 0 falls |
| D09BigMode4 (10 s, real latency) | big, int8 RAM | 12,221 | 9.5 m, 0 falls |

The stand-ins (Mac brain + Chip's measured timing) are replaced by these. Chip still holds the big brain.
