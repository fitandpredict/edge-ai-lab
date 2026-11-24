# Episode 30 results: a maze brain with memory, on Chip

Measured 24–25 Nov 2026. ESP32-S3 N16R8 DevKitC-1 clone at 240 MHz, Arduino core 3.3.12, over USB.
Everything is reproduced by `tools/run_all.sh`. The JSON files in `results/` hold the raw numbers.

## The brain

- **Inputs (20 numbers):**
  - 6 laser distances
  - visited counts of the 4 neighbouring cells and of this cell (odometry memory)
  - gyro heading (cos, sin)
  - goal offset in the robot's frame
  - last action
  - a bias term
- **Network:** LSTM with 128 units, then a 64-unit tanh layer, then 4 actions: forward / left / right / back.
  It decides once per cell; a 200 Hz PID does the driving.
- **Size:** the actor has **85,316 parameters = 341.3 kB in float32**, exactly the "341 KB" in the LinkedIn post.
  On Chip the two LSTM biases are merged, giving 84,804 parameters: **331 KiB float32, 83 KiB int8**.
- **Training:** RecurrentPPO (sb3-contrib) on random 5×5 perfect mazes only, 5M decisions, about 30 min on the
  Mac CPU (`results/train_*.csv`). The first seed stalled at about 17% solved and seed 1 trained cleanly
  (`train_lstm_seed0_stalled.*`).

## Chip (`chip_info.json`, `chip_bench.json`, `chip_check_*.json`)

| How the brain runs on Chip | ms per decision |
|---|---|
| float32, weights read from flash, plain C | 14.07 |
| float32, weights in PSRAM, plain C | 9.50 |
| float32, weights in PSRAM, SIMD dot product | **6.78** |
| float32, weights in internal RAM | does not fit (331 KiB vs 254 KiB free) |
| int8 weights in internal RAM, plain C | **5.75** |

- **Same answers as the Mac:**
  - float32: Chip picks the Mac's action on **1,565 / 1,565** decisions (50 unseen 7×7 mazes), with logits within 4e-6.
  - int8: Chip's int8 brain matches the int8 numpy twin on **1,608 / 1,608** decisions, and still solves **50 / 50** of these mazes.
- **Round trip over USB:** 7.4 ms per decision (float32), 6.4 ms (int8).
- **Budget:** at 0.62 m/s an 18 cm cell takes about 290 ms, so the brain uses about 2% of it.
  - The post's "~3 ms/cell" isn't something we can match on Chip: we measure 5.8–6.8 ms.
  - Our float32 run is memory-bound: 331 KiB of weights stream from PSRAM every decision (compare ep 18).

## The race, cell level (`race.json`, `race_loops.json`)

Setup: 200 unseen mazes per size, deterministic. "path" is the cells driven divided by the shortest path, so 1.0 is perfect.

| Brain | 5×5 solved | 7×7 solved (path) | 9×9 solved (path) | 7×7 with loops |
|---|---|---|---|---|
| **LSTM + visited cells (Chip's brain)** | 100% | **99% (1.35)** | 85% (1.36) | 89% |
| MLP + visited cells | 100% | 98% (1.38) | 78% | 89% |
| LSTM, no visited cells | 100% | 92% (1.18) | 71% | 93% |
| MLP, no visited cells (no memory at all) | 98% | 85% (1.08) | 64% | 76% |
| left-hand rule (no learning) | 100% | 100% (2.0) | 100% (2.29) | **79%** |
| flood fill (no learning, micromouse classic) | 100% | **100% (1.17)** | **100% (1.35)** | 100% |
| random | 36% | 22% | 26% | 30% |

**Honest reading of the race:**
- Trained only on 5×5, the brain solves unseen 7×7 mazes (99%), which reproduces the post's zero-shot claim. It starts to slip on 9×9 (85%).
- Flood fill wins everywhere, with no training. It's what real micromice use.
- The left-hand rule is perfect on simple mazes but takes about twice the path. It fails on 21% of mazes with loops.
- Memory matters most when the visited-cell map is gone: with no memory at all the brain falls to 85% on 7×7.
- With the visited-cell map, a plain MLP is nearly as good as the LSTM (98% vs 99%). The "memory" mostly lives in the odometry map.

## Physics: MuJoCo rover, Chip as the brain (`physics_chip_7x7.json`)

The physical rover has:
- 2 velocity-controlled wheels with a torque limit
- a gyro with noise
- 6 rangefinders (2 forward, 2 diagonal at 25°, 2 side)
- encoders

It runs a 200 Hz PID, and Chip makes one decision per cell, about 3 cm before the centre, so it can chain straights at 0.62 m/s.

| | 50 unseen 7×7 mazes |
|---|---|
| solved | **98%** (49/50) |
| decisions / shortest path (median) | 1.29 |
| sim time per maze (median) | 20.3 s |
| Chip time per decision | 6.84 ms |

- **First physics run: 84%.** The odometry under-read distance by about 6% (the tyre squashes), so the rover stopped short and pinned itself on walls.
  - Fixed like a real micromouse: a calibrated rolling radius, centring on one side wall, and using the front laser to stop at the centre.
- **Wall bumps:** in 4 mazes the brain asked to drive into a wall 2–3 times, and the driver refused. In the one failure it kept asking (160 times).
  - The post's "0 wall-collision hits" holds for our physical rover: the driver never drives into a wall. But the brain does sometimes ask.

## Clips and runs for the video

- `results/clips/chip_7x7_s{3,11,27}.mp4`, `chip_5x5_s3.mp4`: MuJoCo, Chip driving, laser rays visible.
- `results/clips/nomemory_7x7_s60002.mp4` vs `memory_7x7_s60002.mp4`: the same maze.
  - With no memory, the rover ping-pongs between two cells: 65 times at the cell level, still looping after 41 decisions in physics.
  - With memory, it solves the maze in 26 decisions (shortest path 13).
- `results/runs/*.json`: full runs (poses, wheel angles, laser readings, every decision with Chip's µs). Bench 3D replay:
  `node resources/threejs/render.mjs 30-chip-solves-a-maze/video/three/maze_run.html out.mp4 --query "run=/30-chip-solves-a-maze/code/results/runs/chip_7x7_s27.json&v=top"`

## What this doesn't show

- **No real rover.** The rover is a MuJoCo model plus a bench render. Only the brain and its timing are real hardware.
- **No sensor noise in physics.** The MuJoCo rangefinders have no noise, the floor is perfect and wheel slip is mild. A real VL53L0X is noisier, with ±3–5% error and some missed reflections.
- **Training and observations are ours.** We trained at the cell level (one step = one decision) with our own observation and reward. This isn't the post's code, and their 94% is their number.
- **The goal position is given.** Like the post (and flood fill), the brain knows where the goal is relative to the start.
