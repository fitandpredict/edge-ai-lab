# Episode 27 build log and measured numbers

The trained brains are in `sim/brains/` (`<run>.npz`); the training logs, clips and result JSONs referenced
below stayed in the video workspace and aren't in this folder. `tools/sweep.py` regenerates the tables.

Everything here was measured by us, 20 Nov 2026, unless marked otherwise.

## Pipeline

| Part | Where | Files |
|---|---|---|
| The drone (constants, MuJoCo XML) | shared | `sim/drone_model.py` |
| Training: Brax PPO on a fly-to-a-point task, 8,192 drones at once | Colab A100-SXM4-40GB | `colab/drone_env.py`, `colab/train_drone.py`, `colab/run_all.sh`, `colab/setup_playground.sh` |
| Physics check: JAX rigid body vs MuJoCo | Colab | `colab/check_physics.py` |
| Flying it (plain MuJoCo, CPU) + experiments | Mac | `sim/drone_loop.py`, `tools/sweep.py` |
| Export → C header (float32 + int8) | Mac | `sim/export_policy.py` |
| The brain at 100 Hz over USB | ESP32-S3 N16R8 | `firmware/drone_brain/` |
| Swarm shot data | Mac | `sim/export_crowd.py` → `video/three/drones/` |

**The drone** is the Crazyflie Brushless as identified by Gräfe, Scherer, Hönig, Trimpe, "How to Model Your
Crazyflie Brushless" (arXiv 2603.05944, ICRA 2026; code CrazyflieBrushJAX, MIT): 44.5 g, 7 cm motor to motor
(diagonal 10 cm), PWM→thrust and PWM→torque cubic fits, a 50 ms first-order motor lag, max thrust 2.64× its
weight, hover at PWM 0.554. One change: their Izz (7.19e-5) breaks the inertia triangle rule MuJoCo enforces,
so Izz = Ixx + Iyy = 6.9e-5 (−4%). Their repo's training code doesn't run as published (missing config key
`use_pose_delays`, a `get_obs` call with the wrong number of arguments), so the task, reward and trainer are ours.

**Why not MJX:** our first env stepped MuJoCo through MJX (10 RK4 substeps per control step): 28k steps/s
on the A100. The drone is one rigid body with no contacts, so the trainer integrates it in plain JAX
(`body_step`, semi-implicit Euler at 1 ms): **1.5M steps/s**, 53× faster. Checked against MuJoCo with the
same random thrusts and gusts: **0.5 mm apart after 0.6 s of tumbling** (`check_physics.py`). The Mac then
flies the brain in real MuJoCo, so every result below is also a transfer from one simulator to another.

**The task:** fly to a target point and hold it; a new target every ~3 s; random gusts up to 0.2 N; per
episode randomised thrust (±15% overall, ±5% per motor), yaw torque (±20%), motor lag (±20%), battery (max
PWM 0.75–1), 0–4 ms control delay; messy starts (tilted up to 34°, spinning, moving). 100 Hz, 10 s episodes.
Observation 27 floats (target offset clipped to 1 m, velocity, body x/y axes, body rates, last 3 actions);
action 4 motor PWMs.

## Training (`results/colab/<run>/`)

| Run | Brain | Rate | Steps | Wall time (incl. ~25 s compile) | Steps/s |
|---|---|---|---|---|---|
| main | 27 → 64 → 64 → 4, swish, tanh: **6,212 numbers** | 100 Hz | 157M | **2 min 10 s** | 1.52M |
| tiny | 27 → 16 → 16 → 4: **788 numbers** | 100 Hz | 157M | 2 min 1 s | |
| hz50 | 64 × 64 | 50 Hz | 100M+ | 1 min 54 s | |
| hz25 | 64 × 64 | 25 Hz | 100M+ | 1 min 58 s | |
| hz200 | 64 × 64 | 200 Hz | 200M+ | 2 min 15 s | 1.83M |

Final eval crash rates: tiny 1.6%, hz50 0.4%, hz25 0.4%, main 0%. All five trained in one ~12-minute Colab
session; the VM was stopped right after (`colab stop`).

The learning curve (`main/progress.jsonl`, eval on 256 fresh drones, 10 s each):

| Eval | Practice steps | Crashed |
|---|---|---|
| 0 | 0 | 100% |
| 1 | 7.9M | 100% |
| 2 | 15.7M | 77% |
| 3 | 23.6M | **0%** |
| 20 | 157M | 0% |

From eval 2 to eval 3 took 2.3 s of wall time. (The first two evals are slow because of compilation.)

**v1 → v2 reward fix:** the first main brain (`main_v1_no_bullseye/`) hovered a steady 7 cm off the target
(offset −5.7, +3.2, +2.8 cm, wobble only 2 mm): its reward was almost flat between 0 and 7 cm. Adding a sharp
"bullseye" term `2·exp(−(d/0.04 m)²)` brought it to **1.3 cm** (v2, everything else unchanged).

## Flying it in MuJoCo on the Mac (`sim/drone_loop.py`, numpy brain, 5.8 µs per step on the M-series CPU)

- Hover at (0,0,1): mean error 2.6 cm, 0 crashes.
- v1 brain (before the fix): course of 4 waypoints from 0.3 m, a 0.3 N gust (70% of its weight), int8
  weights: all flown with no crash.

Swarm export (`export_crowd.py`, 64 MuJoCo flights per stage from messy starts): random motors 64/64 crashed,
7.9M steps 64/64, 15.7M 24/64, 23.6M 0/64, final 0/64.

## How late can the brain be? (`tools/sweep.py delay`, `results/delay.json`)

Course (14 s): take off from 0.3 m, hover at (0,0,1), waypoints at 3, 6, 9 s, a 0.25 N sideways gust at 10 s.
The extra delay is on top of the 0–4 ms the brain was trained with.

| Extra delay | Crashed | Mean error |
|---|---|---|
| 0 ms | no | 19 cm (mostly while flying between waypoints) |
| 5 ms | no | 19 cm |
| 10 ms | no | 19 cm |
| 15 ms | no | 33 cm (wobbling) |
| 20 ms | **yes, at 5.9 s** | — |
| 25 ms | yes, at 1.6 s | — |
| 30 ms | yes, at 1.4 s | — |
| 40 ms | yes, at 1.1 s | — |
| 60 ms | yes, at 0.9 s | — |

The dog (episode 26): fine at 10 ms, stumbling at 20 ms, down at 30 ms. The drone goes down 10 ms sooner.

## The generations (`results/clips/gen_*.mp4`, `generations.mp4`: one take-off from 0.3 m per snapshot)

- random motors: flips over in 0.4 s
- 7.9M steps: never leaves the floor
- 15.7M steps: finds "up" but not "stop" and climbs away into the sky
- 23.6M steps: overshoots the target, then comes back to it
- 157M steps: sits on the target

## Brain size and int8 (`tools/sweep.py size`, `results/size.json`, same 14 s course)

| Brain | Numbers | float32 | int8 | Course |
|---|---|---|---|---|
| main | 6,212 | 24 KiB | 6 KiB | no crash, 19.0 cm (int8 19.0 cm) |
| tiny | 788 | 3 KiB | under 1 KiB | no crash, 19.3 cm (int8 19.3 cm) |

The dog's brain in episode 26 was 190,860 numbers. Firmware for both compiles (`firmware/drone_brain`,
25 KB of RAM used).

## Loop rate (`tools/sweep.py rate`, `results/rate.json`, same course)

| Brain trained at | Run at | Result |
|---|---|---|
| 200 Hz | 200 Hz | no crash, 35 cm |
| 100 Hz | 200 Hz | no crash, 19 cm |
| 100 Hz | 100 Hz | no crash, 19 cm |
| 50 Hz | 50 Hz | no crash, 16 cm |
| 100 Hz | 50 Hz | no crash, 18 cm |
| 25 Hz | 25 Hz | no crash, 15 cm |
| 100 Hz | 25 Hz | **crash at 4.2 s** |

A slow brain flies fine if it learned at that speed (the motors themselves lag 50 ms, longer than a 25 Hz
step). What crashes it is running slower than it practised. Same story as the delay test: the brain
practised with 0–4 ms of delay and goes down at +20 ms.

## Chip flies it (ESP32-S3 N16R8 at 240 MHz, 21 Nov 2026, `results/chip_*.json`)

The sim runs in lockstep on the Mac: every 10 ms of flight it sends Chip the 27 numbers over USB and waits for
the 4 motor commands, so the USB round trip doesn't add flight delay. (On a real drone the brain would sit on
the drone: 0.26 ms of thinking is well inside the 0–4 ms it practised with.)

**Chip vs Mac** (`chip_check.py`, 200 random observations): largest difference per motor command
5.8e-7 (float32) and 4.1e-2 (int8).

| Brain on Chip | Mode | µs per step | Share of the 10 ms step | Course (14 s, waypoints + gust) |
|---|---|---|---|---|
| main (6,212) | float32, plain C, weights in flash | 539 | 5.4% | |
| main | float32, **SIMD** (PSRAM or internal RAM) | **258** | **2.6%** | no crash, 19.0 cm (same as the Mac) |
| main | int8, plain C | 615 | 6.2% | no crash, 19.0 cm |
| tiny (788) | float32, plain C | 96 | 1.0% | |
| tiny | float32, SIMD | **66** | 0.7% | no crash, 19.3 cm |
| tiny | int8, plain C | 133 | 1.3% | |

USB round trip (Mac → Chip → Mac): 0.90 ms median for main, 0.71 ms for tiny. The int8 path is slower than
float32 here because it has no SIMD version in this firmware (plain C int8 loops); it only saves memory, which
this brain doesn't need (24 KiB float32, 312 KiB of internal RAM free).

**The latency test with Chip as the brain** (`results/delay_chip.json`): identical to the Mac run, to the
millimetre: no crash up to +10 ms, wobbling at +15 ms (33.6 cm), crash at 5.9 s at +20 ms, within 1.6 s from
+25 ms. Chip leaves 97% of every step free, but if its answers arrived 20 ms late the drone would fall.

## A brain that practises with lag (21 Nov 2026, `results/delay_lag40*.json`)

Same task and reward as main, but every episode draws a control delay of 0–40 ms (the env's delay buffer
now reaches back several steps), 200M steps, one more Colab session (stopped after). Two versions:

| Run | Remembers its last | Inputs | Numbers | Wall time | Final eval crashes |
|---|---|---|---|---|---|
| lag40 | 6 commands (60 ms) | 39 | 6,980 | 2 min 25 s | 0% |
| lag40_h3 | 3 commands (30 ms) | 27 | 6,212 | 2 min 13 s | 2.7% |

The 14 s course with extra delay (the delay is on top of nothing: these brains get no built-in lag on the Mac):

| Extra delay | main (practised 0–4 ms) | lag40_h3 | lag40 | lag40 on Chip |
|---|---|---|---|---|
| 0 ms | 19.0 cm | 18.2 cm | **16.3 cm** | 16.3 cm |
| 10 ms | 18.8 cm | 18.1 cm | 16.4 cm | 16.4 cm |
| 15 ms | 33.5 cm, wobbling | 18.1 cm | 16.3 cm | 16.3 cm |
| 20 ms | **crash at 5.9 s** | 18.1 cm | 16.4 cm | 16.4 cm |
| 30 ms | crash at 1.4 s | 18.3 cm | 16.5 cm | 16.5 cm |
| 40 ms | crash at 1.1 s | 28.8 cm | **17.2 cm** | 17.2 cm |
| 60 ms | crash at 0.9 s | crash at 1.7 s | crash at 1.6 s | crash at 1.6 s |

- Practising with lag moved the crash point from +20 ms to past +40 ms, and it costs nothing: with no lag
  the lag-practised brain is slightly *more* accurate (16.3 vs 19.0 cm).
- Remembering more of its own recent commands helps at the edge: at +40 ms 17 cm (6 remembered) vs 29 cm (3).
- Past what it practised (+60 ms), both go down within 2 s.
- On Chip (`flash.sh lag40`, 284 µs per step with SIMD, matches the Mac to 3.9e-7): every number identical
  to the Mac.
- So the headline: a slow brain isn't what crashes a drone. A brain that's late *without expecting it* is.

Clip: `results/clips/lag_30ms_untrained_vs_practised.mp4` (left: main, 30 ms late, crashes; right: lag40,
30 ms late, flies the course).

Chip currently has **lag40** flashed (`tools/flash.sh main` to go back).

## Not done yet

