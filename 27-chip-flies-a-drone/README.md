# 27 · Can a $5 Microcontroller Fly a Drone?

A simulated Crazyflie-sized quadcopter learns to fly with reinforcement learning on a rented GPU (about 2 minutes,
8,192 drones at once), then an ESP32-S3 runs the learned brain and flies it in MuJoCo over USB, 100 times a second.
Then the experiment: how late can the brain's commands be before the drone falls? And does a brain that practises
with lag survive it? Includes the training env and scripts, seven trained brains, the ESP32-S3 firmware and the
sim that puts the board in charge.

## Run it

**Watch a trained brain fly (Mac/Linux, no board):**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python sim/drone_loop.py --policy sim/brains/main.npz --secs 14 --start 0,0,0.3 \
  --targets "0,0,1;1,0,1.5@3;1,1,1@6;0,0,1@9" --gust 0.25@10@90 --video out/main.mp4
.venv/bin/python sim/drone_loop.py --policy sim/brains/main.npz --delay-ms 20 --secs 14 ...    # falls
.venv/bin/python sim/drone_loop.py --policy sim/brains/lag40.npz --delay-ms 40 --secs 14 ...   # flies
.venv/bin/python tools/sweep.py delay --policy sim/brains/lag40.npz                           # the whole table
```

**Put the brain on an ESP32-S3** (N16R8, Arduino core 3.3.x, `arduino-cli`): plug the board in, then

```bash
tools/flash.sh main          # builds, flashes, checks the board against the Mac, benchmarks every mode
.venv/bin/python sim/drone_loop.py --brain chip --port /dev/cu.usbmodem1101 --secs 14 --video out/chip.mp4
.venv/bin/python tools/sweep.py delay --brain chip --port /dev/cu.usbmodem1101
```

`tools/flash.sh tiny` (788 numbers) and `tools/flash.sh lag40` work the same way (lag40 remembers 6 past
commands, so give the Mac `--hist 6` when it flies with `--brain chip`). Make a header from any brain with
`sim/export_policy.py sim/brains/<name>.npz --name <name>`.

**Train your own** (Google Colab A100 via the `colab` CLI; about 2–2.5 minutes per brain):

```bash
colab new -s drone --gpu A100
for f in colab/setup_playground.sh colab/drone_env.py colab/train_drone.py sim/drone_model.py; do
  colab upload -s drone $f /content/$(basename $f); done
colab ssh -s drone -- bash /content/setup_playground.sh
colab ssh -s drone -- bash -lc 'cd /content && LD_LIBRARY_PATH=/usr/lib64-nvidia venv-mjp/bin/python train_drone.py --name main --hidden 64 64'
colab ssh -s drone -- bash -lc 'cd /content && LD_LIBRARY_PATH=/usr/lib64-nvidia venv-mjp/bin/python train_drone.py --name lag40 --max-delay-ms 40 --hist 6 --steps 200e6'
colab download -s drone /content/runs/main/final.npz sim/brains/my_main.npz
colab stop -s drone          # an idle A100 still burns units
```

`colab/check_physics.py` compares the trainer's plain-JAX rigid body with MuJoCo (0.5 mm apart after 0.6 s).

## My results

All measured on 20–21 Nov 2026 (full log in `RESULTS.md`):

- Training (Brax PPO, 8,192 drones, 157M steps, Colab A100): 2 min 10 s. Crashed test flights: 100 % at 8M
  steps, 77 % at 16M, 0 % at 24M.
- The brain: 27 inputs → 64 → 64 → 4 motor commands, 6,212 numbers. A 788-number brain flies the course too.
- On the ESP32-S3: 258 µs per step with SIMD (2.6 % of the 10 ms step), 539 µs plain C, 615 µs int8 (no SIMD
  path yet); the tiny brain 66 µs. Matches the Mac to 6e-7.
- Extra delay (14 s course, waypoints and a gust): fine at 10 ms, wobbling at 15 ms, crashes at 20 ms.
- Brain trained with 0–40 ms of lag: flies at 40 ms (17 cm mean error), crashes at 60 ms; with no lag it's slightly
  more accurate than the original (16 vs 19 cm). Same numbers on the board.
- Loop rate: a brain trained at 25 Hz flies; the 100 Hz brain run at 25 Hz crashes at 4.2 s.

## What this doesn't show

- It's a simulation: no real drone flew. The laptop waits for the board every step, so USB time isn't counted.
- The GPU did the learning; the board only runs it. One training run per brain.

## Credits

- Drone model (mass, inertia, thrust and torque fits, motor lag): Gräfe, Scherer, Hönig, Trimpe,
  "How to Model Your Crazyflie Brushless" (arXiv 2603.05944),
  [CrazyflieBrushJAX](https://github.com/Data-Science-in-Mechanical-Engineering/CrazyflieBrushJAX), MIT.
  One change: Izz = Ixx + Iyy (−4 %) so MuJoCo accepts the inertia.
- Training stack: [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Apache 2.0), Brax, JAX.
- `sim/hud.py` draws labels with Press Start 2P (SIL OFL) if `sim/PressStart2P-Regular.ttf` is present.
