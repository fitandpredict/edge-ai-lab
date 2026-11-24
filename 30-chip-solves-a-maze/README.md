# 30 · My $5 Microcontroller Solved a Maze It's Never Seen

A maze rover with six laser rangefinders, wheel encoders and a gyro, simulated in MuJoCo. Its brain is a recurrent RL
policy (a 128-unit LSTM, RecurrentPPO, 85,316 parameters, 341 kB in float32) trained only on random 5×5 mazes. It runs on
an ESP32-S3 over USB: the sim sends what the rover's sensors measured, the board sends back forward / left / right / back,
once per 18 cm cell. A 200 Hz PID does the driving.

Inspired by Likith Surya Karthik Nulu's "Maze Hier-RL" LinkedIn post. This is my own re-implementation, not his code.

## Results (from `RESULTS.md`)
- On Chip: 6.8 ms per decision (float32, weights in PSRAM, SIMD), 5.8 ms int8; the float32 weights don't fit in internal RAM
- Chip picks the same action as the Mac on 1,565 / 1,565 decisions (50 unseen 7×7 mazes)
- Unseen mazes, cell level: 99% solved on 7×7, 85% on 9×9; physics with Chip driving: 98% (49/50) on 7×7
- Flood fill (no learning) still wins: 100%, 1.17× the shortest path. The left-hand rule: 100% but 2.0× the path, and 79% on mazes with loops (the brain: 89%)

## Run it
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python sim/rover.py --brain numpy --model models/lstm_final.zip --size 7 --seed 3 --video out/run.mp4 --verbose
.venv/bin/python sim/race.py --sizes 5 7 9 --mazes 200          # every brain vs left-hand rule vs flood fill
```

**Put the brain on an ESP32-S3** (N16R8 with PSRAM, Arduino core 3.3.x, `arduino-cli`), board on USB:
```bash
.venv/bin/python sim/export_brain.py models/lstm_final.zip   # writes firmware/maze_brain/policy_maze.h
tools/run_all.sh                                              # flash, benchmark, Chip-vs-Mac check, race, physics, clips
.venv/bin/python sim/rover.py --brain chip --size 7 --seed 3 --verbose
```

**Train your own** (CPU, about 30 min for 5M decisions):
```bash
.venv/bin/python sim/train.py --brain lstm --seed 1 --steps 5000000
.venv/bin/python sim/train.py --brain mlp --no-visits --steps 3000000   # the "no memory" ablation
```
`sim/rover.py --naive` reproduces the first physics run (no wheel calibration, no wall referencing: 84%).

## Files
- `sim/maze.py` mazes, the six-laser model and the cell-level training env (20 inputs)
- `sim/rover.py` the MuJoCo rover: wheels, gyro, rangefinders, 200 Hz PID, one brain decision per cell
- `sim/train.py`, `sim/race.py`, `sim/chip.py`, `sim/export_brain.py`
- `firmware/maze_brain/` the ESP32-S3 brain: LSTM step in C, five ways to run it (flash / PSRAM / SIMD / internal RAM / int8)
- `models/` the four trained brains, `results/` the raw numbers

What this doesn't show: no real rover (it's a sim), and the sim's lasers have no noise.
