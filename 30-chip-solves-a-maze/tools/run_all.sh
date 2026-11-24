#!/usr/bin/env bash
# Every number in episode 30, from the trained models. Chip must be on USB.
#   tools/run_all.sh            # writes results/*.json and results/runs/*.json, results/clips/*.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PORT=${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1)}
[ -n "$PORT" ] || { echo "Chip not found: plug the ESP32-S3 in over USB"; exit 1; }
FQBN='esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi'
BUILD=${TMPDIR:-/tmp}/maze_brain
MODEL=${MODEL:-models/lstm_final.zip}
mkdir -p results/runs results/clips

echo "== 1. export the brain and flash Chip"
$PY sim/export_brain.py "$MODEL" | tee results/export.txt
arduino-cli compile --fqbn "$FQBN" --build-path "$BUILD" firmware/maze_brain | grep -E "Sketch uses|Global" | tee results/firmware_size.txt
arduino-cli upload --fqbn "$FQBN" --input-dir "$BUILD" -p "$PORT" firmware/maze_brain >/dev/null
sleep 3

echo "== 2. Chip: info, speed in every mode, same answers as the Mac (float32 and int8)"
$PY sim/chip.py info --port "$PORT" --out results/chip_info.json
$PY sim/chip.py bench --port "$PORT" --out results/chip_bench.json
$PY sim/chip.py check --port "$PORT" --model "$MODEL" --mode 2 --mazes 50 --out results/chip_check_f32.json
$PY sim/chip.py check --port "$PORT" --model "$MODEL" --mode 4 --mazes 50 --out results/chip_check_int8.json

echo "== 3. the race on unseen mazes (cell level, deterministic)"
$PY sim/race.py --sizes 5 7 9 --mazes 200 --out results/race.json
$PY sim/race.py --sizes 7 --mazes 200 --loops 8 --out results/race_loops.json

echo "== 4. physics: Chip drives the MuJoCo rover on 50 unseen 7x7 mazes"
$PY sim/rover.py --brain chip --port "$PORT" --model "$MODEL" --size 7 --eval 50 --out results/physics_chip_7x7.json | tail -1

echo "== 5. hero runs for the video (logs for the bench replay + MuJoCo clips)"
for seed in 3 11 27; do
  $PY sim/rover.py --brain chip --port "$PORT" --model "$MODEL" --size 7 --seed $seed \
    --log results/runs/chip_7x7_s$seed.json --video results/clips/chip_7x7_s$seed.mp4
done
$PY sim/rover.py --brain chip --port "$PORT" --model "$MODEL" --size 5 --seed 3 \
  --log results/runs/chip_5x5_s3.json --video results/clips/chip_5x5_s3.mp4
echo done
