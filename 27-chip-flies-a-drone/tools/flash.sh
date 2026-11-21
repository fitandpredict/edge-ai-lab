#!/usr/bin/env bash
# Build and flash Chip's drone brain with one policy header, then check it against the Mac.
#   tools/flash.sh main          # or tiny
#   PORT=/dev/cu.usbmodem1101 tools/flash.sh main
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-main}
PORT=${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1)}
[ -n "$PORT" ] || { echo "Chip not found: plug the ESP32-S3 in over USB"; exit 1; }
FQBN='esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi'
BUILD=${TMPDIR:-/tmp}/drone_brain_$NAME
arduino-cli compile --fqbn "$FQBN" --build-property "compiler.cpp.extra_flags=-DPOLICY_HEADER=\"policy_$NAME.h\"" \
  --build-path "$BUILD" firmware/drone_brain
arduino-cli upload --fqbn "$FQBN" --input-dir "$BUILD" -p "$PORT" firmware/drone_brain
sleep 2
NPZ=sim/brains/$NAME.npz
.venv/bin/python sim/chip_check.py --port "$PORT" --npz "$NPZ"
echo "now: .venv/bin/python sim/drone_loop.py --brain chip --port $PORT --secs 20 --video results/clips/chip_$NAME.mp4"
