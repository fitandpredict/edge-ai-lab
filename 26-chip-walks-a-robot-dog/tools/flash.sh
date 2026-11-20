#!/usr/bin/env bash
# Build and flash Chip's dog brain with one policy header, then check it against the Mac.
#   tools/flash.sh ours_small            # or ours_big / big (DeepMind's pretrained)
#   PORT=/dev/cu.usbmodem1101 tools/flash.sh ours_small
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-ours_small}
PORT=${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1)}
[ -n "$PORT" ] || { echo "Chip not found: plug the ESP32-S3 in over USB"; exit 1; }
FQBN='esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi'
BUILD=${TMPDIR:-/tmp}/dog_brain_$NAME
arduino-cli compile --fqbn "$FQBN" --build-property "compiler.cpp.extra_flags=-DPOLICY_HEADER=\"policy_$NAME.h\"" \
  --build-path "$BUILD" firmware/dog_brain
arduino-cli upload --fqbn "$FQBN" --input-dir "$BUILD" -p "$PORT" firmware/dog_brain
sleep 2
NPZ=sim/$NAME.npz
.venv/bin/python sim/chip_check.py --port "$PORT" --npz "$NPZ"
echo "now: .venv/bin/python sim/go1_loop.py --brain chip --port $PORT --secs 20 --video results/clips/chip_$NAME.mp4"
