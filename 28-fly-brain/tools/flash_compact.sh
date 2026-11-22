#!/usr/bin/env bash
# Build the 319-neuron compact brain, flash it to 0x210000 and the episode 28 firmware, then time it.
# usage: tools/flash_compact.sh [port]      (ESP32-S3 N16R8, arduino-cli with esp32:esp32 core 3.3.x)
set -euo pipefail
PORT=${1:-/dev/cu.usbmodem1101}
FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'
ESPTOOL=$(ls -d ~/Library/Arduino15/packages/esp32/tools/esptool_py/*/esptool ~/.arduino15/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1)
PYTHONPATH=brain .venv/bin/python brain/make_compact.py
arduino-cli compile --fqbn "$FQBN" --build-property 'compiler.optimization_flags=-O3 -ffp-contract=off' \
  --build-property 'upload.maximum_size=2097152' --build-path "$PWD/data/build_small" firmware/esp32_fly_small
"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 write_flash 0x210000 data/bundles/compact/connectome.fcl
arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir "$PWD/data/build_small" firmware/esp32_fly_small
sleep 4
.venv/bin/python brain/time_board.py "$PORT"
