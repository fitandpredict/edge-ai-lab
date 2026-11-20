#!/usr/bin/env bash
# Every MuJoCo clip episode 26 uses, rendered from real runs (1080p30, the bright look).
#   tools/render_clips.sh [nochip|chip-small|chip-big|all]
# chip-* clips drive the dog with the ESP32-S3 live: flash the matching brain first (tools/flash.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=../video/assets/footage
PORT=${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)}
mkdir -p "$OUT"
R() { local name=$1; shift; echo "== $name"; $PY sim/go1_loop.py --terrain full "$@" --video "$OUT/$name.mp4" --json "results/clips/$name.json" > "results/clips/$name.log" 2>&1 || { echo "FAILED $name"; tail -5 "results/clips/$name.log"; exit 1; }; grep -E "\"falls\"|distance_m|chip_infer_us_median" "results/clips/$name.log" || true; }
WHAT=${1:-nochip}

if [[ $WHAT == nochip || $WHAT == all ]]; then
  R D02Faceplant --brain random --secs 10.5 --view 2.4 -14 100
  R D05Min2 --brain npz --policy results/colab/small/snap_01.npz --secs 6 --cam fixed --lookat 0.6 0 0.25 --view 3.2 -14 120 \
    --hud '{"clock":{"label":"MINUTE","value":"2:01"},"counter":{"pos":"bl"}}'
  R D05Min4 --brain npz --policy results/colab/small/snap_05.npz --secs 6 --cam fixed --lookat 0.6 0 0.25 --view 3.2 -14 120 \
    --hud '{"clock":{"label":"MINUTE","value":"3:40"},"counter":{"pos":"bl"}}'
  R D05Min6 --brain npz --policy results/colab/small/snap_11.npz --secs 6 --cam fixed --lookat 0.6 0 0.25 --view 3.2 -14 120 \
    --hud '{"clock":{"label":"MINUTE","value":"5:49"},"counter":{"pos":"bl"}}'
  R D05Min8 --brain npz --policy results/colab/small/snap_17.npz --secs 9 --cam fixed --lookat 0.6 0 0.25 --view 3.2 -14 120 \
    --hud '{"clock":{"label":"MINUTE","value":"7:57","color":"GREEN"},"counter":{"pos":"bl","color":"GREEN"}}'
  R D05Small --brain npz --policy sim/ours_small.npz --secs 12 --size 960x1080 --view 2.6 -14 120 \
    --hud '{"pills":[{"text":"SMALL BRAIN","color":"GREEN","pos":"tc"}],"counter":{"pos":"bc"}}'
  R D05Big --brain npz --policy sim/ours_big.npz --secs 12 --size 960x1080 --view 2.6 -14 120 \
    --hud '{"pills":[{"text":"BIG BRAIN","color":"BLUE","pos":"tc"}],"counter":{"pos":"bc"}}'

  for d in 0 10 20 30; do
    c=YELLOW; [[ $d == 30 ]] && c=RED; [[ $d == 0 ]] && c=GREEN
    R D08Delay$d --brain onnx --secs 9 --delay-ms $d --view 2.4 -14 140 \
      --hud "{\"pills\":[{\"text\":\"THINKING DELAY: $d MS\",\"color\":\"$c\",\"pos\":\"tr\"}],\"counter\":{\"pos\":\"bl\"},\"fallen\":true}"
  done
  R D11OutroDusk --brain npz --policy sim/ours_small.npz --secs 17 --stop-at 12 --look dusk --view 3.4 -10 160
fi

if [[ $WHAT == chip-small || $WHAT == all ]]; then
  R D01Hook --brain chip --chip-mode 3 --port "$PORT" --secs 14 --view 2.2 -14 120 \
    --hud '{"pills":[{"text":"BRAIN: ESP32-S3","color":"GREEN","pos":"tr"}],"chip_us":{"pos":"tr2"}}'
  R D07ChipWalk --brain chip --chip-mode 3 --port "$PORT" --secs 22 --view 2.6 -16 60 \
    --hud '{"pills":[{"text":"CHIP IN CHARGE","color":"GREEN","pos":"tr"}],"chip_us":{"pos":"tr2"},"counter":{"pos":"bl"},"fallen":true}'
fi

if [[ $WHAT == chip-big || $WHAT == all ]]; then
  B() { R D09BigMode$1 --brain chip --chip-mode $1 --real-latency --port "$PORT" --secs $4 --view 2.4 -14 140 \
      --hud "{\"pills\":[{\"text\":\"BIG BRAIN · $2\",\"color\":\"$3\",\"pos\":\"tr\"}],\"chip_us\":{\"pos\":\"tr2\"},\"counter\":{\"pos\":\"bl\"},\"fallen\":true}"; }
  B 0 "FLASH · PLAIN C" ORANGE 10
  B 1 "PSRAM · PLAIN C" YELLOW 20
  B 2 "PSRAM · SIMD" YELLOW 10
  B 4 "INT8 · FAST RAM" GREEN 10
fi

# Stand-ins for the Chip clips when the board isn't plugged in: the same brains on the Mac, with Chip's
# measured time per step (results/chip_check_ours_*.json). Replace with chip-small / chip-big before the final.
if [[ $WHAT == placeholder ]]; then
  R D01Hook --brain npz --policy sim/ours_small.npz --secs 14 --view 2.2 -14 120 \
    --hud '{"pills":[{"text":"BRAIN: ESP32-S3","color":"GREEN","pos":"tr"},{"text":"CHIP: 575 µS","color":"GREEN","pos":"tr2"}]}'
  R D07ChipWalk --brain npz --policy sim/ours_small.npz --secs 22 --view 2.6 -16 60 \
    --hud '{"pills":[{"text":"CHIP IN CHARGE","color":"GREEN","pos":"tr"},{"text":"CHIP: 575 µS","color":"GREEN","pos":"tr2"}],"counter":{"pos":"bl"},"fallen":true}'
  P() { R D09BigMode$1 --brain npz --policy sim/ours_big.npz --real-latency --think-us $5 --secs $4 --view 2.4 -14 140 \
      --hud "{\"pills\":[{\"text\":\"BIG BRAIN · $2\",\"color\":\"$3\",\"pos\":\"tr\"},{\"text\":\"CHIP: $5 µS\",\"color\":\"$3\",\"pos\":\"tr2\"}],\"counter\":{\"pos\":\"bl\"},\"fallen\":true}"; }
  P 0 "FLASH · PLAIN C" ORANGE 10 30849
  P 1 "PSRAM · PLAIN C" YELLOW 20 20541
  P 2 "PSRAM · SIMD" YELLOW 10 14392
  P 4 "INT8 · FAST RAM" GREEN 10 12226
fi

if [[ $WHAT == retime ]]; then
  R D02Faceplant --brain random --secs 10.5 --view 2.4 -14 100
  R D05Min8 --brain npz --policy results/colab/small/snap_17.npz --secs 9 --cam fixed --lookat 0.6 0 0.25 --view 3.2 -14 120 \
    --hud '{"clock":{"label":"MINUTE","value":"7:57","color":"GREEN"},"counter":{"pos":"bl","color":"GREEN"}}'
  R D11OutroDusk --brain npz --policy sim/ours_small.npz --secs 17 --stop-at 12 --look dusk --view 3.4 -10 160
fi

# Multi-beat clips (footage scenes that span several beats)
cd "$OUT"
cat_clips() { local out=$1; shift; local args=() fl="" n=0; for spec in "$@"; do IFS=: read f ss d <<< "$spec"; args+=(-ss "$ss" -t "$d" -i "$f.mp4"); fl+="[$n:v]"; n=$((n+1)); done
  ffmpeg -loglevel error -y "${args[@]}" -filter_complex "${fl}concat=n=$n:v=1:a=0[v]" -map "[v]" -c:v libx264 -crf 18 -pix_fmt yuv420p "$out.mp4"; echo "assembled $out"; }
# a slow push-in for shots where the dog stands or lies still (a frozen-looking frame reads as a stuck video)
pushin() { ffmpeg -loglevel error -y -i "$1.mp4" -vf "scale=3840:2160,zoompan=z='min(1+0.00045*on,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p "$2.mp4"; }
[[ -f D05Min6.mp4 ]] && cat_clips D05Stand_raw D05Min2:0:2.5 D05Min4:0:2.5 D05Min6:0:2.6 && pushin D05Stand_raw D05Stand
[[ -f D11OutroDusk.mp4 ]] && pushin D11OutroDusk D11OutroPush
[[ -f D05Min8.mp4 ]] && pushin D05Min8 D05Min8Push
[[ -f D08Delay30.mp4 ]] && cat_clips D08Delay D08Delay0:0:9.6 D08Delay10:0:4.3 D08Delay20:0:6.8 D08Delay30:0:6
[[ -f D09BigMode4.mp4 ]] && # the flash run is down by 2 s; its last 2.5 s hold the final frame (the dog lies still) to fit the voice
[[ -f D09BigMode0.mp4 ]] && ffmpeg -loglevel error -y -i D09BigMode0.mp4 -vf tpad=stop_mode=clone:stop_duration=2.6 -c:v libx264 -crf 18 -pix_fmt yuv420p D09BigMode0_long.mp4
[[ -f D09BigMode0_long.mp4 ]] && pushin D09BigMode0_long D09BigMode0_push
[[ -f D09BigMode1.mp4 ]] && ffmpeg -loglevel error -y -ss 2 -t 14.2 -i D09BigMode1.mp4 -c:v libx264 -crf 18 -pix_fmt yuv420p D09BigMode1_cut.mp4 && pushin D09BigMode1_cut D09BigMode1_push
[[ -f D09BigMode4.mp4 ]] && cat_clips D09Modes D09BigMode0_push:0:12.5 D09BigMode1_push:0:14.2 D09BigMode2:0:8.6 D09BigMode4:0:7.4
[[ -f D05Big.mp4 ]] && ffmpeg -loglevel error -y -i D05Small.mp4 -i D05Big.mp4 -filter_complex hstack -c:v libx264 -pix_fmt yuv420p -crf 18 D05Duo.mp4
true
