# 28 · I Put a Real Fly's Brain in a Virtual Body

A 48,311-neuron piece of a real male fruit fly's connectome (MaleCNS v1.0, Janelia) runs on an ESP32-S3 and
decides when, and which way, a simulated fly escapes a looming ball. Its body is flybody, flown in MuJoCo by
flybody's trained flight controller along a real fly's recorded escape. The whole brain needs 1.6 s per step,
so in real time the fly gets hit. Then I prune the brain until it's fast enough: 319 neurons (the eyes' looming
detectors and the escape neurons, 0.22 % of the connections) decide in 2.2 ms and the fly escapes in real time.

## Run it

You need Python 3.10 (flybody's requirement), an ESP32-S3 N16R8 for the board runs, and `arduino-cli` with the
`esp32:esp32` core 3.3.x. Everything runs from this folder.

**1. The brain** (the connectome bundle from [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai), MIT; data CC-BY 4.0):

```bash
git clone https://github.com/slvDev/esp32-ai
(cd esp32-ai && scripts/fetch_model.sh fly)        # downloads and verifies artifacts/fly (~87 MB)
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python brain/check_gold.py                # decodes the graph, checks the numpy model against the released gold
```

**2. The body** ([flybody](https://github.com/TuragaLab/flybody), Apache-2.0; data GPL-3.0+ on
[Figshare](https://doi.org/10.25378/janelia.25309105)):

```bash
git clone https://github.com/TuragaLab/flybody && .venv/bin/pip install -e flybody
# From the Figshare record, unzip into data/: trained-fly-policies.zip and datasets_flight-imitation.zip
#   -> data/trained-fly-policies/flight/ and data/flight-imitation/
.venv-tf/bin/python sim/export_flight_weights.py    # one-off: TF checkpoint -> data/flight_policy.npz (see the file)
PYTHONPATH=flybody MUJOCO_GL=glfw .venv/bin/python sim/fly_flight.py 3 --video out/flight.mp4
```

On Linux without a display use `MUJOCO_GL=egl`.

**3. Link them.** Flash the full brain with esp32-ai's own `scripts/deploy.sh fly`, then:

```bash
export PYTHONPATH=flybody MUJOCO_GL=glfw
.venv/bin/python sim/link.py --side left --brain-dt 0.01 --log out/link_left.json --video out/link_left.mp4   # the world waits: escapes
.venv/bin/python sim/link.py --side left --brain-dt 1.6 --log out/link_truetime.json --video out/link_truetime.mp4  # real time: hit
.venv/bin/python sim/link.py --replay out/link_left.json --video out/again.mp4      # re-fly a saved decision, no board
```

**4. Throw brain away.**

```bash
.venv/bin/python brain/prune.py                     # which pruned brains still escape (numpy, matches the board)
PYTHONPATH=brain .venv/bin/python brain/speed_needed.py   # how fast each brain must be in real time
tools/flash_compact.sh /dev/cu.usbmodem1101         # 319-neuron brain + this folder's firmware, then times it
.venv/bin/python sim/link.py --side left --brain-dt 0 --log out/link_compact_realtime.json --video out/compact.mp4
```

`brain/make_bundle.py <name> count 12` (or `path 1`, `direct 0`) builds pruned bundles that keep all 48,311
neuron slots, for esp32-ai's `deploy.sh fly` (`ARTIFACTS=` / `MANIFEST=`, see the file).

## My results

ESP32-S3 N16R8, Arduino core 3.3.10, 21–22 Nov 2026 (full list in `RESULTS.md`):

- Full brain: 48,311 neurons, 9,462,135 connections, 14.2 MB in flash; **1.56 s per step** (1.60 s per cycle).
- Numpy model = board to 1.1e-8 on every step of the live run.
- World waits for the brain (10 ms per step, my assumption): escapes at step 30 on the correct side, both
  sides; the ball passes 1.04 cm from the fly (radius 0.6 cm).
- Real time, full brain: the ball arrives at 440 ms, before Chip's second step finishes: hit.
- Pruned (all still escape on the right side): contact count ≥ 3 (39.1 % of connections) ~480 ms per step;
  ≥ 12 (9.4 %) ~200 ms; eyes + escape neurons only (0.22 %) 113 ms with all slots, **2.2 ms** compacted to 319
  neurons. Same-size random drops escape too.
- Real time, compact brain: escapes at 188 ms after 90 steps; ball centre 1.62 cm away.

## Caveats

- One behaviour only (escape from a looming ball). All connections are excitatory and it's a rate model, no
  spikes; 10 ms per step is my assumption, not a measured fly number.
- The brain decides *whether and which side*; the body flies a real fly's recorded evasion (Muijres et al.,
  Science 2014) away from that side. The body didn't learn the escape.
- The Mac waits for the board every step in the paused runs; USB time isn't counted in the real-time ones
  beyond the board's own reported compute time.

## Files

- `brain/fcl.py`: reads the FCL1 graph and runs the rate model in numpy (`FLY_BUNDLE` = path to `artifacts/fly`).
- `brain/check_gold.py`, `activity.py`, `numpy_drives.py`: check numpy against the released gold and the live board run.
- `brain/prune.py`, `speed_needed.py`: the throw-away-brain experiments. `make_bundle.py`, `make_compact.py`: board bundles.
- `brain/time_board.py`: times decisions on whatever graph is flashed.
- `sim/link.py`: board decides (pass 1), flybody flies it (pass 2). `flight_policy.py`: flybody's flight controller in numpy.
- `sim/fly_flight.py`, `pick_evasions.py`, `score_evasions.py`, `escape_window.py`: checks and choices behind the flight.
- `sim/still.py`: transparent stills of the fly (the thumbnails).
- `firmware/esp32_fly_small/`: esp32-ai's fly sketch, changed to accept fewer than 48,311 neurons (builds against
  `esp32-ai/runtime/`). `tools/flash_compact.sh` builds and flashes it.

Third party: connectome data MaleCNS v1.0 (Janelia, CC-BY 4.0) via slvDev/esp32-ai (code MIT, Viacheslav
Sierbov; licence in `firmware/esp32_fly_small/LICENSE-esp32-ai`); flybody (Vaxenburg et al., Nature 2025,
Janelia + Google DeepMind; code Apache-2.0, data GPL-3.0+, not included here); flight trajectories from
Muijres et al. 2014/2015 inside flybody's dataset; MuJoCo (Apache-2.0).
