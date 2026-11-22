# Results (ESP32-S3 N16R8, 21–22 Nov 2026)

| What | Value | Where from |
|---|---|---|
| Graph | 48,311 neurons, 9,462,135 connections, 14,174,411 bytes, fnv1a d66d1348 | board INFO |
| PSRAM block cache | 530 of 1,510 graph blocks (6.1 MB) | board INFO |
| One step, full brain | 1.56 s compute, 1.60 s per cycle (25 steps, 1,556,956–1,563,009 µs) | board |
| Numpy vs board | max difference 1.1e-8 over the live left run | brain/activity.py |
| Active neurons at the escape | 3,620 of 48,311 (> 0.001) | brain/activity.py |
| World waits (10 ms/step) | escape at step 30 (310 ms), correct side both runs; closest 1.04 / 1.03 cm | sim/link.py |
| Real time, full brain | ball arrives 440 ms, before step 1 finishes: hit (closest 0.04 cm) | sim/link.py --brain-dt 1.6 |
| Latest safe decision | ~330 ms (345 ms is clipped) | sim/escape_window.py |
| Pruning: count ≥ 3 / ≥ 12 / ≥ 40 | 39.1 % / 9.4 % / 1.8 % of connections: all escape | brain/prune.py |
| Random drops of the same size | all escape | brain/prune.py |
| 1-hop pathway | 1,288 neurons, 130,864 connections: escapes | brain/prune.py |
| Eyes + escape neurons only | 319 neurons, 21,288 connections (0.22 %): escapes at step 23 | brain/prune.py |
| Board, all 48,311 slots | full 1,560 · count ≥ 3 ~480 · count ≥ 12 ~200 · direct 113 ms per step | brain/time_board.py |
| Board, compact 319 neurons | 2.2 ms per step, same drives | tools/flash_compact.sh |
| Real time, compact | escape at 188 ms after 90 steps; ball centre 1.62 cm away | sim/link.py --brain-dt 0 |
| flybody flight controller in numpy | reward 0.83–0.93 per step vs 0.41–0.52 doing nothing | sim/fly_flight.py |
