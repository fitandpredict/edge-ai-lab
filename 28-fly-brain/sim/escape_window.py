"""Fly the escape decided at different times; find the latest decision that still clears the target."""
import sys
sys.path.insert(0, 'sim'); import link
for t in (0.31, 0.33, 0.345, 0.36, 0.375, 0.39):
    log = {'side': 'left', 'outcome': {'result': 'escape', 'danger_seen': 'left', 't': t}}
    r = link.fly(log, None)
    print(f'decision at {t*1000:.0f} ms: closest {r["closest_cm"]:.2f} cm, hit={r["caught_in_flight"]}', flush=True)
