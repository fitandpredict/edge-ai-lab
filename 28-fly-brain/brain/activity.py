"""Replay the live left run (out/link_left.json, full brain on the board) through the numpy brain:
check every step's escape drives against what the board reported, and save the whole-brain activity."""
import json, numpy as np
import fcl
g = np.load('data/graph.npz'); N = int(g['N'])
circ = fcl.circuit(); b = fcl.Brain(fcl.matrix(N, g['rows'], g['srcs'], g['counts']), circ)
log = json.load(open('out/link_left.json'))
states, worst = [], 0.0
for s in log['steps']:
    dl, dr = b.step(*s['looms'])
    worst = max(worst, abs(dl - s['drives'][0]), abs(dr - s['drives'][1]))
    states.append(b.state.astype(np.float16))
states = np.array(states)
np.savez_compressed('out/activity_left.npz', states=states, looms=np.array([s['looms'] for s in log['steps']]))
act = (states[-1] > 0.001).sum()
print(f'{len(states)} steps; worst |numpy - board| drive difference {worst:.2e}; '
      f'neurons above 0.001 at the escape step: {act} of {N}; file out/activity_left.npz')
