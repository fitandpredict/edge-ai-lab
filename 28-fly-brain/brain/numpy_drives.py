"""Left-escape drives from the numpy brain for the live left run, beside the board's (for the 9d table)."""
import json, numpy as np
import fcl
g = np.load('data/graph.npz'); N = int(g['N'])
b = fcl.Brain(fcl.matrix(N, g['rows'], g['srcs'], g['counts']), fcl.circuit())
log = json.load(open('out/link_left.json'))
rows = [{'k': s['k'], 'board': s['drives'], 'mac': list(b.step(*s['looms']))} for s in log['steps']]
json.dump(rows, open('out/drives_board_vs_mac.json', 'w'), indent=1)
print(max(abs(r['board'][i] - r['mac'][i]) for r in rows for i in (0, 1)))
for r in rows[-5:]: print(r['k'], f"{r['board'][0]:.8f}", f"{r['mac'][0]:.8f}")
