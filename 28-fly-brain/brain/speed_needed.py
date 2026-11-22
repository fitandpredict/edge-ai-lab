"""Real time: each neural step costs dt of world time. What is the largest dt that still saves the fly?
Same scene and loom formula as sim/link.py; numpy brain (matches the board's gold)."""
import sys, numpy as np
sys.path.insert(0, 'sim'); import link
import fcl
g = np.load('data/graph.npz'); N = int(g['N']); rows, srcs, counts = g['rows'], g['srcs'], g['counts']
circ = fcl.circuit(); ins, outs, amp, thr = circ
on = np.zeros(N, bool); on[np.r_[ins['L'], ins['R'], outs['L'], outs['R']]] = True
brains = {'full': np.ones(len(rows), bool), 'direct (0.22%)': on[rows] & on[srcs], 'count>=12 (9.4%)': counts >= 12}
target, t_hit, _ = link.target_path('left')
def trial(W, dt):
    b = fcl.Brain(W, circ)
    for k in range(10000):
        t = k * dt
        looms, dist = link.looms_at(t, target)
        if dist <= link.CATCH or t >= t_hit: return ('caught', k, t)
        dl, dr = b.step(*looms)
        if dl > thr or dr > thr: return ('escape', k, t + dt)
for name, keep in brains.items():
    W = fcl.matrix(N, rows[keep], srcs[keep], counts[keep])
    best = None
    for dt in (0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.050, 0.113):
        r = trial(W, dt)
        print(f'{name:18s} dt={dt*1000:5.0f} ms -> {r[0]} at step {r[1]}, t={r[2]*1000:.0f} ms')
        if r[0] == 'escape': best = dt
    print(f'  => largest step that escapes: {best*1000 if best else None} ms\n')
