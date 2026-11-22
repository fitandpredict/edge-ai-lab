"""How much of the fly brain can go before the escape breaks?

Strategies (all keep the 48,311 neuron slots so the escape circuit indices stay valid):
  weak    drop edges whose contact count is below c
  random  drop the same number of edges at random (control)
  path    keep only edges between neurons that are within k hops downstream of the looming inputs
          AND within k hops upstream of the escape outputs
Test: the left-approach scenario from the live link run (out/link_left.json looms), and its mirror.
Pass = escapes on the correct side; we report the step it fires (live board: step 30).
"""
import json, numpy as np, scipy.sparse as sp
import sys
import fcl
ONLY_PATH = '--path' in sys.argv

g = np.load('data/graph.npz'); N = int(g['N']); rows, srcs, counts = g['rows'], g['srcs'], g['counts']
circ = fcl.circuit(); ins, outs, amp, thr = circ
looms = [s['looms'] for s in json.load(open('out/link_left.json'))['steps']]
looms += [looms[-1]] * 15   # keep looming at the last level if it hasn't fired yet


def run(keep):
    W = fcl.matrix(N, rows[keep], srcs[keep], counts[keep])
    res = {}
    for name, seq in (('left', looms), ('right', [l[::-1] for l in looms])):
        b = fcl.Brain(W, circ); fired = None
        for k, (l, r) in enumerate(seq):
            dl, dr = b.step(l, r)
            if dl > thr or dr > thr:
                fired = (k, 'left' if dl > dr else 'right'); break
        res[name] = fired
    ok = res['left'] and res['right'] and res['left'][1] == 'left' and res['right'][1] == 'right'
    return ok, res


def hops(start, A, k):
    seen = np.zeros(N, bool); seen[start] = True; front = seen.copy()
    for _ in range(k):
        front = (A @ front.astype(np.float32)) > 0
        front &= ~seen; seen |= front
    return seen


E = len(rows)
print(f'full: {E} edges', run(np.ones(E, bool)))
rng = np.random.default_rng(0)
print('\n-- weak edges dropped --')
for c in () if ONLY_PATH else (2, 3, 5, 8, 12, 20, 40):
    keep = counts >= c; ok, res = run(keep)
    print(f'count >= {c:2d}: {keep.mean()*100:5.1f}% edges kept, {np.count_nonzero(np.bincount(rows[keep], minlength=N) + np.bincount(srcs[keep], minlength=N))} neurons used, ok={ok} {res}')
    rk = np.zeros(E, bool); rk[rng.choice(E, keep.sum(), replace=False)] = True
    ok2, res2 = run(rk); print(f'   random same size: ok={ok2} {res2}')
print('\n-- escape pathway only (k hops downstream of the eyes AND upstream of the escape neurons) --')
A = sp.csr_matrix((np.ones(E, np.float32), (rows, srcs)), shape=(N, N))   # dst <- src
AT = A.T.tocsr()
for k in (() if '--direct' in sys.argv else (0, 1, 2, 3)):
    down = hops(np.r_[ins['L'], ins['R']], A, k)
    up = hops(np.r_[outs['L'], outs['R']], AT, k)
    on = down & up
    keep = on[rows] & on[srcs]; ok, res = run(keep)
    print(f'k={k}: {on.sum()} neurons, {keep.sum()} edges ({keep.mean()*100:.3f}%), ok={ok} {res}')

if '--direct' in sys.argv:
    on = np.zeros(N, bool); on[np.r_[ins['L'], ins['R'], outs['L'], outs['R']]] = True
    keep = on[rows] & on[srcs]
    print(f'direct eye->escape only: {on.sum()} neurons, {keep.sum()} edges, ok/res', run(keep))
