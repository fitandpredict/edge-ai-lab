"""Decode the graph and check the numpy rate model against the released escape gold."""
import time, numpy as np
import fcl
t = time.time(); N, order, rows, srcs, counts = fcl.read(); print(f'decoded {N} neurons, {len(rows)} edges in {time.time()-t:.1f} s')
np.savez_compressed('data/graph.npz', N=N, order=order, rows=rows.astype(np.int32), srcs=srcs.astype(np.int32), counts=counts.astype(np.int32))
circ = fcl.circuit(); b = fcl.Brain(fcl.matrix(N, rows, srcs, counts), circ)
gold, atol, rtol = fcl.escape_gold()
outs = np.r_[circ[1]['L'], circ[1]['R']]
worst = 0
for c, (looms, exp) in enumerate(gold):
    b.reset()
    for s in range(len(looms)):
        b.step(*looms[s]); got = b.state[outs]
        err = np.abs(got - exp[s]) - (atol + rtol * np.abs(exp[s])); worst = max(worst, err.max())
    print(f'case {c}: final drives L {exp[-1][:4].mean():.4f} R {exp[-1][4:].mean():.4f}')
print('PASS' if worst <= 0 else f'FAIL by {worst}', f'(atol {atol}, rtol {rtol})')
