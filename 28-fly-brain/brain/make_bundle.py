"""Build a pruned fly bundle that deploy.sh accepts: same 48,311 neurons and circuit, fewer connections.

usage: .venv/bin/python brain/make_bundle.py <name> (path K | count C | direct 0)
writes data/bundles/<name>/{connectome.fcl, gold-device-order.bin, escape-circuit.json, escape-gold.bin,
neurons.csv, model_bundle.json}. Gold files reuse the released cases' inputs and initial states; the
expected outputs are recomputed on the pruned graph with the float32 reference (fcl.Brain).
Flash (full-size bundles, in a clone of github.com/slvDev/esp32-ai): ARTIFACTS=<abs bundle> MANIFEST=<abs bundle>/model_bundle.json scripts/deploy.sh fly
"""
import hashlib, json, shutil, struct, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, lz4.block
import fcl

SRC = fcl.BUNDLE


def sha(b): return hashlib.sha256(b).hexdigest()


def varints(vals):
    out = bytearray()
    for v in vals:
        v = int(v)
        while True:
            b = v & 0x7F; v >>= 7
            out.append(b | (0x80 if v else 0))
            if not v: break
    return bytes(out)


def encode(N, order, rows, srcs, counts, rpb=32):
    o = np.lexsort((srcs, rows)); rows, srcs, counts = rows[o], srcs[o], counts[o]
    deg = np.bincount(rows, minlength=N)
    starts = np.r_[0, np.cumsum(deg)]
    out = bytearray(b'FCL1' + struct.pack('<4I', 1, N, len(rows), rpb) + order.astype('<u4').tobytes())
    for r0 in range(0, N, rpb):
        r1 = min(N, r0 + rpb); a, b = starts[r0], starts[r1]
        d = []
        for r in range(r0, r1):
            s = srcs[starts[r]:starts[r + 1]]
            if len(s): d.append(np.r_[s[0], np.diff(s)])
        delta = np.concatenate(d) if d else np.zeros(0, np.int64)
        raw = varints(deg[r0:r1]) + varints(delta) + varints(counts[a:b])
        comp = lz4.block.compress(raw, store_size=False, mode='high_compression')
        out += struct.pack('<4I', r1 - r0, b - a, len(raw), len(comp)) + comp
    return bytes(out)


def main():
    name, kind, val = sys.argv[1], sys.argv[2], int(sys.argv[3])
    g = np.load('data/graph.npz'); N = int(g['N']); order = g['order']
    rows, srcs, counts = g['rows'].astype(np.int64), g['srcs'].astype(np.int64), g['counts'].astype(np.int64)
    ins, outs, amp, thr = circ = fcl.circuit()
    if kind == 'count':
        keep = counts >= val
    elif kind == 'direct':
        on = np.zeros(N, bool); on[np.r_[ins['L'], ins['R'], outs['L'], outs['R']]] = True
        keep = on[rows] & on[srcs]
    else:
        E = len(rows)
        A = sp.csr_matrix((np.ones(E, np.float32), (rows, srcs)), shape=(N, N))   # dst <- src
        def hops(start, M):
            seen = np.zeros(N, bool); seen[start] = True; front = seen.copy()
            for _ in range(val):
                front = (M @ front.astype(np.float32)) > 0; front &= ~seen; seen |= front
            return seen
        down = hops(np.r_[ins['L'], ins['R']], A)        # reachable from inputs
        up = hops(np.r_[outs['L'], outs['R']], A.T.tocsr())   # can reach outputs
        on = down & up; keep = on[rows] & on[srcs]
    rows, srcs, counts = rows[keep], srcs[keep], counts[keep]
    out = Path('data/bundles') / name; out.mkdir(parents=True, exist_ok=True)
    graph = encode(N, order, rows, srcs, counts)
    (out / 'connectome.fcl').write_bytes(graph)
    gsha = sha(graph)
    W = fcl.matrix(N, rows, srcs, counts)

    # Graph gold: same cases, initial states and inputs; recomputed expected states.
    raw = (SRC / 'gold-device-order.bin').read_bytes()
    magic, ver, n, cases, steps, atol, rtol = struct.unpack_from('<4s4I2f', raw)
    buf = bytearray(raw[:28]); pos = 28
    for _ in range(cases):
        s0 = np.frombuffer(raw, '<f4', N, pos); pos += 4 * N; buf += s0.tobytes(); s = s0.copy()
        for _ in range(steps):
            x = np.frombuffer(raw, '<f4', N, pos); pos += 8 * N
            s = (0.5 * s + 0.5 * np.tanh(W @ s + x)).astype(np.float32)
            buf += x.tobytes() + s.astype('<f4').tobytes()
    (out / 'gold-device-order.bin').write_bytes(bytes(buf))

    c = json.loads((SRC / 'escape-circuit.json').read_text())
    c['graph_sha256'] = gsha
    c['evidence'] = {'note': f'pruned for episode 28 ({kind} {val}): {len(rows)} of 9462135 edges kept'}
    cbytes = (json.dumps(c, indent=2) + '\n').encode(); (out / 'escape-circuit.json').write_bytes(cbytes)

    eraw = (SRC / 'escape-gold.bin').read_bytes()
    hdr = struct.unpack_from('<4s5I2f', eraw); _, _, _, ecases, esteps, O, _, _ = hdr
    ebuf = bytearray(eraw[:32]); pos = 32
    b = fcl.Brain(W, circ); oidx = np.r_[outs['L'], outs['R']]
    for _ in range(ecases):
        looms = np.frombuffer(eraw, '<f4', esteps * 2, pos).reshape(esteps, 2); pos += esteps * (2 + O) * 4
        b.reset(); res = []
        for l, r in looms:
            b.step(l, r); res.append(b.state[oidx].copy())
        ebuf += looms.astype('<f4').tobytes() + np.array(res, '<f4').tobytes()
    (out / 'escape-gold.bin').write_bytes(bytes(ebuf))
    shutil.copy(SRC / 'neurons.csv', out / 'neurons.csv')

    m = json.loads((SRC / 'model_bundle.json').read_text())
    m['bundle_id'] = f'fly-escape-v1-{name}'
    for role, fname in (('graph', 'connectome.fcl'), ('graph_gold', 'gold-device-order.bin'),
                        ('circuit', 'escape-circuit.json'), ('escape_gold', 'escape-gold.bin'), ('neurons', 'neurons.csv')):
        data = (out / fname).read_bytes(); m['files'][role].update(bytes=len(data), sha256=sha(data))
    m['graph']['edges'] = int(len(rows))
    m['circuit']['graph_sha256'] = gsha
    m['gold']['graph']['graph_sha256'] = gsha
    m['gold']['escape']['graph_sha256'] = gsha
    m['gold']['escape']['circuit_sha256'] = sha(cbytes)
    (out / 'model_bundle.json').write_text(json.dumps(m, indent=2) + '\n')
    # Round-trip check: decode what we wrote.
    N2, o2, r2, s2, c2 = fcl.read(out / 'connectome.fcl')
    assert N2 == N and len(r2) == len(rows) and np.array_equal(np.sort(c2), np.sort(counts))
    print(f'{name}: {len(rows)} edges ({len(rows)/9462135*100:.2f}%), graph {len(graph)/1e6:.2f} MB -> {out}')


if __name__ == '__main__':
    main()
