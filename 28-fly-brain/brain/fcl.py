"""Read/write the episode 01 FCL1 connectome graph and run its rate model in numpy (float32 reference).

Format: firmware/esp32_fly_small/MODEL_FORMATS.md (from slvDev/esp32-ai). Dynamics (scalar reference):
    scale[row] = 0.7 / max(sum(counts[row]), 1)
    next = 0.5 * state + 0.5 * tanh(sum(count * scale * state[src]) + input)
"""
import json, struct
from pathlib import Path
import numpy as np
import lz4.block
import scipy.sparse as sp

import os
# The fly bundle from slvDev/esp32-ai (`scripts/fetch_model.sh fly` puts it in artifacts/fly). Point FLY_BUNDLE at it.
BUNDLE = Path(os.environ.get('FLY_BUNDLE', 'esp32-ai/artifacts/fly')).resolve()


def _varints(buf, pos, n):
    out = np.empty(n, np.int64)
    for i in range(n):
        v = shift = 0
        while True:
            b = buf[pos]; pos += 1
            v |= (b & 0x7F) << shift
            if b < 0x80: break
            shift += 7
        out[i] = v
    return out, pos


def read(path=BUNDLE / 'connectome.fcl'):
    """Returns (N, order, rows, srcs, counts) with rows/srcs in execution indices."""
    raw = Path(path).read_bytes()
    assert raw[:4] == b'FCL1'
    ver, N, E, rpb = struct.unpack_from('<4I', raw, 4)
    order = np.frombuffer(raw, '<u4', N, 20).copy()
    pos = 20 + 4 * N
    rows, srcs, counts = [], [], []
    row0 = 0
    while row0 < N:
        nrow, nedge, dlen, clen = struct.unpack_from('<4I', raw, pos); pos += 16
        buf = lz4.block.decompress(raw[pos:pos + clen], uncompressed_size=dlen); pos += clen
        deg, p = _varints(buf, 0, nrow)
        delta, p = _varints(buf, p, nedge)
        cnt, p = _varints(buf, p, nedge)
        assert p == dlen and deg.sum() == nedge
        r = np.repeat(np.arange(row0, row0 + nrow), deg)
        starts = np.r_[0, np.cumsum(deg)[:-1]]
        # Delta encoding restarts from zero per row.
        seg = np.repeat(starts, deg)
        base = np.r_[0, np.cumsum(delta)][seg]
        idx = np.cumsum(delta) - base
        rows.append(r); srcs.append(idx); counts.append(cnt)
        row0 += nrow
    rows, srcs, counts = (np.concatenate(a) for a in (rows, srcs, counts))
    assert len(rows) == E
    return N, order, rows, srcs, counts


def matrix(N, rows, srcs, counts):
    """Row-normalised float32 sparse W, so that W @ state = sum(count * scale * state[src])."""
    tot = np.bincount(rows, weights=counts, minlength=N)
    scale = (0.7 / np.maximum(tot, 1)).astype(np.float32)
    w = (counts.astype(np.float32) * scale[rows]).astype(np.float32)
    return sp.csr_matrix((w, (rows, srcs)), shape=(N, N), dtype=np.float32)


def circuit(path=BUNDLE / 'escape-circuit.json'):
    c = json.loads(Path(path).read_text())
    ins = {s: np.array([e['index'] for e in c['inputs'][s]]) for s in 'LR'}
    outs = {s: np.array([e['index'] for e in c['outputs'][s]]) for s in 'LR'}
    return ins, outs, np.float32(c['amplitude']), np.float32(c['threshold'])


class Brain:
    def __init__(self, W, circ):
        self.W = W; self.ins, self.outs, self.amp, self.threshold = circ
        self.N = W.shape[0]; self.state = np.zeros(self.N, np.float32)

    def reset(self):
        self.state[:] = 0

    def step(self, loom_l, loom_r):
        x = np.zeros(self.N, np.float32)
        x[self.ins['L']] = self.amp * np.float32(loom_l); x[self.ins['R']] = self.amp * np.float32(loom_r)
        self.state = (0.5 * self.state + 0.5 * np.tanh(self.W @ self.state + x)).astype(np.float32)
        return float(self.state[self.outs['L']].mean()), float(self.state[self.outs['R']].mean())


def escape_gold(path=BUNDLE / 'escape-gold.bin'):
    raw = Path(path).read_bytes()
    magic, ver, N, cases, steps, O = struct.unpack_from('<4s5I', raw, 0)
    atol, rtol = struct.unpack_from('<2f', raw, 24)
    pos, out = 32, []
    for _ in range(cases):
        looms = np.frombuffer(raw, '<f4', steps * 2, pos).reshape(steps, 2); pos += steps * 8
        exp = np.frombuffer(raw, '<f4', steps * O, pos).reshape(steps, O); pos += steps * O * 4
        out.append((looms, exp))
    return out, atol, rtol
