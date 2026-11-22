"""Build a compact fly bundle: only the eye (LC4/LPLC2) and escape (DNp01/02/04/11) neurons, renumbered,
with every connection among them. For the episode 28 firmware copy (firmware/esp32_fly_small), which
accepts fewer than 48,311 nodes.

usage: PYTHONPATH=brain .venv/bin/python brain/make_compact.py  ->  data/bundles/compact/
"""
import csv, hashlib, io, json, shutil, struct
from pathlib import Path
import numpy as np
import fcl
from make_bundle import encode, sha

SRC = fcl.BUNDLE
g = np.load('data/graph.npz'); N0 = int(g['N']); order0 = g['order']
rows, srcs, counts = g['rows'].astype(np.int64), g['srcs'].astype(np.int64), g['counts'].astype(np.int64)
ins, outs, amp, thr = fcl.circuit()
keep_nodes = np.sort(np.r_[ins['L'], ins['R'], outs['L'], outs['R']])       # old execution indices
n = len(keep_nodes)
new_of = -np.ones(N0, np.int64); new_of[keep_nodes] = np.arange(n)
ek = (new_of[rows] >= 0) & (new_of[srcs] >= 0)
r, s, c = new_of[rows[ek]], new_of[srcs[ek]], counts[ek]
order = np.argsort(np.argsort(order0[keep_nodes]))   # canonical ranks among kept neurons
out = Path('data/bundles/compact'); out.mkdir(parents=True, exist_ok=True)
graph = encode(n, order, r, s, c); (out / 'connectome.fcl').write_bytes(graph); gsha = sha(graph)
W = fcl.matrix(n, r, s, c)

# Circuit with new indices.
circ = json.loads((SRC / 'escape-circuit.json').read_text())
circ['graph_sha256'] = gsha
circ['order_sha256'] = sha(order.astype('<u4').tobytes())
for kind in ('inputs', 'outputs'):
    for side in 'LR':
        for e in circ[kind][side]:
            e['index'] = int(new_of[e['index']])
circ['evidence'] = {'note': f'episode 28 compact: {n} of 48311 neurons (eyes + escape neurons), {len(r)} edges'}
cbytes = (json.dumps(circ, indent=2) + '\n').encode(); (out / 'escape-circuit.json').write_bytes(cbytes)
cins = {sd: np.array([e['index'] for e in circ['inputs'][sd]]) for sd in 'LR'}
couts = {sd: np.array([e['index'] for e in circ['outputs'][sd]]) for sd in 'LR'}

# neurons.csv subset, renumbered.
rd = list(csv.DictReader(io.StringIO((SRC / 'neurons.csv').read_text())))
buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rd[0].keys()), lineterminator='\n'); w.writeheader()
for new, old in enumerate(keep_nodes):
    row = dict(rd[old]); row['execution_index'] = str(new); row['canonical_index'] = str(int(order[new])); w.writerow(row)
(out / 'neurons.csv').write_text(buf.getvalue())

# Graph gold: original cases restricted to the kept neurons, recomputed.
raw = (SRC / 'gold-device-order.bin').read_bytes()
magic, ver, _, cases, steps, atol, rtol = struct.unpack_from('<4s4I2f', raw)
gb = bytearray(struct.pack('<4s4I2f', magic, ver, n, cases, steps, atol, rtol)); pos = 28
for _ in range(cases):
    st = np.frombuffer(raw, '<f4', N0, pos)[keep_nodes].copy(); pos += 4 * N0; gb += st.tobytes()
    for _ in range(steps):
        x = np.frombuffer(raw, '<f4', N0, pos)[keep_nodes].copy(); pos += 8 * N0
        st = (0.5 * st + 0.5 * np.tanh(W @ st + x)).astype(np.float32); gb += x.tobytes() + st.tobytes()
(out / 'gold-device-order.bin').write_bytes(bytes(gb))

# Escape gold.
eraw = (SRC / 'escape-gold.bin').read_bytes()
magic, ver, _, ecases, esteps, O, eatol, ertol = struct.unpack_from('<4s5I2f', eraw)
eb = bytearray(struct.pack('<4s5I2f', magic, ver, n, ecases, esteps, O, eatol, ertol)); pos = 32
b = fcl.Brain(W, (cins, couts, amp, thr)); oidx = np.r_[couts['L'], couts['R']]
for _ in range(ecases):
    looms = np.frombuffer(eraw, '<f4', esteps * 2, pos).reshape(esteps, 2); pos += esteps * (2 + O) * 4
    b.reset(); res = []
    for l, rr in looms:
        b.step(l, rr); res.append(b.state[oidx].copy())
    eb += looms.astype('<f4').tobytes() + np.array(res, '<f4').tobytes()
(out / 'escape-gold.bin').write_bytes(bytes(eb))

m = json.loads((SRC / 'model_bundle.json').read_text())
m['bundle_id'] = 'fly-escape-v1-compact'
for role, fname in (('graph', 'connectome.fcl'), ('graph_gold', 'gold-device-order.bin'), ('circuit', 'escape-circuit.json'),
                    ('escape_gold', 'escape-gold.bin'), ('neurons', 'neurons.csv')):
    d = (out / fname).read_bytes(); m['files'][role].update(bytes=len(d), sha256=sha(d))
m['graph'].update(nodes=n, edges=int(len(r)), order_sha256=circ['order_sha256'])
m['circuit'].update(graph_sha256=gsha, order_sha256=circ['order_sha256'])
m['gold']['graph'].update(graph_sha256=gsha, order_sha256=circ['order_sha256'])
m['gold']['escape'].update(graph_sha256=gsha, circuit_sha256=sha(cbytes))
(out / 'model_bundle.json').write_text(json.dumps(m, indent=2) + '\n')
print(f'compact: {n} neurons, {len(r)} edges, graph {len(graph)} bytes -> {out}')
