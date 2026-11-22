"""Time decisions on the board for whatever graph is flashed: ZERO, then 8 LOOM steps (left eye 0.5)."""
import json, struct, sys, time, serial
s = serial.Serial(sys.argv[1] if len(sys.argv) > 1 else '/dev/cu.usbmodem1101', 921600, timeout=0.2)
s.write(b'STOP\n'); time.sleep(3); s.reset_input_buffer()
def ask(cmd, want, timeout=8):
    s.write(cmd.encode() + b'\n'); end = time.time() + timeout
    while time.time() < end:
        l = s.readline().decode(errors='replace').strip()
        if l.startswith('{') and f'"event":"{want}"' in l: return json.loads(l)
    raise TimeoutError(cmd)
info = ask('INFO', 'info')
ask('ZERO', 'zeroed')
h = struct.pack('>f', 0.5).hex(); z = struct.pack('>f', 0.0).hex()
us = []
for _ in range(8):
    r = ask(f'LOOM {h} {z}', 'loom'); us.append(r['compute_us'])
print(json.dumps({'edges_bytes': info['model_bytes'], 'fnv': info['model_fnv1a'], 'cached_blocks': info['cached_blocks'],
                  'blocks': info['blocks'], 'compute_ms': [round(u / 1000, 2) for u in us],
                  'median_ms': sorted(us)[4] / 1000, 'last_drives': r['drives']}))
