"""Rank the recorded evasions by how sharply they turn, per direction (heading of the CoM velocity)."""
import h5py, numpy as np
f = h5py.File('data/flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5')
dt = float(f['timestep_seconds'][()])
types = {k: set(f['trajectory_type_indices'][k][()].tolist()) for k in f['trajectory_type_indices']}
rows = []
for k in f['trajectories']:
    i = int(k); q = f['trajectories'][k]['com_qpos'][()]; v = f['trajectories'][k]['com_qvel'][()]
    hd = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn = np.degrees(hd[-1] - hd[0])
    speed = np.linalg.norm(v[:, :3], axis=1)
    # onset: first step where heading has changed by >10 degrees from the first 10 ms mean
    base = hd[:50].mean(); on = np.argmax(np.abs(hd - base) > np.radians(10))
    t = next(n for n, s in types.items() if i in s)
    rows.append((i, t, len(q) * dt, turn, speed.mean(), on * dt, np.ptp(q[:, 2])))
print('dt', dt, 'units: pos as stored; speed = mean |v|')
for t in ['evasion_original', 'evasion_reflected']:
    rr = sorted([r for r in rows if r[1] == t], key=lambda r: r[3])
    print('==', t, 'n', len(rr), 'turn range', round(rr[0][3]), round(rr[-1][3]))
    for r in rr[:3] + rr[-3:]:
        print(f'  #{r[0]:03d} {r[2]*1000:.0f} ms turn {r[3]:+.0f} deg, speed {r[4]:.1f}, onset {r[5]*1000:.0f} ms, dz {r[6]:.2f}')
