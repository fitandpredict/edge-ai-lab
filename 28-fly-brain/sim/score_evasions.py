"""Fly the policy on moderate evasions (60-150 deg turn, onset >= 15 ms) and rank by tracking."""
import sys, h5py, numpy as np
from flybody.fly_envs import flight_imitation
from flight_policy import FlightPolicy
D = 'data/flight-imitation/'
f = h5py.File(D + 'flight-dataset_saccade-evasion_augmented.hdf5')
orig = sorted(f['trajectory_type_indices']['evasion_original'][()].tolist())
cands = []
for i in orig:
    v = f['trajectories'][f'{i:03d}']['com_qvel'][()]
    hd = np.unwrap(np.arctan2(v[:, 1], v[:, 0])); turn = np.degrees(hd[-1] - hd[0])
    on = np.argmax(np.abs(hd - hd[:50].mean()) > np.radians(10)) * 2e-4
    if 60 <= abs(turn) <= 150 and on >= 0.015: cands.append((i, turn, on, len(v)))
print(len(cands), 'candidates')
env = flight_imitation(D + 'flight-dataset_saccade-evasion_augmented.hdf5', D + 'wing_pattern_fmech.npy',
                       randomize_start_step=False, terminal_com_dist=float('inf'))
pi = FlightPolicy()
out = []
for i, turn, on, n in cands:
    env.task.set_next_trajectory_index(i)
    ts = env.reset(); d = []
    while not ts.last():
        ts = env.step(pi(ts.observation)); d.append(np.linalg.norm(ts.observation['walker/ref_displacement'][0]))
    out.append((np.mean(d), np.max(d), i, turn, on, n * 2e-4, len(d) * 2e-4))
for r in sorted(out)[:12]:
    print(f'#{r[2]:03d} turn {r[3]:+.0f} onset {r[4]*1000:.0f} ms len {r[5]*1000:.0f} ms flown {r[6]*1000:.0f} ms '
          f'mean err {r[0]:.3f} max {r[1]:.3f}')
