"""Chip's connectome decides; flybody's fly escapes.

Pass 1 (decide, live on the ESP32-S3): a dark looming target flies at the cruising fly from one side.
Each neural step the Mac computes how big the target looks to each eye (the same loom formula as the
episode 01 firmware) and sends it to the board with `LOOM l r`. The board runs one step of the
48,311-neuron connectome and returns the escape-neuron drives. Escape = a side's drive crosses the
threshold (0.05, the firmware's), danger side = the larger drive. Or the target reaches the fly: caught.
`--brain-dt` is how much world time passes per neural step:
    0.010  the world waits for the brain (each step stands for 10 ms; our assumption, stated on screen)
    1.6    real time for the full brain: the world does not wait for Chip's 1.6 s per step
    0      true real time: each step advances the world by the compute time the board reports

Pass 2 (fly, on the Mac): flybody's trained flight policy flies a reference made of straight cruising up
to the escape moment, then a real fly's recorded evasion (Muijres et al. 2014) that turns away from the
danger side. The target is drawn as a dark sphere on its straight path. Nothing in pass 2 changes the
decision; pass 1's log is saved and replayed.

usage: PYTHONPATH=flybody MUJOCO_GL=glfw .venv/bin/python sim/link.py --side left --brain-dt 0.01 \
           [--port /dev/cu.usbmodem1101] [--log out/link_left.json] [--video out/link_left.mp4]
       ... --replay out/link_left.json      (skip the board, fly from a saved decision)
"""
import argparse, json, math, struct, time
import numpy as np

# --- the scene (cm, s; flybody uses cm) -------------------------------------------------------------
CRUISE_V = 14.2          # cm/s, the chosen evasion's starting speed
TARGET_R = 0.6           # cm radius of the looming target
TARGET_V = 25.0          # cm/s
TARGET_D0 = 11.0         # cm away at t = 0 (looming starts at FAR = 10 cm)
BEARING = math.radians(70)   # target comes from 70 degrees off the fly's heading, on the chosen side
FAR, NEAR, CATCH = 10.0, 2.0, 0.5   # loom 0 at FAR, 1 at NEAR; caught at CATCH (centre distance, cm)
THRESHOLD = 0.05         # firmware ESCAPE_THRESHOLD
EVASION = {'left': 53, 'right': 145}   # danger on the left -> #053 turns right; #145 is its mirror


def target_path(side):
    """Fly cruises along +x from the origin. The target aims at where the fly will be at arrival."""
    t_hit = TARGET_D0 / TARGET_V
    hit = np.array([CRUISE_V * t_hit, 0.0])
    s = 1 if side == 'left' else -1           # +y is the fly's left
    u = np.array([math.cos(BEARING), s * math.sin(BEARING)])
    start = hit + u * TARGET_D0
    return lambda t: (hit + u * (TARGET_D0 - TARGET_V * t)), t_hit, start


def looms_at(t, target):
    fly = np.array([CRUISE_V * t, 0.0])
    d = target(t) - fly
    dist = float(np.hypot(*d))
    size = 2 * math.atan(TARGET_R / max(dist, CATCH))
    far, near = 2 * math.atan(TARGET_R / FAR), 2 * math.atan(TARGET_R / NEAR)
    level = min(1.0, max(0.0, (size - far) / (near - far)))
    bearing = math.atan2(d[1], d[0])          # heading is +x; positive = left
    right = 0.5 * (1 - math.sin(bearing))
    return [level * (1 - right), level * right], dist


def hexf(x):
    return struct.pack('>f', x).hex()


def decide(port, side, brain_dt, max_steps=1000):
    import serial
    s = serial.Serial(port, 921600, timeout=0.2)
    s.write(b'STOP\n'); time.sleep(2.5); s.reset_input_buffer()

    def ask(cmd, want, timeout=6):
        s.write(cmd.encode() + b'\n'); end = time.time() + timeout
        while time.time() < end:
            line = s.readline().decode(errors='replace').strip()
            if line.startswith('{') and f'"event":"{want}"' in line:
                return json.loads(line)
        raise TimeoutError(cmd)

    ask('ZERO', 'zeroed')
    target, t_hit, _ = target_path(side)
    log = {'side': side, 'brain_dt': brain_dt, 'scene': dict(CRUISE_V=CRUISE_V, TARGET_R=TARGET_R,
           TARGET_V=TARGET_V, TARGET_D0=TARGET_D0, BEARING_DEG=70, FAR=FAR, NEAR=NEAR, CATCH=CATCH),
           'steps': [], 'outcome': None}
    real = brain_dt <= 0      # real time: the world advances by Chip's measured compute time per step
    t = 0.0
    for k in range(max_steps):
        if not real:
            t = k * brain_dt
        looms, dist = looms_at(t, target)
        if dist <= CATCH or t >= t_hit:
            log['outcome'] = {'result': 'caught', 't': min(t, t_hit), 'step': k}
            break
        wall = time.time()
        r = ask(f'LOOM {hexf(looms[0])} {hexf(looms[1])}', 'loom')
        step = {'k': k, 't': t, 'dist': dist, 'looms': looms, 'drives': r['drives'],
                'outputs': r.get('outputs'), 'compute_us': r['compute_us'], 'wall_s': time.time() - wall}
        log['steps'].append(step)
        print(f"step {k:3d} t={t*1000:7.1f} ms dist={dist:5.2f} looms={looms[0]:.2f}/{looms[1]:.2f} "
              f"drives={r['drives'][0]:.4f}/{r['drives'][1]:.4f} ({r['compute_us']/1e6:.2f} s)", flush=True)
        dl, dr = r['drives']
        step_dt = r['compute_us'] / 1e6 if real else brain_dt
        if dl > THRESHOLD or dr > THRESHOLD:
            seen = 'left' if dl > dr else 'right'
            # The decision takes effect after this step's world time has passed.
            log['outcome'] = {'result': 'escape', 'danger_seen': seen, 't': t + step_dt, 'step': k}
            break
        if real:
            t += step_dt
    print('outcome:', log['outcome'])
    return log


def fly(log, video, fps=30, slowmo=None, size=(720, 1280)):
    """Fly the decided escape in flybody and render it. Returns a summary dict."""
    import h5py, mediapy
    from dm_control import composer
    from dm_control.locomotion.arenas import floors
    from flybody.fruitfly import fruitfly
    from flybody.tasks.flight_imitation import FlightImitationWBPG
    from flybody.tasks.pattern_generators import WingBeatPatternGenerator
    from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader
    from flight_policy import FlightPolicy

    D = 'data/flight-imitation/'
    dt = 2e-4
    out = log['outcome']
    target, t_hit, _ = target_path(log['side'])
    h = h5py.File(D + 'flight-dataset_saccade-evasion_augmented.hdf5')
    if out['result'] == 'escape':
        ev = EVASION[out['danger_seen']]
        q = h['trajectories'][f'{ev:03d}']['com_qpos'][()]; v = h['trajectories'][f'{ev:03d}']['com_qvel'][()]
        t_esc = out['t']
    else:   # caught: keep cruising past the catch point
        ev, t_esc = None, None
        q = h['trajectories']['053']['com_qpos'][:1]
    # Straight cruise before the escape (or for the whole run if caught), joined to the recording.
    q0 = q[0].copy()
    n_cruise = int(round((t_esc if t_esc is not None else t_hit + 0.1) / dt))
    tt = np.arange(n_cruise) * dt
    cq = np.tile(q0, (n_cruise, 1)); cq[:, 0] = CRUISE_V * tt; cq[:, 1] = 0
    cv = np.zeros((n_cruise, 6)); cv[:, 0] = CRUISE_V
    if ev is not None:
        rq = q.copy(); rq[:, 0] += CRUISE_V * n_cruise * dt - rq[0, 0]; rq[:, 1] -= rq[0, 1]
        ref_q, ref_v = np.vstack([cq, rq]), np.vstack([cv, v])
    else:
        ref_q, ref_v = cq, cv

    arena = floors.Floor()
    body = arena.mjcf_model.worldbody.add('body', name='threat', mocap=True, pos=[0, 0, -5])
    body.add('geom', type='sphere', size=[TARGET_R], rgba=[0.12, 0.1, 0.14, 1], contype=0, conaffinity=0)
    arena.mjcf_model.worldbody.add('camera', name='director', pos=[0, 0, 3], fovy=40)
    loader = InferenceFlightTrajectoryLoader(); loader.set_next_trajectory(ref_q, ref_v)
    task = FlightImitationWBPG(walker=fruitfly.FruitFly, arena=arena,
                               wbpg=WingBeatPatternGenerator(base_pattern_path=D + 'wing_pattern_fmech.npy'),
                               traj_generator=loader, terminal_com_dist=float('inf'), initialize_qvel=True,
                               force_actuators=False, disable_legs=True, time_limit=len(ref_q) * dt,
                               joint_filter=0., future_steps=5)
    env = composer.Environment(time_limit=len(ref_q) * dt, task=task, strip_singleton_obs_buffer_dim=True)
    pi = FlightPolicy()
    ts = env.reset()
    phys = env.physics
    mocap = phys.model.body('threat').mocapid[0]
    z = ref_q[0, 2]
    cam = phys.model.camera('director').id
    side_sign = 1 if log['side'] == 'left' else -1
    shot = {'eye': None, 'look': None}

    def aim(fly_p, tgt_p, a=0.12):
        """Director camera: keep fly and target in frame; behind the fly, on the side away from the target."""
        gap = float(np.linalg.norm(tgt_p - fly_p))
        w = 0.35 if gap < 12 else 0.0
        look = (1 - w) * fly_p + w * tgt_p
        dist = float(np.clip(0.9 * gap + 2.5, 3.0, 9.0))
        dirv = np.array([-0.7, -side_sign * 0.55, 0.45]); dirv /= np.linalg.norm(dirv)
        eye = look + dist * dirv
        if shot['eye'] is None:
            shot['eye'], shot['look'] = eye, look
        shot['eye'] = (1 - a) * shot['eye'] + a * eye
        shot['look'] = (1 - a) * shot['look'] + a * look
        fwd = shot['look'] - shot['eye']; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        # MuJoCo cameras look along -z with +y up: columns are (x=right, y=up, z=-forward).
        phys.data.cam_xpos[cam] = shot['eye']
        phys.data.cam_xmat[cam] = np.stack([right, up, -fwd], axis=1).ravel()
    every = max(1, int(round(1 / fps / dt / (slowmo or 1))))
    root = phys.bind(task._walker.root_body)
    frames, k, min_d, hit_k = [], 0, 1e9, None
    while not ts.last():
        if hit_k is not None and k - hit_k > 0.02 / dt:   # stop 20 ms after the target reaches the fly
            break
        t = k * dt
        p = target(t); phys.data.mocap_pos[mocap] = [p[0], p[1], z]
        ts = env.step(pi(ts.observation)); k += 1
        min_d = min(min_d, float(np.linalg.norm(np.array([p[0], p[1], z]) - root.xpos)))
        if hit_k is None and min_d < TARGET_R:
            hit_k = k
        if video and k % every == 0:
            aim(root.xpos.copy(), np.array([p[0], p[1], z]))
            frames.append(phys.render(size[0], size[1], camera_id=cam))
    if video:
        mediapy.write_video(video, frames, fps=fps)
    return {'evasion': ev, 'flown_s': k * dt, 'closest_cm': min_d, 'caught_in_flight': min_d < TARGET_R + 0.15, 'frames': len(frames),
            'slowmo': (1 / fps) / (every * dt)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', choices=['left', 'right'], default='left')
    ap.add_argument('--brain-dt', type=float, default=0.01)
    ap.add_argument('--port', default='/dev/cu.usbmodem1101')
    ap.add_argument('--log'); ap.add_argument('--video'); ap.add_argument('--replay')
    ap.add_argument('--slowmo', type=float, default=20.0, help='render slow-down factor')
    a = ap.parse_args()
    log = json.load(open(a.replay)) if a.replay else decide(a.port, a.side, a.brain_dt)
    if a.log and not a.replay:
        json.dump(log, open(a.log, 'w'), indent=1)
    if a.video:
        print('flight:', fly(log, a.video, slowmo=a.slowmo))
