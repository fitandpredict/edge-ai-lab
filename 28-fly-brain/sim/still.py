"""Transparent PNG cut-outs of the fly (and the looming ball) at a moment of a real run, for thumbnails and Short covers.

usage: PYTHONPATH=flybody MUJOCO_GL=glfw .venv/bin/python sim/still.py out.png --replay out/link_left.json --at 0.36 \
           [--what fly|ball|both] [--view AZ EL DIST] [--size 2400]
Replays a saved decision exactly as link.py's pass 2 does, stops at --at seconds of world time, renders one
square frame from a close camera aimed at the fly, and keeps only the chosen bodies' pixels (MuJoCo's
segmentation render, so no colour keying), cropped to them. Episode 27's sim/still.py, for the fly.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import link  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument('out')
    a.add_argument('--replay', default='out/link_left.json')
    a.add_argument('--at', type=float, default=0.36)
    a.add_argument('--what', choices=['fly', 'ball', 'both'], default='fly')
    a.add_argument('--view', type=float, nargs=3, default=[-140, 18, 1.1], help='azimuth, elevation (deg), distance (cm)')
    a.add_argument('--size', type=int, default=2400)
    args = a.parse_args()

    import h5py
    from dm_control import composer
    from dm_control.locomotion.arenas import floors
    from flybody.fruitfly import fruitfly
    from flybody.tasks.flight_imitation import FlightImitationWBPG
    from flybody.tasks.pattern_generators import WingBeatPatternGenerator
    from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader
    from flight_policy import FlightPolicy

    log = json.load(open(args.replay))
    D, dt, out = 'data/flight-imitation/', 2e-4, log['outcome']
    target, t_hit, _ = link.target_path(log['side'])
    h = h5py.File(D + 'flight-dataset_saccade-evasion_augmented.hdf5')
    ev = link.EVASION[out['danger_seen']] if out['result'] == 'escape' else None
    t_esc = out['t'] if ev is not None else None
    q = h['trajectories'][f'{ev if ev is not None else 53:03d}']['com_qpos'][()]
    v = h['trajectories'][f'{ev if ev is not None else 53:03d}']['com_qvel'][()]
    q0 = q[0].copy()
    n_cruise = int(round((t_esc if t_esc is not None else t_hit + 0.1) / dt))
    tt = np.arange(n_cruise) * dt
    cq = np.tile(q0, (n_cruise, 1)); cq[:, 0] = link.CRUISE_V * tt; cq[:, 1] = 0
    cv = np.zeros((n_cruise, 6)); cv[:, 0] = link.CRUISE_V
    if ev is not None:
        rq = q.copy(); rq[:, 0] += link.CRUISE_V * n_cruise * dt - rq[0, 0]; rq[:, 1] -= rq[0, 1]
        ref_q, ref_v = np.vstack([cq, rq]), np.vstack([cv, v])
    else:
        ref_q, ref_v = cq, cv

    arena = floors.Floor()
    body = arena.mjcf_model.worldbody.add('body', name='threat', mocap=True, pos=[0, 0, -5])
    body.add('geom', name='threat_geom', type='sphere', size=[link.TARGET_R], rgba=[0.12, 0.1, 0.14, 1], contype=0, conaffinity=0)
    arena.mjcf_model.worldbody.add('camera', name='still', pos=[0, 0, 3], fovy=30)
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
    phys.model.vis.global_.offwidth = phys.model.vis.global_.offheight = args.size
    mocap = phys.model.body('threat').mocapid[0]
    z = ref_q[0, 2]
    root = phys.bind(task._walker.root_body)
    for k in range(int(round(args.at / dt))):
        p = target(k * dt); phys.data.mocap_pos[mocap] = [p[0], p[1], z]
        ts = env.step(pi(ts.observation))

    fly_p = root.xpos.copy()
    az, el, dist = math.radians(args.view[0]), math.radians(args.view[1]), args.view[2]
    look = fly_p if args.what == 'fly' else 0.5 * (fly_p + phys.data.mocap_pos[mocap])
    eye = look + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fwd = look - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cam = phys.model.camera('still').id
    phys.forward()
    phys.data.cam_xpos[cam] = eye
    phys.data.cam_xmat[cam] = np.stack([right, up, -fwd], axis=1).ravel()
    walker = [i for i in range(phys.model.ngeom)
              if phys.model.body(phys.model.geom_bodyid[i]).name.startswith(task._walker.mjcf_model.model + '/')]
    ball = phys.model.geom('threat_geom').id
    shown = (set(walker) if args.what != 'ball' else set()) | ({ball} if args.what != 'fly' else set())
    phys.model.geom_rgba[[i for i in range(phys.model.ngeom) if i not in shown], 3] = 0   # nothing else can hide it
    phys.model.site_rgba[:, 3] = 0
    rgb = phys.render(args.size, args.size, camera_id=cam)
    seg = phys.render(args.size, args.size, camera_id=cam, segmentation=True)[..., 0]

    # The fly = geoms on the walker's own bodies (not the floor, the ball or the task's cyan reference dots).
    keep = np.zeros(seg.shape, bool)
    if args.what in ('fly', 'both'):
        keep |= np.isin(seg, walker)
    if args.what in ('ball', 'both'):
        keep |= seg == ball
    alpha = (keep * 255).astype(np.uint8)
    im = Image.fromarray(np.dstack([rgb, alpha]), 'RGBA')
    im = im.crop(im.getbbox())
    im.save(args.out)
    print(args.out, im.size, 'fly at', np.round(fly_p, 2), 't =', args.at)


if __name__ == '__main__':
    main()
