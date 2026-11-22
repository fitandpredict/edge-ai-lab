"""Fly flybody's flight-imitation task with the numpy policy; report tracking error, save a video.

usage: MUJOCO_GL=glfw .venv/bin/python sim/fly_flight.py [episodes] [--video out/flight.mp4]
"""
import sys
import numpy as np
import mediapy
from flybody.fly_envs import flight_imitation
from flight_policy import FlightPolicy

D = 'data/flight-imitation/'
episodes = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
video = sys.argv[sys.argv.index('--video') + 1] if '--video' in sys.argv else None

env = flight_imitation(D + 'flight-dataset_saccade-evasion_augmented.hdf5', D + 'wing_pattern_fmech.npy',
                       random_state=np.random.RandomState(0), randomize_start_step=False)
pi = FlightPolicy()
if '--baseline' in sys.argv:
    pi.__call__ = None
    act = lambda o: np.zeros(env.action_spec().shape)
else:
    act = pi
cam = [env.physics.model.camera(i).name for i in range(env.physics.model.ncam)].index('walker/side')
print('control dt', env.control_timestep(), 'obs', pi.flatten(env.reset().observation).shape)

frames = []
for ep in range(episodes):
    ts = env.reset()
    steps, rew, dists = 0, 0.0, []
    while not ts.last():
        ts = env.step(act(ts.observation))
        steps += 1
        rew += ts.reward or 0
        dists.append(np.linalg.norm(ts.observation['walker/ref_displacement'][0]))
        if video and ep == 0 and steps % 10 == 0:
            frames.append(env.physics.render(480, 640, camera_id=cam))
    print(f'episode {ep}: {steps} steps ({steps * env.control_timestep():.2f} s), '
          f'mean reward {rew / steps:.3f}, mean |ref displacement| {np.mean(dists):.3f}, '
          f'final {dists[-1]:.3f}, terminated early: {steps * env.control_timestep() < 0.59}')
if video:
    mediapy.write_video(video, frames, fps=30)
    print('wrote', video, len(frames), 'frames')
