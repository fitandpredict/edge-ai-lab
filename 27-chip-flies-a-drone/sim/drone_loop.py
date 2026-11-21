"""Fly the drone in plain MuJoCo (CPU) on the Mac, with a learned brain in the loop.

Brains:
  npz     our numpy MLP from train_drone.py (reference)       --policy runs/main/final.npz [--int8]
  chip    the ESP32-S3 over USB serial: 27 floats out, 4 floats + inference µs back
  hover   constant hover PWM (physics sanity check: it should just sit there, then drift)
  random  random motors (the "before it learned anything" shot)

Experiments:
  --delay-ms 20         every command reaches the motors 20 ms late (on top of whatever the brain takes)
  --hz 50               run the brain at 50 Hz (the policy must be trained at that rate, or it's a mismatch)
  --gust 0.3@4          a 0.3 N push sideways at t = 4 s (decays over 0.15 s, like training)
  --targets "0,0,1;1,0,1.5@3;..."   waypoints (x,y,z@time); default: hover at (0,0,1)
  --video out.mp4 [--size 1280x720] [--hud hud.json] [--cam chase|side|wide]

Prints a JSON summary: position error, crash time, brain µs per step.
"""

import argparse
import collections
import json
import struct
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drone_model as dm  # noqa: E402

SYNC_OUT, SYNC_IN = b"\xaa\x55", b"\x55\xaa"


# ---------------------------------------------------------------- brains
class NpzBrain:
    def __init__(self, path, int8=False):
        z = np.load(path)
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
        self.int8 = int8
        if int8:  # per-output-channel int8 weights, per-layer int8 activations (what the firmware does)
            q = []
            for W, b in self.layers:
                s = np.abs(W).max(axis=0) / 127.0
                s[s == 0] = 1
                q.append((np.clip(np.round(W / s), -127, 127) * s, b))
            self.layers = q

    def __call__(self, obs):
        t = time.perf_counter()
        x = (obs.astype(np.float32) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            if self.int8:
                sx = max(np.abs(x).max(), 1e-8) / 127.0
                x = np.round(x / sx) * sx
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = x / (1 + np.exp(-x))
        return np.tanh(x), (time.perf_counter() - t) * 1e6


class ChipBrain:
    """Frame out: AA 55 + 27 float32. Frame in: 55 AA + 4 float32 + uint32 inference µs."""

    def __init__(self, port, mode=None):
        import serial
        self.ser = serial.Serial(port, 921600, timeout=1.0)
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        if mode is not None:
            self.ser.write(str(mode).encode())
            print("chip:", self.ser.readline().decode(errors="ignore").strip())
        self.chip_us, self.round_us = [], []

    def __call__(self, obs):
        t = time.perf_counter()
        self.ser.write(SYNC_OUT + obs.astype("<f4").tobytes())
        buf = b""
        while not buf.endswith(SYNC_IN):
            c = self.ser.read(1)
            if not c:
                raise TimeoutError("no answer from Chip (is the firmware flashed?)")
            buf = (buf + c)[-2:]
        body = self.ser.read(4 * dm.ACT_SIZE + 4)
        a = np.frombuffer(body[: 4 * dm.ACT_SIZE], "<f4").copy()
        self.chip_us.append(struct.unpack("<I", body[-4:])[0])
        us = (time.perf_counter() - t) * 1e6
        self.round_us.append(us)
        return a, us


class HoverBrain:
    def __call__(self, obs):
        return np.full(4, 2 * dm.HOVER_PWM - 1, np.float32), 0.0


class RandomBrain:
    def __init__(self, seed=3):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, 4).astype(np.float32), 0.0


# ---------------------------------------------------------------- the drone
def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_obs(data, target, hist):
    pos, quat, vel, omega = data.qpos[0:3], data.qpos[3:7], data.qvel[0:3], data.qvel[3:6]
    err = target - pos
    err = err * min(1.0, 1.0 / (np.linalg.norm(err) + 1e-6))
    R = quat_to_rot(quat)
    return np.concatenate([err, vel, R[:, 0], R[:, 1], omega, hist.reshape(-1)]).astype(np.float32)


def parse_targets(s):
    out = []
    for part in s.split(";"):
        xyz, _, t = part.partition("@")
        out.append((float(t or 0), np.array([float(v) for v in xyz.split(",")])))
    return sorted(out, key=lambda p: p[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brain", choices=["npz", "chip", "hover", "random"], default="npz")
    p.add_argument("--policy")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--port")
    p.add_argument("--chip-mode")
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--hist", type=int, help="past actions the brain sees (default: from the npz, else 3)")
    p.add_argument("--delay-ms", type=float, default=0.0)
    p.add_argument("--secs", type=float, default=10.0)
    p.add_argument("--start", default="0,0,1", help="x,y,z")
    p.add_argument("--targets", default="0,0,1")
    p.add_argument("--gust", action="append", default=[], help="N@seconds[@angle_deg]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--cam", default="chase", choices=["chase", "side", "wide"])
    p.add_argument("--view", help="camera 'distance elevation azimuth' (overrides --cam's)")
    p.add_argument("--orbit", type=float, default=0.0, help="camera turns this many degrees per second")
    p.add_argument("--out-fps", type=int, help="write the video at this rate (fps 240 + out-fps 30 = 8x slow motion)")
    p.add_argument("--lookat", help="fixed camera target 'x y z' instead of following the drone")
    p.add_argument("--hud")
    p.add_argument("--json")
    p.add_argument("--trace", help="write the flight path (npz: t, pos, quat, target)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    brain = {"npz": lambda: NpzBrain(args.policy, args.int8), "chip": lambda: ChipBrain(args.port, args.chip_mode),
             "hover": HoverBrain, "random": RandomBrain}[args.brain]()

    model = mujoco.MjModel.from_xml_string(dm.xml(floor_contact=True, pretty=True))
    data = mujoco.MjData(model)
    bid = model.body("drone").id
    tgt_mocap = model.body("target").mocapid[0]
    data.qpos[0:3] = [float(v) for v in args.start.split(",")]
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    targets = parse_targets(args.targets)
    gusts = []
    for g in args.gust:
        f, t, *ang = (float(v) for v in g.split("@"))
        a = np.deg2rad(ang[0] if ang else 0.0)
        gusts.append((t, f * np.array([np.cos(a), np.sin(a), 0.0])))

    n_sub = int(round(1.0 / args.hz / dm.SIM_DT))
    delay_sub = int(round(args.delay_ms / 1000 / dm.SIM_DT))
    queue = collections.deque()           # (apply at substep index, pwm)
    pwm_now = np.full(4, dm.HOVER_PWM)
    omega_m = np.full(4, dm.HOVER_PWM * dm.OMEGA_MAX)
    ad = np.exp(-dm.MOTOR_A * dm.SIM_DT)
    bd = dm.OMEGA_MAX * (1 - ad)
    n_hist = args.hist or (int((brain.layers[0][0].shape[0] - 15) // 4) if isinstance(brain, NpzBrain) else dm.HIST)
    hist = np.zeros((n_hist, 4), np.float32)
    gust = np.zeros(3)

    renderer = writer = hud = None
    if args.video:
        import imageio.v2 as imageio
        w, h = (int(v) for v in args.size.split("x"))
        renderer = mujoco.Renderer(model, h, w)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = {"chase": (0.9, -18, 135), "side": (1.6, -8, 90),
                                                    "wide": (2.4, -14, 120)}[args.cam]
        if args.view:
            cam.distance, cam.elevation, cam.azimuth = (float(v) for v in args.view.split())
        fixed = np.array([float(v) for v in args.lookat.split()]) if args.lookat else None
        cam.lookat[:] = data.qpos[0:3] if fixed is None else fixed
        writer = imageio.get_writer(args.video, fps=args.out_fps or args.fps, quality=9, macro_block_size=8)
        if args.hud:
            from hud import Hud
            hud = Hud(json.loads(args.hud if args.hud.lstrip().startswith("{") else Path(args.hud).read_text()), w, h)

    trail = []
    step, sub_i = 0, 0
    t_frame = 0.0
    crash_t = None
    errs, brain_us, trace = [], [], []
    target = targets[0][1]
    total_steps = int(args.secs * args.hz)
    while step < total_steps:
        t = data.time
        for tt, xyz in targets:
            if t >= tt - 1e-9:
                target = xyz
        data.mocap_pos[tgt_mocap] = target

        if crash_t is None:
            obs = make_obs(data, target, hist)
            obs[0:15] += rng.normal(0, 0.01, 15).astype(np.float32) * 0  # no sensor noise on the Mac
            act, us = brain(obs)
            act = np.clip(np.asarray(act, np.float64), -1, 1)
            brain_us.append(us)
            hist = np.concatenate([act[None].astype(np.float32), hist[:-1]], axis=0)
            queue.append((sub_i + delay_sub, (act + 1) / 2))
        else:
            queue.clear()
            pwm_now = np.zeros(4)                 # motors cut after a crash, like a kill switch

        for _ in range(n_sub):
            while queue and queue[0][0] <= sub_i:
                pwm_now = queue.popleft()[1]
            for gt, gf in gusts:
                if abs(data.time - gt) < dm.SIM_DT / 2:
                    gust = gust + gf
            gust = gust * np.exp(-dm.SIM_DT / 0.15)
            omega_m = ad * omega_m + bd * pwm_now
            pp = np.clip(omega_m / dm.OMEGA_MAX, 0, 1.2)
            thrust = dm.pwm_to_thrust(pp)
            yaw = dm.J_M * (-dm.MOTOR_A * omega_m + dm.MOTOR_A * dm.OMEGA_MAX * pwm_now) * 1e-3 + dm.pwm_to_torque(pp)
            data.ctrl[:] = np.concatenate([thrust, yaw])
            data.xfrc_applied[bid, 0:3] = gust
            mujoco.mj_step(model, data)
            sub_i += 1

            if renderer is not None and data.time >= t_frame - 1e-9:
                t_frame += 1.0 / args.fps
                look = data.qpos[0:3].copy() if fixed is None else fixed
                look[2] = max(look[2], 0.25)
                k = 1 - 0.9 ** (30 / args.fps)                 # same follow speed at any frame rate
                cam.lookat[:] = (1 - k) * cam.lookat + k * look
                cam.azimuth += args.orbit / args.fps
                renderer.update_scene(data, cam)
                trail.append(data.qpos[0:3].copy())
                del trail[:-int(1.5 * args.fps)]           # the last 1.5 s of flight, as a fading ribbon
                scn = renderer.scene
                for k in range(1, len(trail)):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    a_ = k / len(trail)
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                        np.array([0.13, 0.55, 0.95, 0.15 + 0.6 * a_], np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, trail[k - 1], trail[k])
                    scn.ngeom += 1
                frame = renderer.render()
                if hud is not None:
                    frame = hud.draw(frame, t=data.time, dist=0.0, fallen=crash_t is not None,
                                     us=int(np.median(brain.chip_us[-20:])) if getattr(brain, "chip_us", None) else None)
                writer.append_data(frame)

        pos = data.qpos[0:3].copy()
        trace.append((data.time, *pos, *data.qpos[3:7], *target))
        up = quat_to_rot(data.qpos[3:7])[2, 2]
        if crash_t is None and (pos[2] < 0.03 or (pos[2] < 0.08 and up < 0.5) or np.abs(pos).max() > 8):
            crash_t = data.time
        if crash_t is None and data.time > 1.0:
            errs.append(np.linalg.norm(target - pos))
        step += 1

    if writer is not None:
        writer.close()
    errs = np.array(errs) if errs else np.array([np.nan])
    out = {"brain": args.brain, "policy": args.policy, "int8": args.int8, "hz": args.hz,
           "delay_ms": args.delay_ms, "secs": args.secs, "gusts": args.gust, "targets": args.targets,
           "crashed": crash_t is not None, "crash_t": None if crash_t is None else round(crash_t, 3),
           "pos_err_mean_m": round(float(np.nanmean(errs)), 4),
           "pos_err_final_m": round(float(np.linalg.norm(target - data.qpos[0:3])), 4),
           "brain_us_median": round(float(np.median(brain_us)), 1) if brain_us else None}
    if getattr(brain, "chip_us", None):
        out["chip_infer_us_median"] = int(np.median(brain.chip_us))
        out["round_trip_us_median"] = int(np.median(brain.round_us))
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    if args.trace:
        np.savez(args.trace, trace=np.array(trace))


if __name__ == "__main__":
    main()
