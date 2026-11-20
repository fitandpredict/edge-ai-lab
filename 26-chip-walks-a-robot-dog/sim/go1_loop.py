"""Walk the Unitree Go1 in plain MuJoCo (CPU) with a learned policy as its brain.

The brain can be:
  onnx  the policy file run by onnxruntime on the Mac (reference)
  npz   the same weights run by our own numpy MLP (checks the export)
  chip  the ESP32-S3 over USB serial: Mac sends the 48 observations, Chip sends back 12 actions

usage:
  python sim/go1_loop.py --brain onnx --secs 10 --video out/onnx.mp4
  python sim/go1_loop.py --brain chip --port /dev/cu.usbmodem101 --secs 20 --video out/chip.mp4
  python sim/go1_loop.py --brain onnx --delay-ms 40          # how late can the brain be?

Prints a JSON summary: forward-speed error vs the command, falls, brain time per step.
Observation layout and constants follow MuJoCo Playground's sim2sim/play_go1_joystick.py (Apache 2.0).
"""

import argparse
import json
import struct
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENES = {
    "flat": HERE / "go1" / "scene_mjx_feetonly_flat_terrain.xml",
    "rough": HERE / "go1" / "scene_mjx_feetonly_rough_terrain.xml",
    "full": HERE / "go1" / "scene_mjx_fullcollisions_flat_terrain.xml",  # body can hit the floor (renders)
}
CTRL_DT, SIM_DT, ACTION_SCALE = 0.02, 0.004, 0.5
SYNC_OUT, SYNC_IN = b"\xaa\x55", b"\x55\xaa"


class OnnxBrain:
    def __init__(self, path):
        import onnxruntime as rt
        self.s = rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, obs):
        t = time.perf_counter()
        a = self.s.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
        return a, (time.perf_counter() - t) * 1e6


class NpzBrain:
    """Plain MLP from export_policy.py: optional obs normalisation, hidden layers, tanh head."""

    def __init__(self, path, int8=False):
        z = np.load(path)
        self.mean, self.std = z.get("obs_mean"), z.get("obs_std")
        n = int(z["n_layers"])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.int8 = int8
        if int8:  # what Chip's int8 mode does: per-output-channel int8 weights
            q = []
            for W, b in self.layers:
                s = np.abs(W).max(axis=0) / 127.0
                s[s == 0] = 1
                q.append((np.clip(np.round(W / s), -127, 127) * s, b))
            self.layers = q
        self.act = str(z["activation"])
        self.head = str(z["head"])

    def __call__(self, obs):
        t = time.perf_counter()
        x = obs.astype(np.float32)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            if self.int8:  # activations quantised per layer, like the firmware
                sx = max(np.abs(x).max(), 1e-8) / 127.0
                x = np.round(x / sx) * sx
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0) if self.act == "relu" else x / (1 + np.exp(-x))  # swish
        if self.head == "tanh":
            x = np.tanh(x)
        return x, (time.perf_counter() - t) * 1e6


class ChipBrain:
    """Frame out: AA 55 + 48 float32. Frame in: 55 AA + 12 float32 + uint32 inference µs."""

    def __init__(self, port, n_obs=48, n_act=12, mode=None):
        import serial
        self.n_obs, self.n_act = n_obs, n_act
        self.ser = serial.Serial(port, 921600, timeout=1.0)
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        if mode is not None:
            self.ser.write(str(mode).encode())
            print("chip:", self.ser.readline().decode(errors="ignore").strip())
        self.chip_us = []

    def __call__(self, obs):
        t = time.perf_counter()
        self.ser.write(SYNC_OUT + obs.astype("<f4").tobytes())
        buf = b""
        while not buf.endswith(SYNC_IN):
            c = self.ser.read(1)
            if not c:
                raise TimeoutError("no answer from Chip (is the firmware flashed?)")
            buf = (buf + c)[-2:]
        body = self.ser.read(4 * self.n_act + 4)
        a = np.frombuffer(body[: 4 * self.n_act], "<f4").copy()
        self.chip_us.append(struct.unpack("<I", body[-4:])[0])
        return a, (time.perf_counter() - t) * 1e6


class RandomBrain:
    """Flails: random joint targets, for the 'learning to walk means falling over' shot."""

    def __init__(self, seed=3):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, 12).astype(np.float32), 0.0


def make_brain(args):
    if args.brain == "random":
        return RandomBrain()
    if args.brain == "onnx":
        return OnnxBrain(args.policy or HERE / "go1_policy.onnx")
    if args.brain == "npz":
        return NpzBrain(args.policy or HERE / "big.npz", int8=args.int8)
    return ChipBrain(args.port, mode=args.chip_mode)


def get_obs(model, data, default, last_action, command):
    imu = data.site_xmat[model.site("imu").id].reshape(3, 3)
    return np.hstack([
        data.sensor("local_linvel").data,
        data.sensor("gyro").data,
        imu.T @ np.array([0, 0, -1.0]),
        data.qpos[7:] - default,
        data.qvel[6:],
        last_action,
        command,
    ]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brain", choices=["onnx", "npz", "chip", "random"], default="onnx")
    p.add_argument("--policy", help="policy file (default: the pretrained Go1 policy)")
    p.add_argument("--int8", action="store_true", help="npz brain: simulate Chip's int8 mode")
    p.add_argument("--chip-mode", type=int, help="firmware mode 0-4 (see dog_brain.ino)")
    p.add_argument("--port", default="/dev/cu.usbmodem101")
    p.add_argument("--terrain", choices=list(SCENES), default="flat")
    p.add_argument("--secs", type=float, default=10.0)
    p.add_argument("--cmd", type=float, nargs=3, default=[1.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    p.add_argument("--delay-ms", type=float, default=0.0, help="apply each action this much later")
    p.add_argument("--real-latency", action="store_true",
                   help="apply each action after the brain's own measured compute time (as if on the robot)")
    p.add_argument("--think-us", type=float, help="with --real-latency and a non-Chip brain: pretend each step takes this long")
    p.add_argument("--push", type=float, default=0.0, help="sideways shove (N) every 4 s")
    p.add_argument("--video", help="write an mp4 (30 fps, --size, bright look)")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--cam", default="track", help="track | fixed")
    p.add_argument("--view", type=float, nargs=3, default=[2.4, -16, 135], metavar=("DIST", "ELEV", "AZIM"))
    p.add_argument("--lookat", type=float, nargs=3, default=[1.2, 0, 0.25], help="fixed camera target")
    p.add_argument("--hud", help="JSON: overlay pills and counters (see hud.py)")
    p.add_argument("--stop-at", type=float, help="command 0 m/s from this time on")
    p.add_argument("--look", default="day", help="day | dusk")
    p.add_argument("--json", help="also write the summary here")
    args = p.parse_args()

    scene = SCENES[args.terrain]
    if args.video and args.terrain in ("flat", "full"):  # the bright look
        from look import PRETTY
        scene = PRETTY[("full_" if args.terrain == "full" else "") + args.look]
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    default = np.array(model.keyframe("home").qpos[7:])
    n_sub = int(round(CTRL_DT / SIM_DT))
    trunk = model.body("trunk").id
    command = np.array(args.cmd, dtype=np.float32)

    brain = make_brain(args)
    last_action = np.zeros(12, np.float32)
    delay_steps = int(round(args.delay_ms / (SIM_DT * 1000)))
    pending = []  # (apply_at_substep, ctrl)

    renderer = writer = hud = None
    if args.video:
        import imageio.v2 as imageio
        from look import style
        from hud import Hud
        W, H = map(int, args.size.split("x"))
        style(model, args.look)
        renderer = mujoco.Renderer(model, H, W)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = args.view
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = args.lookat if args.cam == "fixed" else [data.xpos[trunk][0], data.xpos[trunk][1], 0.22]
        hud = Hud(json.loads(Path(args.hud).read_text()) if args.hud and Path(args.hud).exists()
                  else json.loads(args.hud) if args.hud else {}, W, H)
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30, quality=9, macro_block_size=8,
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        next_frame, n_frames = 0.0, 0

    steps = int(args.secs / CTRL_DT)
    brain_us, vx_err, falls, fallen = [], [], 0, False
    substep, busy_until, skipped = 0, 0, 0
    for k in range(steps):
        if args.stop_at is not None and k * CTRL_DT >= args.stop_at:
            command[:] = 0
        if substep >= busy_until:
            obs = get_obs(model, data, default, last_action, command)
            a, us = brain(obs)
            brain_us.append(us)
            last_action = np.asarray(a, np.float32)
            lat = delay_steps
            if args.real_latency:  # the action lands only once the brain has finished thinking
                think_us = brain.chip_us[-1] if isinstance(brain, ChipBrain) else (args.think_us or us)
                lat += int(np.ceil(think_us / (SIM_DT * 1e6)))
                busy_until = substep + lat  # still thinking at the next tick? that tick is skipped
            pending.append((substep + lat, last_action * ACTION_SCALE + default))
        else:
            skipped += 1
        if args.push and k % int(4 / CTRL_DT) == int(2 / CTRL_DT):
            data.xfrc_applied[trunk, 1] = args.push
        for _ in range(n_sub):
            while pending and pending[0][0] <= substep:
                data.ctrl[:] = pending.pop(0)[1]
            mujoco.mj_step(model, data)
            substep += 1
            if writer is not None and data.time >= next_frame - 1e-9:
                if args.cam == "track":  # chase: ease the target toward the trunk, ignore its tumbling
                    tx, ty = data.xpos[trunk][0], data.xpos[trunk][1]
                    cam.lookat[0] += 0.15 * (tx - cam.lookat[0])
                    cam.lookat[1] += 0.15 * (ty - cam.lookat[1])
                renderer.update_scene(data, cam)
                frame = renderer.render()
                writer.append_data(hud.draw(frame, t=data.time, dist=float(data.qpos[0]), fallen=fallen,
                                            us=(brain.chip_us[-1] if isinstance(brain, ChipBrain) and brain.chip_us else None)))
                next_frame += 1 / 30
                n_frames += 1
        data.xfrc_applied[trunk] = 0
        up_z = data.site_xmat[model.site("imu").id].reshape(3, 3)[2, 2]
        down = data.xpos[trunk][2] < 0.15 or up_z < 0.3
        if down and not fallen:
            falls += 1
        fallen = down
        if k > 50:  # skip the first second (standing up to speed)
            vx_err.append(data.sensor("local_linvel").data[0] - command[0])

    out = {
        "brain": args.brain + ("-int8" if args.int8 else ""),
        "terrain": args.terrain,
        "secs": args.secs,
        "cmd": args.cmd,
        "delay_ms": args.delay_ms,
        "real_latency": args.real_latency,
        "skipped_ticks": skipped,
        "falls": falls,
        "fallen_at_end": bool(fallen),
        "vx_rmse": float(np.sqrt(np.mean(np.square(vx_err)))) if vx_err else None,
        "distance_m": float(data.qpos[0]),
        "brain_us_median": float(np.median(brain_us)),
    }
    if isinstance(brain, ChipBrain):
        out["chip_infer_us_median"] = float(np.median(brain.chip_us))
        out["chip_infer_us_max"] = float(np.max(brain.chip_us))
        out["chip_mode"] = args.chip_mode
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
    if writer is not None:
        writer.close()
        print("video:", args.video, n_frames, "frames")


if __name__ == "__main__":
    main()
