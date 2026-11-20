"""A transparent PNG of the Go1 at a moment of a real run, for thumbnails and Short covers.

usage: python sim/still.py out.png --policy sim/ours_small.npz --at 2.4 [--delay-ms 30] [--view 2.0 -8 120]
Runs the brain on the Mac (full-collision Go1), renders one 2560x1440 frame at --at seconds with a magenta
sky and an invisible floor, and keys the magenta out (soft edge), then crops to the dog.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from go1_loop import ACTION_SCALE, CTRL_DT, SIM_DT, NpzBrain, OnnxBrain, get_obs
from look import PRETTY, style

KEY = np.array([255, 0, 255])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--policy", default=str(Path(__file__).parent / "ours_small.npz"))
    a.add_argument("--onnx", action="store_true")
    a.add_argument("--at", type=float, default=2.4)
    a.add_argument("--delay-ms", type=float, default=0.0)
    a.add_argument("--view", type=float, nargs=3, default=[2.0, -8, 120])
    args = a.parse_args()

    model = mujoco.MjModel.from_xml_path(str(PRETTY["full_day"]))
    model.vis.global_.offwidth, model.vis.global_.offheight = 2560, 1440
    model.opt.timestep = SIM_DT
    style(model)
    for t in range(model.ntex):  # skybox -> flat magenta
        if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            adr, n = model.tex_adr[t], model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t]
            px = model.tex_data[adr:adr + n].reshape(-1, model.tex_nchannel[t])
            px[:, :3] = KEY
    model.geom_rgba[model.geom("floor").id, 3] = 0.0
    model.mat_rgba[:, 3] = np.where(np.arange(model.nmat) == model.geom_matid[model.geom("floor").id], 0.0, model.mat_rgba[:, 3])
    model.vis.headlight.ambient[:] = 0.45
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    default = np.array(model.keyframe("home").qpos[7:])
    brain = OnnxBrain(Path(__file__).parent / "go1_policy.onnx") if args.onnx else NpzBrain(args.policy)
    cmd = np.array([1.0, 0, 0], np.float32)
    last = np.zeros(12, np.float32)
    pending, delay, sub = [], int(round(args.delay_ms / (SIM_DT * 1000))), 0
    while data.time < args.at:
        act, _ = brain(get_obs(model, data, default, last, cmd))
        last = np.asarray(act, np.float32)
        pending.append((sub + delay, last * ACTION_SCALE + default))
        for _ in range(int(round(CTRL_DT / SIM_DT))):
            while pending and pending[0][0] <= sub:
                data.ctrl[:] = pending.pop(0)[1]
            mujoco.mj_step(model, data)
            sub += 1
    r = mujoco.Renderer(model, 1440, 2560)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = args.view
    trunk = model.body("trunk").id
    cam.lookat[:] = [data.xpos[trunk][0], data.xpos[trunk][1], max(0.12, data.xpos[trunk][2] * 0.8)]
    r.update_scene(data, cam)
    img = r.render().astype(np.float32)
    d = np.linalg.norm(img - KEY, axis=2)
    alpha = np.clip((d - 40) / 90, 0, 1)
    # de-spill: pull magenta out of soft edge pixels
    rgb = img.copy()
    spill = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]
    rgb[..., 0] -= np.clip(spill, 0, None) * (1 - alpha)
    rgb[..., 2] -= np.clip(spill, 0, None) * (1 - alpha)
    out = np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8)
    im = Image.fromarray(out, "RGBA")
    im = im.crop(im.getbbox())
    im.save(args.out)
    print(args.out, im.size)


if __name__ == "__main__":
    main()
