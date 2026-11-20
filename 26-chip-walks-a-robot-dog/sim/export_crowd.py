"""Record real Go1 motion clips for the three.js crowd shot (8,192 dogs replaying them).

usage: python sim/export_crowd.py
writes video/three/go1/:
  <mesh>.stl          low-poly convex hulls of the Go1 parts, in MuJoCo's own mesh frame
  crowd.bin           float32 [clip][frame][geom][x y z qw qx qy qz] (MuJoCo world, z up)
  crowd.json          geom -> mesh + colour, clip kinds, frame count, fps
Clips (full-collision Go1, 8 s at 30 fps): random flailing (falls), an untrained-ish brain that stands,
a late brain that stumbles and falls, and good walkers.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from go1_loop import ACTION_SCALE, CTRL_DT, SIM_DT, NpzBrain, OnnxBrain, RandomBrain, SCENES, get_obs

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "video" / "three" / "go1"
SECS, FPS = 8.0, 30


def visual_geoms(model):
    return [g for g in range(model.ngeom)
            if model.geom_group[g] == 2 and not model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]


def export_meshes(model, geoms):
    names = {}
    for g in geoms:
        mid = model.geom_dataid[g]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        names[g] = name
        if (OUT / f"{name}.stl").exists() and name in getattr(export_meshes, "done", set()):
            continue
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        hull = trimesh.Trimesh(model.mesh_vert[a:a + n]).convex_hull
        if len(hull.faces) > 320:  # ~300 triangles a part: 8,192 dogs x 13 parts still has to draw
            import fast_simplification
            v, f = fast_simplification.simplify(np.asarray(hull.vertices, np.float32),
                                                np.asarray(hull.faces, np.int32), target_reduction=1 - 300 / len(hull.faces))
            hull = trimesh.Trimesh(v, f)
        hull.export(OUT / f"{name}.stl")
        export_meshes.done = getattr(export_meshes, "done", set()) | {name}
    return names


def run_clip(model, brain, delay_ms=0.0, cmd=(1.0, 0, 0), geoms=()):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    default = np.array(model.keyframe("home").qpos[7:])
    n_sub = int(round(CTRL_DT / SIM_DT))
    command = np.array(cmd, np.float32)
    last = np.zeros(12, np.float32)
    pending, delay = [], int(round(delay_ms / (SIM_DT * 1000)))
    frames, next_t, sub = [], 0.0, 0
    q = np.zeros(4)
    while len(frames) < SECS * FPS:
        a, _ = brain(get_obs(model, data, default, last, command))
        last = np.asarray(a, np.float32)
        pending.append((sub + delay, last * ACTION_SCALE + default))
        for _ in range(n_sub):
            while pending and pending[0][0] <= sub:
                data.ctrl[:] = pending.pop(0)[1]
            mujoco.mj_step(model, data)
            sub += 1
            if data.time >= next_t - 1e-9 and len(frames) < SECS * FPS:
                row = []
                for g in geoms:
                    mujoco.mju_mat2Quat(q, data.geom_xmat[g])
                    row.append(np.concatenate([data.geom_xpos[g], q]))
                frames.append(row)
                next_t += 1 / FPS
    return np.asarray(frames, np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENES["full"]))
    model.opt.timestep = SIM_DT
    geoms = visual_geoms(model)
    names = export_meshes(model, geoms)
    code = HERE.parent
    plan = ([("flail", lambda s=s: RandomBrain(s), 0.0) for s in range(12)]
            + [("stand", lambda: NpzBrain(code / "results/colab/small/snap_05.npz"), 0.0)] * 4
            + [("stumble", lambda: OnnxBrain(HERE / "go1_policy.onnx"), d) for d in (35, 40, 45, 50)]
            + [("walk", lambda: NpzBrain(HERE / "ours_small.npz"), 0.0)] * 4)
    clips, kinds = [], []
    for kind, make, delay in plan:
        clips.append(run_clip(model, make(), delay, geoms=geoms))
        kinds.append(kind)
        print(kind, delay, clips[-1].shape, flush=True)
    arr = np.stack(clips)
    arr.tofile(OUT / "crowd.bin")
    meta = {"fps": FPS, "frames": int(arr.shape[1]), "clips": kinds, "geoms": [names[g] for g in geoms],
            "colour": ["body" if names[g] == "trunk" else "trim" if names[g] == "hip" else "legs" for g in geoms]}
    (OUT / "crowd.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT / "crowd.bin", arr.shape, f"{arr.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
