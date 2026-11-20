"""The channel look for MuJoCo renders: bright sky and floor (scene_pretty_*.xml), a coloured Go1."""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PRETTY = {k: HERE / "go1" / f"scene_pretty_{k}.xml" for k in ("day", "dusk", "full_day", "full_dusk")}

BODY = np.array([0x5F, 0xBF, 0x73, 255]) / 255   # the hero green, a touch darker than GREEN for 3D shading
LEGS = np.array([0x2B, 0x32, 0x40, 255]) / 255   # graphite
TRIM = np.array([0xEC, 0xEF, 0xF4, 255]) / 255   # hips: light trim


def style(model, look="day"):
    """Recolour the dog's visual meshes in place (physics untouched)."""
    for g in range(model.ngeom):
        if model.geom_contype[g] or model.geom_conaffinity[g] or model.geom_group[g] != 2:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) or ""
        model.geom_matid[g] = -1
        model.geom_rgba[g] = BODY if name == "trunk" else TRIM if name == "hip" else LEGS
