"""The rover in MuJoCo: wheels, gyro, six laser rangefinders, a 200 Hz PID, and a brain once per cell.

  python sim/rover.py --brain numpy --size 7 --seed 3 --video results/clips/rover_7x7.mp4
  python sim/rover.py --brain chip  --size 7 --seed 3 --video results/clips/chip_7x7.mp4

Two layers, like the robot would run them:
  spinal cord  200 Hz: wheel encoders + gyro -> odometry; PID holds heading, centres between the side
               walls, drives 18 cm cells, turns in place. Chains straights without stopping.
  slow brain   once per cell, ~3 cm before the centre ("peek ahead"): six ToF readings + odometry
               memory -> forward / left / right / back. Runs on the Mac (numpy) or on Chip over USB.

The brain only ever sees what the rover's own sensors measured (rangefinders, encoders, gyro). The one
preprocessing step: the two diagonal sensors are reduced to "is there a side wall in the next cell", the
way the training sim reports them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from maze import CELL_M, DIRS, TOF_MAX_M, TURN, Maze, MazeEnv  # noqa: E402

WALL_T, WALL_H = 0.012, 0.05
WHEEL_R, TRACK = 0.016, 0.072
WHEEL_R_ODO = 0.0150   # effective rolling radius (the tyre squashes): calibrated by driving a known distance
VERBOSE = False
NAIVE = False          # --naive: the first version (nominal wheel radius, no wall referencing) for the 'it drifts' clip
V_MAX = 0.62            # m/s, the post's straight speed
DT = 0.005              # 200 Hz control
# ToF sensors: (name, x, y, yaw) in the rover frame, +x forward, +y left
TOF = [("fl", 0.045, 0.018, 0.0), ("fr", 0.045, -0.018, 0.0), ("dl", 0.04, 0.03, np.radians(25)),
       ("dr", 0.04, -0.03, -np.radians(25)), ("l", 0.0, 0.036, np.pi / 2), ("r", 0.0, -0.036, -np.pi / 2)]


def scene_xml(maze, start, goal):
    n = maze.n
    W = n * CELL_M
    geoms = []

    def wall(x0, y0, x1, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx = max(abs(x1 - x0) / 2, WALL_T / 2) + (WALL_T / 2 if x1 != x0 else 0)
        hy = max(abs(y1 - y0) / 2, WALL_T / 2) + (WALL_T / 2 if y1 != y0 else 0)
        geoms.append(f'<geom type="box" pos="{cx:.4f} {cy:.4f} {WALL_H / 2}" size="{hx:.4f} {hy:.4f} {WALL_H / 2}" material="wall"/>')

    for x in range(n):
        for y in range(n):
            if not maze.open(x, y, 0):
                wall(x * CELL_M, (y + 1) * CELL_M, (x + 1) * CELL_M, (y + 1) * CELL_M)
            if not maze.open(x, y, 1):
                wall((x + 1) * CELL_M, y * CELL_M, (x + 1) * CELL_M, (y + 1) * CELL_M)
            if y == 0 and not maze.open(x, y, 2):
                wall(x * CELL_M, 0, (x + 1) * CELL_M, 0)
            if x == 0 and not maze.open(x, y, 3):
                wall(0, y * CELL_M, 0, (y + 1) * CELL_M)
    sx, sy = (start[0] + 0.5) * CELL_M, (start[1] + 0.5) * CELL_M
    gx, gy = (goal[0] + 0.5) * CELL_M, (goal[1] + 0.5) * CELL_M
    sensors = "".join(f'<site name="tof_{nm}" pos="{x} {y} 0.02" zaxis="{np.cos(a):.4f} {np.sin(a):.4f} 0" size="0.004" rgba="0.1 0.1 0.1 1"/>'
                      for nm, x, y, a in TOF)
    rf = "".join(f'<rangefinder name="rf_{nm}" site="tof_{nm}" cutoff="{TOF_MAX_M}"/>' for nm, *_ in TOF)
    return f"""
<mujoco model="maze rover">
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/>
    <rgba rangefinder="1 0.9 0.1 1"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.85 0.86 0.88" width="256" height="256"/>
    <texture name="floor" type="2d" builtin="flat" rgb1="0.93 0.93 0.94" width="64" height="64"/>
    <material name="floor" texture="floor" reflectance="0.05"/>
    <material name="wall" rgba="0.16 0.18 0.24 1"/>
    <material name="body" rgba="0.2 0.5 0.95 1"/>
    <material name="tyre" rgba="0.08 0.08 0.08 1"/>
  </asset>
  <worldbody>
    <light pos="{W / 2} {W / 2} 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="{W} -1 2" dir="-0.3 0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom type="plane" size="{4 * W} {4 * W} 0.1" pos="{W / 2} {W / 2} 0" material="floor"/>
    <geom type="cylinder" size="0.045 0.0005" pos="{gx} {gy} 0.0006" rgba="0.1 0.75 0.3 1" contype="0" conaffinity="0"/>
    {''.join(geoms)}
    <body name="rover" pos="{sx} {sy} {WHEEL_R + 0.001}">
      <freejoint/>
      <geom type="box" size="0.045 0.032 0.012" pos="0 0 0.008" material="body" mass="0.12"/>
      <geom type="box" size="0.025 0.02 0.004" pos="-0.005 0 0.024" rgba="0.1 0.1 0.12 1" mass="0.02"/>
      <geom type="sphere" size="0.006" pos="0.035 0 -0.0085" condim="1" priority="1" mass="0.005"/>
      <geom type="sphere" size="0.006" pos="-0.035 0 -0.0085" condim="1" priority="1" mass="0.005"/>
      <site name="imu" pos="0 0 0"/>
      {sensors}
      <body name="wl" pos="0 {TRACK / 2} 0"><joint name="wl" type="hinge" axis="0 1 0" damping="0.0005" armature="0.0001"/>
        <geom type="cylinder" size="{WHEEL_R} 0.005" quat="0.7071 0.7071 0 0" material="tyre" contype="0" conaffinity="0" mass="0.01"/><geom type="sphere" size="{WHEEL_R}" rgba="0 0 0 0" friction="1.5 0.02 0.001" mass="0.001"/></body>
      <body name="wr" pos="0 {-TRACK / 2} 0"><joint name="wr" type="hinge" axis="0 1 0" damping="0.0005" armature="0.0001"/>
        <geom type="cylinder" size="{WHEEL_R} 0.005" quat="0.7071 0.7071 0 0" material="tyre" contype="0" conaffinity="0" mass="0.01"/><geom type="sphere" size="{WHEEL_R}" rgba="0 0 0 0" friction="1.5 0.02 0.001" mass="0.001"/></body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="ml" joint="wl" kv="0.004" ctrlrange="-60 60" forcelimited="true" forcerange="-0.015 0.015"/>
    <velocity name="mr" joint="wr" kv="0.004" ctrlrange="-60 60" forcelimited="true" forcerange="-0.015 0.015"/>
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu" noise="0.002"/>
    <jointpos name="enc_l" joint="wl"/><jointpos name="enc_r" joint="wr"/>
    {rf}
  </sensor>
</mujoco>"""


class Rover:
    """Sim + spinal cord. Odometry and every reading the brain gets come from the sensors only."""

    def __init__(self, maze, start, goal, heading):
        self.maze, self.goal = maze, tuple(goal)
        self.m = mujoco.MjModel.from_xml_string(scene_xml(maze, start, goal))
        self.d = mujoco.MjData(self.m)
        yaw0 = np.pi / 2 - heading * np.pi / 2          # heading 0 = +y
        self.d.qpos[3:7] = [np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]
        mujoco.mj_forward(self.m, self.d)
        self.start = np.array([(start[0] + 0.5) * CELL_M, (start[1] + 0.5) * CELL_M])
        self.pos = self.start.copy()                    # odometry (start cell is known)
        self.yaw = yaw0                                 # gyro-integrated
        self.enc = np.zeros(2)
        self.sid = {nm: self.m.sensor(f"rf_{nm}").adr[0] for nm, *_ in TOF}
        self.t = 0.0
        self.frames, self.log = [], []

    # --- sensors -------------------------------------------------------------------------------------
    def tof(self):
        r = {nm: self.d.sensordata[a] for nm, a in self.sid.items()}
        return {k: (TOF_MAX_M if v < 0 else v) for k, v in r.items()}

    def cell(self):
        return tuple(np.clip(np.floor(self.pos / CELL_M).astype(int), 0, self.maze.n - 1))

    def heading(self):
        return int(np.round((np.pi / 2 - self.yaw) / (np.pi / 2))) % 4

    def _odometry(self):
        e = np.array([self.d.sensordata[self.m.sensor("enc_l").adr[0]], self.d.sensordata[self.m.sensor("enc_r").adr[0]]])
        ds = (e - self.enc) * (WHEEL_R if NAIVE else WHEEL_R_ODO)
        self.enc = e
        self.yaw += self.d.sensordata[self.m.sensor("gyro").adr[0] + 2] * DT
        s = ds.mean()
        self.pos += s * np.array([np.cos(self.yaw), np.sin(self.yaw)])

    def _step(self, v, w, render=None):
        wl, wr = (v - w * TRACK / 2) / WHEEL_R, (v + w * TRACK / 2) / WHEEL_R
        self.d.ctrl[:] = [wl, wr]
        for _ in range(int(DT / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        self.t += DT
        self._odometry()
        if len(self.log) == 0 or self.t - self.log[-1][0] >= 1 / 60 - 1e-9:
            q = self.d.qpos[3:7]
            yaw = 2 * np.arctan2(q[3], q[0])
            wl, wr = self.enc
            self.log.append([round(self.t, 4), round(float(self.d.qpos[0]), 5), round(float(self.d.qpos[1]), 5), round(float(yaw), 5),
                             round(float(wl), 4), round(float(wr), 4)] + [round(float(v), 4) for v in self.tof().values()])
        if render:
            render(self)

    # --- spinal cord ---------------------------------------------------------------------------------
    def turn(self, quarter, render=None):
        target = self.yaw - quarter * np.pi / 2
        target = np.pi / 2 - np.round((np.pi / 2 - target) / (np.pi / 2)) * np.pi / 2
        for _ in range(400):
            err = np.arctan2(np.sin(target - self.yaw), np.cos(target - self.yaw))
            if abs(err) < 0.01:
                break
            self._step(0.0, np.clip(14 * err, -9, 9), render)
        self.yaw = target   # gyro snaps to the grid, as micromice do after a turn

    def drive_to(self, cell, v_end, render=None, peek=None):
        """Drive to the centre of cell; leave at speed v_end (0 = stop). peek() is called ~3 cm before."""
        tgt = (np.array(cell) + 0.5) * CELL_M
        h = self.heading()
        fwd = DIRS[h].astype(float)
        target_yaw = np.pi / 2 - h * np.pi / 2
        peeked = False
        v = float(np.hypot(*self.d.qvel[:2]))
        for _ in range(2000):
            togo = float(np.dot(tgt - self.pos, fwd))
            t = self.tof()
            front = (t["fl"] + t["fr"]) / 2 + 0.045 + WALL_T / 2      # sensor to wall face, from the rover centre
            if not NAIVE and front < CELL_M * 0.9 and togo < CELL_M * 0.6:
                # a wall ahead of the target cell: trust the lasers over the wheels for the last few cm
                togo = front - CELL_M / 2
                self.pos = tgt - togo * fwd
            if not peeked and peek and togo < 0.03:
                peeked = True
                v_end = peek()
            if togo <= 0.002:
                break
            v_cap = np.sqrt(max(v_end, 0.0) ** 2 + 2 * 3.0 * max(togo, 0))   # brake at 3 m/s^2
            v = min(V_MAX, v + 4.0 * DT, v_cap)
            v = max(v, 0.08)
            # hold heading, centre between side walls when both are close
            lat = 0.0
            side = CELL_M / 2 - WALL_T / 2 - 0.036      # sensor-to-wall when centred
            if t["l"] < 0.12 and t["r"] < 0.12:
                lat = t["l"] - t["r"]
            elif not NAIVE and t["l"] < 0.10:
                lat = 2 * (t["l"] - side)
            elif not NAIVE and t["r"] < 0.10:
                lat = 2 * (side - t["r"])
            yaw_err = np.arctan2(np.sin(target_yaw - self.yaw), np.cos(target_yaw - self.yaw))
            self._step(v, 8 * yaw_err + 6 * lat, render)
        if v_end == 0:
            for _ in range(20):
                self._step(0.0, 0.0, render)
        self.pos = tgt.copy() if v_end == 0 else self.pos   # re-zero odometry at a stop (cell centre)

    # --- what the brain sees -------------------------------------------------------------------------
    def observation(self, env, look_ahead=0.0):
        """Build the brain's 20 numbers from the rangefinders + odometry. look_ahead: metres left to the
        centre (the front and side readings are corrected to the centre, like the training sim)."""
        t = self.tof()
        half = WALL_T / 2
        front = (t["fl"] + t["fr"]) / 2 + 0.045 + half - look_ahead
        # 25° diagonals: a side wall in the next cell answers at ~0.17-0.19 m, anything else is further
        diag = lambda v: CELL_M * 1.414 if v < 0.235 - look_ahead else 2 * CELL_M * 1.414
        if front < CELL_M * 0.75:
            dl = dr = CELL_M / 2 * 1.414
        else:
            dl, dr = diag(t["dl"]), diag(t["dr"])
        d = np.array([front, front, dl, dr, t["l"] + 0.036 + half, t["r"] + 0.036 + half])
        return np.clip(d, 0.03, TOF_MAX_M) / TOF_MAX_M


class Camera:
    """Renders every `every`-th control step straight into an MP4 (frames are not kept in memory)."""

    def __init__(self, rover, size=(1280, 720), every=4, path=None):
        import imageio
        self.w = imageio.get_writer(path, fps=int(round(1 / (DT * every))), quality=8, macro_block_size=8)
        self.last = None
        self.r = mujoco.Renderer(rover.m, size[1], size[0])
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
        self.cam = mujoco.MjvCamera()
        W = rover.maze.n * CELL_M
        self.cam.lookat[:] = [W / 2, W / 2 - 0.05 * W, 0]
        self.cam.distance = 1.55 * W + 0.2
        self.cam.elevation = -62
        self.cam.azimuth = 90
        self.every, self.k, self.frames = every, 0, []

    def __call__(self, rover):
        self.k += 1
        if self.k % self.every:
            return
        self.r.update_scene(rover.d, self.cam, self.opt)
        self.last = self.r.render()
        self.w.append_data(self.last)

    def close(self, hold_s=1.0):
        for _ in range(int(hold_s / (DT * self.every))):
            self.w.append_data(self.last)
        self.w.close()


def run(maze, start, goal, heading, brain, video=None, max_cells=None, visits=True, log=None):
    """brain(obs) -> action int; brain.reset() at the start."""
    rover = Rover(maze, start, goal, heading)
    env = MazeEnv(sizes=(maze.n,), noise=0.0, visits=visits)   # used for its observation layout only (odometry memory)
    env.use(maze, start, goal, heading)
    env.reset()
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
    cam = Camera(rover, path=video) if video else None
    brain.reset()
    max_cells = max_cells or 4 * maze.n ** 2
    decisions, bumps, think_us = 0, 0, []
    cell = tuple(start)
    t0 = time.time()

    def decide(look_ahead):
        nonlocal decisions
        env.cell, env.heading = cell, rover.heading()
        o = env._obs()
        o[:6] = rover.observation(env, look_ahead)
        a = brain(o)
        decisions += 1
        if VERBOSE:
            us = getattr(brain, "last_us", None)
            print(f"cell ({cell[0]},{cell[1]})  lasers {' '.join(f'{v * 1.2:4.2f}' for v in o[:6])} m  ->  {['FORWARD', 'LEFT', 'RIGHT', 'BACK'][a]:7s}"
                  + (f"  {us / 1000:.2f} ms" if us else ""), flush=True)
        events.append(dict(t=round(rover.t, 4), cell=[int(c) for c in cell], heading=int(env.heading), action=int(a),
                           obs=[round(float(v), 4) for v in o], us=getattr(brain, "last_us", None)))
        return a

    events = []
    a = decide(0.0)
    while decisions <= max_cells and cell != tuple(goal):
        # turn in place if needed
        q = TURN[a]
        if q:
            rover.turn(q, cam)
        h = rover.heading()
        env.last = a
        if not maze.open(*cell, h):
            bumps += 1        # the brain asked to drive into a wall: count it, don't drive
            a = decide(0.0)
            continue
        nxt = (cell[0] + DIRS[h][0], cell[1] + DIRS[h][1])
        plan = {}

        def peek():
            # 3 cm before the next centre: decide now, keep speed if the answer is "forward" into open space
            nonlocal cell
            cell_save = cell
            cell = nxt
            env.visits[nxt] += 1
            plan["a"] = decide(0.03)
            cell = cell_save
            go_on = plan["a"] == 0 and maze.open(*nxt, h) and nxt != tuple(goal)
            return V_MAX if go_on else 0.0

        rover.drive_to(nxt, 0.0, cam, peek)
        cell = nxt
        env.cell = cell
        if "a" not in plan:   # very short hop: decide at the centre
            env.visits[cell] += 1
            plan["a"] = decide(0.0)
        a = plan["a"]
    res = dict(solved=cell == tuple(goal), decisions=decisions, bumps=bumps, sim_seconds=round(rover.t, 2),
               shortest=int(maze.bfs(tuple(start))[tuple(goal)]), wall_seconds=round(time.time() - t0, 1))
    if log:
        closed = [[x, y, d] for x in range(maze.n) for y in range(maze.n) for d in range(4) if not maze.open(x, y, d)]
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(json.dumps(dict(n=maze.n, cell_m=CELL_M, closed=closed, start=list(start), goal=list(goal),
            fields=["t", "x", "y", "yaw", "enc_l", "enc_r"] + [f"tof_{nm}" for nm, *_ in TOF],
            samples=rover.log, decisions=events, result={k: v for k, v in res.items()})))
        res["log"] = str(log)
    if video:
        cam.close()
        res["video"] = str(video)
    return res


class LocalBrain:
    def __init__(self, model, int8=False):
        from export_brain import NumpyBrain, load
        self.b = NumpyBrain(load(model)[0], int8=int8)
        self.us = []

    def reset(self):
        self.b.reset()

    def __call__(self, o):
        t = time.perf_counter()
        a = int(np.argmax(self.b(o)))
        self.us.append((time.perf_counter() - t) * 1e6)
        return a


class SB3Brain:
    """Any trained model straight from its .zip (used for the MLP ablation brains)."""

    def __init__(self, model):
        sys.path.insert(0, str(HERE))
        from race import Learned
        self.l = Learned(model)

    def reset(self):
        self.l.reset(None)

    def __call__(self, o):
        return self.l.act(o, None)


class ChipLink:
    def __init__(self, port=None, mode=2):
        from chip import ChipBrain
        self.c = ChipBrain(port, mode)

    def reset(self):
        self.c.reset()

    def __call__(self, o):
        a = int(self.c(o)[1])
        self.last_us = self.c.us[-1]
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["numpy", "int8", "chip", "sb3"], default="numpy")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="print one line per decision")
    ap.add_argument("--naive", action="store_true", help="first version: no wheel calibration, no wall referencing")
    ap.add_argument("--eval", type=int, default=0, help="run N unseen mazes (seeds 50000+) and print the summary")
    ap.add_argument("--out")
    ap.add_argument("--model", default=str(ROOT / "models" / "lstm_final.zip"))
    ap.add_argument("--size", type=int, default=7)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mode", type=int, default=2, help="Chip mode (see maze_brain.ino)")
    ap.add_argument("--port")
    ap.add_argument("--video")
    ap.add_argument("--log", help="write the run (poses, sensor readings, decisions) as JSON for the bench 3D replay")
    a = ap.parse_args()
    global NAIVE, VERBOSE
    NAIVE, VERBOSE = a.naive, a.verbose
    if a.brain == "chip":
        brain = ChipLink(a.port, a.mode)
    elif a.brain == "sb3":
        brain = SB3Brain(a.model)
    else:
        brain = LocalBrain(a.model, int8=a.brain == "int8")
    vis = "novis" not in Path(a.model).name
    if a.eval:
        rows = []
        for k in range(a.eval):
            m = Maze(a.size, 50_000 + k)
            r = run(m, (a.size // 2, 0), (a.size - 1, a.size - 1), 0, brain, visits=vis)
            rows.append(r)
            print(json.dumps(dict(seed=50_000 + k, **r)), flush=True)
        ok = [r for r in rows if r["solved"]]
        summ = dict(model=Path(a.model).name, brain=a.brain, size=a.size, mazes=a.eval, solve_rate=len(ok) / a.eval,
                    bumps_total=sum(r["bumps"] for r in rows),
                    decisions_vs_shortest=round(float(np.median([r["decisions"] / max(r["shortest"], 1) for r in ok])), 2) if ok else None,
                    sim_seconds_median=round(float(np.median([r["sim_seconds"] for r in ok])), 2) if ok else None)
        if a.brain == "chip":
            summ["chip_us_mean"] = round(float(np.mean(brain.c.us)), 1)
        print(json.dumps(summ))
        if a.out:
            Path(a.out).write_text(json.dumps(dict(summary=summ, runs=rows), indent=1))
        return
    maze = Maze(a.size, a.seed)
    start, goal = (a.size // 2, 0), (a.size - 1, a.size - 1)
    res = run(maze, start, goal, 0, brain, a.video, max_cells=a.max_cells, visits=vis, log=a.log)
    if a.brain == "chip":
        res["chip_us_mean"] = round(float(np.mean(brain.c.us)), 1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
