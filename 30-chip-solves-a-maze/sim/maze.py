"""Walled mazes, the rover's sensors as numbers, and the cell-level world the brain learns in.

The brain is hierarchical, like the robot it would drive:
  slow brain   once per cell: look at 20 numbers, pick forward / left / right / back
  spinal cord  a PID on wheels + gyro drives the 18 cm to the next cell (rover.py, MuJoCo)

Training happens at the cell level (one step = one decision), which is what the brain actually sees.

Coordinates: cell (x, y), x right, y up. Headings 0=N(+y) 1=E(+x) 2=S 3=W.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

DIRS = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])
CELL_M = 0.18          # micromouse cell
TOF_MAX_M = 1.2        # VL53L0X long-range mode tops out around here
OBS_DIM = 20
ACTIONS = ["forward", "left", "right", "back"]   # relative turn, then drive one cell
TURN = [0, -1, 1, 2]


class Maze:
    """Perfect maze (one path between any two cells), recursive backtracker. walls[x, y, d] = True if closed."""

    def __init__(self, n, seed=None, loops=0):
        self.n = n
        rng = np.random.default_rng(seed)
        w = np.ones((n, n, 4), dtype=bool)
        seen = np.zeros((n, n), dtype=bool)
        stack = [(0, 0)]
        seen[0, 0] = True
        while stack:
            x, y = stack[-1]
            opts = [d for d in range(4) if self.inside(x + DIRS[d][0], y + DIRS[d][1]) and not seen[x + DIRS[d][0], y + DIRS[d][1]]]
            if not opts:
                stack.pop()
                continue
            d = opts[rng.integers(len(opts))]
            nx, ny = x + DIRS[d][0], y + DIRS[d][1]
            w[x, y, d] = w[nx, ny, (d + 2) % 4] = False
            seen[nx, ny] = True
            stack.append((nx, ny))
        # optional extra openings: a maze with loops (where wall-following can circle forever)
        for _ in range(loops):
            x, y, d = rng.integers(n), rng.integers(n), rng.integers(4)
            nx, ny = x + DIRS[d][0], y + DIRS[d][1]
            if self.inside(nx, ny):
                w[x, y, d] = w[nx, ny, (d + 2) % 4] = False
        self.walls = w

    def inside(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n

    def open(self, x, y, d):
        return not self.walls[x, y, d]

    def bfs(self, start):
        """Steps from start to every cell."""
        dist = -np.ones((self.n, self.n), dtype=int)
        dist[start] = 0
        q = [start]
        for c in q:
            for d in range(4):
                if self.open(*c, d):
                    nb = (c[0] + DIRS[d][0], c[1] + DIRS[d][1])
                    if dist[nb] < 0:
                        dist[nb] = dist[c] + 1
                        q.append(nb)
        return dist

    def corridor(self, x, y, d):
        """How many cells you can see straight ahead before a wall (0 = wall right here)."""
        k = 0
        while self.open(x, y, d):
            x, y = x + DIRS[d][0], y + DIRS[d][1]
            k += 1
        return k


def tof_readings(maze, cell, heading, rng=None, noise=0.0):
    """Six VL53L0X distances in metres, from the cell centre: front-left, front-right, diag-left,
    diag-right, left, right. Front pair sees down the corridor; diagonals see the next cell's side walls."""
    x, y = cell
    f, l, r = heading, (heading - 1) % 4, (heading + 1) % 4
    half = CELL_M / 2
    front = half + maze.corridor(x, y, f) * CELL_M
    left = half + maze.corridor(x, y, l) * CELL_M
    right = half + maze.corridor(x, y, r) * CELL_M

    def diag(side):
        # 45° ray: hits the side wall of the next cell ahead if there is one, else passes through
        if not maze.open(x, y, f):
            return half * 1.414
        nx, ny = x + DIRS[f][0], y + DIRS[f][1]
        return CELL_M * 1.414 if not maze.open(nx, ny, side) else 2 * CELL_M * 1.414

    d = np.array([front, front, diag(l), diag(r), left, right])
    if rng is not None and noise:
        d = d * (1 + noise * rng.standard_normal(6))
    return np.clip(d, 0.03, TOF_MAX_M)


class MazeEnv(gym.Env):
    """One step = one cell decision. Observation (20 numbers, all from the robot's own sensors + odometry):
      0-5   six ToF distances / TOF_MAX
      6-9   visited count (odometry memory) of the cell ahead / left / right / behind, scaled
      10    visits of this cell, scaled
      11-12 heading from the gyro as cos, sin
      13-14 goal offset in the robot's frame (cells / 8, clipped); the goal is given as "x cells over, y up"
      15-18 last action one-hot
      19    bias 1.0
    """

    metadata = {"render_modes": []}

    def __init__(self, sizes=(5,), noise=0.03, seed=None, loops=0, max_steps_per_cell=4, visits=True):
        self.sizes, self.noise, self.loops, self.k = sizes, noise, loops, max_steps_per_cell
        self.visits_on = visits   # False: no odometry visited-cell memory in the observation (ablation)
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-3, 3, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(4)
        self.fixed = None

    def use(self, maze, start, goal, heading=0):
        """Pin the next reset to one maze (evaluation)."""
        self.fixed = (maze, start, goal, heading)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.fixed:
            self.maze, self.cell, self.goal, self.heading = self.fixed
        else:
            n = int(self.rng.choice(self.sizes))
            self.maze = Maze(n, int(self.rng.integers(1 << 30)), loops=self.loops)
            cells = [(x, y) for x in range(n) for y in range(n)]
            i, j = self.rng.choice(len(cells), 2, replace=False)
            self.cell, self.goal = cells[i], cells[j]
            self.heading = int(self.rng.integers(4))
        self.cell = tuple(self.cell)
        self.visits = np.zeros((self.maze.n, self.maze.n))
        self.visits[self.cell] = 1
        self.last = 0
        self.t = 0
        self.bumps = 0
        self.path = [self.cell]
        self.limit = self.k * self.maze.n ** 2
        return self._obs(), {}

    def _neighbour_visits(self, d):
        x, y = self.cell
        if not self.maze.open(x, y, d):
            return -1.0          # wall: the ToF already says so, keep it distinct
        return min(self.visits[x + DIRS[d][0], y + DIRS[d][1]], 3) / 3

    def _obs(self):
        h = self.heading
        tof = tof_readings(self.maze, self.cell, h, self.rng, self.noise) / TOF_MAX_M
        vis = [self._neighbour_visits((h + t) % 4) for t in (0, -1, 1, 2)]
        gx, gy = self.goal[0] - self.cell[0], self.goal[1] - self.cell[1]
        fwd, rgt = DIRS[h], DIRS[(h + 1) % 4]
        g_f, g_r = gx * fwd[0] + gy * fwd[1], gx * rgt[0] + gy * rgt[1]
        ang = -h * np.pi / 2
        last = np.zeros(4)
        last[self.last] = 1
        o = np.concatenate([tof, vis, [min(self.visits[self.cell], 3) / 3], [np.cos(ang), np.sin(ang)],
                            np.clip([g_f / 8, g_r / 8], -2, 2), last, [1.0]])
        if not self.visits_on:
            o[6:11] = [0.0 if v >= 0 else -1.0 for v in vis] + [0.0]   # keep "wall here", drop the counts
        return o.astype(np.float32)

    def step(self, a):
        a = int(a)
        self.t += 1
        self.heading = (self.heading + TURN[a]) % 4
        self.last = a
        x, y = self.cell
        r = -0.02
        if self.maze.open(x, y, self.heading):
            self.cell = (x + DIRS[self.heading][0], y + DIRS[self.heading][1])
            v = self.visits[self.cell]
            r += 0.1 if v == 0 else -0.05 * min(v, 4)
            self.visits[self.cell] += 1
            self.path.append(self.cell)
        else:
            r -= 1.0               # drove into a wall
            self.bumps += 1
        done = self.cell == tuple(self.goal)
        if done:
            r += 10.0
        trunc = self.t >= self.limit
        return self._obs(), r, done, trunc, {"bumps": self.bumps}
