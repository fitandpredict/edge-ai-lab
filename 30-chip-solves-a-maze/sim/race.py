"""Every brain on the same unseen mazes. Deterministic: the brain's best action every time, no dice.

  python sim/race.py --sizes 5 7 9 --mazes 200 --out results/race.json
  python sim/race.py --loops 8 ...        # mazes with loops (wall-following can circle forever)

Racers:
  lstm, mlp, lstm_novis, mlp_novis   trained brains (models/<name>_final.zip)
  left_hand                          keep your left hand on the wall (no learning)
  flood_fill                         micromouse classic: knows where the goal is, maps walls as it goes,
                                     always heads downhill on a distance map (no learning)
  random                             a coin flip each cell
Metrics per size: solve rate, cells driven / shortest path (1.0 = perfect), wall bumps.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from maze import DIRS, Maze, MazeEnv  # noqa: E402


def episode(env, policy):
    obs, _ = env.reset()
    policy.reset(env)
    done = trunc = False
    while not (done or trunc):
        obs, _, done, trunc, info = env.step(policy.act(obs, env))
    return done, env.t, env.bumps, list(env.path)


class Learned:
    def __init__(self, path):
        self.recurrent = "lstm" in Path(path).name
        if self.recurrent:
            from sb3_contrib import RecurrentPPO
            self.m = RecurrentPPO.load(path, device="cpu")
        else:
            from stable_baselines3 import PPO
            self.m = PPO.load(path, device="cpu")

    def reset(self, env):
        self.state, self.start = None, np.ones(1, bool)

    def act(self, obs, env):
        if self.recurrent:
            a, self.state = self.m.predict(obs, state=self.state, episode_start=self.start, deterministic=True)
            self.start = np.zeros(1, bool)
        else:
            a, _ = self.m.predict(obs, deterministic=True)
        return int(a)


class LeftHand:
    """Turn left if you can, else straight, else right, else back (only reads its own wall sensors)."""

    def reset(self, env):
        pass

    def act(self, obs, env):
        x, y = env.cell
        for a, t in ((1, -1), (0, 0), (2, 1), (3, 2)):
            if env.maze.open(x, y, (env.heading + t) % 4):
                return a
        return 3


class FloodFill:
    """Assume unknown walls are open, flood distances from the goal, step to the lowest neighbour,
    learn walls as you see them (the six ToF sensors see the walls of the current cell)."""

    def reset(self, env):
        n = env.maze.n
        self.known = np.zeros((n, n, 4), bool)   # known closed walls
        for x in range(n):
            self.known[x, 0, 2] = self.known[x, n - 1, 0] = True
        for y in range(n):
            self.known[0, y, 3] = self.known[n - 1, y, 1] = True

    def act(self, obs, env):
        m, (x, y) = env.maze, env.cell
        for d in range(4):
            if not m.open(x, y, d):
                self.known[x, y, d] = True
                nx, ny = x + DIRS[d][0], y + DIRS[d][1]
                if m.inside(nx, ny):
                    self.known[nx, ny, (d + 2) % 4] = True
        n = m.n
        dist = np.full((n, n), 10**6)
        dist[env.goal] = 0
        q = [tuple(env.goal)]
        for c in q:
            for d in range(4):
                if not self.known[c[0], c[1], d]:
                    nb = (c[0] + DIRS[d][0], c[1] + DIRS[d][1])
                    if dist[nb] > dist[c] + 1:
                        dist[nb] = dist[c] + 1
                        q.append(nb)
        best = min((d for d in range(4) if not self.known[x, y, d]), key=lambda d: dist[x + DIRS[d][0], y + DIRS[d][1]])
        return [0, 2, 3, 1][(best - env.heading) % 4]   # absolute direction -> forward/right/back/left


class Random:
    def reset(self, env):
        self.rng = np.random.default_rng(0)

    def act(self, obs, env):
        return int(self.rng.integers(4))


def mazes_for(n, count, loops):
    rng = np.random.default_rng(1000 + n)   # unseen: training draws from a different stream
    out = []
    for k in range(count):
        m = Maze(n, 50_000 + 97 * n + k, loops=loops)
        cells = [(x, y) for x in range(n) for y in range(n)]
        i, j = rng.choice(len(cells), 2, replace=False)
        out.append((m, cells[i], cells[j], int(rng.integers(4))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--mazes", type=int, default=200)
    ap.add_argument("--loops", type=int, default=0)
    ap.add_argument("--racers", nargs="+",
                    default=["lstm", "mlp", "lstm_novis", "mlp_novis", "left_hand", "flood_fill", "random"])
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    for name in a.racers:
        path = ROOT / "models" / f"{name}_final.zip"
        if name == "left_hand":
            pol = LeftHand()
        elif name == "flood_fill":
            pol = FloodFill()
        elif name == "random":
            pol = Random()
        elif path.exists():
            pol = Learned(str(path))
        else:
            print(f"skip {name}: no {path.name}")
            continue
        res[name] = {}
        for n in a.sizes:
            solved, ratio, bumps, cells = 0, [], 0, []
            for m, s, g, h in mazes_for(n, a.mazes, a.loops):
                env = MazeEnv(sizes=(n,), noise=0.0, loops=a.loops, visits=not name.endswith("novis"))
                env.use(m, s, g, h)
                ok, t, b, path_ = episode(env, pol)
                solved += ok
                bumps += b
                if ok:
                    ratio.append((len(path_) - 1) / max(m.bfs(s)[g], 1))
            r = dict(solve_rate=round(solved / a.mazes, 3),
                     path_vs_shortest=round(float(np.median(ratio)), 2) if ratio else None,
                     bumps_per_maze=round(bumps / a.mazes, 3))
            res[name][f"{n}x{n}"] = r
            print(f"{name:11s} {n}x{n}  solved {r['solve_rate']:.0%}  path/shortest {r['path_vs_shortest']}  bumps {r['bumps_per_maze']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(mazes=a.mazes, loops=a.loops, results=res), indent=1))


if __name__ == "__main__":
    main()
