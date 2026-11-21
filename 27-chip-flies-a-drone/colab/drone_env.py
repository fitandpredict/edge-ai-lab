"""Fly-to-a-point quadcopter task in MJX, on MuJoCo Playground's MjxEnv base (for Brax PPO on Colab).

The drone and the observation layout come from sim/drone_model.py (copied next to this file on the VM),
so the Mac loop flies exactly what was trained. Per control step (ctrl_dt, default 10 ms = 100 Hz):
  the policy's 4 actions -> PWM, held through the substeps (optionally arriving `delay` ms late),
  each 1 ms substep: first-order motor lag, thrust + yaw torque from the paper's fits, plus a gust force.

Domain randomisation lives in the per-episode info (no model swapping): thrust per motor, yaw torque,
motor lag, battery (max PWM), the gust, sensor noise and a small control delay. Rewards: be near the
target, stay level, don't spin, don't twitch; hitting the floor or flying off ends the episode.
"""

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

import drone_model as dm


def default_config():
    return config_dict.create(
        ctrl_dt=0.01, sim_dt=dm.SIM_DT, episode_s=10.0,
        target_box=((-1.5, -1.5, 0.5), (1.5, 1.5, 2.0)),
        retarget_every_s=3.0,         # mean time between new targets (keyboard steering later)
        min_delay_ms=0.0,             # control delay sampled per episode in [min, max] (serial round trip);
        max_delay_ms=4.0,             # may span several control steps (the "practise with lag" brains)
        hist=3,                       # past actions in the observation
        thrust_dr=0.15, motor_dr=0.1, torque_dr=0.2, lag_dr=0.2, battery_min=0.75,
        gust_n=0.2, gust_every_s=2.0, gust_tau_s=0.15,
        obs_noise=0.01,
        tilt0=0.6, vel0=1.0, omega0=3.0,  # how messy the start is
        crash_penalty=50.0,
    )


def quat_to_rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def thrust_fit(p):
    return -0.23009526 * p**3 + 0.56176458 * p**2 - 0.0433191 * p


def torque_fit(p):
    return -0.0003396 * p**3 + 0.00087032 * p**2 + 0.0002896 * p


def make_obs(data, target, hist):
    pos, quat, vel, omega = data.qpos[0:3], data.qpos[3:7], data.qvel[0:3], data.qvel[3:6]
    err = target - pos
    err = err * jnp.minimum(1.0, 1.0 / (jnp.linalg.norm(err) + 1e-6))
    R = quat_to_rot(quat)
    axes = jnp.concatenate([R[:, 0], R[:, 1]])
    return jnp.concatenate([err, vel, axes, omega, hist.reshape(-1)])


@struct.dataclass
class Body:
    """The drone's state, laid out like MuJoCo's free joint: qpos = pos (world) + quat (w,x,y,z),
    qvel = linear velocity (world) + angular velocity (body frame)."""
    qpos: jax.Array
    qvel: jax.Array


# rigid-body constants, the same numbers MuJoCo gets from drone_model.xml()
_I = jnp.array(dm.INERTIA)
_MXY = jnp.array(dm.MOTOR_XY)
_YAW = jnp.array(dm.YAW_SIGN)


def body_step(b, thrust, yaw, gust, dt):
    """One semi-implicit Euler step of a rigid body: 4 thrusts along body z at the motor arms,
    4 yaw torques, gravity, a world-frame gust force at the centre of mass."""
    q = b.qpos[3:7]
    R = quat_to_rot(q)
    v, w = b.qvel[0:3], b.qvel[3:6]
    f_world = R[:, 2] * jnp.sum(thrust) + jnp.array([0.0, 0.0, -dm.G * dm.MASS]) + gust
    tau = jnp.array([jnp.sum(_MXY[:, 1] * thrust), -jnp.sum(_MXY[:, 0] * thrust), jnp.sum(_YAW * yaw)])
    v = v + dt * f_world / dm.MASS
    w = w + dt * (tau - jnp.cross(w, _I * w)) / _I
    pos = b.qpos[0:3] + dt * v
    wq = jnp.concatenate([jnp.zeros(1), w])
    q = q + 0.5 * dt * _quat_mul(q, wq)
    q = q / jnp.linalg.norm(q)
    return Body(jnp.concatenate([pos, q]), jnp.concatenate([v, w]))


class DroneHover(mjx_env.MjxEnv):

    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        self._mj_model = mujoco.MjModel.from_xml_string(dm.xml(floor_contact=False))
        self._mjx_model = None   # physics is body_step() in plain JAX; the MuJoCo model is for reference
        c = self._config
        self._lo = jnp.array(c.target_box[0])
        self._hi = jnp.array(c.target_box[1])
        self._dmin = int(round(c.min_delay_ms / 1000 / c.sim_dt))
        self._dmax = int(round(c.max_delay_ms / 1000 / c.sim_dt))
        self._nbuf = -(-self._dmax // self.n_substeps) + 2     # commands kept: enough to reach back max delay

    # ---- the base-class plumbing
    @property
    def xml_path(self):
        return "drone_model.xml(floor_contact=False)"

    @property
    def action_size(self):
        return dm.ACT_SIZE

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    # ---- episodes
    def _random_target(self, rng):
        return jax.random.uniform(rng, (3,), minval=self._lo, maxval=self._hi)

    def reset(self, rng):
        c = self._config
        ks = jax.random.split(rng, 15)
        pos = self._random_target(ks[0])
        target = self._random_target(ks[1])
        axis = jax.random.normal(ks[2], (3,))
        axis = axis / (jnp.linalg.norm(axis) + 1e-6)
        ang = jax.random.uniform(ks[3], (), minval=0.0, maxval=c.tilt0)
        yaw = jax.random.uniform(ks[4], (), minval=-jnp.pi, maxval=jnp.pi)
        q_tilt = jnp.concatenate([jnp.cos(ang / 2)[None], axis * jnp.sin(ang / 2)])
        q_yaw = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        quat = _quat_mul(q_tilt, q_yaw)
        vel = jax.random.uniform(ks[5], (3,), minval=-c.vel0, maxval=c.vel0)
        omega = jax.random.uniform(ks[6], (3,), minval=-c.omega0, maxval=c.omega0)
        qpos = jnp.concatenate([pos, quat])
        qvel = jnp.concatenate([vel, omega])
        data = Body(qpos, qvel)

        u = lambda k, n, a: jax.random.uniform(k, (n,), minval=1 - a, maxval=1 + a)
        thrust_k = u(ks[7], 1, c.thrust_dr) * u(ks[8], 4, c.motor_dr * 0.5)
        info = {
            "rng": ks[13],
            "target": target,
            "hist": jnp.zeros((c.hist, 4)),
            "omega_m": jnp.full((4,), dm.HOVER_PWM * dm.OMEGA_MAX) * jax.random.uniform(ks[9], (4,), minval=0.8, maxval=1.1),
            "gust": jnp.zeros(3),
            "thrust_k": thrust_k,
            "torque_k": u(ks[10], 1, c.torque_dr),
            "lag_a": dm.MOTOR_A * u(ks[11], 1, c.lag_dr),
            "battery": jax.random.uniform(ks[12], (), minval=c.battery_min, maxval=1.0),
            "delay_sub": jax.random.randint(ks[14], (), self._dmin, self._dmax + 1),
            "pwm_buf": jnp.full((self._nbuf, 4), dm.HOVER_PWM),   # [0] = this step's command, [j] = j steps ago
        }
        obs = make_obs(data, target, info["hist"])
        metrics = {"dist": jnp.zeros(()), "tilt": jnp.zeros(()), "crash": jnp.zeros(())}
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        c = self._config
        info = dict(state.info)
        rng, k_tgt, k_new, k_gust, k_gmag, k_gdir, k_noise, k_mot = jax.random.split(info["rng"], 8)
        action = jnp.clip(action, -1.0, 1.0)
        pwm_new = jnp.minimum((action + 1.0) / 2.0, info["battery"])
        buf = jnp.concatenate([pwm_new[None], info["pwm_buf"][:-1]], axis=0)
        n = self.n_substeps

        # gust: a random kick now and then, decaying
        dt = self.dt
        kick = jax.random.uniform(k_gust) < dt / c.gust_every_s
        gdir = jax.random.normal(k_gdir, (3,)) * jnp.array([1.0, 1.0, 0.4])
        gdir = gdir / (jnp.linalg.norm(gdir) + 1e-6)
        gust = info["gust"] * jnp.exp(-dt / c.gust_tau_s) + kick * gdir * jax.random.uniform(k_gmag, (), maxval=c.gust_n)

        a_lag = info["lag_a"][0]
        ad = jnp.exp(-a_lag * c.sim_dt)
        bd = dm.OMEGA_MAX * (1 - ad)
        sub_noise = jax.random.normal(k_mot, (self.n_substeps, 4)) * 1.0

        def sub(carry, xs):
            data, om = carry
            i, nz = xs
            # the command in force at substep i was issued j steps ago: the newest with j*n + i >= delay
            j = jnp.clip(-((i - info["delay_sub"]) // n), 0, self._nbuf - 1)
            pwm = buf[j]
            om_next = ad * om + bd * pwm + nz
            p = jnp.clip(om_next / dm.OMEGA_MAX, 0.0, 1.2)
            thrust = thrust_fit(p) * info["thrust_k"]
            yaw = (dm.J_M * (-a_lag * om_next + a_lag * dm.OMEGA_MAX * pwm) * 1e-3 + torque_fit(p)) * info["torque_k"][0]
            data = body_step(data, thrust, yaw, gust, c.sim_dt)
            return (data, om_next), None

        (data, om), _ = jax.lax.scan(sub, (state.data, info["omega_m"]),
                                     (jnp.arange(self.n_substeps), sub_noise))

        # new target now and then
        retarget = jax.random.uniform(k_tgt) < dt / c.retarget_every_s
        target = jnp.where(retarget, self._random_target(k_new), info["target"])

        hist = jnp.concatenate([action[None], info["hist"][:-1]], axis=0)
        pos, quat, vel, omega = data.qpos[0:3], data.qpos[3:7], data.qvel[0:3], data.qvel[3:6]
        R = quat_to_rot(quat)
        d = jnp.linalg.norm(info["target"] - pos)   # reward against the target it was flying to
        up = R[2, 2]
        r = (jnp.exp(-(d / 1.0) ** 2) + 2.0 * jnp.exp(-(d / 0.2) ** 2) + 2.0 * jnp.exp(-(d / 0.04) ** 2)
             + 0.3 * up
             - 0.02 * jnp.linalg.norm(omega)
             - 0.02 * jnp.linalg.norm(vel)
             - 0.2 * jnp.sum((action - info["hist"][0]) ** 2))
        crash = ((pos[2] < 0.03) | (pos[2] > 5.0) | (jnp.abs(pos[0]) > 5.0) | (jnp.abs(pos[1]) > 5.0)
                 | (jnp.linalg.norm(vel) > 10.0) | (jnp.linalg.norm(omega) > 35.0)
                 | jnp.any(jnp.isnan(data.qpos)))
        r = jnp.where(crash, r - c.crash_penalty, r)

        info.update(rng=rng, target=target, hist=hist, omega_m=om, gust=gust, pwm_buf=buf)
        obs = make_obs(data, target, hist)
        obs = obs.at[0:15].add(jax.random.normal(k_noise, (15,)) * c.obs_noise)
        metrics = {**state.metrics, "dist": d, "tilt": jnp.arccos(jnp.clip(up, -1, 1)), "crash": crash.astype(jnp.float32)}
        return state.replace(data=data, obs=obs, reward=r, done=crash.astype(jnp.float32),
                             metrics=metrics, info=info)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
