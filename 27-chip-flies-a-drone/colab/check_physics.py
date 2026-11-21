"""JAX rigid body (drone_env.body_step) vs MuJoCo on drone_model.xml: same thrusts, compare trajectories."""
import jax, jax.numpy as jnp, mujoco, numpy as np
import drone_model as dm, drone_env as de

m = mujoco.MjModel.from_xml_string(dm.xml(floor_contact=False)); d = mujoco.MjData(m)
d.qpos[:] = [0, 0, 1, 1, 0, 0, 0]; mujoco.mj_forward(m, d)
print("mujoco Izz", m.body_inertia[1])
rng = np.random.default_rng(0)
b = de.Body(jnp.array(d.qpos), jnp.array(d.qvel))
step = jax.jit(lambda b, t, y, g: de.body_step(b, t, y, g, dm.SIM_DT))
worst = 0
for k in range(600):                       # 0.6 s, new random-ish thrusts every 10 ms
    if k % 10 == 0:
        p = np.clip(dm.HOVER_PWM + rng.normal(0, 0.05, 4), 0, 1)
        thrust = dm.pwm_to_thrust(p); yaw = dm.pwm_to_torque(p)
        gust = rng.normal(0, 0.05, 3)
    d.ctrl[:] = np.concatenate([thrust, yaw]); d.xfrc_applied[1, :3] = gust
    mujoco.mj_step(m, d)
    b = step(b, jnp.array(thrust), jnp.array(yaw), jnp.array(gust))
    worst = max(worst, np.abs(np.array(b.qpos[:3]) - d.qpos[:3]).max())
print("after 0.6 s  mujoco pos", d.qpos[:3].round(4), "quat", d.qpos[3:7].round(4))
print("             jax    pos", np.array(b.qpos[:3]).round(4), "quat", np.array(b.qpos[3:7]).round(4))
print("max position gap (m)", round(float(worst), 5), "omega", d.qvel[3:].round(3), np.array(b.qvel[3:]).round(3))
