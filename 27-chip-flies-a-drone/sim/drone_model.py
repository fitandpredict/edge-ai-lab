"""The drone: a Crazyflie Brushless-sized quadcopter, shared by the trainer (MJX on Colab) and the Mac loop.

Pure Python, no JAX: both sides import the constants and the MuJoCo XML from here, so they fly the same
drone. Physics constants are the identified Crazyflie Brushless model from Gräfe, Scherer, Hönig, Trimpe,
"How to Model Your Crazyflie Brushless" (arXiv 2603.05944, ICRA 2026), code
github.com/Data-Science-in-Mechanical-Engineering/CrazyflieBrushJAX (MIT): mass, inertia, arm length,
PWM -> thrust / torque polynomials, first-order motor lag, rotor inertia.

Motor numbering (body frame, x forward, y left), same as the paper's XML:
  0: (+x, -y)  1: (-x, -y)  2: (-x, +y)  3: (+x, +y); yaw torque signs -, +, -, +.

Observation (27 floats), identical on both sides:
  0:3   target - position, clipped to 1 m length   (world frame)
  3:6   linear velocity                            (world frame)
  6:12  body x and y axes in world (rotation matrix columns 0 and 1, row-major per column)
  12:15 angular velocity                           (body frame, MuJoCo free-joint qvel[3:6])
  15:27 the last 3 actions, newest first           (each 4 values in [-1, 1])
Action: 4 values in [-1, 1] -> PWM in [0, 1] per motor.
"""

import math

MASS = 0.0445                      # kg
INERTIA = (3.3e-5, 3.6e-5, 6.9e-5)   # kg m^2; the paper identifies Izz 7.19e-5, which breaks the
                                     # triangle rule MuJoCo enforces, so Izz = Ixx + Iyy (-4%)
ARM = 0.0353                       # m, motor offset along x and y (quadcopter_dimension / 2)
OMEGA_MAX = 2.9e3                  # rad/s at PWM 1 (motor_b)
MOTOR_A = 1 / 0.05                 # 1/s, first-order motor lag (time constant 50 ms)
J_M = 0.5e-7 * 1e3                 # rotor inertia term, as in the paper's code
YAW_SIGN = (-1.0, 1.0, -1.0, 1.0)
MOTOR_XY = ((ARM, -ARM), (-ARM, -ARM), (-ARM, ARM), (ARM, ARM))

SIM_DT = 0.001                     # physics step (s)
OBS_SIZE = 27
ACT_SIZE = 4
HIST = 3                           # past actions in the observation
G = 9.81


def pwm_to_thrust(p):
    """N per motor, p in [0, 1] (the paper's cubic fit)."""
    return -0.23009526 * p**3 + 0.56176458 * p**2 - 0.0433191 * p


def pwm_to_torque(p):
    """N m per motor (drag torque, the paper's cubic fit)."""
    return -0.0003396 * p**3 + 0.00087032 * p**2 + 0.0002896 * p


def hover_pwm():
    lo, hi = 0.0, 1.0
    want = MASS * G / 4
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pwm_to_thrust(mid) < want else (lo, mid)
    return (lo + hi) / 2


HOVER_PWM = hover_pwm()            # ~0.47
MAX_THRUST_TO_WEIGHT = 4 * pwm_to_thrust(1.0) / (MASS * G)


def xml(floor_contact=False, pretty=False):
    """MuJoCo model. Training: no floor contact (hitting the floor ends the episode anyway).
    pretty=True adds the bright scene, propellers and a shadow for renders on the Mac."""
    ix, iy, iz = INERTIA
    motors = "\n".join(
        f'      <site name="m{i}" pos="{x} {y} 0.008" size="0.004"/>' for i, (x, y) in enumerate(MOTOR_XY))
    thrust = "\n".join(f'    <motor site="m{i}" gear="0 0 1 0 0 0" ctrllimited="false"/>' for i in range(4))
    yaw = "\n".join(f'    <motor site="m{i}" gear="0 0 0 0 0 {s}" ctrllimited="false"/>'
                    for i, s in enumerate(YAW_SIGN))
    con = 'contype="1" conaffinity="1"' if floor_contact else 'contype="0" conaffinity="0"'
    if pretty:
        props = "\n".join(
            f'      <geom name="prop{i}" type="cylinder" pos="{x} {y} 0.012" size="0.024 0.0012" '
            f'rgba="{"0.98 0.55 0.15 0.55" if i in (0, 3) else "0.2 0.6 1.0 0.55"}" {con} mass="0"/>\n'
            f'      <geom type="cylinder" pos="{x} {y} 0.004" size="0.0045 0.006" rgba="0.15 0.15 0.18 1" {con} mass="0"/>\n'
            f'      <geom type="capsule" fromto="0 0 0 {x} {y} 0" size="0.0025" rgba="0.92 0.92 0.95 1" {con} mass="0"/>'
            for i, (x, y) in enumerate(MOTOR_XY))
        scene = """
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.3 0.3 0.3" specular="0 0 0"/><quality shadowsize="4096"/>
    <global offwidth="1920" offheight="1080"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.62 0.82 1.0" rgb2="0.95 0.98 1.0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.95 0.97" rgb2="0.84 0.88 0.92" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="40 40" specular="0" shininess="0" reflectance="0"/>
  </asset>"""
        floor = f'    <geom name="floor" type="plane" size="20 20 0.1" material="floor" {con}/>\n' \
                f'    <light pos="2 -2 6" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1" castshadow="true"/>'
        body_rgba = "0.13 0.55 0.95 1"
        target = '    <body name="target" mocap="true" pos="0 0 1"><geom type="sphere" size="0.025" ' \
                 'rgba="1.0 0.78 0.1 0.8" contype="0" conaffinity="0"/></body>'
    else:
        props = scene = ""
        floor = f'    <geom name="floor" type="plane" size="20 20 0.1" {con}/>'
        body_rgba = "0.3 0.3 0.8 1"
        target = ""
    return f"""<mujoco model="crazyflie_brushless_ep27">
  <option timestep="{SIM_DT}" gravity="0 0 -{G}" integrator="RK4"/>
  <compiler autolimits="true"/>{scene}
  <worldbody>
{floor}
{target}
    <body name="drone" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{MASS}" diaginertia="{ix} {iy} {iz}"/>
      <geom name="body" type="box" size="0.017 0.017 0.006" rgba="{body_rgba}" {con} mass="0"/>
{props}
{motors}
    </body>
  </worldbody>
  <actuator>
{thrust}
{yaw}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    print(f"hover PWM {HOVER_PWM:.4f}, max thrust/weight {MAX_THRUST_TO_WEIGHT:.2f}")
    print(f"motor time constant {1 / MOTOR_A * 1000:.0f} ms")
