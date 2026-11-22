"""flybody's trained flight policy, ported from its TF SavedModel to numpy.

Architecture (read from the saved graph): concat(obs, sorted keys) -> Linear -> LayerNorm -> tanh
-> Linear -> elu -> Linear -> elu -> mean head (Linear). The std head is ignored: we act on the mean,
as flybody's TestPolicyWrapper does by default.
Weights: data/flight_policy.npz, exported from data/trained-fly-policies/flight (variable index order).
"""
import numpy as np

KEYS = ['walker/accelerometer', 'walker/actuator_activation', 'walker/gyro', 'walker/joints_pos',
        'walker/joints_vel', 'walker/ref_displacement', 'walker/ref_root_quat', 'walker/velocimeter',
        'walker/world_zaxis']


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class FlightPolicy:
    def __init__(self, path='data/flight_policy.npz'):
        w = np.load(path)
        self.b0, self.w0 = w['0'], w['1']
        self.ln_beta, self.ln_gamma = w['2'], w['3']
        self.b1, self.w1 = w['4'], w['5']
        self.b2, self.w2 = w['6'], w['7']
        self.bm, self.wm = w['8'], w['9']

    @staticmethod
    def flatten(obs):
        return np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in KEYS])

    def __call__(self, obs):
        x = self.flatten(obs) if isinstance(obs, dict) else obs
        h = x @ self.w0 + self.b0
        mu, var = h.mean(), h.var()
        h = np.tanh((h - mu) / np.sqrt(var + 1e-5) * self.ln_gamma + self.ln_beta)
        h = _elu(h @ self.w1 + self.b1)
        h = _elu(h @ self.w2 + self.b2)
        return h @ self.wm + self.bm
