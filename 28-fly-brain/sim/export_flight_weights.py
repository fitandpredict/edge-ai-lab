"""Export flybody's trained flight policy (a TF SavedModel) to data/flight_policy.npz for sim/flight_policy.py.

Needs TensorFlow only to read the checkpoint (any recent version; a separate venv is fine):
    python -m venv .venv-tf && .venv-tf/bin/pip install tensorflow numpy
    .venv-tf/bin/python sim/export_flight_weights.py
The 12 variables are saved in the checkpoint's index order, which is what FlightPolicy expects.
"""
import numpy as np
import tensorflow as tf

r = tf.train.load_checkpoint('data/trained-fly-policies/flight/variables/variables')
W = {str(i): r.get_tensor(f'_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE') for i in range(12)}
np.savez('data/flight_policy.npz', **W)
print({k: v.shape for k, v in W.items()})
