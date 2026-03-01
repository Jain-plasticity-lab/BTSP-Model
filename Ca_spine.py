"""
Spine Calcium Module 


"""

import numpy as np
from Params import params


def update_ca_spine(ca_spine, pre_spike, nmda_v, dt, noise):

    dca  = -ca_spine / params["tau_ca_spine"] * dt

    dca += params["alpha_nmda"] * pre_spike * nmda_v * dt

    dca += noise * np.sqrt(dt)

    return float(dca)


def initialize_ca_spine():
    return 0.0
