"""
Eligibility Trace Module — Single Synapse

"""

import numpy as np
from Params import params


def compute_timing_kernel(offset):
    
    return float(np.exp(-(offset ** 2) / (2.0 * params["tau_btsp"] ** 2)))


def update_eligibility(elig, ca_spine, timing_kernel, dt):
    
    delig  = -elig / params["tau_elig"] * dt
    delig += (timing_kernel * ca_spine) / params["tau_elig"] * dt
    return float(delig)


def initialize_eligibility():

    return 0.0
