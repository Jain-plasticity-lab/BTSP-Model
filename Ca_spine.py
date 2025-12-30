"""
Spine Calcium Module
"""

import numpy as np
from Params import params


def update_ca_spine(ca_spine_current, pre_spike, nmda_v, is_active, dt, noise):
    """
    Updates the spinal calcium concentration
    """    
    # Decay term
    dca = -ca_spine_current / params["tau_ca_spine"] * dt
    
    # Influx through NMDA activation
    dca[is_active] += params["alpha_nmda"] * pre_spike * nmda_v[is_active] * dt
    
    # noise
    dca += noise * np.sqrt(dt)
    
    return dca


def initialize_ca_spine(n_syn):
    """
    This function initialises  spine calcium to zero
    """
    return np.zeros(n_syn)