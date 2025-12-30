"""
Eligibility Trace Module
"""

import numpy as np
from Params import params


def compute_timing_kernel(offset):
    """
    timing kernel
    """
    timing_kernel = np.exp(-(offset**2) / (2 * params["tau_btsp"]**2))
    
    return timing_kernel


def update_eligibility(elig_current, ca_spine, timing_kernel, is_active, dt):
    """
    Updates the eligibility trace 
    """    
    # Decay term 
    delig = -elig_current / params["tau_elig"] * dt
    
    # Active synapse operation
    delig[is_active] += (timing_kernel * ca_spine[is_active]) / params["tau_elig"] * dt
    
    return delig


def initialize_eligibility(n_syn):
    """
    Eligibility for each synapse initialised with 0
    """
    return np.zeros(n_syn)