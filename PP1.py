"""
PP1 (Phosphatase) Module

Very Early Development model
"""

import numpy as np
from Params import params


def update_pp1(pp1_current, ca_spine, is_active, dt, noise):
    """
    Updates Phophotase dynamics
    """    
    # Baseline 
    dpp1 = (params["pp1_baseline"] - pp1_current) / params["tau_pp1"] * dt
    
    # Kinase inhibition
    dpp1[is_active] -= params["alpha_pp1"] * ca_spine[is_active] * dt
    
    # Noise
    dpp1 += noise * np.sqrt(dt)
    
    return dpp1


def initialize_pp1(n_syn):
    """
    Currently the module initialises pp1 levels to the parameter of baseline PP1
    """
    return np.ones(n_syn) * params["pp1_baseline"]