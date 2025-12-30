"""
Dendritic Calcium Module
"""

import numpy as np
from Params import params


def update_ca_dend(ca_dend_current, ca_release, dt, noise):
    """
    Works with dendritic calcium concentration
    """
    dca_dend = (
        -ca_dend_current / params["tau_ca_dend"] + ca_release
    ) * dt
    
    dca_dend += noise * np.sqrt(dt)
    
    return dca_dend


def initialize_ca_dend():
    """
    This function initialises Dendritic Calcium to 0 currently
    """
    return 0.0