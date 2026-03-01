"""
Dendritic Calcium Module
Handles calcium dynamics in the dendrite

"""

import numpy as np
from Params import params


def update_ca_dend(ca_dend_current, ca_release, dt, noise):
    dca_dend = (
        (-(ca_dend_current) / (params["tau_ca_dend"])) + ca_release
    ) * dt
    
    dca_dend += noise * np.sqrt(dt)
    
    return dca_dend


def initialize_ca_dend():
    return 0.0