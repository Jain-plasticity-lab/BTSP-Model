"""
IP3 Pathway Module
"""

import numpy as np
from Params import params


def ip3r_open_probability(ca_dend, ip3, ip3_peak):
    """
    IP3R opening affinity
    """
    # IP3 threshold 
    if ip3_peak > 0:
        ip3_gate = 1.0 if ip3 >= params["ip3r_threshold"] * ip3_peak else 0.0
    else:
        ip3_gate = 0.0
    
    # Activation
    ca_act = ca_dend**params["ip3r_n_act"] / (
        ca_dend**params["ip3r_n_act"] + params["ip3r_ka"]**params["ip3r_n_act"]
    )
    
    # Inactivation
    ca_inact = params["ip3r_ki"]**params["ip3r_n_inact"] / (
        ca_dend**params["ip3r_n_inact"] + params["ip3r_ki"]**params["ip3r_n_inact"]
    )
    
    # Combined affinity
    p_open = (ip3_gate * ca_act * ca_inact) * 10
    
    return p_open


def update_ip3(ip3_current, post_spike, dt):
    """
    Updates IP3 concentration
    """
    dip3 = (
        -ip3_current / params["tau_ip3"] + 
        params["alpha_ip3_post"] * post_spike
    ) * dt
    
    return dip3


def update_ca_store(ca_store_current, ip3_current, ca_dend, ip3_peak, dt, noise):
    """
    Intracellular calcium store mechanics
    """
    
    p_open = ip3r_open_probability(ca_dend, ip3_current, ip3_peak)
    
    # Ca release from internal stores 
    release = params["alpha_release"] * ip3_current * p_open
    
    # SERCA mechanism (Primitive)
    refill = (params["ca_store_max"] - ca_store_current) / params["tau_store_refill"]
    
    dca_store = (refill - release) * dt + noise * np.sqrt(dt)
    
    return dca_store, release