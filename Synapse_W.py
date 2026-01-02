"""
Synaptic Weight Module
Handles synaptic weight updates based on plasticity rules
"""

import numpy as np
from Params import params


def update_weights(
    w_current,
    camkii,
    elig,
    pp1,
    is_active,
    camkii_fired,
    camkii_fire_time,
    timing_kernel,
    current_time,
    dt,
    noise
):
    """
    Update synaptic weights based on BTSP rule
    
    Args:
        w_current: Current weights (array for all synapses)
        camkii: CaMKII activity level
        elig: Eligibility trace (array)
        pp1: PP1 levels (array)
        is_active: Boolean array indicating active synapses
        camkii_fired: Boolean indicating if CaMKII has fired
        camkii_fire_time: Time when CaMKII fired (or None)
        timing_kernel: BTSP timing kernel
        current_time: Current simulation time
        dt: Time step
        noise: Noise array
        
    Returns:
        dw: Change in weights (array)
    """
    n_syn = len(w_current)
    dw = np.zeros(n_syn)
    
    # Check conditions for plasticity
    plasticity_mask = (
        is_active & 
        (camkii > 0.1) &  # Only if CaMKII has successfully fired
        (elig > 0) & 
        (pp1 < 0.75)
    )
    
    # Update weights for synapses meeting plasticity conditions
    if np.any(plasticity_mask) and camkii_fired:
        # Timing-dependent gain (decays after CaMKII fires)
        timing_gain = np.exp(-(current_time - camkii_fire_time) / 30.0)
        
        current_w = w_current[plasticity_mask]
        
        dw[plasticity_mask] = (
            params["eta"]
            * timing_kernel
            * timing_gain
            * camkii
            * (params["w_max"] - current_w)
            * elig[plasticity_mask]
            * dt
        )
        
        dw += noise * np.sqrt(dt) * plasticity_mask
    
    return dw


def initialize_weights(n_syn):
    """
    Initialize synaptic weight array
    
    Args:
        n_syn: Number of synapses
        
    Returns:
        w: Initialized weight array
    """
    return np.ones(n_syn) * params["w_init"]


def clip_weights(w):
    """
    Clip weights to valid range
    
    Args:
        w: Weight array
        
    Returns:
        w_clipped: Clipped weight array
    """
    return np.clip(w, params["w_min"], params["w_max"])