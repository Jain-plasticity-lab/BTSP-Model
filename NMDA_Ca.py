"""
NMDA Pathway Module
Handles NMDA receptor dynamics and voltage-dependent gating (Primitive Implementation)
"""

import numpy as np
from Params import params


def compute_voltage_attenuation(syn_distances, v_soma):
    """
    Compute distance-based voltage attenuation at each synapse
    
    Inputs:
        syn_distances: Array of synapse distances from soma (µm)
        v_soma: Voltage at the soma (mV)
        
    Returns:
        v_spine: Array of spine voltages at each synapse
    """
    attenuation = np.exp(-syn_distances / params["lambda_d"]) * 4
    v_spine = params["v_rest"] + (v_soma - params["v_rest"]) * attenuation
    
    return v_spine


def compute_nmda_voltage_dependence(v_spine):
    """
    Computes NMDA receptor voltage dependence (Mg2+ block removal)
    
    Inputs:
        v_spine: Spine voltage (mV) - array defined in previous function
        
    Returns:
        nmda_v: NMDA voltage-dependent factor (0-1)
    """
    nmda_v = 1.0 / (1.0 + params["mg_block"] * np.exp(-0.062 * v_spine)) #Measure of the state of operations of NMDA block
    
    return nmda_v


def get_somatic_voltage(post_spike):
    """
    Calculate somatic voltage based on the voltage spikes at the postsynaptic dendrite
    
    Args:
        post_spike: Postsynaptic spike indicator (0 or 1)
        
    Returns:
        v_soma: Somatic voltage (mV)
    """
    v_soma = params["v_rest"] + post_spike * (params["v_peak"] - params["v_rest"])
    
    return v_soma