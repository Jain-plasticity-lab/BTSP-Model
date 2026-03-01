"""
NMDA Pathway Module 

"""

import numpy as np
from Params import params


def compute_voltage_attenuation(distance, v_soma):
    
    attenuation = np.exp(-distance / params["lambda_d"]) 

    v_spine     = params["v_rest"] + (v_soma - params["v_rest"]) * attenuation

    return float(v_spine)


def compute_nmda_voltage_dependence(v_spine):
    
    return float(1.0 / (1.0 + params["mg_block"] * np.exp(-0.062 * v_spine)))


def get_somatic_voltage(post_spike):
    
    return float(params["v_rest"] + post_spike * (params["v_peak"] - params["v_rest"]))
