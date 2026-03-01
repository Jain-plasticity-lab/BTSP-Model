"""
PP1 (Protein Phosphatase 1) Module
"""

import numpy as np
from numba import jit
from Params import params



@jit(nopython=True, cache=True)
def _ca_dependent_pp1_inhibition(ca_spine, k_half, n_hill):
    
    ca_pow = ca_spine ** n_hill
    k_pow  = k_half  ** n_hill
    return ca_pow / (ca_pow + k_pow) #fraction of PP1 inhibited


@jit(nopython=True, cache=True)
def _pp1_substrate_dephosphorylation(pp1_active, substrate_phospho, k_dephos):

    return k_dephos * pp1_active * substrate_phospho


@jit(nopython=True, cache=True)
def _kinase_phosphatase_competition(camkii, pp1_active, substrate, k_phos, k_dephos):
    
    phosphorylation   = k_phos   * camkii    * (1.0 - substrate)
    dephosphorylation = k_dephos * pp1_active *         substrate
    return phosphorylation - dephosphorylation


def update_pp1(pp1, ca_spine, dt, noise):
    
    pp1_baseline = params.get("pp1_baseline", 1.0)
    tau_pp1      = params.get("tau_pp1",      30.0)
    alpha_pp1    = params.get("alpha_pp1",     1.0)

    recovery      = (pp1_baseline - pp1) / tau_pp1
    ca_suppression = -alpha_pp1 * ca_spine

    dpp1  = (recovery + ca_suppression) * dt
    dpp1 += noise * np.sqrt(dt)
    return float(dpp1)




def compute_pp1_tag_strength(pp1_level, threshold=None):
    
    baseline  = params.get("pp1_baseline", 1.0)
    threshold = threshold if threshold is not None else 0.5 * baseline
    return float(max(0.0, (threshold - pp1_level) / threshold))


def compute_net_phosphorylation(camkii, pp1_active, current_phos, dt):
    
    k_phos   = params.get("k_phosphorylation",   1.0)
    k_dephos = params.get("k_dephosphorylation",  0.8)
    d_phos   = _kinase_phosphatase_competition(
        camkii, pp1_active, current_phos, k_phos, k_dephos) * dt
    return float(np.clip(current_phos + d_phos, 0.0, 1.0))


def initialize_pp1():
    return float(params.get("pp1_baseline", 1.0))


#For quick testing
def initialize_pp1_params():
    defaults = {
        "pp1_baseline":        1.0,
        "tau_pp1":            10.0,
        "alpha_pp1":           0.5,
        "pp1_k_half_ca":       0.5,
        "pp1_n_hill":          2.0,
        "k_phosphorylation":   1.0,
        "k_dephosphorylation": 0.8,
        "pp1_tag_threshold":   0.5,
        "sigma_pp1":           0.01,
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return params
