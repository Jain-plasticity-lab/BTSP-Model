"""
L-Type Calcium Channel (VGCC) Module — Mahajan & Nadkarni 2019 Implementation

References

Mahajan & Nadkarni (2019) 
"""

import numpy as np
from Params import params

# Physical constants
_Z_CA    = 2
_FARADAY = 96485.0    # C/mol
_R       = 8.314      # J/(mol K)


class LTCCState:
    #LTCC gating variables for one spine + one dendritic compartment.

    __slots__ = ('m_spine', 'h_spine', 'm_dend', 'h_dend')





def _m_inf(V):

    #Activation steady-state

    um = params.get("ltcc_um", -20.0) #half activation
    km = params.get("ltcc_km",   5.0) #slope
    return 1.0 / (1.0 + np.exp(-(V - um) / km))


def _h_inf(V):
    #Inactivation steady-state
    uh = params.get("ltcc_uh", -65.0) #half inactivation
    kh = params.get("ltcc_kh",   7.0) #slope
    return 1.0 / (1.0 + np.exp((V - uh) / kh))




def _temp_factor():
    #for tau_h
    temp = params.get("temperature", 35.0)
    Q10  = params.get("ltcc_Q10",     3.0)
    return Q10 ** ((temp - 22.0) / 10.0)


#Gate Update

def _update_gate(gate, ss, tau_ms, dt_ms):
    
    tau_ms = max(tau_ms, 1e-6)       

    return float(np.clip(ss + (gate - ss) * np.exp(-dt_ms / tau_ms), 0.0, 1.0))



# CHANNEL DENSITY


def get_channel_density(distance):
    
    profile = params.get("ltcc_density_profile", "exponential")

    if profile == "exponential":
        density = np.exp(-distance / params.get("ltcc_lambda", 100.0))
    elif profile == "uniform":
        density = 1.0
    else:
        density = max(0.0, 1.0 - distance / params.get("ltcc_max_dist", 300.0))

    return max(density, params.get("ltcc_min_density", 0.1))


#GHK CUrrent

def _ghk_current(g, m, h, V, Ca_intracelluar):
    
    Ca_ext = params.get("ca_ext", 2000.0)   # µM
    temp   = params.get("temperature", 35.0)
    T_K    = temp + 273.15

    # zFV/RT in dimensionless units (V in mV → multiply by 1e-3)
    zFV_RT = (_Z_CA * _FARADAY * V * 1e-3) / (_R * T_K)

    if abs(zFV_RT) < 1e-6:
        # L'Hopital limit as V → 0
        I = g * m * m * h * (Ca_intracelluar - Ca_ext)
    else:
        I = (g * m * m * h * zFV_RT
             * (Ca_intracelluar - Ca_ext * np.exp(-zFV_RT))
             / (1.0 - np.exp(-zFV_RT)))

    # Influx convention: current is positive when Ca2+ enters (V depolarised,
    # Ca_in << Ca_ext).  The GHK expression is negative under those conditions
    # so we negate and floor at 0.

    return float(max(-I, 0.0))


# LTCC to SPine

def update_ltcc_spine(state, Ca_spine, V_spine, distance, dt, noise):
    
    dt_ms      = dt * 1000.0
    tf         = _temp_factor()

    # Activation: tau_m = 0.08 ms — at dt=10 ms this gate is always at steady-state.
    tau_m_ms   = params.get("ltcc_tau_m", 0.08)         # no Q10 (moot)
    state.m_spine = _update_gate(state.m_spine, _m_inf(V_spine), tau_m_ms, dt_ms)

    # Inactivation: tau_h = 300 ms / Q10_factor — dynamically meaningful.
    tau_h_ms   = params.get("ltcc_tau_h", 300.0) / tf
    state.h_spine = _update_gate(state.h_spine, _h_inf(V_spine), tau_h_ms, dt_ms)

    # GHK current
    density = get_channel_density(distance)
    g_eff   = params.get("g_ltcc", 0.0005) * density
    I       = _ghk_current(g_eff, state.m_spine, state.h_spine, V_spine, Ca_spine)

    # I in µM/ms; multiply by dt_ms → µM per step
    return I * dt_ms + noise


#LTCC Dendrite

def update_ltcc_dend(state, Ca_dend, V_soma, dt, noise):
    
    
    dt_ms    = dt * 1000.0
    tf       = _temp_factor()

    tau_m_ms = params.get("ltcc_tau_m", 0.08)
    state.m_dend = _update_gate(state.m_dend, _m_inf(V_soma), tau_m_ms, dt_ms)

    tau_h_ms = params.get("ltcc_tau_h", 300.0) / tf
    state.h_dend = _update_gate(state.h_dend, _h_inf(V_soma), tau_h_ms, dt_ms)

    g_eff = params.get("g_ltcc_dend", 0.0003)
    I     = _ghk_current(g_eff, state.m_dend, state.h_dend, V_soma, Ca_dend)

    return I * dt_ms + noise




def initialize_ltcc_state():
    
    state  = LTCCState()
    V_rest = params.get("v_rest", -70.0)

    m0 = float(_m_inf(V_rest))
    h0 = float(_h_inf(V_rest))

    state.m_spine = m0
    state.h_spine = h0
    state.m_dend  = m0
    state.h_dend  = h0
    return state


#param defintion for quick testing 

def initialize_ltcc_params():
    
    defaults = {
        "g_ltcc":               0.0005,
        "g_ltcc_dend":          0.0003,
        "ca_ext":            2000.0,
        "v_rest":              -70.0,
        "ltcc_um":             -20.0,
        "ltcc_km":               5.0,
        "ltcc_tau_m":            0.08,
        "ltcc_uh":             -65.0,
        "ltcc_kh":               7.0,
        "ltcc_tau_h":          300.0,
        "ltcc_density_profile": "exponential",
        "ltcc_lambda":         100.0,
        "ltcc_min_density":      0.1,
        "temperature":          35.0,
        "ltcc_Q10":              3.0,
        "sigma_ltcc_spine":    0.001,
        "sigma_ltcc_dend":    0.0005,
    }
    for key, val in defaults.items():
        params.setdefault(key, val)
    return params
