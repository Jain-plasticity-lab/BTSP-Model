# -------------------------------------------------------------------
# L-Type Voltage-Gated Calcium Channel (LTCC) Model
#
# This module implements a biophysically detailed LTCC model including:
# - Hodgkin-Huxley style voltage gating
# - Calcium-dependent inactivation (CDI)
# - Distance-dependent channel density
# - Temperature scaling (Q10)
#
# Outputs calcium influx in units of µM per timestep
# -------------------------------------------------------------------

import numpy as np
from Params import params


class LTCCState:
    """
    Container for all LTCC gating variables.

    Each channel has:
    m   : voltage-dependent activation gate
    h   : voltage-dependent inactivation gate
    hCa : calcium-dependent inactivation gate

    Spine variables are stored per synapse.
    Dendritic variables are stored as a single compartment.
    """
    def __init__(self, n_syn):
        self.m_spine = np.zeros(n_syn)      # Activation gate (spines)
        self.h_spine = np.ones(n_syn)       # Voltage inactivation gate (spines)
        self.hCa_spine = np.ones(n_syn)     # Ca-dependent inactivation (spines)
        self.m_dend = 0.0                   # Activation gate (dendrite)
        self.h_dend = 1.0                   # Voltage inactivation gate (dendrite)
        self.hCa_dend = 1.0                 # Ca-dependent inactivation (dendrite)


def alpha_m(V):
    """
    Voltage-dependent forward rate for activation gate (m).

    Derived from experimental LTCC kinetics.
    Implements numerical safeguards near singularities.
    
    Arg:
        V: Membrane voltage (mV)
    Returns:
        alpha_m: Forward rate constant (1/ms)
    """
    V_shift = params.get("ltcc_v_shift", 0.0)  # Voltage shift parameter
    V_adj = V - V_shift
    
    # Avoid division by zero at V = -40
    if np.isscalar(V_adj):
        if abs(V_adj + 40) < 0.01:
            return 0.1
        return 0.055 * (V_adj + 27) / (1 - np.exp(-(V_adj + 27) / 3.8))
    else:
        alpha = np.zeros_like(V_adj)
        mask = np.abs(V_adj + 40) < 0.01
        alpha[mask] = 0.1
        alpha[~mask] = 0.055 * (V_adj[~mask] + 27) / (1 - np.exp(-(V_adj[~mask] + 27) / 3.8))
        return alpha


def beta_m(V):
    """
    Deactivation rate constant
    
    Args:
        V: Membrane voltage (mV)
    Returns:
        beta_m: Backward rate constant (1/ms)
    """
    V_shift = params.get("ltcc_v_shift", 0.0)
    V_adj = V - V_shift
    return 0.94 * np.exp(-(V_adj + 63) / 17)


def alpha_h(V):
    """
    Voltage-dependent inactivation rate (inactivation onset)
    
    Args:
        V: Membrane voltage (mV)
    Returns:
        alpha_h: Forward rate constant (1/ms)
    """
    return 0.000457 * np.exp(-(V + 13) / 50)


def beta_h(V):
    """
    Voltage-dependent recovery from inactivation
    
    Args:
        V: Membrane voltage (mV)
    Returns:
        beta_h: Backward rate constant (1/ms)
    """
    return 0.0065 / (1 + np.exp(-(V + 15) / 28))


def m_inf(V):
    """Steady-state activation"""
    a = alpha_m(V)
    b = beta_m(V)
    return a / (a + b)


def h_inf(V):
    """Steady-state voltage inactivation"""
    a = alpha_h(V)
    b = beta_h(V)
    return a / (a + b)


def tau_m(V):
    """Activation time constant (ms)"""
    a = alpha_m(V)
    b = beta_m(V)
    return 1.0 / (a + b)


def tau_h(V):
    """Voltage inactivation time constant (ms)"""
    a = alpha_h(V)
    b = beta_h(V)
    return 1.0 / (a + b)


def ca_dependent_inactivation(Ca, dt_ms):
    """
    Calcium-dependent inactivation (CDI)
    Models Ca2+/calmodulin-dependent channel inactivation
    
    Args:
        Ca: Calcium concentration (µM)
        dt_ms: Time step (ms)
    Returns:
        dhCa: Change in Ca-dependent inactivation gate
    """
    # CDI parameters
    K_CDI = params.get("ltcc_K_CDI", 1.0)      # Half-inactivation Ca (µM)
    tau_CDI = params.get("ltcc_tau_CDI", 50.0)  # CDI time constant (ms)
    
    # Steady-state CDI
    hCa_inf = 1.0 / (1.0 + (Ca / K_CDI)**2)
    
    return hCa_inf, tau_CDI


def get_channel_density(distance):
    """
    Distance-dependent channel density
    Higher density in proximal dendrites, decreases with distance
    
    Args:
        distance: Distance from soma (µm) - can be scalar or array
    Returns:
        density: Relative channel density (0 to 1)
    """
    density_type = params.get("ltcc_density_profile", "exponential")
    
    if density_type == "exponential":
        # Exponential decay with distance
        lambda_decay = params.get("ltcc_lambda", 100.0)  # Space constant (µm)
        density = np.exp(-distance / lambda_decay)
        
    elif density_type == "gaussian":
        # Gaussian distribution peaked at optimal distance
        optimal_dist = params.get("ltcc_optimal_dist", 75.0)
        sigma_dist = params.get("ltcc_sigma_dist", 50.0)
        density = np.exp(-((distance - optimal_dist)**2) / (2 * sigma_dist**2))
        
    elif density_type == "uniform":
        # Uniform density
        density = np.ones_like(distance) if hasattr(distance, '__len__') else 1.0
        
    else:
        # Linear decay
        max_dist = params.get("ltcc_max_dist", 300.0)
        density = np.maximum(0, 1 - distance / max_dist)
    
    # Ensure minimum density
    min_density = params.get("ltcc_min_density", 0.1)
    return np.maximum(density, min_density)


def update_ltcc_gates_spine(state, V_spine, Ca_spine, dt, is_active):
    """
    Update LTCC gating variables for spines using Hodgkin-Huxley kinetics
    
    Args:
        state: LTCCState object
        V_spine: Spine voltages (mV) - array
        Ca_spine: Spine calcium (µM) - array
        dt: Time step (s)
        is_active: Boolean mask of active synapses
    """
    dt_ms = dt * 1000  # Convert to ms
    
    # Temperature correction (Q10 = 3)
    temp = params.get("temperature", 35.0)  # Celsius
    Q10 = params.get("ltcc_Q10", 3.0)
    temp_factor = Q10 ** ((temp - 22) / 10)
    
    # Only update active synapses
    active_V = V_spine[is_active]
    active_Ca = Ca_spine[is_active]
    
    # Activation gate (m)
    m_ss = m_inf(active_V)
    tau_m_val = tau_m(active_V) / temp_factor
    dm = (m_ss - state.m_spine[is_active]) / tau_m_val * dt_ms
    state.m_spine[is_active] += dm
    state.m_spine[is_active] = np.clip(state.m_spine[is_active], 0, 1)
    
    # Voltage-dependent inactivation (h)
    h_ss = h_inf(active_V)
    tau_h_val = tau_h(active_V) / temp_factor
    dh = (h_ss - state.h_spine[is_active]) / tau_h_val * dt_ms
    state.h_spine[is_active] += dh
    state.h_spine[is_active] = np.clip(state.h_spine[is_active], 0, 1)
    
    # Calcium-dependent inactivation (hCa)
    hCa_ss, tau_hCa = ca_dependent_inactivation(active_Ca, dt_ms)
    dhCa = (hCa_ss - state.hCa_spine[is_active]) / (tau_hCa / temp_factor) * dt_ms
    state.hCa_spine[is_active] += dhCa
    state.hCa_spine[is_active] = np.clip(state.hCa_spine[is_active], 0, 1)


def update_ltcc_gates_dend(state, V_soma, Ca_dend, dt):
    """
    Update LTCC gating variables for dendrite
    
    Args:
        state: LTCCState object
        V_soma: Somatic voltage (mV)
        Ca_dend: Dendritic calcium (µM)
        dt: Time step (s)
    """
    dt_ms = dt * 1000
    
    temp = params.get("temperature", 35.0)
    Q10 = params.get("ltcc_Q10", 3.0)
    temp_factor = Q10 ** ((temp - 22) / 10)
    
    # Activation gate
    m_ss = m_inf(V_soma)
    tau_m_val = tau_m(V_soma) / temp_factor
    dm = (m_ss - state.m_dend) / tau_m_val * dt_ms
    state.m_dend = np.clip(state.m_dend + dm, 0, 1)
    
    # Voltage inactivation
    h_ss = h_inf(V_soma)
    tau_h_val = tau_h(V_soma) / temp_factor
    dh = (h_ss - state.h_dend) / tau_h_val * dt_ms
    state.h_dend = np.clip(state.h_dend + dh, 0, 1)
    
    # Ca-dependent inactivation
    hCa_ss, tau_hCa = ca_dependent_inactivation(Ca_dend, dt_ms)
    dhCa = (hCa_ss - state.hCa_dend) / (tau_hCa / temp_factor) * dt_ms
    state.hCa_dend = np.clip(state.hCa_dend + dhCa, 0, 1)


def compute_ltcc_current_spine(state, V_spine, Ca_spine, distances, is_active):
    """
    Compute L-type calcium current for spines
    Includes realistic driving force and channel density
    
    Args:
        state: LTCCState object
        V_spine: Spine voltages (mV) - array
        Ca_spine: Spine calcium (µM) - array
        distances: Synapse distances (µm) - array
        is_active: Boolean mask
    Returns:
        I_LTCC: Calcium current (µM/s) - array
    """
    # Constants
    g_LTCC = params.get("g_ltcc", 0.0005)  # Conductance (µM/s per mV driving force)
    Ca_ext = params.get("ca_ext", 2000.0)  # External Ca (µM)
    z = 2  # Valence of Ca2+
    F = 96485  # Faraday constant (C/mol)
    R = 8.314  # Gas constant (J/(mol·K))
    T_kelvin = params.get("temperature", 35.0) + 273.15
    
    # Nernst potential for calcium
    E_Ca = (R * T_kelvin) / (z * F) * np.log(Ca_ext / np.maximum(Ca_spine, 0.1)) * 1000  # mV
    
    # Driving force
    driving_force = V_spine - E_Ca
    
    # Channel density (distance-dependent)
    density = get_channel_density(distances)
    
    # Total current: I = g * m^2 * h * hCa * density * (V - E_Ca)
    # m^2 because LTCC has two activation gates
    I_LTCC = np.zeros_like(V_spine)
    I_LTCC[is_active] = (g_LTCC * 
                         state.m_spine[is_active]**2 * 
                         state.h_spine[is_active] * 
                         state.hCa_spine[is_active] * 
                         density[is_active] * 
                         driving_force[is_active])
    
    # Only allow influx (positive current = inward Ca)
    I_LTCC = np.maximum(I_LTCC, 0)
    
    return I_LTCC


def compute_ltcc_current_dend(state, V_soma, Ca_dend):
    """
    Compute L-type calcium current for dendrite
    
    Args:
        state: LTCCState object
        V_soma: Somatic voltage (mV)
        Ca_dend: Dendritic calcium (µM)
    Returns:
        I_LTCC: Calcium current (µM/s)
    """
    g_LTCC = params.get("g_ltcc_dend", 0.0003)
    Ca_ext = params.get("ca_ext", 2000.0)
    z = 2
    F = 96485
    R = 8.314
    T_kelvin = params.get("temperature", 35.0) + 273.15
    
    E_Ca = (R * T_kelvin) / (z * F) * np.log(Ca_ext / max(Ca_dend, 0.1)) * 1000
    driving_force = V_soma - E_Ca
    
    # Dendritic attenuation
    attenuation = params.get("ltcc_dend_attenuation", 0.3)
    
    I_LTCC = (g_LTCC * attenuation * 
              state.m_dend**2 * 
              state.h_dend * 
              state.hCa_dend * 
              driving_force)
    
    return max(I_LTCC, 0)


def update_ltcc_spine(state, Ca_spine, V_spine, distances, is_active, dt, noise):
    """
    Full update for spine LTCC: gates + current
    
    Args:
        state: LTCCState object
        Ca_spine: Current spine Ca (µM) - array
        V_spine: Spine voltage (mV) - array
        distances: Synapse distances (µm) - array
        is_active: Boolean mask
        dt: Time step (s)
        noise: Noise array
    Returns:
        dCa_ltcc: Change in spine Ca from LTCC (µM)
    """
    # Update gating variables
    update_ltcc_gates_spine(state, V_spine, Ca_spine, dt, is_active)
    
    # Compute current
    I_LTCC = compute_ltcc_current_spine(state, V_spine, Ca_spine, distances, is_active)
    
    # Convert to concentration change
    dCa_ltcc = I_LTCC * dt + noise
    
    return dCa_ltcc


def update_ltcc_dend(state, Ca_dend, V_soma, dt, noise):
    """
    Full update for dendritic LTCC: gates + current
    
    Args:
        state: LTCCState object
        Ca_dend: Current dendritic Ca (µM)
        V_soma: Somatic voltage (mV)
        dt: Time step (s)
        noise: Noise value
    Returns:
        dCa_ltcc: Change in dendritic Ca from LTCC (µM)
    """
    # Update gating variables
    update_ltcc_gates_dend(state, V_soma, Ca_dend, dt)
    
    # Compute current
    I_LTCC = compute_ltcc_current_dend(state, V_soma, Ca_dend)
    
    # Convert to concentration change
    dCa_ltcc = I_LTCC * dt + noise
    
    return dCa_ltcc


def initialize_ltcc_state(n_syn):
    """
    Initialize LTCC state at resting potential
    
    Args:
        n_syn: Number of synapses
    Returns:
        state: LTCCState object with initialized gates
    """
    state = LTCCState(n_syn)
    
    # Initialize at resting potential
    V_rest = params.get("v_rest", -70.0)
    
    # Set gates to steady-state values at rest
    state.m_spine[:] = m_inf(V_rest)
    state.h_spine[:] = h_inf(V_rest)
    state.hCa_spine[:] = 1.0  # No Ca-dependent inactivation at rest
    
    state.m_dend = m_inf(V_rest)
    state.h_dend = h_inf(V_rest)
    state.hCa_dend = 1.0
    
    return state

#These parameters already defined in Params.py but also mentioned here for faster computation(To be implemented in other modules)
def initialize_ltcc_params():
    """
    Initialize LTCC-related parameters with robust defaults
    """
    default_ltcc_params = {
        # Conductances
        "g_ltcc": 0.0005,              # Spine LTCC conductance (µM/s per mV)
        "g_ltcc_dend": 0.0003,         # Dendritic LTCC conductance
        
        # External calcium
        "ca_ext": 2000.0,              # External Ca concentration (µM)
        
        # Voltage parameters
        "ltcc_v_shift": 0.0,           # Voltage shift for gating (mV)
        "v_rest": -70.0,               # Resting potential (mV)
        
        # CDI parameters
        "ltcc_K_CDI": 1.0,             # Half-inactivation Ca for CDI (µM)
        "ltcc_tau_CDI": 50.0,          # CDI time constant (ms)
        
        # Spatial distribution
        "ltcc_density_profile": "exponential",  # exponential, gaussian, linear, uniform are the possible distributions
        "ltcc_lambda": 100.0,          # Space constant for exponential (µm)
        "ltcc_optimal_dist": 75.0,     # Optimal distance for gaussian (µm)
        "ltcc_sigma_dist": 50.0,       # Std dev for gaussian (µm)
        "ltcc_max_dist": 300.0,        # Max distance for linear (µm)
        "ltcc_min_density": 0.1,       # Minimum relative density
        
        # Temperature
        "temperature": 35.0,           # Temperature (Celsius)
        "ltcc_Q10": 3.0,              # Temperature sensitivity
        
        # Dendritic scaling
        "ltcc_dend_attenuation": 0.3,  # Dendritic LTCC scaling factor
        
        # Noise
        "sigma_ltcc_spine": 0.00001,   # Spine LTCC noise (reduced)
        "sigma_ltcc_dend": 0.000005,   # Dendritic LTCC noise (reduced)
    }
    
    for key, value in default_ltcc_params.items():
        if key not in params:
            params[key] = value
    
    return params
