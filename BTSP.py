"""
BTSP Main Simulation Module 
Central script that coordinates all modules to run the BTSP simulation
"""

import numpy as np
from Params import params
from IP3_Ca import update_ip3, update_ca_store, ip3r_open_probability
from NMDA_Ca import compute_voltage_attenuation, compute_nmda_voltage_dependence, get_somatic_voltage
from Ca_spine import update_ca_spine, initialize_ca_spine
from Ca_dend import update_ca_dend, initialize_ca_dend
from CamKII import update_camkii_potential, update_camkii_stochastic, initialize_camkii
from ET import update_eligibility, compute_timing_kernel, initialize_eligibility
from PP1 import update_pp1, initialize_pp1
from Synapse_W import update_weights, initialize_weights, clip_weights
from LTCC import (
    update_ltcc_spine, 
    update_ltcc_dend, 
    initialize_ltcc_params,
    initialize_ltcc_state
)


def run_btsp(
    offset,
    active_synapses,
    T=120.0,
    dt=0.01,
    return_traces=False,
    seed=None,
    use_ltcc=True
):
    """
    Run BTSP simulation with modular components 
    
    Args:
        offset: Timing difference (t_post - t_pre)
        active_synapses: List of synapse indices with presynaptic input
        T: Total simulation time (s)
        dt: Time step (s)
        return_traces: If True, return full time traces
        seed: Random seed for reproducibility
        use_ltcc: If True, include LTCC dynamics (default: True)
        
    Returns:
        If return_traces=False:
            dw: Final weight change for each synapse
            syn_distances: Synapse distances
            camkii_fire_time: Time when CaMKII fired
        If return_traces=True:
            time, w, Ca_sp, PP1, elig, CaMKII, Ca_dend, IP3, Ca_store, 
            syn_distances, nmda_v, IP3R_prob, CaMKII_potential, CaMKII_fired, 
            camkii_fire_time, Ca_ltcc_spine, Ca_ltcc_dend
    """
    # Initialize random number generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(1234)
    
    # Time array
    time = np.arange(0, T, dt)
    n = len(time)
    n_syn = params["n_syn"]
    
    # Initialize LTCC parameters and state (after n_syn is defined)
    ltcc_state = None
    if use_ltcc:
        initialize_ltcc_params()
        ltcc_state = initialize_ltcc_state(n_syn)
    
    # Create boolean mask for active synapses
    is_active = np.zeros(n_syn, dtype=bool)
    is_active[active_synapses] = True
    
    # Synapse distances (evenly spaced from 1 to 300 µm)
    syn_distances = np.linspace(1, 300, n_syn)
    
    # Initialize state arrays
    Ca_sp = np.zeros((n_syn, n))
    elig = np.zeros((n_syn, n))
    PP1 = np.zeros((n_syn, n))
    w = np.zeros((n_syn, n))
    
    # Initialize per-synapse states
    Ca_sp[:, 0] = initialize_ca_spine(n_syn)
    elig[:, 0] = initialize_eligibility(n_syn)
    PP1[:, 0] = initialize_pp1(n_syn)
    w[:, 0] = initialize_weights(n_syn)
    
    # Global state arrays
    IP3 = np.zeros(n)
    Ca_store = np.ones(n) * params["ca_store_max"]
    Ca_dend = np.zeros(n)
    CaMKII = np.zeros(n)
    CaMKII_potential = np.zeros(n)
    CaMKII_fired = np.zeros(n, dtype=bool)
    IP3R_prob = np.zeros(n)
    
    # LTCC contribution tracking (optional)
    Ca_ltcc_spine = np.zeros((n_syn, n)) if return_traces else None
    Ca_ltcc_dend = np.zeros(n) if return_traces else None
    
    # Initialize global states
    Ca_dend[0] = initialize_ca_dend()
    CaMKII[0], camkii_state = initialize_camkii()
    
    # Track IP3 peak
    ip3_peak = 0.0
    
    # Event times
    t_pre = 10.0
    t_post = 10.0 + offset
    
    # Precompute timing kernel
    timing_kernel = compute_timing_kernel(offset)
    
    # Preallocate noise arrays
    noise_ca_spine = rng.normal(0, 1, (n_syn, n)) * params["sigma_ca_spine"]
    noise_pp1 = rng.normal(0, 1, (n_syn, n)) * params["sigma_pp1"]
    noise_w = rng.normal(0, 1, (n_syn, n)) * params["sigma_w"]
    noise_dend = rng.normal(0, 1, n) * params["sigma_dend"]
    noise_camkii = rng.normal(0, 1, n) * params["sigma_camkii"]
    
    if use_ltcc:
        noise_ltcc_spine = rng.normal(0, 1, (n_syn, n)) * params.get("sigma_ltcc_spine", 0.01)
        noise_ltcc_dend = rng.normal(0, 1, n) * params.get("sigma_ltcc_dend", 0.005)
    
    # Simulation loop
    for i in range(n - 1):
        t = time[i]
        
        # === Spike timing ===
        post = 1.0 if abs(t - t_post) < dt else 0.0
        pre = 1.0 if abs(t - t_pre) < dt else 0.0
        
        # === Voltage and NMDA ===
        V_soma = get_somatic_voltage(post)
        V_spine = compute_voltage_attenuation(syn_distances, V_soma)
        nmda_v = compute_nmda_voltage_dependence(V_spine)
        
        # === Global IP3 dynamics ===
        dIP3 = update_ip3(IP3[i], post, dt)
        IP3[i + 1] = max(IP3[i] + dIP3, 0)
        
        # Update IP3 peak tracker
        if IP3[i] > ip3_peak:
            ip3_peak = IP3[i]
        
        # === Ca store dynamics ===
        dCa_store, release = update_ca_store(
            Ca_store[i], IP3[i], Ca_dend[i], ip3_peak, dt, noise_dend[i]
        )
        Ca_store[i + 1] = np.clip(
            Ca_store[i] + dCa_store, 0, params["ca_store_max"]
        )
        
        # Store IP3R probability
        IP3R_prob[i] = ip3r_open_probability(Ca_dend[i], IP3[i], ip3_peak)
        
        # === Dendritic Ca dynamics (with LTCC) ===
        dCa_dend = update_ca_dend(Ca_dend[i], release, dt, noise_dend[i])
        
        # Add LTCC contribution to dendrite
        if use_ltcc and ltcc_state is not None:
            dCa_ltcc_d = update_ltcc_dend(ltcc_state, Ca_dend[i], V_soma, dt, noise_ltcc_dend[i])
            dCa_dend += dCa_ltcc_d
            if return_traces:
                Ca_ltcc_dend[i] = dCa_ltcc_d
        
        Ca_dend[i + 1] = max(Ca_dend[i] + dCa_dend, 0)
        
        # === CaMKII dynamics ===
        # Get delayed Ca
        delay_steps = int(params["camkii_delay"] / dt)
        ca_dend_delayed = Ca_dend[i - delay_steps] if i >= delay_steps else 0.0
        
        # Update potential CaMKII
        dCaMKII_pot = update_camkii_potential(
            CaMKII_potential[i], ca_dend_delayed, dt
        )
        CaMKII_potential[i + 1] = max(CaMKII_potential[i] + dCaMKII_pot, 0)
        
        # Update stochastic CaMKII
        dCaMKII, fired_now = update_camkii_stochastic(
            CaMKII[i],
            CaMKII_potential[i + 1],
            ca_dend_delayed,
            camkii_state,
            t,
            dt,
            noise_camkii[i],
            rng
        )
        CaMKII[i + 1] = max(CaMKII[i] + dCaMKII, 0)
        CaMKII_fired[i] = fired_now
        
        # === Spine Ca dynamics (with LTCC) ===
        dCa_sp = update_ca_spine(
            Ca_sp[:, i], pre, nmda_v, is_active, dt, noise_ca_spine[:, i]
        )
        
        # Add LTCC contribution to spines
        if use_ltcc and ltcc_state is not None:
            dCa_ltcc_sp = update_ltcc_spine(
                ltcc_state, Ca_sp[:, i], V_spine, syn_distances, is_active, dt, noise_ltcc_spine[:, i]
            )
            dCa_sp += dCa_ltcc_sp
            if return_traces:
                Ca_ltcc_spine[:, i] = dCa_ltcc_sp
        
        Ca_sp[:, i + 1] = np.maximum(Ca_sp[:, i] + dCa_sp, 0)
        
        # === Eligibility trace ===
        delig = update_eligibility(
            elig[:, i], Ca_sp[:, i], timing_kernel, is_active, dt
        )
        elig[:, i + 1] = np.maximum(elig[:, i] + delig, 0)
        
        # === PP1 dynamics ===
        dPP1 = update_pp1(PP1[:, i], Ca_sp[:, i], is_active, dt, noise_pp1[:, i])
        PP1[:, i + 1] = np.maximum(PP1[:, i] + dPP1, 0)
        
        # === Weight update ===
        dw = update_weights(
            w[:, i],
            CaMKII[i],
            elig[:, i],
            PP1[:, i],
            is_active,
            camkii_state.fired_once,
            camkii_state.fire_time,
            timing_kernel,
            t,
            dt,
            noise_w[:, i]
        )
        w[:, i + 1] = clip_weights(w[:, i] + dw)
    
    # Return results
    if return_traces:
        return (
            time, w, Ca_sp, PP1, elig, CaMKII, Ca_dend, IP3, Ca_store,
            syn_distances, nmda_v, IP3R_prob, CaMKII_potential, 
            CaMKII_fired, camkii_state.fire_time, Ca_ltcc_spine, Ca_ltcc_dend
        )
    
    return w[:, -1] - w[:, 0], syn_distances, camkii_state.fire_time
