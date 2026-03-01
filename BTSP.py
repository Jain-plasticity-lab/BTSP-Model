"""

BTSP Central Simulation Script

"""

import numpy as np
from numba import jit

from Params import params

from IP3 import (
    USE_KINETIC_IP3R,
    update_ip3_spine, update_ca_store, get_ip3_at_er,
    sample_er_distance, compute_ip3_diffusion_params,
    initialize_ip3r_kinetic_state, ip3r_open_probability,
)
from NMDA_Ca import compute_voltage_attenuation, compute_nmda_voltage_dependence, get_somatic_voltage
from Ca_spine import update_ca_spine, initialize_ca_spine
from Ca_dend import update_ca_dend, initialize_ca_dend
from CamKII import update_camkii_potential, update_camkii_stochastic, initialize_camkii
from ET import update_eligibility, compute_timing_kernel, initialize_eligibility
from PP1 import update_pp1, initialize_pp1
from Synapse_W import update_weights, initialize_weights, clip_weights
from LTCC import update_ltcc_spine, update_ltcc_dend, initialize_ltcc_params, initialize_ltcc_state


@jit(nopython=True, cache=True)
def _spike_arrays(time, t_pre, t_post, dt):
    n    = len(time)
    post = np.zeros(n, dtype=np.float32)
    pre  = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if abs(time[i] - t_post) < dt: post[i] = 1.0
        if abs(time[i] - t_pre)  < dt: pre[i]  = 1.0
    return post, pre


@jit(nopython=True, cache=True)
def _make_noise(n, s_ca, s_pp1, s_w, s_dend, s_camkii, seed):
    np.random.seed(seed)
    nca     = np.random.normal(0, 1, n).astype(np.float32) * s_ca
    npp1    = np.random.normal(0, 1, n).astype(np.float32) * s_pp1
    nw      = np.random.normal(0, 1, n).astype(np.float32) * s_w
    ndend   = np.random.normal(0, 1, n).astype(np.float32) * s_dend
    ncamkii = np.random.normal(0, 1, n).astype(np.float32) * s_camkii
    return nca, npp1, nw, ndend, ncamkii


#RUN SCRIPT

def run_btsp(offset, T=120.0, dt=0.01,
             return_traces=False, seed=1234, use_ltcc=True):
    """
    Run BTSP simulation for one synapse.

    """
    rng  = np.random.default_rng(seed)
    time = np.arange(0, T, dt, dtype=np.float32)
    n    = len(time)

    #single synapse at 50 µm from the Soma
    syn_distance = float(params.get("syn_distance", 50.0))   # µm

    
    er_dist_um = sample_er_distance(rng)
    er_delay_steps, ip3_atten = compute_ip3_diffusion_params(er_dist_um, dt)

    ip3r_state = initialize_ip3r_kinetic_state() if USE_KINETIC_IP3R else None

    ltcc_state = None
    if use_ltcc:
        initialize_ltcc_params()
        ltcc_state = initialize_ltcc_state()   


#Intialising lists of important quantities 

    Ca_sp =np.zeros(n , dtype=np.float32)
    elig=np.zeros(n , dtype=np.float32)
    PP1=np.zeros(n , dtype=np.float32)
    w = np.zeros(n, dtype=np.float32)

    # Shared dendritic / ER compartment (always 1-D)
    IP3_spine = np.zeros(n, dtype=np.float32)
    IP3_er    = np.zeros(n, dtype=np.float32)
    Ca_store  = np.ones(n, dtype=np.float32) * params["ca_store_max"]
    Ca_dend   = np.zeros(n, dtype=np.float32)
    CaMKII    = np.zeros(n, dtype=np.float32)
    CaMKII_pot = np.zeros(n, dtype=np.float32)
    IP3R_prob  = np.zeros(n, dtype=np.float32)
    CaMKII_fired = np.zeros(n, dtype=bool)

    # LTCC trace arrays 
    Ca_ltcc_sp = np.zeros(n, dtype=np.float32) if (use_ltcc and return_traces) else None
    Ca_ltcc_dend = np.zeros(n, dtype=np.float32) if (use_ltcc and return_traces) else None

    # Initial conditions 
    Ca_sp[0]  = initialize_ca_spine()
    elig[0]   = initialize_eligibility()
    PP1[0]    = initialize_pp1()
    w[0]      = initialize_weights()
    Ca_dend[0] = initialize_ca_dend()
    CaMKII[0], camkii_state = initialize_camkii()

    #  Noise 
    nca, npp1, nw_n, ndend, ncamkii = _make_noise(
        n,
        params["sigma_ca_spine"], params["sigma_pp1"], params["sigma_w"],
        params["sigma_dend"],     params["sigma_camkii"],
        seed,
    )

    # LTCC noise allocated only when used
    if use_ltcc:
        nltcc_sp   = rng.normal(0, 1, n).astype(np.float32) * params.get("sigma_ltcc_spine", 0.001)
        nltcc_dend = rng.normal(0, 1, n).astype(np.float32) * params.get("sigma_ltcc_dend",  0.0005)
    else:
        nltcc_sp   = None
        nltcc_dend = None

    # Precompute spikes & voltages 
    t_pre, t_post   = 10.0, 10.0 - offset
    post_sp, pre_sp = _spike_arrays(time, t_pre, t_post, dt)

    # Somatic voltage 
    V_soma_arr = np.array([get_somatic_voltage(p) for p in post_sp], dtype=np.float32)

    # Spine voltage: distance-attenuated 
    V_spine_arr = np.array(
        [compute_voltage_attenuation(syn_distance, V) for V in V_soma_arr],
        dtype=np.float32,
    )

    # NMDA voltage dependence
    nmda_v_arr = np.array(
        [compute_nmda_voltage_dependence(Vs) for Vs in V_spine_arr],
        dtype=np.float32,
    )

    # Timing & constants 
    timing_kernel     = compute_timing_kernel(offset)
    delay_steps_ckii  = int(params["camkii_delay"] / dt)
    ca_store_max      = params["ca_store_max"]
    ip3_peak_er       = 0.0

    #  Main loop 
    for i in range(n - 1):
        post   = float(post_sp[i])
        pre    = float(pre_sp[i])
        V_soma  = float(V_soma_arr[i])
        V_spine = float(V_spine_arr[i])
        nmda_v  = float(nmda_v_arr[i])

        # IP3 production in spine 
        IP3_spine[i+1] = max(
            IP3_spine[i] + update_ip3_spine(IP3_spine[i], pre, dt), #changed from post
            0.0,
        )

        # IP3 diffusion: spine → ER  
        ip3_at_er   = get_ip3_at_er(IP3_spine, i, er_delay_steps, ip3_atten)
        IP3_er[i]   = ip3_at_er
        ip3_peak_er = max(ip3_peak_er, ip3_at_er)

        # ER Ca store update 
        dCa_store, release = update_ca_store(
            Ca_store[i], ip3_at_er, Ca_dend[i], ip3_peak_er, dt, ndend[i], ip3r_state
        )
        Ca_store[i+1] = np.clip(Ca_store[i] + dCa_store, 0.0, ca_store_max)
        if return_traces:
            IP3R_prob[i] = ip3r_open_probability(ip3r_state)

        #  Dendritic Ca (ER release + LTCC) 
        dCa_dend = update_ca_dend(Ca_dend[i], release, dt, ndend[i])
        if use_ltcc and ltcc_state is not None:
            d_ltcc_d = update_ltcc_dend(ltcc_state, Ca_dend[i], V_soma, dt, nltcc_dend[i])
            dCa_dend += d_ltcc_d
            if Ca_ltcc_dend is not None:
                Ca_ltcc_dend[i] = d_ltcc_d
        Ca_dend[i+1] = max(Ca_dend[i] + dCa_dend, 0.0)

        #  CaMKII 
        ca_del = Ca_dend[i - delay_steps_ckii] if i >= delay_steps_ckii else 0.0
        CaMKII_pot[i+1] = max(
            CaMKII_pot[i] + update_camkii_potential(CaMKII_pot[i], ca_del, dt),
            0.0,
        )
        dCaMKII, fired = update_camkii_stochastic(
            CaMKII[i], CaMKII_pot[i+1], ca_del, camkii_state, time[i], dt, ncamkii[i], rng
        )
        CaMKII[i+1] = max(CaMKII[i] + dCaMKII, 0.0)
        if return_traces:
            CaMKII_fired[i] = fired

        #  Spine Ca (NMDA + LTCC) 
        dCa_sp = update_ca_spine(Ca_sp[i], pre, nmda_v, dt, nca[i])
        if use_ltcc and ltcc_state is not None:
            d_ltcc_sp = update_ltcc_spine(
                ltcc_state, Ca_sp[i], V_spine, syn_distance, dt, nltcc_sp[i]
            )
            dCa_sp += d_ltcc_sp
            if Ca_ltcc_sp is not None:
                Ca_ltcc_sp[i] = d_ltcc_sp
        Ca_sp[i+1] = max(Ca_sp[i] + dCa_sp, 0.0)

        #  Eligibility trace 
        elig[i+1] = max(
            elig[i] + update_eligibility(elig[i], Ca_sp[i], timing_kernel, dt),
            0.0,
        )

        #  PP1 
        PP1[i+1] = max(
            PP1[i] + update_pp1(PP1[i], Ca_sp[i], dt, npp1[i]),
            0.0,
        )

        #  Synaptic weight 
        dw = update_weights(
            w[i], CaMKII[i], elig[i], PP1[i],
            camkii_state.fired_once, camkii_state.fire_time,
            timing_kernel, time[i], dt, nw_n[i],
        )
        w[i+1] = clip_weights(w[i] + dw)


    if return_traces:
        return (
            time, w, Ca_sp, PP1, elig, CaMKII, Ca_dend,
            IP3_spine, IP3_er, Ca_store,
            syn_distance, float(nmda_v_arr[-1]),
            IP3R_prob, CaMKII_pot, CaMKII_fired,
            camkii_state.fire_time,
            Ca_ltcc_sp, Ca_ltcc_dend,
            er_dist_um,
        )

    return float(w[-1] - w[0]), syn_distance, camkii_state.fire_time
