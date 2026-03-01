"""
IP3 Pathway Module 
Based on Adeoye et al. 2022 

"""

import numpy as np
from numba import jit, float64
from numba.experimental import jitclass
from Params import params


USE_KINETIC_IP3R = True   # True → 4-state kinetic | False → simple bell-curve

TAU_IP3          = params.get("tau_ip3", 15.0)
ALPHA_IP3_POST   = params.get("alpha_ip3_post", 10.0) 
CA_STORE_MAX     = params.get("ca_store_max", 100.0)
TAU_STORE_REFILL = params.get("tau_store_refill", 10.0)
ALPHA_RELEASE    = params.get("alpha_release", 5.0) #changed from 2
K_ER_LEAK        = params.get("k_er_leak", 0.0001)

# IP3 diffusion coefficient in dendrites 
D_IP3            = params.get("d_ip3_um2_ms", 0.3)

LAMBDA_IP3       = params.get("lambda_ip3_um",np.sqrt(D_IP3 * TAU_IP3))   # µm

# Stochastic ER distance range (µm) 
ER_DIST_MIN_UM   = params.get("er_dist_min_um", 10.0)
ER_DIST_MAX_UM   = params.get("er_dist_max_um", 60.0)

# Kinetic model — Adeoye et al. 2022
A1, N_O, K_OD = 17.05043,  2.473407, 0.909078
A2, N_A, K_AD = 18.49186,  0.093452, 1.955650
A3, N_I, K_ID = 234.0259, 56.84823,  0.089938
J01, J12, J22  = 303.1635, 323.0063, 4.814111
J23, J45       =   5.356155, 5.625616
EJ01, EJ45     = 301.3284,  2.648741

K_INACT_SCALE = 50.0   # Artificially scales inactive-state efflux to tune steady-state p_I

# Safety nets
_CA_MIN   = 0.01    
_IP3_MIN  = 0.01    
_RATE_MAX = 1e4     

#Stochastic ER distance
def sample_er_distance(rng=None):

    if rng is not None:
        er_dist_um = float(rng.uniform(ER_DIST_MIN_UM, ER_DIST_MAX_UM))
    else:
        er_dist_um = float(np.random.uniform(ER_DIST_MIN_UM, ER_DIST_MAX_UM))
    return er_dist_um

#IP3 Diffusion
def compute_ip3_diffusion_params(er_dist_um, dt):
    
    dt_ms = dt * 1000.0

    delay_ms    = (er_dist_um ** 2) / (2.0 * D_IP3)
    delay_steps = max(1, int(round(delay_ms / dt_ms)))

    #exponential decay
    ip3_atten   = float(np.exp(-er_dist_um / max(LAMBDA_IP3, 1e-6)))/5

    return delay_steps, ip3_atten


#Kinetic State

@jitclass([('p_R', float64), ('p_A', float64), ('p_O', float64), ('p_I', float64)])
class IP3RKineticState:
    def __init__(self):
        self.p_R = 1.0
        self.p_A = 0.0
        self.p_O = 0.0
        self.p_I = 0.0


def initialize_ip3r_kinetic_state():
    return IP3RKineticState()


#Kinetic Model

@jit(nopython=True, cache=True)
def _occupancy_params(ip3):
    K_O = A1 * ip3**N_O / (ip3**N_O + K_OD**N_O)
    K_A = A2 * ip3**N_A / (ip3**N_A + K_AD**N_A)
    K_I = A3 * ip3**N_I / (ip3**N_I + K_ID**N_I)
    return K_O, K_A, K_I


@jit(nopython=True, cache=True)
def _cap(r):
    return r if r < _RATE_MAX else _RATE_MAX


@jit(nopython=True, cache=True)
def _transition_rates(ca, ip3):
    K_O, K_A, K_I = _occupancy_params(ip3)
    ca2, ca3, ca5 = ca*ca, ca**3, ca**5

    denom_RA = 1.0/(J01*ca) + 1.0/(J12*ca2)
    denom_OI = 1.0/(J23*ca3) + 1.0/(J45*ca5)
    denom_RI = 1.0/(EJ01*ca) + 1.0/(EJ45*ca5)

    Ka_ca2  = max(K_A * ca2, 1e-300)
    Ko_ca2  = max(K_O * ca2, 1e-300)
    Ki_ca5  = max(K_I * ca5, 1e-300)
    J22_ca2 = max(J22 * ca2, 1e-300)

    k_RA = _cap(1.0 / max(denom_RA, 1e-300))
    k_AR = _cap(1.0 / max(Ka_ca2 * denom_RA, 1e-300))
    k_AO = _cap(J22_ca2 / Ka_ca2)
    k_OA = _cap(J22_ca2 / Ko_ca2)
    k_OI = _cap(1.0 / max(Ko_ca2 * denom_OI, 1e-300))
    k_IO = _cap(1.0 / max(Ki_ca5 * denom_OI, 1e-300))
    k_RI = _cap(1.0 / max(denom_RI, 1e-300))
    k_IR = _cap(1.0 / max(Ki_ca5 * denom_RI, 1e-300))

    return k_RA, k_AR, k_AO, k_OA, k_OI, k_IO, k_RI, k_IR


@jit(nopython=True, cache=True)
def _update_kinetic_state(state, ca_dend, ip3, dt_ms):
    ca  = max(ca_dend, _CA_MIN)
    ip3 = max(ip3,     _IP3_MIN)

    k_RA, k_AR, k_AO, k_OA, k_OI, k_IO, k_RI, k_IR = _transition_rates(ca, ip3)

    dp_R = (k_IR*state.p_I + k_AR*state.p_A - (k_RA + k_RI)*state.p_R)                  * dt_ms
    dp_A = (k_RA*state.p_R + k_OA*state.p_O - (k_AR + k_AO)*state.p_A)                  * dt_ms
    dp_O = (k_AO*state.p_A + k_IO*state.p_I - (k_OA + k_OI)*state.p_O)                  * dt_ms
    dp_I = (k_OI*state.p_O + k_RI*state.p_R - (k_IO + k_IR)*K_INACT_SCALE*state.p_I)    * dt_ms

    state.p_R += dp_R
    state.p_A += dp_A
    state.p_O += dp_O
    state.p_I += dp_I

    total = state.p_R + state.p_A + state.p_O + state.p_I
    if total > 0.0:
        inv = 1.0 / total
        state.p_R = max(0.0, min(1.0, state.p_R * inv))
        state.p_A = max(0.0, min(1.0, state.p_A * inv))
        state.p_O = max(0.0, min(1.0, state.p_O * inv))
        state.p_I = max(0.0, min(1.0, state.p_I * inv))
    else:
        state.p_R = 1.0
        state.p_A = 0.0
        state.p_O = 0.0
        state.p_I = 0.0


def ip3r_open_probability(state):
    return state.p_O if state is not None else 0.0


# ============================================================================
# SIMPLE IP3R MODEL (fallback)
# ============================================================================
@jit(nopython=True, cache=True)
def _simple_ip3r_open_prob(ca_dend, ip3, ip3_peak):
    if ip3_peak <= 0.0 or ip3 < 0.3 * ip3_peak:
        return 0.0
    ca_act   = ca_dend**2 / (ca_dend**2 + 0.25)
    ca_inact = 1.0 / (ca_dend + 1.0)
    return ca_act * ca_inact


@jit(nopython=True, cache=True)
def update_ip3_spine(ip3_current, post_spike, dt):
    
    return (ALPHA_IP3_POST * post_spike - (ip3_current / TAU_IP3)) * dt * 100.0 #arbitary scaling


#IP3 at ER
def get_ip3_at_er(ip3_spine_history, current_step, delay_steps, ip3_atten):
    
    src = current_step - delay_steps
    raw = float(ip3_spine_history[src]) if src >= 0 else 0.0
    return max(0.0, raw * ip3_atten)


def update_ca_store(ca_store, ip3_er, ca_dend, ip3_peak_er, dt, noise, ip3r_state=None):
    dt_ms = dt * 1000.0

    if USE_KINETIC_IP3R and ip3r_state is not None:
        _update_kinetic_state(ip3r_state, ca_dend, ip3_er, dt_ms)
        p_open = min(ip3r_state.p_O, 1.0)
    else:
        p_open = _simple_ip3r_open_prob(ca_dend, ip3_er, ip3_peak_er)

    driving = max(0.0, float(ca_store) - float(ca_dend))
    release = ALPHA_RELEASE * p_open * driving
    refill  = (CA_STORE_MAX - ca_store) / TAU_STORE_REFILL
    leak    = K_ER_LEAK * driving

    dca_store = (refill - release - leak) * dt + noise * np.sqrt(dt)
    return dca_store, release + leak
