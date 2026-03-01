"""
Synaptic Weight Module 

"""

import numpy as np
from Params import params


def update_weights(
    w,
    camkii,
    elig,
    pp1,
    camkii_fired,
    camkii_fire_time,
    timing_kernel,
    current_time,
    dt,
    noise,
):
    
    dw = 0.0

    plasticity_cond = (
        (camkii > 0.5)
        and (elig > 0)
        and (pp1 < 0.75)
        and camkii_fired
    )

    if plasticity_cond:
        timing_gain = np.exp(-(current_time - camkii_fire_time) / 30.0)
        dw = (
            params["eta"]
            * timing_kernel #trying out with no kernel and no gain terms 
            * timing_gain
            * camkii
            * (params["w_max"] - w)
            * elig
            * dt
        )
        dw += noise * np.sqrt(dt)

    return float(dw)


def initialize_weights():
    
    return float(params["w_init"])


def clip_weights(w):
    
    return float(np.clip(w, params["w_min"], params["w_max"]))
