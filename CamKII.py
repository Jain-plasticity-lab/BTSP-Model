"""
CaMKII Module
Handles CaMKII activation dynamics with stochastic firing
"""

import numpy as np
from Params import params


def camkii_activation(ca, theta, n=4):

    return ca**n / (ca**n + theta**n)


class CaMKIIState:
    
    #Class to maintain CaMKII state variables for stochastic dynamics

    def __init__(self):
        self.can_fire = False
        self.fired_once = False
        self.fire_time = None
        self.last_attempt_time = None
        self.in_refractory = False
        
    def reset(self):
        """Reset the state"""
        self.can_fire = False
        self.fired_once = False
        self.fire_time = None
        self.last_attempt_time = None
        self.in_refractory = False


def update_camkii_potential(camkii_potential_current, ca_dend_delayed, dt):
    # Deterministic CaMKII signature 

    drive = ca_dend_delayed
    
    dcamkii_potential = (
        -camkii_potential_current / params["tau_camkii"] +
        params["alpha_camkii"] * camkii_activation(drive, params["camkii_theta"])
    ) * dt
    
    return dcamkii_potential


def update_camkii_stochastic(
    camkii_current,
    camkii_potential,
    ca_dend_delayed,
    state,
    current_time,
    dt,
    noise,
    rng
):
    #Stochastic firing 

    drive = ca_dend_delayed
    fired_now = False
    
    # Check if CaMKII wants to fire (potential rises above threshold)
    if not state.can_fire and camkii_potential > 0.7 and not state.in_refractory:
        state.can_fire = True
        state.last_attempt_time = current_time
    
    # If CaMKII can fire and hasn't fired yet
    if state.can_fire and not state.fired_once:
        # Generate random number to decide if it fires
        fire_roll = rng.random()
        
        if fire_roll < params["camkii_fire_prob"]:
            # Successful firing!
            state.fired_once = True
            state.fire_time = current_time
            fired_now = True
            # Set CaMKII to its potential value
            dcamkii = camkii_potential - camkii_current
        else:
            # Failed to fire - go into refractory period
            dcamkii = params["camkii_noise_level"] * rng.normal(1, 0.1) - camkii_current
            state.can_fire = False
            state.in_refractory = True
            state.last_attempt_time = current_time
    else:
        dcamkii = 0.0
    
    # Check if refractory period is over
    if state.in_refractory and (current_time - state.last_attempt_time) > params["camkii_refractory"]:
        state.in_refractory = False
    
    # If CaMKII has fired, follow the potential dynamics
    if state.fired_once:
        dcamkii = (
            -(camkii_current*3) / params["tau_camkii"] +
            params["alpha_camkii"] * camkii_activation(drive, params["camkii_theta"])
        ) * dt
        dcamkii += noise * np.sqrt(dt * drive)
    elif not fired_now:
        # Maintain at noise level
        dcamkii = params["camkii_noise_level"] * rng.normal(1, 0.1) - camkii_current
    
    return dcamkii, fired_now


def initialize_camkii():
    
    return 0.0, CaMKIIState()