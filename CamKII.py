"""
CaMKII Module
Handles CaMKII activation dynamics with stochastic firing
"""

import numpy as np
from Params import params

def camkii_activation(ca, theta, n=4):
    """
    Hill function: models cooperative calcium binding to calmodulin
    
    Args:
        ca: Calcium concentration
        theta: Activation threshold
        n=4 reflects binding of 4 Ca2+ ions to calmodulin
        
    Returns:
        activation: CaMKII activation level (0-1)
    """

    return ca**n / (ca**n + theta**n)

    # When ca < theta: activation is low (kinase mostly inactive)
    # When ca > theta: activation approaches 1 (kinase fully active)


class CaMKIIState:
    #Class to maintain CaMKII state variables for stochastic dynamics
    
    def __init__(self):
        self.can_fire = False  # Is CaMKII eligible to attempt firing?

        self.fired_once = False #Has CaMKII successfully fired yet?

        self.fire_time = None #When did successful firing occur? 

        self.last_attempt_time = None #Last time firing was attempted

        self.in_refractory = False #Is CaMKII in the refractory period after failed attempt?
        
    def reset(self):
        #Reset the states
        self.can_fire = False
        self.fired_once = False
        self.fire_time = None
        self.last_attempt_time = None
        self.in_refractory = False


def update_camkii_potential(camkii_potential_current, ca_dend_delayed, dt):
    """
    # Deterministic CaMKII dynamics (no stochasticity)
    
    Args:
        camkii_potential_current: Current CaMKII level
        ca_dend_delayed: Delayed dendritic Ca (for CaMKII activation)
        dt: Time step
        
    Returns:
        dcamkii_potential: Change in potential CaMKII
    """
    drive = ca_dend_delayed
    
    dcamkii_potential = (
        -camkii_potential_current / params["tau_camkii"] #  First term: natural decay of CaMKII activity 
        + params["alpha_camkii"] * camkii_activation(drive, params["camkii_theta"]) # Second term: calcium-driven increase in CaMKII activation 
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
    """
    Update CaMKII with stochastic firing dynamics
    
    Args:
        camkii_current: Current CaMKII level
        camkii_potential: Potential (deterministic) CaMKII level
        ca_dend_delayed: Delayed dendritic Ca
        state: CaMKIIState object tracking firing state
        current_time: Current simulation time
        dt: Time step
        noise: Noise term
        rng: Random number generator
        
    Returns:
        dcamkii: Change in CaMKII
        fired_now: Boolean indicating if CaMKII fired in this timestep
    """
    drive = ca_dend_delayed
    fired_now = False
    
    # Check if CaMKII wants to fire (potential rises above threshold)
    if not state.can_fire and camkii_potential > 0.1 and not state.in_refractory:
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
    
    # If CaMKII has fired, follow the determinisitic dynamics
    if state.fired_once:
        dcamkii = (
            -camkii_current / params["tau_camkii"] +
            params["alpha_camkii"] * camkii_activation(drive, params["camkii_theta"])
        ) * dt
        dcamkii += noise * np.sqrt(dt * drive)
    elif not fired_now:
        # Maintain at noise level
        dcamkii = params["camkii_noise_level"] * rng.normal(1, 0.1) - camkii_current
    
    return dcamkii, fired_now


def initialize_camkii():
    """
    Initialize CaMKII variables
    
    Returns:
        camkii: Initial CaMKII value
        state: Initialized CaMKIIState object
    """
    return 0.0, CaMKIIState()
