"""
BTSP Model Parameters
Contains all model parameters for the synaptic plasticity simulation
"""

params = {
    "n_syn": 300,

    # Weight parameters
    "w_init": 0.4,
    "w_min": 0.0,
    "w_max": 4.0,
    "eta": 0.22,
    "sigma_w": 0.02,
    "tau_forward": 0.69,
    "tau_backward": 1.31,

    # Spine Ca2+ (NMDAR)
    "alpha_nmda": 10.0,
    "tau_ca_spine": 0.2,
    "sigma_ca_spine": 0.05,
    "mg_block": 0.062,

    # Voltage
    "v_rest": -70,
    "v_peak": -15,

    # IP3 & ER
    "alpha_ip3_post": 5.0, 
    "tau_ip3": 6.0,
    "ca_store_max": 8.0,
    "alpha_release": 3.5,
    "tau_store_refill": 40.0,
    
    # IP3R dynamics parameters
    "ip3r_ka": 0.02,      # Ca concentration for half-maximal activation (µM)
    "ip3r_ki": 1.0,       # Ca concentration for half-maximal inactivation (µM)
    "ip3r_threshold": 0.3,  # IP3 threshold (30% of peak)
    "ip3r_n_act": 2,      #  activation
    "ip3r_n_inact": 2,    # # Conductances
    "g_ltcc": 0.0005,              # Spine LTCC conductance (µM/s per mV)
    "g_ltcc_dend": 0.0003,         # Dendritic LTCC conductance
        

    # Dendritic Ca2+
    "tau_ca_dend": 3.0,
    "sigma_dend": 0.04,

    # CaMKII dynamics (delayed & stochastic)
    "alpha_camkii": 1.5,
    "tau_camkii": 8.2,
    "camkii_theta": 0.2, 
    "camkii_delay": 10.0,
    "sigma_camkii": 0.08,
    "camkii_fire_prob": 0.7,  
    "camkii_refractory": 5.0,  
    "camkii_noise_level": 0.05,  

    # Eligibility trace (decaying)
    "tau_elig": 20.0,

    # Synaptic tag (phosphatase / PP1)
    "alpha_pp1": 1.0,
    "tau_pp1": 30.0,
    "pp1_baseline": 1.0,
    "sigma_pp1": 0.03,

    # BTSP timing kernel (Gaussian)
    "tau_btsp": 2.5,
    
    # Distance-based attenuation of Somatic Voltage
    "lambda_d": 200.0,  # Space constant in µm

    # LTCC parameter conductances
    "g_ltcc": 0.0005,              # Spine LTCC conductance (µM/s per mV)
    "g_ltcc_dend": 0.0003,         # Dendritic LTCC conductance
        
    # LTCC External calcium
    "ca_ext": 2000.0,              # External Ca concentration (µM)
        
    # LTCC Voltage parameters
    "ltcc_v_shift": 0.0,           # Voltage shift for gating (mV)
    "v_rest": -70.0,               # Resting potential (mV)
        
    # LTCC CDI parameters
    "ltcc_K_CDI": 1.0,             # Half-inactivation Ca for CDI (µM)
    "ltcc_tau_CDI": 50.0,          # CDI time constant (ms)
        
    # LTCC Spatial distribution
    "ltcc_density_profile": "exponential",  # exponential, gaussian, linear, uniform
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
