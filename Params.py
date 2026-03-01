"""
BTSP Model Parameters
Contains all model parameters for the synaptic plasticity simulation.

Single-synapse version: n_syn=1, one active dendritic spine + one dendritic compartment.
LTCC parameters updated to match Mahajan & Nadkarni 2019 (Table 1).
"""

params = {
    # ── Synapse count ─────────────────────────────────────────────────────────
    "n_syn": 1,

    # ── Synapse geometry ─────────────────────────────────────────────────────
    # Single spine placed at 150 µm from soma (mid-apical dendrite, typical CA1)
    "syn_distance": 50.0,   # µm

    # ── Weight parameters ─────────────────────────────────────────────────────
    "w_init":         0.4,
    "w_min":          0.0,
    "w_max":          4.0,
    "eta":            0.62,   # learning rate (changed from 0.22, 26/1/26)
    "sigma_w":        0.02,
    "tau_forward":    0.69,
    "tau_backward":   1.31,

    # ── Spine Ca2+ (NMDAR) ────────────────────────────────────────────────────
    "alpha_nmda":     20.0, #from 10
    "tau_ca_spine":    0.2,
    "sigma_ca_spine":  0.05,
    "mg_block":        0.062,

    # ── Voltage ───────────────────────────────────────────────────────────────
    "v_rest":  -70,   # mV
    "v_peak":    0,   # mV — plateau potential seen by LTCCs (after fast Na+ spike)

    # ── IP3 & ER ──────────────────────────────────────────────────────────────
    "alpha_ip3_post":   5.0,
    "tau_ip3":        325.0,
    "ip3_quantum":  10000.0,
    "ca_store_max":     8.0,
    "alpha_release":    1.0,   # (12/2/26 from 0.01; 16/1/26 from 1.0)
    "tau_store_refill": 40.0,

    # IP3R dynamics
    "ip3r_ka":         0.02,   # uM  Ca for half-maximal activation
    "ip3r_ki":         1.0,    # uM  Ca for half-maximal inactivation
    "ip3r_threshold":  0.3,    # IP3 threshold (30% of peak)
    "ip3r_n_act":      2,
    "ip3r_n_inact":    2,

    # ── Dendritic Ca2+ ────────────────────────────────────────────────────────
    "tau_ca_dend":  10,   # s  (16/1/26 from 10)
    "sigma_dend":    0.04,

    # ── CaMKII dynamics ───────────────────────────────────────────────────────
    "alpha_camkii":        1.5,      # (26/2/26 from 2)
    "tau_camkii":          8.2,
    "camkii_theta":        0.3,
    "camkii_delay":       10.0,
    "sigma_camkii":        0.08,
    "camkii_fire_prob":    0.7,
    "camkii_refractory":   5.0,
    "camkii_noise_level":  0.05,

    # ── Eligibility trace ─────────────────────────────────────────────────────
    "tau_elig": 20.0,

    # ── PP1 ───────────────────────────────────────────────────────────────────
    "alpha_pp1":    1.0,
    "tau_pp1":     30.0,
    "pp1_baseline": 1.0,
    "sigma_pp1":    0.03,

    # ── BTSP timing kernel (Gaussian) ─────────────────────────────────────────
    "tau_btsp": 2.5,

    # ── Voltage attenuation ───────────────────────────────────────────────────
    "lambda_d": 200.0,   # dendritic space constant (um)

    # ── LTCC conductances ─────────────────────────────────────────────────────
    # BUGFIX: was 0.0005 / 0.0003 in earlier params (100x too small).
    "g_ltcc":      0.0015,    # spine LTCC  (uM ms-1 mV-1)
    "g_ltcc_dend": 0.0000043,    # dendritic LTCC

    # ── LTCC compartment geometry ─────────────────────────────────────────────
    "spine_shell_depth": 0.2e-6,   # m  (0.2 um submembrane shell, spine head)
    "dend_shell_depth":  1.0e-6,   # m  (1.0 um submembrane shell, dendrite)

    # ── LTCC external calcium ─────────────────────────────────────────────────
    "ca_ext":  2000.0,   # uM  (2 mM physiological extracellular Ca)
    "ca_rest":    0.1,   # uM  resting intracellular Ca

    # ── LTCC gating — Mahajan & Nadkarni 2019, Table 1 ───────────────────────
    # Activation gate m:    m_inf = 1 / (1 + exp(-(V - um) / km))
    "ltcc_um":    -20.0,   # mV  activation half-point
    "ltcc_km":      5.0,   # mV  activation slope
    "ltcc_tau_m":   0.08,  # ms  activation tau (instantaneous vs dt=10 ms)

    # Inactivation gate h:  h_inf = 1 / (1 + exp((V - uh) / kh))
    "ltcc_uh":   -65.0,   # mV  inactivation half-point
    "ltcc_kh":     7.0,   # mV  inactivation slope
    "ltcc_tau_h": 300.0,  # ms  inactivation tau (slow; dynamically meaningful)

    # ── Temperature / Q10 ─────────────────────────────────────────────────────
    # Q10 applied only to tau_h (tau_m is instantaneous, Q10 correction is moot).
    "temperature": 35.0,   # Celsius
    "ltcc_Q10":     3.0,   # temperature coefficient

    # ── LTCC channel density ──────────────────────────────────────────────────
    # Evaluated at syn_distance for single-synapse model.
    "ltcc_density_profile": "exponential",
    "ltcc_lambda":      100.0,   # um  exponential space constant
    "ltcc_min_density":   0.1,   # minimum relative density

    # ── LTCC noise ────────────────────────────────────────────────────────────
    "sigma_ltcc_spine": 0.001,   # uM/sqrt(s)
    "sigma_ltcc_dend":  0.0005,  # uM/sqrt(s)

    # ── Realistic somatic voltage model ───────────────────────────────────────
    "ltcc_theta_freq":   8.0,    # Hz   hippocampal theta frequency
    "ltcc_theta_amp":    6.0,    # mV   theta oscillation amplitude
    "ltcc_bg_rate":      5.0,    # Hz   background AP firing rate
    "ltcc_ap_tau_decay": 0.008,  # s    AP waveform decay time constant
    "ltcc_ap_tau_rise":  0.001,  # s    AP waveform rise time constant
}
