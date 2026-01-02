"""
Main Execution Script
Run this script to execute the full BTSP simulation and generate plots
"""

import numpy as np
from Params import params
from Plots import (
    plot_stochastic_camkii_simulations,
    plot_weight_vs_camkii_timing,
    plot_weight_change_vs_camkii_time,
    plot_synaptic_traces
)


def main():
    """
    Main execution function - runs all simulations and generates plots
    """
    # Set a random seed
    rng = np.random.default_rng(1234)
    
    # Select the active synapses
    active_syn = rng.choice(params["n_syn"], size=10, replace=False).tolist() #Chooses 10 random synapses to be active synapses out of the entire pool
    
    # 1. Plot stochastic CaMKII activation
    print("\n" + "="*70)
    print("SECTION 1: Stochastic CaMKII Activation")
    print("="*70)
    plot_stochastic_camkii_simulations(
        n_sims=100,
        active_syn=active_syn,
        offset=0.75 #Manually setting the offset(i.e the time gap between Pre and Post pairing)
    )
    
    # 2. Analyze weight change vs CaMKII timing
    print("\n" + "="*70)
    print("SECTION 2: Weight Change vs CaMKII Timing Analysis")
    print("="*70)
    offsets = np.linspace(-6, 6, 13) # Choosing the following offsets (-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6)
    plot_weight_vs_camkii_timing(
        offsets=offsets,
        n_trials=100,
        active_syn=active_syn
    )
    
    # 3. Plot weight change as a function of CaMKII activation time
    print("\n" + "="*70)
    print("SECTION 3: Weight Change vs CaMKII Activation Time (0-120s)")
    print("="*70)
    offsets_subset = np.linspace(-6, 6, 7)
    plot_weight_change_vs_camkii_time(
        offsets=offsets_subset,
        active_syn=active_syn,
        n_trials=100
    )
    
    # 4. Plot detailed synaptic traces
    print("\n" + "="*70)
    print("SECTION 4: Detailed Synaptic Traces")
    print("="*70)
    plot_synaptic_traces(
        offset=2.0,
        active_syn=active_syn
    )
    
    print("\n" + "="*70)
    print("All simulations completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main() #Calls the main program which starts the cascade