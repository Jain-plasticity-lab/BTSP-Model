"""
Plotting Module
Handles all visualization and plotting functions for BTSP simulations
"""

### OFFSET = TIME BETWEEN PRE AND POST STIMULATION (+VE MEANS POST BEFORE PRE)

import numpy as np
import matplotlib.pyplot as plt
from Params import params
from BTSP_LTCC import run_btsp 


def plot_stochastic_camkii_simulations(n_sims=50, active_syn=None, offset=0.75):
    """
    Plot multiple simulations showing stochastic CaMKII activation
    
    Args:
        n_sims: Number of simulations to run
        active_syn: List of active synapse indices
        offset: Timing offset for BTSP
    """
    if active_syn is None:
        rng = np.random.default_rng(1234)
        active_syn = rng.choice(params["n_syn"], size=10, replace=False).tolist()
    
    print(f"Running {n_sims} simulations of stochastic CaMKII activation...")
    
    (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    camkii_fire_times = []
    
    for sim in range(n_sims):
        results = run_btsp(offset, active_syn, return_traces=True, seed=1234+sim)
        time, w_tr, Ca_sp, PP1, elig, CaMKII, Ca_dend, IP3, Ca_store, \
            distances, nmda, IP3R_prob, CaMKII_pot, CaMKII_fired, fire_time,Ca_ltcc_spine, Ca_ltcc_dend = results #Calling all possible relevant parameters to plot, all quantities not plotted in this code 
        
        # Plot CaMKII trace
        alpha = 0.3 if sim > 0 else 0.8
        linewidth = 2 if sim == 0 else 0.5
        color = 'blue' if sim == 0 else 'gray'
        ax1.plot(time, CaMKII, alpha=alpha, linewidth=linewidth, color=color)
        
        # Track fire times
        if fire_time is not None:
            camkii_fire_times.append(fire_time)
            ax1.axvline(fire_time, color='red', alpha=0.1, linewidth=0.5)
    
    # Plot potential CaMKII (deterministic)
    ax1.plot(time, CaMKII_pot, 'k--', label='Potential (no stochasticity)', linewidth=2)
    ax1.set_ylabel('CaMKII Activity', fontsize=12)
    ax1.set_title(f'Stochastic CaMKII Activation ({n_sims} simulations, p_fire={params["camkii_fire_prob"]})', 
                  fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of firing times
    ax2.hist(camkii_fire_times, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Distribution of CaMKII Activation Times (mean={np.mean(camkii_fire_times):.2f}s)', 
                  fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"CaMKII fired in {len(camkii_fire_times)}/{n_sims} simulations")
    print(f"Mean firing time: {np.mean(camkii_fire_times):.2f} ± {np.std(camkii_fire_times):.2f}s")
    
    return camkii_fire_times


def plot_weight_vs_camkii_timing(offsets=None, n_trials=100, active_syn=None):
    """
    Analyze and plot weight change vs CaMKII activation time for different offsets
    
    Args:
        offsets: Array of timing offsets to test
        n_trials: Number of trials per offset
        active_syn: List of active synapse indices
    """
    if offsets is None:
        offsets = np.linspace(-6, 6, 13)
    
    if active_syn is None:
        rng = np.random.default_rng(1234)
        active_syn = rng.choice(params["n_syn"], size=10, replace=False).tolist()
    
    print("\nAnalyzing weight change vs CaMKII activation time...")
    
    # Store results
    results = {offset: {'fire_times': [], 'weight_changes': []} for offset in offsets}
    
    for idx, off in enumerate(offsets):
        print(f"Offset {idx+1}/{len(offsets)}: {off:.2f}s")
        for trial in range(n_trials):
            dw, _, fire_time = run_btsp(off, active_syn, seed=1234+idx*100+trial)
            if fire_time is not None:
                results[off]['fire_times'].append(fire_time)
                results[off]['weight_changes'].append(dw[active_syn].mean())
    
    # Plot results
    axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Weight change vs firing time for each offset
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(offsets)))
    for idx, off in enumerate(offsets):
        if len(results[off]['fire_times']) > 0:
            ax.scatter(results[off]['fire_times'], results[off]['weight_changes'], 
                      alpha=0.6, s=50, c=[colors[idx]], label=f'{off:.1f}s')
    ax.set_xlabel('CaMKII Activation Time (s)', fontsize=12)
    ax.set_ylabel('Weight Change (Δw)', fontsize=12)
    ax.set_title('Weight Change vs CaMKII Activation Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Offset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Mean weight change vs offset
    ax = axes[0, 1]
    mean_dw = [np.mean(results[off]['weight_changes']) if len(results[off]['weight_changes']) > 0 else 0 
               for off in offsets]
    sem_dw = [np.std(results[off]['weight_changes'])/np.sqrt(len(results[off]['weight_changes'])) 
              if len(results[off]['weight_changes']) > 1 else 0 for off in offsets]
    ax.errorbar(offsets, mean_dw, yerr=sem_dw, fmt='o-', capsize=4, linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Post − Pre Timing (s)', fontsize=12)
    ax.set_ylabel('Mean Δw ± SEM', fontsize=12)
    ax.set_title('BTSP Timing Curve (Stochastic CaMKII)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean firing time vs offset
    ax = axes[1, 0]
    mean_fire_times = [np.mean(results[off]['fire_times']) if len(results[off]['fire_times']) > 0 else np.nan 
                       for off in offsets]
    std_fire_times = [np.std(results[off]['fire_times']) if len(results[off]['fire_times']) > 1 else 0 
                      for off in offsets]
    ax.errorbar(offsets, mean_fire_times, yerr=std_fire_times, fmt='s-', capsize=4, linewidth=2, color='red')
    ax.set_xlabel('Post − Pre Timing (s)', fontsize=12)
    ax.set_ylabel('Mean CaMKII Activation Time (s)', fontsize=12)
    ax.set_title('CaMKII Activation Time vs Offset', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Success rate vs offset
    ax = axes[1, 1]
    success_rates = [len(results[off]['fire_times']) / n_trials * 100 for off in offsets]
    ax.plot(offsets, success_rates, 'o-', linewidth=2, markersize=8, color='green')
    ax.axhline(params["camkii_fire_prob"]*100, color='gray', linestyle='--', 
               label=f'Expected ({params["camkii_fire_prob"]*100:.0f}%)', alpha=0.5)
    ax.set_xlabel('Post − Pre Timing (s)', fontsize=12)
    ax.set_ylabel('CaMKII Activation Success Rate (%)', fontsize=12)
    ax.set_title('Proportion of Successful CaMKII Activations', fontsize=14)
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nOverall statistics across all offsets:")
    all_fire_times = [ft for off in offsets for ft in results[off]['fire_times']]
    all_weight_changes = [wc for off in offsets for wc in results[off]['weight_changes']]
    print(f"Total successful activations: {len(all_fire_times)}/{len(offsets)*n_trials}")
    print(f"Mean firing time: {np.mean(all_fire_times):.2f} ± {np.std(all_fire_times):.2f}s")
    print(f"Mean weight change: {np.mean(all_weight_changes):.3f} ± {np.std(all_weight_changes):.3f}")
    
    return results


def plot_weight_change_vs_camkii_time(offsets=None, active_syn=None, n_trials=100):
    """
    Plot how weight change varies with CaMKII activation time for different offsets
    
    For each offset, bins trials by their CaMKII activation time and plots
    the mean weight change as a function of activation time.
    
    Args:
        offsets: Array of timing offsets to test
        active_syn: List of active synapse indices
        n_trials: Number of trials per offset
    """
    if offsets is None:
        offsets = np.linspace(-6, 6, 13)  #offsets are the time gaps between Pre and Post-synaptic stimulations
    
    if active_syn is None:
        rng = np.random.default_rng(1234)
        active_syn = rng.choice(params["n_syn"], size=10, replace=False).tolist()
    
    print("\nAnalyzing weight change as a function of CaMKII activation time...")
    
    # Store results
    results = {offset: {'fire_times': [], 'weight_changes': []} for offset in offsets}
    
    for idx, off in enumerate(offsets):
        print(f"Offset {idx+1}/{len(offsets)}: {off:.2f}s")
        for trial in range(n_trials):
            dw, _, fire_time = run_btsp(off, active_syn, seed=1234+idx*100+trial)
            if fire_time is not None:
                results[off]['fire_times'].append(fire_time)
                results[off]['weight_changes'].append(dw[active_syn].mean())
    
    # Create time bins (0-120 seconds)
    time_bins = np.linspace(0, 120, 25)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Plot
    ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(offsets)))
    
    for idx, off in enumerate(offsets):
        if len(results[off]['fire_times']) > 0:
            fire_times = np.array(results[off]['fire_times'])
            weight_changes = np.array(results[off]['weight_changes'])
            
            # Bin the data
            binned_weights = []
            binned_sems = []
            valid_centers = []
            
            for i in range(len(time_bins)-1):
                mask = (fire_times >= time_bins[i]) & (fire_times < time_bins[i+1])
                if np.sum(mask) > 0:
                    binned_weights.append(np.mean(weight_changes[mask]))
                    binned_sems.append(np.std(weight_changes[mask]) / np.sqrt(np.sum(mask)))
                    valid_centers.append(time_centers[i])
            
            if len(valid_centers) > 0:
                ax.errorbar(valid_centers, binned_weights, yerr=binned_sems,
                           fmt='o-', capsize=4, linewidth=2, markersize=6,
                           color=colors[idx], label=f'Offset = {off:.1f}s', alpha=0.8)
    
    ax.set_xlabel('CaMKII Activation Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Weight Change (Δw)', fontsize=14, fontweight='bold')
    ax.set_title('Weight Change vs CaMKII Activation Time Across Offsets', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nStatistics by offset:")
    for off in offsets:
        if len(results[off]['fire_times']) > 0:
            fire_times = np.array(results[off]['fire_times'])
            weight_changes = np.array(results[off]['weight_changes'])
            print(f"  Offset {off:.2f}s:")
            print(f"    Mean activation time: {np.mean(fire_times):.2f} ± {np.std(fire_times):.2f}s")
            print(f"    Mean weight change: {np.mean(weight_changes):.3f} ± {np.std(weight_changes):.3f}")
            print(f"    Success rate: {len(fire_times)}/{n_trials} ({100*len(fire_times)/n_trials:.1f}%)")
    
    return results


def plot_synaptic_traces(offset=2.0, active_syn=None):
    """
    Plot detailed synaptic traces for active vs inactive synapses
    
    Args:
        offset: Timing offset
        active_syn: List of active synapse indices
    """
    if active_syn is None:
        rng = np.random.default_rng(1234)
        active_syn = rng.choice(params["n_syn"], size=10, replace=False).tolist() #This allots the active synapses if there are none 
    
    print("Running trace simulation...")
    
    results = run_btsp(offset, active_syn, return_traces=True)
    time, w_tr, Ca_sp, PP1, elig, CaMKII, Ca_dend, IP3, Ca_store, \
        distances, nmda, IP3R_prob, CaMKII_pot, CaMKII_fired, fire_time,Ca_ltcc_spine, Ca_ltcc_dend = results #Calling all possible relevant parameters to plot, all quantities not plotted in this code 
    
    inactive_syn = [0]
    
    # Plot 1: Spine Ca, PP1, and Eligibility
    ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    
    #Spine Calcium
    ax[0].plot(time, Ca_sp[active_syn[0]], label=f"Active (d={distances[active_syn[0]]:.1f}µm)", linewidth=2)
    ax[0].set_ylabel("Spine Ca²⁺")
    ax[0].legend()
    
    #Phosphotase
    ax[1].plot(time, PP1[active_syn[0]], label="Active", linewidth=2)
    ax[1].plot(time, PP1[inactive_syn[0]], color='gray', alpha=0.7, label="Inactive")
    ax[1].set_ylabel("PP1")
    ax[1].legend()
    
    #Eligibility
    ax[2].plot(time, elig[active_syn[0]], label="Active", linewidth=2)
    ax[2].plot(time, elig[inactive_syn[0]], color='gray', alpha=0.7, label="Inactive")
    ax[2].set_ylabel("Eligibility")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Global signals (CaMKII, Ca_dend, IP3, Ca_store)
    ax1 = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    
    #CaMKII and Dendritic Calcium
    ax1[0].plot(time, CaMKII, label="CaMKII", linewidth=2)
    ax1[0].plot(time, Ca_dend, color='gray', label="Dendritic Ca", linewidth=2)
    ax1[0].set_ylabel("Dendritic Ca²⁺ vs CaMKII")
    ax1[0].legend()
    
    #IP3
    ax1[1].plot(time, IP3, label="IP3", linewidth=2)
    ax1[1].axhline(y=max(IP3)*0.3, color='red', linestyle='--', label='30% threshold', alpha=0.5)
    ax1[1].set_ylabel("IP3")
    ax1[1].legend()
    
    #Internal Store Calcium
    ax1[2].plot(time, Ca_store, label="Ca Store", linewidth=2)
    ax1[2].set_ylabel("Ca from internal stores")
    ax1[2].set_xlabel("Time (s)")
    ax1[2].legend()
    
    plt.tight_layout()
    plt.show()