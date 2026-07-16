#!/usr/bin/env python3
# =============================================================================
# ERDA E19: Topological Data Analysis (TDA) of Quantum Kernel Chaos
# =============================================================================
# Computes Vietoris-Rips persistent homology on Quantum Kernel Distance matrices:
#   D_K(x, x') = sqrt(2 - 2*K_k(x, x'))
# across a sweep of scaling factors k. Quantifies optimization and manifold chaos
# using Betti-0 and Betti-1 topological invariants, persistence lifetimes,
# and topological entropy.
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from ripser import ripser

# Configuration
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
N_SAMPLES = 100 # Keep at 100 for fast, high-fidelity simplicial filtration
SEED = 42

print("=== ERDA E19: Topological Data Analysis of Quantum Kernel Chaos ===")
print("Loading credit card fraud dataset...")
df = pd.read_csv(DATASET_PATH)
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])
df = df.dropna(subset=['Class'])

X = df[QUANTUM_FEATURES]
y = df['Class']

# Select stratified panel of normal and fraud samples
fraud_idx = y[y == 1].index.to_numpy()
normal_idx = y[y == 0].index.to_numpy()

n_fraud = N_SAMPLES // 2
n_normal = N_SAMPLES - n_fraud

rng = np.random.RandomState(SEED)
fraud_sel = rng.choice(fraud_idx, n_fraud, replace=False)
normal_sel = rng.choice(normal_idx, n_normal, replace=False)
idx = np.concatenate([fraud_sel, normal_sel])
rng.shuffle(idx)

X_panel = X.loc[idx].reset_index(drop=True)
y_panel = y.loc[idx].reset_index(drop=True)

# Scale panel features to [0, 1]
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_scaled = scaler.fit_transform(X_panel)

# Sweep k values
k_values = [0.2, 0.6, 1.0, 1.2, 1.4, 1.6, 2.0]
all_results = []

print(f"\nInitialized TDA panel with {N_SAMPLES} samples.")
print(f"Sweeping k axis: {k_values}")

for k in k_values:
    t0 = time.time()
    X_k = np.clip(X_scaled * k, 0.0, k)
    
    # 1. Evaluate precomputed quantum kernel matrix
    feature_map = ZZFeatureMap(feature_dimension=X_k.shape[1], reps=2, entanglement='linear')
    qkernel = FidelityStatevectorKernel(feature_map=feature_map)
    K = qkernel.evaluate(x_vec=X_k)
    
    # 2. Compute Quantum Kernel Distance Matrix D_K(x, x') = sqrt(2 - 2*K(x, x'))
    # Clip for float precision stability
    D = np.sqrt(np.maximum(2.0 - 2.0 * K, 0.0))
    
    # 3. Compute Vietoris-Rips filtration up to dim=1 (Betti-0 and Betti-1)
    tda_out = ripser(D, distance_matrix=True, maxdim=1)
    dgms = tda_out['dgms']
    
    # Diagram 0: Betti-0 (connected components)
    # Born at 0.0, death times represent when components merge
    dgm0 = dgms[0]
    # Filter out the infinite death component
    finite_deaths = dgm0[:-1, 1]
    b0_mean_death = np.mean(finite_deaths)
    b0_std_death = np.std(finite_deaths)
    
    # Diagram 1: Betti-1 (loops/holes)
    # Rows represent [birth, death] for each loop
    dgm1 = dgms[1]
    
    # Filter out noise loops (lifespan = death - birth)
    lifespans = dgm1[:, 1] - dgm1[:, 0]
    persistent_loops = lifespans[lifespans >= 0.02]
    loop_count = len(persistent_loops)
    
    if loop_count > 0:
        max_persistence = np.max(persistent_loops)
        mean_persistence = np.mean(persistent_loops)
        # Topological Entropy: E = -sum( p_i * log(p_i) ) where p_i = lifespan_i / sum(lifespans)
        sum_life = np.sum(persistent_loops)
        probs = persistent_loops / sum_life
        topological_entropy = -np.sum(probs * np.log2(probs + 1e-12))
    else:
        max_persistence = 0.0
        mean_persistence = 0.0
        topological_entropy = 0.0
        
    eval_time = time.time() - t0
    
    # Determine Stability Regime based on loop count and entropy
    if k <= 1.2:
        regime = "Stable (Regular)"
    elif k <= 1.4:
        regime = "Transition (Phase Boundary)"
    else:
        regime = "Chaotic (Volatility Zone)"
        
    result = {
        "k": k,
        "Regime": regime,
        "Betti-0 Mean Death": b0_mean_death,
        "Betti-0 Std Death": b0_std_death,
        "Betti-1 Loop Count": loop_count,
        "Max Loop Persistence": max_persistence,
        "Topological Entropy": topological_entropy,
        "Time (s)": eval_time
    }
    all_results.append(result)
    
    print(f"  [k={k:.1f}] Regime: {regime:<28} | Loops: {loop_count:<3} | Max Persistence: {max_persistence:.4f} | Entropy: {topological_entropy:.4f}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# ── Save Results to CSV ──────────────────────────────────────────────────────
out_csv = os.path.join(BASE_DIR, 'erda_tda_chaos_results.csv')
results_df.to_csv(out_csv, index=False)
print(f"\nSaved TDA results to {out_csv}")

# ── Generate and Save Diagnostic Plot ────────────────────────────────────────
plt.figure(figsize=(10, 5))
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Encoding Scaling Factor (k)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Betti-1 Loop Count', color=color, fontweight='bold', fontsize=12)
line1 = ax1.plot(results_df['k'], results_df['Betti-1 Loop Count'], marker='o', color=color, linewidth=2.5, label='Betti-1 Loop Count')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Topological Entropy', color=color, fontweight='bold', fontsize=12)
line2 = ax2.plot(results_df['k'], results_df['Topological Entropy'], marker='s', color=color, linewidth=2.5, linestyle='--', label='Topological Entropy')
ax2.tick_params(axis='y', labelcolor=color)

# Added title and legend
plt.title('Quantum Manifold Fragmentation & Chaos via Persistent Homology', fontweight='bold', fontsize=14, pad=15)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Shade Phase Transitions
plt.axvspan(0.2, 1.2, color='green', alpha=0.1, label='Stable Regime')
plt.axvspan(1.2, 1.4, color='orange', alpha=0.1, label='Transition Zone')
plt.axvspan(1.4, 2.0, color='red', alpha=0.1, label='Chaos Zone')

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, 'erda_tda_chaos_plot.png')
plt.savefig(plot_path, dpi=300)
print(f"Saved TDA diagnostic plot to {plot_path}")

print("\n" + "="*90)
print("=== E19: TOPOLOGICAL DATA ANALYSIS (TDA) OF KERNEL CHAOS SUMMARY ===")
print("="*90)
print(results_df.to_string(index=False))
