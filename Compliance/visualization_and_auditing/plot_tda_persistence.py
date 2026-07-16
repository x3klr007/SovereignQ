#!/usr/bin/env python3
import os
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
N_SAMPLES = 100
SEED = 42

print("Generating TDA Persistence Diagrams...")
df = pd.read_csv(DATASET_PATH)
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])
df = df.dropna(subset=['Class'])

X = df[QUANTUM_FEATURES]
y = df['Class']

# Stratification panel
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
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_scaled = scaler.fit_transform(X_panel)

k_values = [0.6, 1.2, 1.6, 2.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Custom color scheme
color_b0 = "#3498DB" # Blue for Betti-0
color_b1 = "#E74C3C" # Red for Betti-1

for idx_k, k in enumerate(k_values):
    ax = axes[idx_k]
    X_k = np.clip(X_scaled * k, 0.0, k)
    
    # Evaluate quantum kernel
    feature_map = ZZFeatureMap(feature_dimension=X_k.shape[1], reps=2, entanglement='linear')
    qkernel = FidelityStatevectorKernel(feature_map=feature_map)
    K = qkernel.evaluate(x_vec=X_k)
    D = np.sqrt(np.maximum(2.0 - 2.0 * K, 0.0))
    
    # Ripser persistent homology
    tda_out = ripser(D, distance_matrix=True, maxdim=1)
    dgms = tda_out['dgms']
    dgm0 = dgms[0]
    dgm1 = dgms[1]
    
    # Set plot bounds based on distance matrix max
    max_val = np.max(D)
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], color="gray", linestyle="--", alpha=0.7)
    
    # Plot Betti-0
    # Betti-0 components are born at 0. Filter out the infinite death feature (represented as inf)
    b0_finite = dgm0[~np.isinf(dgm0[:, 1])]
    ax.scatter(b0_finite[:, 0], b0_finite[:, 1], color=color_b0, marker="o", s=35, alpha=0.7, label=r"$\beta_0$ (Components)")
    
    # Plot Betti-1
    if len(dgm1) > 0:
        ax.scatter(dgm1[:, 0], dgm1[:, 1], color=color_b1, marker="^", s=45, alpha=0.8, label=r"$\beta_1$ (Loops)")
        
    ax.set_title(f"Bandwidth $k = {k}$", fontweight="bold", fontsize=12)
    ax.set_xlim(-0.05, max_val * 1.1)
    ax.set_ylim(-0.05, max_val * 1.1)
    ax.set_xlabel("Birth", fontsize=10)
    ax.set_ylabel("Death", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    if idx_k == 0:
        ax.legend(loc="upper left")

plt.suptitle("Quantum Kernel Vietoris-Rips Persistence Homology Diagrams", fontweight="bold", fontsize=15, y=0.98)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "erda_tda_persistence_diagrams.png")
plt.savefig(out_path, dpi=300)
print(f"Successfully generated and saved persistence diagrams to {out_path}")
