#!/usr/bin/env python3
# =============================================================================
# ERDA: Quantum vs. Classical Kernel Alignment Metric (N9)
# =============================================================================
# Computes the optimal Frobenius-inner-product kernel alignment between the 
# quantum Gram matrix K_Q(k) and the best-fitting classical RBF Gram matrix,
# proving the non-classicality of the QSVM decision space.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize_scalar
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 200
SEED = 42

print("Loading dataset for Kernel Alignment evaluation...")
df = pd.read_csv(DATASET_PATH)
X = df[QUANTUM_FEATURES]
y = df['Class']

fraud_idx = y[y == 1].index.to_numpy()
normal_idx = y[y == 0].index.to_numpy()

n_fraud = min(len(fraud_idx), MAX_SAMPLES // 3)
n_normal = MAX_SAMPLES - n_fraud

rng = np.random.RandomState(SEED)
fraud_sel = rng.choice(fraud_idx, n_fraud, replace=False)
normal_sel = rng.choice(normal_idx, n_normal, replace=False)
idx = np.concatenate([fraud_sel, normal_sel])
rng.shuffle(idx)

X_subset = X.loc[idx].reset_index(drop=True)
y_subset = y.loc[idx].reset_index(drop=True)

X_train, _, _, _ = train_test_split(
    X_subset, y_subset, test_size=0.2, stratify=y_subset, random_state=SEED
)

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_scaled = scaler.fit_transform(X_train)

# Pairwise squared Euclidean distances
X_diff = X_train_scaled[:, np.newaxis, :] - X_train_scaled[np.newaxis, :, :]
D2 = np.sum(X_diff**2, axis=-1)

k_values = np.linspace(0.6, 2.0, 8)
max_alignments = []
optimal_gammas = []

print("\nComputing optimal kernel alignment across k...")

for k in k_values:
    # Scale inputs
    Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
    
    # Compute Quantum Kernel
    feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
    qkernel = FidelityStatevectorKernel(feature_map=feature_map)
    K_Q = qkernel.evaluate(x_vec=Xtr_k)
    
    # Find best-fitting classical RBF kernel by maximizing alignment
    # Alignment(K_1, K_2) = Tr(K_1 * K_2) / (||K_1||_F * ||K_2||_F)
    # To minimize with minimize_scalar, we minimize -Alignment
    
    def neg_alignment(gamma):
        K_RBF = np.exp(-gamma * D2)
        
        # Frobenius inner product and norms
        inner_prod = np.sum(K_Q * K_RBF)
        norm_q = np.sqrt(np.sum(K_Q**2))
        norm_rbf = np.sqrt(np.sum(K_RBF**2))
        
        return -inner_prod / (norm_q * norm_rbf + 1e-15)
        
    res = minimize_scalar(neg_alignment, bounds=(0.01, 100.0), method='bounded')
    best_align = -res.fun
    best_gamma = res.x
    
    max_alignments.append(best_align)
    optimal_gammas.append(best_gamma)
    
    print(f"k = {k:.2f} | Best Classical Alignment = {best_align:.5f} (with gamma = {best_gamma:.2f})")

# Plotting Kernel Alignment and Optimal Gamma
fig, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)

color = '#1ABC9C'
ax1.set_xlabel('Bandwidth Scaling Parameter $k$', fontsize=10)
ax1.set_ylabel('Maximum Classical Kernel Alignment', color=color, fontweight='bold')
line1 = ax1.plot(k_values, max_alignments, marker='o', color=color, linewidth=2.5, markersize=8, label='Max RBF Alignment')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.0, 1.05)
ax1.grid(True, linestyle="--", alpha=0.25)

ax2 = ax1.twinx()
color = '#9B59B6'
ax2.set_ylabel('Optimal RBF Parameter $\\gamma^*$', color=color, fontweight='bold')
line2 = ax2.plot(k_values, optimal_gammas, marker='^', color=color, linestyle='--', linewidth=1.8, markersize=8, label='Optimal $\\gamma^*$')
ax2.tick_params(axis='y', labelcolor=color)

# Adding annotations for non-classicality
# High alignment means the representation is highly classical. Low alignment means high quantum-native uniqueness!
ax1.axvspan(1.1, 1.4, color="#3498DB", alpha=0.1, label="High Classical Alignment Zone")
ax1.axvspan(1.5, 2.0, color="#E74C3C", alpha=0.1, label="Quantum Non-Classical Deviation Zone")

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', fontsize=9.5, frameon=True, facecolor="white", edgecolor="#E2E2E2")

plt.title("Quantum vs. Classical Kernel Alignment & Non-Classicality Sweep (N9)\nFrobenius Alignment between $K_{Q}(k)$ and optimal $K_{RBF}(\\gamma)$ | creditcard.csv subset", fontsize=11, fontweight='bold', pad=15)

out_path = os.path.join(BASE_DIR, 'erda_kernel_alignment.png')
plt.savefig(out_path, bbox_inches="tight")
print(f"\nSuccessfully generated Kernel Alignment plot and saved to {out_path}")
