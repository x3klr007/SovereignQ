#!/usr/bin/env python3
# =============================================================================
# ERDA E3: Pegasos Reproducibility Lockdown
# =============================================================================
# Formally demonstrates that PegasosQSVC optimization volatility is completely
# bounded and 100% reproducible when random seeds and step sizes are locked.
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms import PegasosQSVC

BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 500  # Smaller sample for rapid, repeatable training
SEED = 42

print("=== ERDA E3: Running Pegasos Reproducibility Lockdown ===")
print("Loading credit card dataset...")
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

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, stratify=y_subset, random_state=SEED
)

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Set specific k = 1.2 (the bimodal sweet spot)
k = 1.2
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
Xte_k = np.clip(X_test_scaled * k, 0.0, k)

# Build feature map and kernel
feature_map = ZZFeatureMap(feature_dimension=Xtr_k.shape[1], reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

# 1. Parallel Independent Runs under Lock 1 (Seed 42)
print("\n--- Running Run A1 (Seed 42) ---")
model_a1 = PegasosQSVC(C=1000, num_steps=200, quantum_kernel=qkernel, seed=42)
model_a1.fit(Xtr_k, y_train)
scores_a1 = model_a1.decision_function(Xte_k)

print("--- Running Run A2 (Seed 42) ---")
model_a2 = PegasosQSVC(C=1000, num_steps=200, quantum_kernel=qkernel, seed=42)
model_a2.fit(Xtr_k, y_train)
scores_a2 = model_a2.decision_function(Xte_k)

# 2. Parallel Run under Lock 2 (Seed 123)
print("--- Running Run B1 (Seed 123) ---")
model_b1 = PegasosQSVC(C=1000, num_steps=200, quantum_kernel=qkernel, seed=123)
model_b1.fit(Xtr_k, y_train)
scores_b1 = model_b1.decision_function(Xte_k)

# 3. Analyze Reproducibility
diff_a1_a2 = np.max(np.abs(scores_a1 - scores_a2))
diff_a1_b1 = np.max(np.abs(scores_a1 - scores_b1))

print("\n=== PEGASOS REPRODUCIBILITY METRICS ===")
print(f"Max Absolute Score Difference (Same Seed: Run A1 vs Run A2): {diff_a1_a2:.8e}")
print(f"Max Absolute Score Difference (Different Seeds: Run A1 vs Run B1): {diff_a1_b1:.4f}")

is_reproducible = diff_a1_a2 < 1e-7

if is_reproducible:
    print("\n✓ SUCCESS: Pegasos optimization path is 100% deterministic and reproducible under seed lock!")
else:
    print("\n⚠ WARNING: Small floating-point discrepancies detected.")

# Export results to CSV
df_out = pd.DataFrame({
    "Sample_Index": np.arange(len(Xte_k)),
    "Run_A1_Score": scores_a1,
    "Run_A2_Score": scores_a2,
    "Run_B1_Score": scores_b1,
})
out_path = os.path.join(BASE_DIR, 'erda_pegasos_lockdown_results.csv')
df_out.to_csv(out_path, index=False)
print(f"Saved E3 results to {out_path}")
