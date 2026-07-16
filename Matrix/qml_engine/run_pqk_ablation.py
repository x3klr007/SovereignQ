#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector, Operator

# Configuration
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 200
TEST_SIZE = 0.3
SEEDS = [42, 101, 2023, 7, 88] # 5 random seeds for statistical stability
NUM_SNAPSHOTS = 100            # Number of random measurements for classical shadows
GAMMA_PQK = 2.0                # RBF kernel scale on projected features

print("Loading dataset for the rigorous 4-condition PQK + Shadows Ablation Study...")
df = pd.read_csv(DATASET_PATH)
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])
df = df.dropna(subset=['Class'])

X = df[QUANTUM_FEATURES]
y = df['Class']

# Balanced subset to eliminate class bias
fraud_idx = y[y == 1].index.to_numpy()
normal_idx = y[y == 0].index.to_numpy()

# Classical Shadow State Estimator
class ClassicalShadowEstimator:
    def __init__(self, num_qubits, num_snapshots=100):
        self.num_qubits = num_qubits
        self.num_snapshots = num_snapshots
        self.simulator = AerSimulator()
        
    def generate_shadows_batched(self, state_circuits, seed):
        """
        Generates randomized Pauli (X, Y, Z) measurement shadow snapshots for all state circuits in a single batch.
        """
        rng = np.random.RandomState(seed)
        all_circuits = []
        metadata = []
        
        for idx, state_circuit in enumerate(state_circuits):
            for _ in range(self.num_snapshots):
                qc = state_circuit.copy()
                rotations = rng.randint(0, 3, self.num_qubits)
                for qubit, rot in enumerate(rotations):
                    if rot == 0:    # X basis
                        qc.h(qubit)
                    elif rot == 1:  # Y basis
                        qc.sdg(qubit)
                        qc.h(qubit)
                qc.measure_all()
                all_circuits.append(qc)
                metadata.append((idx, rotations))
                
        result = self.simulator.run(all_circuits, shots=1, seed_simulator=seed).result()
        counts_list = result.get_counts()
        
        if not isinstance(counts_list, list):
            counts_list = [counts_list]
            
        shadows_per_circuit = [[] for _ in range(len(state_circuits))]
        for i, (idx, rotations) in enumerate(metadata):
            bitstring = list(counts_list[i].keys())[0]
            shadows_per_circuit[idx].append((rotations, bitstring))
            
        return shadows_per_circuit

# Shadow expectations for Pauli operators
def estimate_pauli_expectations(shadow, num_qubits):
    """
    Reconstructs the expectation values <X_j>, <Y_j>, <Z_j> from classical shadows.
    """
    X_exp = np.zeros(num_qubits)
    Y_exp = np.zeros(num_qubits)
    Z_exp = np.zeros(num_qubits)
    
    for j in range(num_qubits):
        x_vals, y_vals, z_vals = [], [], []
        for rotations, bitstring in shadow:
            val = 1 if bitstring[num_qubits - 1 - j] == '0' else -1
            if rotations[j] == 0:    # X measurement
                x_vals.append(3.0 * val)
            elif rotations[j] == 1:  # Y measurement
                y_vals.append(3.0 * val)
            elif rotations[j] == 2:  # Z measurement
                z_vals.append(3.0 * val)
                
        X_exp[j] = np.mean(x_vals) if x_vals else 0.0
        Y_exp[j] = np.mean(y_vals) if y_vals else 0.0
        Z_exp[j] = np.mean(z_vals) if z_vals else 0.0
        
    return X_exp, Y_exp, Z_exp

# Exact statevector expectation values
def get_exact_z_expectations(state_vector, num_qubits):
    z_exp = np.zeros(num_qubits)
    for j in range(num_qubits):
        # Construct single-qubit Z operator on qubit j
        op_list = ['I'] * num_qubits
        op_list[j] = 'Z'
        z_op = Operator.from_label(''.join(reversed(op_list)))
        z_exp[j] = np.real(state_vector.expectation_value(z_op))
    return z_exp

# Zero-Order Boundary Estimator
def get_boundary_distance(predict_fn, x, bounds=(0.0, 1.2), max_steps=12, seed=42):
    x_orig = x.copy()
    y_orig = predict_fn(x_orig.reshape(1, -1))[0]
    
    rng_dir = np.random.RandomState(seed)
    directions = rng_dir.normal(size=(5, len(x)))
    min_dist = float('inf')
    
    for d in directions:
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-5:
            continue
        d /= d_norm
        
        low = 0.0
        high = 0.8
        step_best = float('inf')
        for _ in range(max_steps):
            mid = (low + high) / 2
            x_cand = np.clip(x_orig + d * mid, bounds[0], bounds[1])
            y_cand = predict_fn(x_cand.reshape(1, -1))[0]
            if y_cand != y_orig:
                step_best = mid
                high = mid
            else:
                low = mid
        min_dist = min(min_dist, step_best)
        
    return min_dist if min_dist != float('inf') else 0.4

# Main simulation loop
results = {
    'Fidelity_Exact': {'F1': [], 'Margin': []},
    'PQK_Exact': {'F1': [], 'Margin': []},
    'Fidelity_Shadows': {'F1': [], 'Margin': []},
    'PQK_Shadows': {'F1': [], 'Margin': []}
}

for seed in SEEDS:
    print(f"\n--- Running sweep for Seed {seed} ---")
    
    # Stratified split per seed
    rng = np.random.RandomState(seed)
    n_fraud = min(len(fraud_idx), MAX_SAMPLES // 2)
    n_normal = MAX_SAMPLES - n_fraud
    
    fraud_sel = rng.choice(fraud_idx, n_fraud, replace=False)
    normal_sel = rng.choice(normal_idx, n_normal, replace=False)
    idx = np.concatenate([fraud_sel, normal_sel])
    rng.shuffle(idx)
    
    X_subset = X.loc[idx].reset_index(drop=True)
    y_subset = y.loc[idx].reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=TEST_SIZE, stratify=y_subset, random_state=seed
    )
    
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    
    n_qubits = X_train_scaled.shape[1]
    
    # 1. Generate Qiskit Statevectors for exact math
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    transpiled_feature_map = transpile(feature_map, AerSimulator())
    
    train_statevectors = [Statevector.from_instruction(transpiled_feature_map.assign_parameters(x * 1.2)) for x in X_train_scaled]
    test_statevectors = [Statevector.from_instruction(transpiled_feature_map.assign_parameters(x * 1.2)) for x in X_test_scaled]
    
    # 2. Generate Classical Shadows for statistical projection
    shadow_estimator = ClassicalShadowEstimator(num_qubits=n_qubits, num_snapshots=NUM_SNAPSHOTS)
    train_circuits = [transpiled_feature_map.assign_parameters(x * 1.2) for x in X_train_scaled]
    test_circuits = [transpiled_feature_map.assign_parameters(x * 1.2) for x in X_test_scaled]
    
    shadows_all = shadow_estimator.generate_shadows_batched(train_circuits + test_circuits, seed=seed)
    train_shadows = shadows_all[:len(X_train_scaled)]
    test_shadows = shadows_all[len(X_train_scaled):]
    
    # Compute representations for all four quadrants
    
    # A) QUADRANT 1: Fidelity (Exact)
    print("Quadrant 1: Evaluating Exact Fidelity Kernel...")
    global_kernel = FidelityStatevectorKernel(feature_map=feature_map)
    K_train_exact_fid = global_kernel.evaluate(x_vec=X_train_scaled * 1.2)
    K_test_exact_fid = global_kernel.evaluate(x_vec=X_test_scaled * 1.2, y_vec=X_train_scaled * 1.2)
    
    clf_exact_fid = SVC(kernel='precomputed', C=1.0)
    clf_exact_fid.fit(K_train_exact_fid, y_train)
    y_pred = clf_exact_fid.predict(K_test_exact_fid)
    results['Fidelity_Exact']['F1'].append(f1_score(y_test, y_pred) * 100)
    
    # B) QUADRANT 2: PQK (Exact Expectations)
    print("Quadrant 2: Evaluating Exact PQK Kernel...")
    train_exact_pqk_feats = np.array([get_exact_z_expectations(sv, n_qubits) for sv in train_statevectors])
    test_exact_pqk_feats = np.array([get_exact_z_expectations(sv, n_qubits) for sv in test_statevectors])
    
    K_train_exact_pqk = np.zeros((len(X_train_scaled), len(X_train_scaled)))
    for i in range(len(X_train_scaled)):
        for j in range(len(X_train_scaled)):
            diff = train_exact_pqk_feats[i] - train_exact_pqk_feats[j]
            K_train_exact_pqk[i, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
            
    K_test_exact_pqk = np.zeros((len(X_test_scaled), len(X_train_scaled)))
    for i in range(len(X_test_scaled)):
        for j in range(len(X_train_scaled)):
            diff = test_exact_pqk_feats[i] - train_exact_pqk_feats[j]
            K_test_exact_pqk[i, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
            
    clf_exact_pqk = SVC(kernel='precomputed', C=1.0)
    clf_exact_pqk.fit(K_train_exact_pqk, y_train)
    y_pred = clf_exact_pqk.predict(K_test_exact_pqk)
    results['PQK_Exact']['F1'].append(f1_score(y_test, y_pred) * 100)
    
    # C) QUADRANT 3: Fidelity via Shadows (Reconstructed Overlaps)
    print("Quadrant 3: Evaluating Shadows-Reconstructed Fidelity Overlaps...")
    # Extract 1-qubit expectations for shadows
    train_shadow_exp = [estimate_pauli_expectations(sh, n_qubits) for sh in train_shadows]
    test_shadow_exp = [estimate_pauli_expectations(sh, n_qubits) for sh in test_shadows]
    
    # overlap metric: product_m 1/2(1 + <X_i><X_j> + <Y_i><Y_j> + <Z_i><Z_j>)
    def compute_shadow_overlap_matrix(exp_a, exp_b):
        K = np.zeros((len(exp_a), len(exp_b)))
        for i in range(len(exp_a)):
            xa, ya, za = exp_a[i]
            for j in range(len(exp_b)):
                xb, yb, zb = exp_b[j]
                overlap = 1.0
                for q in range(n_qubits):
                    q_over = 0.5 * (1.0 + xa[q]*xb[q] + ya[q]*yb[q] + za[q]*zb[q])
                    overlap *= np.clip(q_over, 0.0, 1.0)
                K[i, j] = overlap
        return K
        
    K_train_shadow_fid = compute_shadow_overlap_matrix(train_shadow_exp, train_shadow_exp)
    K_test_shadow_fid = compute_shadow_overlap_matrix(test_shadow_exp, train_shadow_exp)
    
    clf_shadow_fid = SVC(kernel='precomputed', C=1.0)
    clf_shadow_fid.fit(K_train_shadow_fid, y_train)
    y_pred = clf_shadow_fid.predict(K_test_shadow_fid)
    results['Fidelity_Shadows']['F1'].append(f1_score(y_test, y_pred) * 100)
    
    # D) QUADRANT 4: PQK via Shadows (Our baseline upgrade)
    print("Quadrant 4: Evaluating PQK via Shadows...")
    train_shadow_pqk_feats = np.array([exp[2] for exp in train_shadow_exp]) # Exp of Z is the 3rd term (index 2)
    test_shadow_pqk_feats = np.array([exp[2] for exp in test_shadow_exp])
    
    K_train_shadow_pqk = np.zeros((len(X_train_scaled), len(X_train_scaled)))
    for i in range(len(X_train_scaled)):
        for j in range(len(X_train_scaled)):
            diff = train_shadow_pqk_feats[i] - train_shadow_pqk_feats[j]
            K_train_shadow_pqk[i, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
            
    K_test_shadow_pqk = np.zeros((len(X_test_scaled), len(X_train_scaled)))
    for i in range(len(X_test_scaled)):
        for j in range(len(X_train_scaled)):
            diff = test_shadow_pqk_feats[i] - train_shadow_pqk_feats[j]
            K_test_shadow_pqk[i, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
            
    clf_shadow_pqk = SVC(kernel='precomputed', C=1.0)
    clf_shadow_pqk.fit(K_train_shadow_pqk, y_train)
    y_pred = clf_shadow_pqk.predict(K_test_shadow_pqk)
    results['PQK_Shadows']['F1'].append(f1_score(y_test, y_pred) * 100)
    
    # Evasion Robustness Boundary Probing for 5 fraud samples per seed
    print("Running Zero-Order Decision Boundary probing across all 4 quadrants...")
    fraud_indices = np.where(y_test == 1)[0][:5]
    
    d_fid_exact, d_pqk_exact, d_fid_shadow, d_pqk_shadow = [], [], [], []
    
    # Exact Fidelity wrapper
    def pred_fid_exact_wrap(x_single):
        K_val = global_kernel.evaluate(x_vec=x_single, y_vec=X_train_scaled * 1.2)
        return clf_exact_fid.predict(K_val)
        
    # Exact PQK wrapper
    def pred_pqk_exact_wrap(x_single):
        sv = Statevector.from_instruction(transpiled_feature_map.assign_parameters(x_single[0]))
        f_vec = get_exact_z_expectations(sv, n_qubits)
        K_val = np.zeros((1, len(X_train_scaled)))
        for j in range(len(X_train_scaled)):
            diff = f_vec - train_exact_pqk_feats[j]
            K_val[0, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
        return clf_exact_pqk.predict(K_val)
        
    # Shadows Fidelity wrapper
    def pred_fid_shadow_wrap(x_single):
        circ = transpiled_feature_map.assign_parameters(x_single[0])
        sh = shadow_estimator.generate_shadows_batched([circ], seed=seed)[0]
        single_exp = [estimate_pauli_expectations(sh, n_qubits)]
        K_val = compute_shadow_overlap_matrix(single_exp, train_shadow_exp)
        return clf_shadow_fid.predict(K_val)
        
    # Shadows PQK wrapper
    def pred_pqk_shadow_wrap(x_single):
        circ = transpiled_feature_map.assign_parameters(x_single[0])
        sh = shadow_estimator.generate_shadows_batched([circ], seed=seed)[0]
        f_vec = estimate_pauli_expectations(sh, n_qubits)[2]
        K_val = np.zeros((1, len(X_train_scaled)))
        for j in range(len(X_train_scaled)):
            diff = f_vec - train_shadow_pqk_feats[j]
            K_val[0, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
        return clf_shadow_pqk.predict(K_val)
        
    for idx_f in fraud_indices:
        x_samp = X_test_scaled[idx_f]
        d_fid_exact.append(get_boundary_distance(pred_fid_exact_wrap, x_samp * 1.2, bounds=(0.0, 1.2), seed=seed))
        d_pqk_exact.append(get_boundary_distance(pred_pqk_exact_wrap, x_samp * 1.2, bounds=(0.0, 1.2), seed=seed))
        d_fid_shadow.append(get_boundary_distance(pred_fid_shadow_wrap, x_samp * 1.2, bounds=(0.0, 1.2), seed=seed))
        d_pqk_shadow.append(get_boundary_distance(pred_pqk_shadow_wrap, x_samp * 1.2, bounds=(0.0, 1.2), seed=seed))
        
    results['Fidelity_Exact']['Margin'].append(np.median(d_fid_exact))
    results['PQK_Exact']['Margin'].append(np.median(d_pqk_exact))
    results['Fidelity_Shadows']['Margin'].append(np.median(d_fid_shadow))
    results['PQK_Shadows']['Margin'].append(np.median(d_pqk_shadow))

print("\n--- Sweeping scaling factor k to map kernel concentration variance statistical envelope ---")
k_sweep = np.linspace(0.5, 2.0, 4)
# Save variance matrices for each seed to plot shaded standard-error region
global_variances = {k: [] for k in k_sweep}
pqk_exact_variances = {k: [] for k in k_sweep}
global_shadow_variances = {k: [] for k in k_sweep}
pqk_shadow_variances = {k: [] for k in k_sweep}

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    sub_idx = rng.choice(len(X_train_scaled), 20, replace=False)
    sub_data = X_train_scaled[sub_idx]
    
    # Shadows setup for this subset
    shadow_estimator_sweep = ClassicalShadowEstimator(num_qubits=n_qubits, num_snapshots=NUM_SNAPSHOTS)
    
    for k_val in k_sweep:
        # A) Global Exact
        K_glob = global_kernel.evaluate(x_vec=sub_data * k_val)
        global_variances[k_val].append(np.std(K_glob))
        
        # B) PQK Exact
        exact_feats = []
        for x in sub_data:
            sv = Statevector.from_instruction(transpiled_feature_map.assign_parameters(x * k_val))
            exact_feats.append(get_exact_z_expectations(sv, n_qubits))
        exact_feats = np.array(exact_feats)
        K_pqk_exact = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                diff = exact_feats[i] - exact_feats[j]
                K_pqk_exact[i, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
        pqk_exact_variances[k_val].append(np.std(K_pqk_exact))
        
        # C) Shadows Global & PQK
        sweep_circuits = [transpiled_feature_map.assign_parameters(x * k_val) for x in sub_data]
        shs = shadow_estimator_sweep.generate_shadows_batched(sweep_circuits, seed=seed)
        sh_exps = [estimate_pauli_expectations(sh, n_qubits) for sh in shs]
        
        # Shadows Global
        K_glob_sh = compute_shadow_overlap_matrix(sh_exps, sh_exps)
        global_shadow_variances[k_val].append(np.std(K_glob_sh))
        
        # Shadows PQK
        sh_pqk_feats = np.array([exp[2] for exp in sh_exps])
        K_pqk_sh = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                diff = sh_pqk_feats[i] - sh_pqk_feats[j]
                K_pqk_sh[i, j] = np.exp(-GAMMA_PQK * np.sum(diff**2))
        pqk_shadow_variances[k_val].append(np.std(K_pqk_sh))

# Aggregate statistical outcomes
stats = {}
for qd in ['Fidelity_Exact', 'PQK_Exact', 'Fidelity_Shadows', 'PQK_Shadows']:
    stats[qd] = {
        'F1_mean': np.mean(results[qd]['F1']),
        'F1_std': np.std(results[qd]['F1']),
        'Margin_mean': np.mean(results[qd]['Margin']),
        'Margin_std': np.std(results[qd]['Margin'])
    }

print("\n=========================================================================")
print("                      rigorous 4-condition ablation results               ")
print("=========================================================================")
print(f"| {'Representation / Quadrant':<30} | {'F1 Score (%)':<16} | {'Median L2 Margin':<16} |")
print("|" + "-"*32 + "|" + "-"*18 + "|" + "-"*18 + "|")
for qd in ['Fidelity_Exact', 'PQK_Exact', 'Fidelity_Shadows', 'PQK_Shadows']:
    name = qd.replace('_', ' ')
    f1_str = f"{stats[qd]['F1_mean']:.2f} ± {stats[qd]['F1_std']:.2f}"
    marg_str = f"{stats[qd]['Margin_mean']:.4f} ± {stats[qd]['Margin_std']:.4f}"
    print(f"| {name:<30} | {f1_str:<16} | {marg_str:<16} |")
print("=========================================================================")

# Plotting with statistical confidence bands and error bars
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

labels = ['Global Fid.\n(Exact)', 'PQK\n(Exact)', 'Global Fid.\n(Shadows)', 'PQK\n(Shadows)']
quadrants = ['Fidelity_Exact', 'PQK_Exact', 'Fidelity_Shadows', 'PQK_Shadows']

f1_means = [stats[q]['F1_mean'] for q in quadrants]
f1_stds = [stats[q]['F1_std'] for q in quadrants]

margin_means = [stats[q]['Margin_mean'] for q in quadrants]
margin_stds = [stats[q]['Margin_std'] for q in quadrants]

# Panel 1: F1 utility comparison
ax1.bar(labels, f1_means, yerr=f1_stds, color=['#E74C3C', '#2ECC71', '#9B59B6', '#1ABC9C'], width=0.45, alpha=0.85, capsize=8, error_kw={'elinewidth':2, 'capthick':2})
ax1.set_ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 110)
ax1.set_title("Classification F1 Utility Sweep (n=5 Seeds)", fontsize=12, fontweight="bold", pad=12)
ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

# Panel 2: Zero-order margin comparison
ax2.bar(labels, margin_means, yerr=margin_stds, color=['#E74C3C', '#2ECC71', '#9B59B6', '#1ABC9C'], width=0.45, alpha=0.85, capsize=8, error_kw={'elinewidth':2, 'capthick':2})
ax2.set_ylabel("Estimated Boundary Distance (L2)", fontsize=11, fontweight="bold")
ax2.set_ylim(0, max(margin_means)*1.4)
ax2.set_title("Zero-Order Certified Boundary Margin (n=5 Seeds)", fontsize=12, fontweight="bold", pad=12)
ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

# Panel 3: Kernel concentration standard deviation sweep
def get_envelope(variances_dict):
    means = [np.mean(variances_dict[k]) for k in k_sweep]
    stds = [np.std(variances_dict[k]) for k in k_sweep]
    return np.array(means), np.array(stds)

g_m, g_s = get_envelope(global_variances)
pq_m, pq_s = get_envelope(pqk_exact_variances)
g_sh_m, g_sh_s = get_envelope(global_shadow_variances)
pq_sh_m, pq_sh_s = get_envelope(pqk_shadow_variances)

# Plotting means and standard error envelopes
ax3.plot(k_sweep, g_m, marker="o", color="#E74C3C", linewidth=2.5, label="Global Fid. (Exact)")
ax3.fill_between(k_sweep, g_m-g_s, g_m+g_s, color="#E74C3C", alpha=0.15)

ax3.plot(k_sweep, pq_m, marker="s", color="#2ECC71", linewidth=2.5, label="PQK (Exact)")
ax3.fill_between(k_sweep, pq_m-pq_s, pq_m+pq_s, color="#2ECC71", alpha=0.15)

ax3.plot(k_sweep, g_sh_m, marker="^", linestyle="--", color="#9B59B6", linewidth=2.0, label="Global Fid. (Shadows)")
ax3.fill_between(k_sweep, g_sh_m-g_sh_s, g_sh_m+g_sh_s, color="#9B59B6", alpha=0.10)

ax3.plot(k_sweep, pq_sh_m, marker="v", linestyle="--", color="#1ABC9C", linewidth=2.0, label="PQK (Shadows)")
ax3.fill_between(k_sweep, pq_sh_m-pq_sh_s, pq_sh_m+pq_sh_s, color="#1ABC9C", alpha=0.10)

ax3.set_xlabel("Encoding scaling parameter (k)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Kernel Standard Deviation", fontsize=11, fontweight="bold")
ax3.set_title("Kernel Variance Envelopes across k-sweep", fontsize=12, fontweight="bold", pad=12)
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(fontsize=8, loc="upper right")

plt.suptitle("Causally Disentangled QML Ablation Matrix: Projected Quantum Kernels vs. Classical Shadows\nCredit Card Fraud Benchmark | 4-Qubit ZZFeatureMap | Statistical Stability Sweep (n=5 Seeds)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "erda_pqk_ablation_results.png")
plt.savefig(out_path, dpi=300)
brain_path = "/home/x3klr007/.gemini/antigravity/brain/e34b53b2-fc57-4d3d-9afe-eeb213434b46/erda_pqk_ablation_results.png"
plt.savefig(brain_path, dpi=300)

print(f"\nPlots generated and saved to:")
print(f"1. {out_path}")
print(f"2. {brain_path}")
