#!/usr/bin/env python3
# =============================================================================
# ERDA: Kernel PGD Adversarial Training on QSVM (N5)
# =============================================================================
# Implements Kernel PGD Adversarial Training (Kernel PGD-AT) to harden the
# precomputed QSVM decision boundary and evaluates robustness gains under attack.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 200
SEED = 42

print("Loading dataset for Kernel Adversarial Training...")
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

# Set k = 1.2 (Geometric Honesty Zone)
k = 1.2
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
Xte_k = np.clip(X_test_scaled * k, 0.0, k)

# Build feature map and quantum kernel
feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

print("\nComputing base clean quantum kernel matrix...")
K_train_clean = qkernel.evaluate(x_vec=Xtr_k)
K_test_clean = qkernel.evaluate(x_vec=Xte_k, y_vec=Xtr_k)

# Fit standard clean QSVM
print("Fitting standard clean QSVM...")
svc_clean = SVC(kernel='precomputed', C=1.0)
svc_clean.fit(K_train_clean, y_train)

# Calculate clean F1 and predictions
y_pred_clean = svc_clean.predict(K_test_clean)
f1_clean = f1_score(y_test, y_pred_clean, zero_division=0)
print(f"Clean QSVM F1 Score = {f1_clean:.4f}")

# --- PGD Adversarial Perturbation ---
print("\nRunning PGD on training samples to synthesize adversarial training set...")
# Decision function: f(x) = sum_i alpha_i * y_i * K(x_i, x) + b
# We want to perform 5 steps of PGD with epsilon = 0.05, step size eta = 0.01

epsilon = 0.05
eta = 0.01
n_steps = 5
delta_fd = 1e-4  # finite difference step

Xtr_adv = Xtr_k.copy()
y_train_signed = np.where(y_train == 1, 1, -1)

# Helper to compute decision function f(x) for arbitrary samples
def get_decision_scores(samples, X_train_ref, svc_model):
    K_ref = qkernel.evaluate(x_vec=samples, y_vec=X_train_ref)
    return svc_model.decision_function(K_ref)

for step in range(n_steps):
    print(f"  PGD Step {step+1}/{n_steps}...")
    
    # Estimate finite difference gradients of decision function with respect to input features
    grads = np.zeros_like(Xtr_adv)
    
    # Base decision scores
    base_scores = get_decision_scores(Xtr_adv, Xtr_k, svc_clean)
    
    for feature_idx in range(4):
        # Perturb +delta
        X_perturbed_plus = Xtr_adv.copy()
        X_perturbed_plus[:, feature_idx] += delta_fd
        scores_plus = get_decision_scores(X_perturbed_plus, Xtr_k, svc_clean)
        
        # Perturb -delta
        X_perturbed_minus = Xtr_adv.copy()
        X_perturbed_minus[:, feature_idx] -= delta_fd
        scores_minus = get_decision_scores(X_perturbed_minus, Xtr_k, svc_clean)
        
        grads[:, feature_idx] = (scores_plus - scores_minus) / (2 * delta_fd)
        
    # Adversarial step to maximize loss (i.e. move in opposite direction of y * gradient)
    for i in range(len(Xtr_adv)):
        y_val = y_train_signed[i]
        direction = -y_val * np.sign(grads[i])
        Xtr_adv[i] += eta * direction
        
        # Clip to epsilon ball around original training samples and clip to boundary [0, k]
        Xtr_adv[i] = np.clip(Xtr_adv[i], Xtr_k[i] - epsilon, Xtr_k[i] + epsilon)
        Xtr_adv[i] = np.clip(Xtr_adv[i], 0.0, k)

print("\nComputing robust (adversarially perturbed) training kernel matrix...")
K_train_robust = qkernel.evaluate(x_vec=Xtr_adv)
K_test_robust = qkernel.evaluate(x_vec=Xte_k, y_vec=Xtr_adv)

print("Fitting Adversarially Trained QSVM (PGD-AT)...")
svc_robust = SVC(kernel='precomputed', C=1.0)
svc_robust.fit(K_train_robust, y_train)

# Calculate robust F1 on clean test data
y_pred_robust = svc_robust.predict(K_test_robust)
f1_robust = f1_score(y_test, y_pred_robust, zero_division=0)
print(f"Robust QSVM Clean F1 Score = {f1_robust:.4f}")

# --- Evaluate Evasion Rates ---
# We run a fast black-box evasion sweep using random search on a subset of test fraud samples
print("\nEvaluating evasion rate on test fraud samples under clean vs robust models...")
test_fraud_idx = np.where(y_test == 1)[0]
test_fraud_samples = Xte_k[test_fraud_idx]

def evaluate_attack(svc_model, X_train_ref, test_samples):
    evaded_count = 0
    total = len(test_samples)
    
    # We do a simple random search attack (20 queries, epsilon=0.08)
    for x_orig in test_samples:
        clean_pred = svc_model.predict(qkernel.evaluate(x_vec=x_orig.reshape(1, -1), y_vec=X_train_ref))[0]
        if clean_pred == 0:
            continue  # already classified as normal
            
        success = False
        for _ in range(20):
            perturbation = np.random.uniform(-0.08, 0.08, size=4)
            x_adv = np.clip(x_orig + perturbation, 0.0, k)
            pred = svc_model.predict(qkernel.evaluate(x_vec=x_adv.reshape(1, -1), y_vec=X_train_ref))[0]
            if pred == 0:
                success = True
                break
        if success:
            evaded_count += 1
            
    return (evaded_count / total) * 100

evaded_clean = evaluate_attack(svc_clean, Xtr_k, test_fraud_samples)
evaded_robust = evaluate_attack(svc_robust, Xtr_adv, test_fraud_samples)

print(f"Clean QSVM Evasion Rate = {evaded_clean:.1f}%")
print(f"Robust (PGD-AT) QSVM Evasion Rate = {evaded_robust:.1f}%")

# Plotting the Evasion Rates and F1 comparison
fig, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)

labels = ['Standard (Clean)', 'Hardened (PGD-AT)']
evasion_rates = [evaded_clean, evaded_robust]
f1_scores = [f1_clean, f1_robust]

x = np.arange(len(labels))
width = 0.35

rects1 = ax1.bar(x - width/2, evasion_rates, width, label='Adversarial Evasion Rate (%)', color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Evasion Rate (%)', color='#E74C3C', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#E74C3C')
ax1.set_ylim(0, 110)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, [f * 100 for f in f1_scores], width, label='Clean Test F1 (%)', color='#3498DB', alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Clean Test F1 (%)', color='#3498DB', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#3498DB')
ax2.set_ylim(0, 110)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax1.set_title("Kernel PGD Adversarial Training Robustness Comparison (N5)\nQSVM at $k=1.2$ | 5-Step PGD Training | Random Evasion Sweep Evaluation", fontsize=11, fontweight='bold', pad=12)
ax1.grid(True, linestyle="--", alpha=0.2, axis='y')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

out_path = os.path.join(BASE_DIR, 'erda_kernel_adversarial_training.png')
plt.savefig(out_path, bbox_inches="tight")
print(f"Successfully generated Kernel Adversarial Training plot and saved to {out_path}")
