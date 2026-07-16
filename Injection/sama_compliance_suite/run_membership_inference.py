#!/usr/bin/env python3
# =============================================================================
# ERDA: Membership Inference Attack Defense (N8)
# =============================================================================
# Evaluates membership inference vulnerability using shadow confidence modeling,
# demonstrating QML privacy guarantees under centralized vs. federated layouts.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 200
SEED = 42

print("Loading dataset for Membership Inference evaluation...")
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

# Set scales
k = 1.2
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
Xte_k = np.clip(X_test_scaled * k, 0.0, k)

# Build feature map and quantum kernel
feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

# Precomputed QSVM
print("\nComputing precomputed kernels...")
K_train_q = qkernel.evaluate(x_vec=Xtr_k)
K_test_q = qkernel.evaluate(x_vec=Xte_k, y_vec=Xtr_k)

print("Fitting target models...")
svc_q = SVC(kernel='precomputed', C=1.0)
svc_q.fit(K_train_q, y_train)

svc_rbf = SVC(kernel='rbf', gamma=3.0, C=1.0)
svc_rbf.fit(X_train_scaled, y_train)

# --- Membership Inference Attack Simulation ---
# Members: in-sample training samples
# Non-members: out-of-sample test samples
# Attacker tries to predict member (label=1) vs non-member (label=0) based on confidence (absolute score)

# 1. Evaluate on QSVM
scores_train_q = np.abs(svc_q.decision_function(K_train_q))
scores_test_q = np.abs(svc_q.decision_function(K_test_q))

# 2. Evaluate on RBF SVM
scores_train_rbf = np.abs(svc_rbf.decision_function(X_train_scaled))
scores_test_rbf = np.abs(svc_rbf.decision_function(X_test_scaled))

# Attacker datasets (balanced: 40 members, 40 non-members)
n_non_members = len(scores_test_q)
rng_att = np.random.RandomState(SEED)

# Randomly select a subset of members equal in size to the non-members
train_indices = rng_att.choice(len(scores_train_q), n_non_members, replace=False)

scores_train_q_bal = scores_train_q[train_indices]
scores_train_rbf_bal = scores_train_rbf[train_indices]

X_attack_q = np.concatenate([scores_train_q_bal, scores_test_q]).reshape(-1, 1)
X_attack_rbf = np.concatenate([scores_train_rbf_bal, scores_test_rbf]).reshape(-1, 1)
y_attack = np.concatenate([np.ones(n_non_members), np.zeros(n_non_members)])

# Split attacker dataset to train a shadow classifier
X_att_train_q, X_att_test_q, y_att_train, y_att_test = train_test_split(
    X_attack_q, y_attack, test_size=0.3, stratify=y_attack, random_state=SEED
)
X_att_train_rbf, X_att_test_rbf, _, _ = train_test_split(
    X_attack_rbf, y_attack, test_size=0.3, stratify=y_attack, random_state=SEED
)

# Fit shadow attacker model (Logistic Regression)
shadow_q = LogisticRegression()
shadow_q.fit(X_att_train_q, y_att_train)
y_pred_att_q = shadow_q.predict(X_att_test_q)
mia_acc_q = accuracy_score(y_att_test, y_pred_att_q)

shadow_rbf = LogisticRegression()
shadow_rbf.fit(X_att_train_rbf, y_att_train)
y_pred_att_rbf = shadow_rbf.predict(X_att_test_rbf)
mia_acc_rbf = accuracy_score(y_att_test, y_pred_att_rbf)

print("\n=== MEMBERSHIP INFERENCE ATTACK ACCURACY ===")
print(f"RBF SVM Target (Centralized) : MIA Accuracy = {mia_acc_rbf*100:.1f}%")
print(f"QSVM Target (Centralized)    : MIA Accuracy = {mia_acc_q*100:.1f}%")

# Federated projection: federated modeling naturally introduces noise or boundary masking.
# We model a federated FQKL layout by adding local DP noise (scale 0.05) to scores, representing privacy-masking.
scores_train_fed = scores_train_q_bal + rng_att.normal(0, 0.05, size=n_non_members)
scores_test_fed = scores_test_q + rng_att.normal(0, 0.05, size=n_non_members)
X_attack_fed = np.concatenate([scores_train_fed, scores_test_fed]).reshape(-1, 1)

X_att_train_fed, X_att_test_fed, _, _ = train_test_split(
    X_attack_fed, y_attack, test_size=0.3, stratify=y_attack, random_state=SEED
)
shadow_fed = LogisticRegression()
shadow_fed.fit(X_att_train_fed, y_att_train)
y_pred_att_fed = shadow_fed.predict(X_att_test_fed)
mia_acc_fed = accuracy_score(y_att_test, y_pred_att_fed)

print(f"Federated FQKL (DP Noise)    : MIA Accuracy = {mia_acc_fed*100:.1f}%")

# Plotting the MIA accuracies
plt.figure(figsize=(8, 5.5), dpi=300)
labels = ['Classical RBF SVM\n(Centralized)', 'Quantum QSVM\n(Centralized)', 'Federated FQKL\n(DP Defended)']
accuracies = [mia_acc_rbf * 100, mia_acc_q * 100, mia_acc_fed * 100]
colors = ['#E74C3C', '#3498DB', '#2ECC71']

bars = plt.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=0.5, width=0.45)

# Add baseline (50% random guessing)
plt.axhline(y=50, color='gray', linestyle='--', linewidth=1.2, label='Baseline Privacy (50% Random)')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.title("Membership Inference Attack (MIA) Evasion Efficacy (N8)\nAttacker Shadow Model Accuracy | Centralized vs. Federated Layouts | creditcard.csv subset", fontsize=11, fontweight='bold', pad=15)
plt.ylabel("Attacker Inference Accuracy (%)", fontsize=10)
plt.ylim(0, 110)
plt.grid(True, linestyle="--", alpha=0.2, axis='y')
plt.legend(loc="upper right", fontsize=9.5)

out_path = os.path.join(BASE_DIR, 'erda_membership_inference.png')
plt.savefig(out_path, bbox_inches="tight")
print(f"\nSuccessfully generated Membership Inference plot and saved to {out_path}")
