#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from exact_boundary_distance import ExactBoundaryDistanceEstimator

# Configuration
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 1000
TEST_SIZE = 0.25
SEED = 42
PANEL_SAMPLES = 50
N_RESAMPLES = 1000

print("Loading dataset for Bootstrap CI Violins...")
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=['Class'])
X = df[QUANTUM_FEATURES]
y = df['Class']

fraud_idx = y[y == 1].index.to_numpy()
normal_idx = y[y == 0].index.to_numpy()
n_fraud = min(len(fraud_idx), MAX_SAMPLES // 3)
n_normal = MAX_SAMPLES - n_fraud

rng = np.random.RandomState(SEED)
idx = np.concatenate([
    rng.choice(fraud_idx, n_fraud, replace=False),
    rng.choice(normal_idx, n_normal, replace=False)
])
rng.shuffle(idx)

X_subset = X.loc[idx].reset_index(drop=True)
y_subset = y.loc[idx].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=TEST_SIZE, stratify=y_subset, random_state=SEED
)

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Select evaluation panel
X_eval = X_test_scaled[:PANEL_SAMPLES]

# 1. Evaluate Classical RBF SVM (gamma=3.0)
print("Evaluating RBF SVM (gamma=3.0)...")
rbf_model = SVC(kernel='rbf', gamma=3.0, C=1.0)
rbf_model.fit(X_train_scaled, y_train)
rbf_estimator = ExactBoundaryDistanceEstimator(rbf_model.decision_function, bounds=(0.0, 1.0), max_bsearch_steps=12)

rbf_dists = []
for x in X_eval:
    d = rbf_estimator.measure_distance(x)
    if np.isfinite(d):
        rbf_dists.append(d)
        
# 2. Evaluate QSVM (k=1.2)
print("Evaluating QSVM (k=1.2)...")
k1 = 1.2
Xtr_k1 = X_train_scaled * k1
Xeval_k1 = X_eval * k1

feature_map1 = ZZFeatureMap(feature_dimension=Xtr_k1.shape[1], reps=2, entanglement='linear')
qkernel1 = FidelityStatevectorKernel(feature_map=feature_map1)
K_train1 = qkernel1.evaluate(x_vec=Xtr_k1)

qsvm_svc1 = SVC(kernel='precomputed', C=1.0)
qsvm_svc1.fit(K_train1, y_train)

# Hyper-optimized precomputed QSVM helper using support vectors
class PrecomputedQSVMHelper:
    def __init__(self, qkernel, svc, X_train_k):
        self.qkernel = qkernel
        self.support_idx = svc.support_
        self.support_vectors = X_train_k[self.support_idx]
        self.dual_coef = svc.dual_coef_[0]
        self.intercept = svc.intercept_[0]

    def decision_function(self, X):
        if len(X.shape) == 1:
            X_2d = X.reshape(1, -1)
        else:
            X_2d = X
        K_sv = self.qkernel.evaluate(x_vec=X_2d, y_vec=self.support_vectors)
        return np.dot(K_sv, self.dual_coef) + self.intercept

helper1 = PrecomputedQSVMHelper(qkernel1, qsvm_svc1, Xtr_k1)
q1_estimator = ExactBoundaryDistanceEstimator(helper1.decision_function, bounds=(0.0, k1), max_bsearch_steps=12)

q1_dists = []
for x in Xeval_k1:
    d = q1_estimator.measure_distance(x)
    if np.isfinite(d):
        q1_dists.append(d / k1) # Normalize to unit domain

# 3. Evaluate QSVM (k=1.4)
print("Evaluating QSVM (k=1.4)...")
k2 = 1.4
Xtr_k2 = X_train_scaled * k2
Xeval_k2 = X_eval * k2

feature_map2 = ZZFeatureMap(feature_dimension=Xtr_k2.shape[1], reps=2, entanglement='linear')
qkernel2 = FidelityStatevectorKernel(feature_map=feature_map2)
K_train2 = qkernel2.evaluate(x_vec=Xtr_k2)

qsvm_svc2 = SVC(kernel='precomputed', C=1.0)
qsvm_svc2.fit(K_train2, y_train)

helper2 = PrecomputedQSVMHelper(qkernel2, qsvm_svc2, Xtr_k2)
q2_estimator = ExactBoundaryDistanceEstimator(helper2.decision_function, bounds=(0.0, k2), max_bsearch_steps=12)

q2_dists = []
for x in Xeval_k2:
    d = q2_estimator.measure_distance(x)
    if np.isfinite(d):
        q2_dists.append(d / k2) # Normalize to unit domain

# Generate bootstrap median distributions
def get_bootstrap_medians(data, n_resamples=N_RESAMPLES):
    rng_bs = np.random.RandomState(SEED)
    medians = []
    n = len(data)
    for _ in range(n_resamples):
        sample = rng_bs.choice(data, size=n, replace=True)
        medians.append(np.median(sample))
    return medians

print("Running bootstrap resampling...")
rbf_bs_medians = get_bootstrap_medians(rbf_dists)
q1_bs_medians = get_bootstrap_medians(q1_dists)
q2_bs_medians = get_bootstrap_medians(q2_dists)

# Save summary metrics
rbf_low, rbf_high = np.percentile(rbf_bs_medians, [2.5, 97.5])
q1_low, q1_high = np.percentile(q1_bs_medians, [2.5, 97.5])
q2_low, q2_high = np.percentile(q2_bs_medians, [2.5, 97.5])

print(f"RBF SVM (gamma=3.0) 95% CI: [{rbf_low:.4f}, {rbf_high:.4f}] (Median = {np.median(rbf_dists):.4f})")
print(f"QSVM (k=1.2) 95% CI: [{q1_low:.4f}, {q1_high:.4f}] (Median = {np.median(q1_dists):.4f})")
print(f"QSVM (k=1.4) 95% CI: [{q2_low:.4f}, {q2_high:.4f}] (Median = {np.median(q2_dists):.4f})")

# Plotting violin plots
fig, ax = plt.subplots(figsize=(8.5, 5.5))

color_r = "#E74C3C" # Red
color_q1 = "#3498DB" # Blue
color_q2 = "#2ECC71" # Green

data_to_plot = [rbf_bs_medians, q1_bs_medians, q2_bs_medians]
positions = [1, 2, 3]

parts = ax.violinplot(data_to_plot, positions, showmeans=False, showmedians=True, showextrema=True)

# Customize violins
colors = [color_r, color_q1, color_q2]
for idx_col, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[idx_col])
    pc.set_edgecolor("black")
    pc.set_alpha(0.7)

parts['cmedians'].set_color("black")
parts['cmedians'].set_linewidth(2.0)
parts['cmins'].set_color("black")
parts['cmaxes'].set_color("black")
parts['cbars'].set_color("black")

# Annotate Confidence Intervals
CIs = [(rbf_low, rbf_high), (q1_low, q1_high), (q2_low, q2_high)]
labels = [r"RBF SVM ($\gamma=3.0$)", r"QSVM ($k=1.2$)", r"QSVM ($k=1.4$)"]

for pos, (low, high), label in zip(positions, CIs, labels):
    ax.text(pos, high + 0.015, f"95% CI:\n[{low:.4f}, {high:.4f}]", 
            ha='center', va='bottom', fontsize=9, fontweight="bold", 
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontweight="bold", fontsize=11)
ax.set_ylabel("Bootstrap Median L2 Boundary Distance (Normalized)", fontweight="bold", fontsize=11)
ax.set_title("95% Bootstrap Confidence Intervals: Complete Margin Separation", fontweight="bold", fontsize=13, pad=15)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.set_ylim(-0.02, max([max(b) for b in data_to_plot]) * 1.25)

plt.tight_layout()

out_path = os.path.join(BASE_DIR, "erda_bootstrap_ci_comparison.png")
plt.savefig(out_path, dpi=300)
print(f"Successfully saved Bootstrap CI comparison plot to {out_path}")
