#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

# Configuration
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 2500
SEED = 42
ALPHA = 0.05

print("Loading dataset for Conformal Prediction Histogram...")
df = pd.read_csv(DATASET_PATH)
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])
df = df.dropna(subset=['Class'])

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

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_subset, y_subset, test_size=0.20, stratify=y_subset, random_state=SEED
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=SEED
)

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_cal = y_cal.to_numpy()
y_test = y_test.to_numpy()

# Train QSVM Base
k = 1.2
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
Xcal_k = np.clip(X_cal_scaled * k, 0.0, k)
Xte_k = np.clip(X_test_scaled * k, 0.0, k)

feature_map = ZZFeatureMap(feature_dimension=Xtr_k.shape[1], reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

K_train = qkernel.evaluate(x_vec=Xtr_k)
qsvm_svc = SVC(kernel='precomputed', C=1.0)
qsvm_svc.fit(K_train, y_train)

K_cal = qkernel.evaluate(x_vec=Xcal_k, y_vec=Xtr_k)
K_test = qkernel.evaluate(x_vec=Xte_k, y_vec=Xtr_k)

# Train Classical RBF SVM
rbf_svc = SVC(kernel='rbf', gamma=3.0, C=1.0)
rbf_svc.fit(X_train_scaled, y_train)

# Conformal Wrapper
class ConformalPredictor:
    def __init__(self, svc_model, is_precomputed=False):
        self.svc = svc_model
        self.is_precomputed = is_precomputed
        self.platt = LogisticRegression(C=1.0)
        self.q_hat = None

    def fit_platt(self, X_cal_raw, K_cal_precomputed, y_cal_labels):
        if self.is_precomputed:
            scores = self.svc.decision_function(K_cal_precomputed)
        else:
            scores = self.svc.decision_function(X_cal_raw)
        self.platt.fit(scores.reshape(-1, 1), y_cal_labels)

    def predict_proba(self, X_raw, K_precomputed):
        if self.is_precomputed:
            scores = self.svc.decision_function(K_precomputed)
        else:
            scores = self.svc.decision_function(X_raw)
        return self.platt.predict_proba(scores.reshape(-1, 1))

    def calibrate_conformal(self, X_cal_raw, K_cal_precomputed, y_cal_labels, alpha=ALPHA):
        p_cal = self.predict_proba(X_cal_raw, K_cal_precomputed)
        n = len(y_cal_labels)
        conformity_scores = [1.0 - p_cal[i, y_cal_labels[i]] for i in range(n)]
        quantile_level = np.ceil((n + 1) * (1.0 - alpha)) / n
        self.q_hat = np.quantile(conformity_scores, min(quantile_level, 1.0))

    def predict_set(self, X_raw, K_precomputed):
        p_test = self.predict_proba(X_raw, K_precomputed)
        prediction_sets = []
        for i in range(len(p_test)):
            p = p_test[i]
            pred_set = []
            if 1.0 - p[0] <= self.q_hat:
                pred_set.append(0)
            if 1.0 - p[1] <= self.q_hat:
                pred_set.append(1)
            prediction_sets.append(pred_set)
        return prediction_sets

# Calibrate
qsvm_cp = ConformalPredictor(qsvm_svc, is_precomputed=True)
qsvm_cp.fit_platt(None, K_cal, y_cal)
qsvm_cp.calibrate_conformal(None, K_cal, y_cal)

rbf_cp = ConformalPredictor(rbf_svc, is_precomputed=False)
rbf_cp.fit_platt(X_cal_scaled, None, y_cal)
rbf_cp.calibrate_conformal(X_cal_scaled, None, y_cal)

# Generate adversarial examples
fraud_test_indices = np.where(y_test == 1)[0]
X_test_adv = X_test_scaled.copy()
perturbation = np.zeros_like(X_test_scaled)
perturbation[:, 0] = -0.15
perturbation[:, 2] = -0.15
X_test_adv[fraud_test_indices] = np.clip(
    X_test_scaled[fraud_test_indices] + perturbation[fraud_test_indices], 0.0, 1.0
)

Xte_k_adv = np.clip(X_test_adv * k, 0.0, k)
K_test_adv = qkernel.evaluate(x_vec=Xte_k_adv, y_vec=Xtr_k)

# Get set size distributions
def get_set_sizes(cp_model, X_raw, K_precomputed):
    pred_sets = cp_model.predict_set(X_raw, K_precomputed)
    return [len(s) for s in pred_sets]

qsvm_clean_sizes = get_set_sizes(qsvm_cp, None, K_test)
qsvm_adv_sizes = get_set_sizes(qsvm_cp, None, K_test_adv)
rbf_clean_sizes = get_set_sizes(rbf_cp, X_test_scaled, None)
rbf_adv_sizes = get_set_sizes(rbf_cp, X_test_adv, None)

# Plotting 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
axes = axes.flatten()

titles = [
    "QSVM Clean State", "QSVM Adversarial State",
    "RBF SVM Clean State", "RBF SVM Adversarial State"
]
datasets = [
    qsvm_clean_sizes, qsvm_adv_sizes,
    rbf_clean_sizes, rbf_adv_sizes
]

colors = ["#2E4053", "#B03A2E", "#2E4053", "#B03A2E"]

for i in range(4):
    ax = axes[i]
    data = datasets[i]
    
    # Calculate counts of 0, 1, 2 sizes
    counts = np.bincount(data, minlength=3)
    x_lbls = ["Anomaly Alert\nSize 0", "Singleton Pred\nSize 1", "Human Review\nSize 2"]
    
    bars = ax.bar(x_lbls, counts, color=colors[i], width=0.5, edgecolor="black", alpha=0.85)
    ax.set_title(titles[i], fontweight="bold", fontsize=12)
    ax.set_ylabel("Count of Test Samples", fontsize=10)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    
    # Add count values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight="bold")

plt.suptitle("CQFP: Split-Conformal Prediction Set-Size Distributions", fontweight="bold", fontsize=14, y=0.98)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "erda_conformal_prediction_set_sizes.png")
plt.savefig(out_path, dpi=300)
print(f"Successfully generated conformal set-size histograms and saved to {out_path}")
