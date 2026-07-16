#!/usr/bin/env python3
# =============================================================================
# ERDA E16: Conformalized Quantum Fraud Predictor (CQFP)
# =============================================================================
# Implements Platt Calibration and Split Conformal Prediction (Angelopoulos-Bates style)
# on QSVM and Classical RBF SVM. Demonstrates mathematically guaranteed coverage
# (95% confidence) and shows that conformal sets gracefully expand to Human Review
# {0, 1} under adversarial attacks, providing a natural evasion detector.
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
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
ALPHA = 0.05  # 95% target coverage

print("=== ERDA E16: Conformalized Quantum Fraud Predictor (CQFP) ===")
print("Loading credit card fraud dataset...")
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

# Split into Train (60%), Calibration (20%), and Test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_subset, y_subset, test_size=0.20, stratify=y_subset, random_state=SEED
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=SEED
)

# Scale features
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_cal = y_cal.to_numpy()
y_test = y_test.to_numpy()

print(f"Dataset split: Train={len(X_train)} | Calibrate={len(X_cal)} | Test={len(X_test)}")

# ── Train Precomputed QSVM ──────────────────────────────────────────────────
k = 1.2
print(f"\nTraining Precomputed QSVM (k={k}) on Train Set...")
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
Xcal_k = np.clip(X_cal_scaled * k, 0.0, k)
Xte_k = np.clip(X_test_scaled * k, 0.0, k)

feature_map = ZZFeatureMap(feature_dimension=Xtr_k.shape[1], reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

# Precompute training kernel
K_train = qkernel.evaluate(x_vec=Xtr_k)
qsvm_svc = SVC(kernel='precomputed', C=1.0)
qsvm_svc.fit(K_train, y_train)

# Precompute calibration and test kernels
K_cal = qkernel.evaluate(x_vec=Xcal_k, y_vec=Xtr_k)
K_test = qkernel.evaluate(x_vec=Xte_k, y_vec=Xtr_k)

# ── Train Classical RBF SVM ──────────────────────────────────────────────────
print("Training Classical RBF SVM (gamma=3.0) on Train Set...")
rbf_svc = SVC(kernel='rbf', gamma=3.0, C=1.0)
rbf_svc.fit(X_train_scaled, y_train)

# ── Conformal Predictor Wrapper with Platt Calibration ────────────────────────

class ConformalPredictor:
    def __init__(self, svc_model, is_precomputed=False):
        self.svc = svc_model
        self.is_precomputed = is_precomputed
        self.platt = LogisticRegression(C=1.0)
        self.q_hat = None

    def fit_platt(self, X_cal_raw, K_cal_precomputed, y_cal_labels):
        # Obtain raw decision scores
        if self.is_precomputed:
            scores = self.svc.decision_function(K_cal_precomputed)
        else:
            scores = self.svc.decision_function(X_cal_raw)
        
        # Fit Platt Calibration: mapping raw score to class probability
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
        
        # conformity scores E_i = 1 - p(x_i)_{y_i}
        conformity_scores = []
        for i in range(n):
            true_label = y_cal_labels[i]
            conformity_scores.append(1.0 - p_cal[i, true_label])
        
        conformity_scores = np.array(conformity_scores)
        # Compute the adjusted quantile
        quantile_level = np.ceil((n + 1) * (1.0 - alpha)) / n
        self.q_hat = np.quantile(conformity_scores, min(quantile_level, 1.0))
        print(f"  Quantile level: {quantile_level:.4f} | Conformal Cutoff q_hat: {self.q_hat:.4f}")

    def predict_set(self, X_raw, K_precomputed):
        p_test = self.predict_proba(X_raw, K_precomputed)
        prediction_sets = []
        for i in range(len(p_test)):
            p = p_test[i]
            # S = { y in {0, 1} : 1 - p_y <= q_hat }
            pred_set = []
            if 1.0 - p[0] <= self.q_hat:
                pred_set.append(0)
            if 1.0 - p[1] <= self.q_hat:
                pred_set.append(1)
            prediction_sets.append(pred_set)
        return prediction_sets

# ── Run Calibration ─────────────────────────────────────────────────────────

print("\n--- Calibrating Platt & Conformal Predictors ---")
qsvm_cp = ConformalPredictor(qsvm_svc, is_precomputed=True)
qsvm_cp.fit_platt(None, K_cal, y_cal)
print("QSVM (k=1.2):")
qsvm_cp.calibrate_conformal(None, K_cal, y_cal)

rbf_cp = ConformalPredictor(rbf_svc, is_precomputed=False)
rbf_cp.fit_platt(X_cal_scaled, None, y_cal)
print("Classical RBF SVM (gamma=3.0):")
rbf_cp.calibrate_conformal(X_cal_scaled, None, y_cal)

# ── Evaluate on Test Set ────────────────────────────────────────────────────

def evaluate_conformal_model(cp_model, X_raw, K_precomputed, y_true):
    pred_sets = cp_model.predict_set(X_raw, K_precomputed)
    n_test = len(y_true)
    
    coverage = 0
    set_sizes = []
    set_types = {
        "Auto-Approve {0}": 0,
        "Auto-Block {1}": 0,
        "Human Review {0, 1}": 0,
        "Anomaly Alert {}": 0
    }
    
    for i in range(n_test):
        s = pred_sets[i]
        true_y = y_true[i]
        
        # Coverage check
        if true_y in s:
            coverage += 1
            
        set_sizes.append(len(s))
        
        # Categorize
        if s == [0]:
            set_types["Auto-Approve {0}"] += 1
        elif s == [1]:
            set_types["Auto-Block {1}"] += 1
        elif s == [0, 1] or s == [1, 0]:
            set_types["Human Review {0, 1}"] += 1
        elif len(s) == 0:
            set_types["Anomaly Alert {}"] += 1
            
    emp_coverage = (coverage / n_test) * 100
    avg_size = np.mean(set_sizes)
    
    results = {
        "Coverage %": emp_coverage,
        "Avg Set Size": avg_size,
        "Auto-Approve %": (set_types["Auto-Approve {0}"] / n_test) * 100,
        "Auto-Block %": (set_types["Auto-Block {1}"] / n_test) * 100,
        "Human Review %": (set_types["Human Review {0, 1}"] / n_test) * 100,
        "Anomaly Alert %": (set_types["Anomaly Alert {}"] / n_test) * 100
    }
    return results, pred_sets

print("\n--- Evaluating Test Set Performance (Clean) ---")
qsvm_clean_res, qsvm_clean_sets = evaluate_conformal_model(qsvm_cp, None, K_test, y_test)
rbf_clean_res, rbf_clean_sets = evaluate_conformal_model(rbf_cp, X_test_scaled, None, y_test)

print("\nQSVM (k=1.2) Clean Results:")
for k_res, v_res in qsvm_clean_res.items():
    print(f"  {k_res}: {v_res:.2f}")

print("\nClassical RBF SVM Clean Results:")
for k_res, v_res in rbf_clean_res.items():
    print(f"  {k_res}: {v_res:.2f}")

# ── Evaluate under Adversarial Evasion ────────────────────────────────────────

def fgsm_perturb(x, grad, eps=0.15, bounds=(0.0, 1.0)):
    # Standard FGSM perturbation step
    x_pert = x + eps * np.sign(grad)
    return np.clip(x_pert, bounds[0], bounds[1])

print("\n--- Simulating Adversarial Evasion Efficacy on Conformal Sets ---")
# We will perturb all fraud samples in the test set to trigger false negatives
fraud_test_indices = np.where(y_test == 1)[0]
n_adv = len(fraud_test_indices)

# We will generate a mock gradient perturbation on the scaled features
# to simulate an active evasive attacker pushing toward normal (reducing score).
# For evaluation, we apply a small evasion vector to the fraud samples.
# (Shift V10 towards normal, which generally has opposite signs)
X_test_adv = X_test_scaled.copy()
perturbation = np.zeros_like(X_test_scaled)
# V10 is the 1st feature (quantum V10, V4, V14, V12)
perturbation[:, 0] = -0.15 # push negative to trigger false negatives
perturbation[:, 2] = -0.15 # push V14 negative

X_test_adv[fraud_test_indices] = np.clip(
    X_test_scaled[fraud_test_indices] + perturbation[fraud_test_indices], 0.0, 1.0
)

# Evaluate conformal sets on attacked test set
Xte_k_adv = np.clip(X_test_adv * k, 0.0, k)
K_test_adv = qkernel.evaluate(x_vec=Xte_k_adv, y_vec=Xtr_k)

qsvm_adv_res, qsvm_adv_sets = evaluate_conformal_model(qsvm_cp, None, K_test_adv, y_test)
rbf_adv_res, rbf_adv_sets = evaluate_conformal_model(rbf_cp, X_test_adv, None, y_test)

print("\nQSVM (k=1.2) Adversarial Results:")
for k_res, v_res in qsvm_adv_res.items():
    print(f"  {k_res}: {v_res:.2f}")

print("\nClassical RBF SVM Adversarial Results:")
for k_res, v_res in rbf_adv_res.items():
    print(f"  {k_res}: {v_res:.2f}")

# ── Save Results ─────────────────────────────────────────────────────────────
all_results = [
    {
        "Model": "Classical RBF SVM (gamma=3.0)",
        "State": "Clean",
        "Coverage %": f"{rbf_clean_res['Coverage %']:.2f}%",
        "Avg Set Size": f"{rbf_clean_res['Avg Set Size']:.2f}",
        "Auto-Approve %": f"{rbf_clean_res['Auto-Approve %']:.1f}%",
        "Auto-Block %": f"{rbf_clean_res['Auto-Block %']:.1f}%",
        "Human Review %": f"{rbf_clean_res['Human Review %']:.1f}%",
        "Anomaly Alert %": f"{rbf_clean_res['Anomaly Alert %']:.1f}%"
    },
    {
        "Model": "Classical RBF SVM (gamma=3.0)",
        "State": "Attacked",
        "Coverage %": f"{rbf_adv_res['Coverage %']:.2f}%",
        "Avg Set Size": f"{rbf_adv_res['Avg Set Size']:.2f}",
        "Auto-Approve %": f"{rbf_adv_res['Auto-Approve %']:.1f}%",
        "Auto-Block %": f"{rbf_adv_res['Auto-Block %']:.1f}%",
        "Human Review %": f"{rbf_adv_res['Human Review %']:.1f}%",
        "Anomaly Alert %": f"{rbf_adv_res['Anomaly Alert %']:.1f}%"
    },
    {
        "Model": "Quantum QSVM (k=1.2)",
        "State": "Clean",
        "Coverage %": f"{qsvm_clean_res['Coverage %']:.2f}%",
        "Avg Set Size": f"{qsvm_clean_res['Avg Set Size']:.2f}",
        "Auto-Approve %": f"{qsvm_clean_res['Auto-Approve %']:.1f}%",
        "Auto-Block %": f"{qsvm_clean_res['Auto-Block %']:.1f}%",
        "Human Review %": f"{qsvm_clean_res['Human Review %']:.1f}%",
        "Anomaly Alert %": f"{qsvm_clean_res['Anomaly Alert %']:.1f}%"
    },
    {
        "Model": "Quantum QSVM (k=1.2)",
        "State": "Attacked",
        "Coverage %": f"{qsvm_adv_res['Coverage %']:.2f}%",
        "Avg Set Size": f"{qsvm_adv_res['Avg Set Size']:.2f}",
        "Auto-Approve %": f"{qsvm_adv_res['Auto-Approve %']:.1f}%",
        "Auto-Block %": f"{qsvm_adv_res['Auto-Block %']:.1f}%",
        "Human Review %": f"{qsvm_adv_res['Human Review %']:.1f}%",
        "Anomaly Alert %": f"{qsvm_adv_res['Anomaly Alert %']:.1f}%"
    }
]

results_df = pd.DataFrame(all_results)
out_path = os.path.join(BASE_DIR, "erda_e16_conformal_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved E16 conformal results to {out_path}")
print("\n" + "="*90)
print("=== E16: CONFORMALIZED QUANTUM FRAUD PREDICTOR (CQFP) SUMMARY ===")
print("="*90)
print(results_df.to_string(index=False))
