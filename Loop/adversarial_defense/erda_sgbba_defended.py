#!/usr/bin/env python3
# =============================================================================
# ERDA E12: True Score-Guided Black-Box Attack (SGBBA) and RLS Defense
# =============================================================================
# Evaluates a true iterative, score-guided black-box attack where perturbations
# are built incrementally from x_best. Demonstrates how Random Logit Scaling
# (RLS) breaks the optimization feedback loop.
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

# Configuration
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 3000
TEST_SIZE = 0.25
SEED = 42
ATTACK_SAMPLES = 50
EPSILON_BUDGET = 0.15  # A realistic 15% perturbation budget

print("=== ERDA E12: True Score-Guided Black-Box Attack (SGBBA) Sweep ===")
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

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=TEST_SIZE, stratify=y_subset, random_state=SEED
)

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

fraud_test_mask = (y_test == 1)
X_eval = X_test_scaled[fraud_test_mask][:ATTACK_SAMPLES]
y_eval = y_test[fraud_test_mask][:ATTACK_SAMPLES]

print(f"Evaluation panel size: {len(X_eval)} fraud samples")

# ── True Score-Guided Attack ──────────────────────────────────────────────

def score_guided_attack(decision_func, x, max_queries=100, eps=EPSILON_BUDGET, bounds=(0.0, 1.0)):
    """
    True score-guided black-box attack.
    Builds the perturbation incrementally by exploring the decision score landscape.
    """
    x_best = x.copy()
    f_best = decision_func(x_best.reshape(1, -1))[0]
    
    if f_best < 0:
        return x_best, 1
        
    d = x.shape[0]
    queries = 1
    
    rng_atk = np.random.RandomState(int(x[0]*1000) % 10000)
    step_size = eps / 3.0  # Incremental step size
    
    for q in range(max_queries):
        if f_best < 0:
            break
            
        # Take an incremental step relative to the current BEST candidate (x_best)
        direction = rng_atk.uniform(-1, 1, size=d)
        norm_val = np.linalg.norm(direction)
        if norm_val > 1e-5:
            direction /= norm_val
            
        x_cand = x_best + direction * step_size
        # Project back to L_infinity ball around original x and domain bounds
        x_cand = np.clip(x_cand, x - eps, x + eps)
        x_cand = np.clip(x_cand, bounds[0], bounds[1])
        
        f_cand = decision_func(x_cand.reshape(1, -1))[0]
        queries += 1
        
        if f_cand < f_best:
            f_best = f_cand
            x_best = x_cand
            
    return x_best, queries

# ── Model Wrappers ────────────────────────────────────────────────────────

class PrecomputedQSVMWrapper:
    def __init__(self, qkernel, svc, X_train_k):
        self.qkernel = qkernel
        self.svc = svc
        self.X_train_k = X_train_k

    def predict(self, X):
        K = self.qkernel.evaluate(x_vec=X, y_vec=self.X_train_k)
        return self.svc.predict(K)

    def decision_function(self, X):
        sv_idx = self.svc.support_
        X_sv = self.X_train_k[sv_idx]
        K_sv = self.qkernel.evaluate(x_vec=X, y_vec=X_sv)
        dual_coef = self.svc.dual_coef_[0]
        intercept = self.svc.intercept_[0]
        scores = np.dot(K_sv, dual_coef) + intercept
        return scores

class RandomLogitScalingWrapper:
    """
    Random Logit Scaling (RLS) Defense.
    Randomly scales the continuous score by a factor alpha ~ Uniform(0.7, 1.3)
    on each query. Sign is preserved, but score differentials are randomized.
    """
    def __init__(self, base_model, seed=42):
        self.base_model = base_model
        self.rng = np.random.RandomState(seed)

    def predict(self, X):
        return self.base_model.predict(X)

    def decision_function(self, X):
        scores = self.base_model.decision_function(X)
        scales = self.rng.uniform(0.7, 1.3, size=scores.shape)
        return scores * scales

# ── Train Models ──────────────────────────────────────────────────────────

print("\nTraining Classical RBF SVM (gamma=3.0) as reference...")
rbf_model = SVC(kernel='rbf', gamma=3.0, C=1.0)
rbf_model.fit(X_train_scaled, y_train)

k = 1.2
print(f"\nTraining Precomputed QSVM (k={k})...")
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)

feature_map = ZZFeatureMap(feature_dimension=Xtr_k.shape[1], reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

K_train = qkernel.evaluate(x_vec=Xtr_k)
qsvm_svc = SVC(kernel='precomputed', C=1.0)
qsvm_svc.fit(K_train, y_train)

qsvm_model = PrecomputedQSVMWrapper(qkernel, qsvm_svc, Xtr_k)

# ── Evaluations ───────────────────────────────────────────────────────────

models_to_test = [
    ("RBF SVM (gamma=3.0) [Standard]", rbf_model, 1.0, (0.0, 1.0)),
    ("QSVM (k=1.2) [Standard]", qsvm_model, k, (0.0, k)),
    ("QSVM (k=1.2) [RLS Defended]", RandomLogitScalingWrapper(qsvm_model), k, (0.0, k))
]

all_results = []

for model_name, model_obj, scale_factor, domain_bounds in models_to_test:
    print(f"\n{'='*60}")
    print(f"Evaluating SGBBA on {model_name}...")
    print(f"{'='*60}")
    
    success_count = 0
    l2_sum = 0.0
    queries_sum = 0
    
    t0 = time.time()
    for idx_s, x_sample in enumerate(X_eval):
        x_domain = x_sample * scale_factor
        
        f_init = model_obj.decision_function(x_domain.reshape(1, -1))[0]
        if f_init < 0:
            success_count += 1
            continue
            
        x_adv, queries = score_guided_attack(model_obj.decision_function, x_domain, eps=EPSILON_BUDGET * scale_factor, bounds=domain_bounds)
        f_adv = model_obj.decision_function(x_adv.reshape(1, -1))[0]
        if f_adv < 0:
            success_count += 1
            l2_sum += np.linalg.norm(x_domain - x_adv) / scale_factor
        queries_sum += queries
        
        if (idx_s + 1) % 10 == 0:
            print(f"  Processed {idx_s + 1}/{len(X_eval)} samples...")
        
    eval_time = time.time() - t0
    
    atk_rate = (success_count / ATTACK_SAMPLES) * 100
    avg_l2 = l2_sum / max(success_count, 1)
    avg_queries = queries_sum / ATTACK_SAMPLES
    
    result = {
        "Model": model_name,
        "Attack Success %": f"{atk_rate:.1f}%",
        "Average L2 (Unit)": f"{avg_l2:.4f}",
        "Average Queries": f"{avg_queries:.1f}",
        "Eval Time (s)": f"{eval_time:.1f}"
    }
    all_results.append(result)
    
    print(f"\n  Results for {model_name}:")
    print(f"    Attack Success: {atk_rate:.1f}%, L2={avg_l2:.4f}, queries={avg_queries:.1f}")
    print(f"    Time:            {eval_time:.1f}s")

# ── Results ──
results_df = pd.DataFrame(all_results)
print("\n" + "="*80)
print("=== E12: SCORE-GUIDED BLACK-BOX ATTACK (SGBBA) RESULTS ===")
print("="*80)
print(results_df.to_string(index=False))
