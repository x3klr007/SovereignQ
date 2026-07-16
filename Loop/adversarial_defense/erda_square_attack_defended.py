#!/usr/bin/env python3
# =============================================================================
# ERDA E12: Defended Square Attack Panel for QSVM using Random Logit Scaling (RLS)
# =============================================================================
# Implements Hamid Dashtbani et al.'s state-of-the-art Random Logit Scaling (RLS)
# defense to protect the smooth QSVM decision boundary against score-based
# query-efficient black-box Square Attacks.
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

# Configuration — identical to E11 for comparability
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 3000
TEST_SIZE = 0.25
SEED = 42
ATTACK_SAMPLES = 50  # Same panel size as E11
EPSILON_BUDGET = 0.6  # Same L_inf budget as E11

print("=== ERDA E12: Defended Square Attack Panel via Random Logit Scaling (RLS) ===")
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

# Fit scaler
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Select evaluation panel (only positive fraud samples)
fraud_test_mask = (y_test == 1)
X_eval = X_test_scaled[fraud_test_mask][:ATTACK_SAMPLES]
y_eval = y_test[fraud_test_mask][:ATTACK_SAMPLES]

print(f"Evaluation panel size: {len(X_eval)} fraud samples")

# ── Attack Implementation ─────────────────────────────────────────────────

def square_attack(decision_func, x, max_queries=150, eps=EPSILON_BUDGET, bounds=(0.0, 1.0)):
    """Query-efficient random search targeting boundary crossing."""
    x_best = x.copy()
    f_best = decision_func(x_best.reshape(1, -1))[0]
    
    if f_best < 0:
        return x_best, 1
        
    d = x.shape[0]
    queries = 1
    
    rng_atk = np.random.RandomState(int(x[0]*1000) % 10000)
    
    for q in range(max_queries):
        if f_best < 0:
            break
            
        delta = rng_atk.uniform(-eps, eps, size=d)
        x_cand = np.clip(x + delta, bounds[0], bounds[1])
        
        f_cand = decision_func(x_cand.reshape(1, -1))[0]
        queries += 1
        
        if f_cand < f_best:
            f_best = f_cand
            x_best = x_cand
            
    return x_best, queries

# ── Precomputed QSVM Wrapper ─────────────────────────────────────────────

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

# ── RLS Defended Wrapper ──────────────────────────────────────────────────

class RandomLogitScalingWrapper:
    """
    Plug-and-play Random Logit Scaling (RLS) Defense layer.
    Rescales the decision score by a random scaling factor alpha ~ Uniform(0.8, 1.2)
    on each query. This preserves the sign (classification accuracy) but completely
    destroys the score-based query-gradient estimations used by Square Attack.
    """
    def __init__(self, base_model, seed=42):
        self.base_model = base_model
        # Use a local generator to keep it consistent
        self.rng = np.random.RandomState(seed)

    def predict(self, X):
        # Sign is preserved, classification is 100% unaffected
        return self.base_model.predict(X)

    def decision_function(self, X):
        scores = self.base_model.decision_function(X)
        # Apply random logit scaling alpha ~ Uniform(0.8, 1.2)
        # Note: we scale each score individually to disrupt trajectory search
        scales = self.rng.uniform(0.8, 1.2, size=scores.shape)
        return scores * scales

# ── Train Reference and QSVM ─────────────────────────────────────────────

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

# ── Set up evaluations ───────────────────────────────────────────────────

models_to_test = [
    ("RBF SVM (gamma=3.0) [Standard]", rbf_model, 1.0, (0.0, 1.0)),
    ("QSVM (k=1.2) [Standard]", qsvm_model, k, (0.0, k)),
    ("QSVM (k=1.2) [RLS Defended]", RandomLogitScalingWrapper(qsvm_model), k, (0.0, k))
]

all_results = []

for model_name, model_obj, scale_factor, domain_bounds in models_to_test:
    print(f"\n{'='*60}")
    print(f"Evaluating Square Attack on {model_name}...")
    print(f"{'='*60}")
    
    square_success_count = 0
    square_l2_sum = 0.0
    square_queries_sum = 0
    
    t0 = time.time()
    for idx_s, x_sample in enumerate(X_eval):
        x_domain = x_sample * scale_factor
        
        f_init = model_obj.decision_function(x_domain.reshape(1, -1))[0]
        if f_init < 0:
            # Already misclassified
            square_success_count += 1
            continue
            
        # Square Attack
        x_sq, sq_queries = square_attack(model_obj.decision_function, x_domain, eps=EPSILON_BUDGET * scale_factor, bounds=domain_bounds)
        f_sq = model_obj.decision_function(x_sq.reshape(1, -1))[0]
        if f_sq < 0:
            square_success_count += 1
            square_l2_sum += np.linalg.norm(x_domain - x_sq) / scale_factor
        square_queries_sum += sq_queries
        
        if (idx_s + 1) % 10 == 0:
            print(f"  Processed {idx_s + 1}/{len(X_eval)} samples...")
        
    eval_time = time.time() - t0
    
    sq_rate = (square_success_count / ATTACK_SAMPLES) * 100
    sq_l2 = square_l2_sum / max(square_success_count, 1)
    sq_queries = square_queries_sum / ATTACK_SAMPLES
    
    result = {
        "Model": model_name,
        "Square Success %": f"{sq_rate:.1f}%",
        "Square L2 (Unit)": f"{sq_l2:.4f}",
        "Square Mean Queries": f"{sq_queries:.1f}",
        "Eval Time (s)": f"{eval_time:.1f}"
    }
    all_results.append(result)
    
    print(f"\n  Results for {model_name}:")
    print(f"    Square Attack: {sq_rate:.1f}% success, L2={sq_l2:.4f}, queries={sq_queries:.1f}")
    print(f"    Time:          {eval_time:.1f}s")

# ── Export ──
results_df = pd.DataFrame(all_results)
print("\n" + "="*80)
print("=== E12: DEFENDED SQUARE ATTACK SWEEP RESULTS ===")
print("="*80)
print(results_df.to_string(index=False))

out_path = os.path.join(BASE_DIR, 'erda_e12_square_attack_defended_results.csv')
results_df.to_csv(out_path, index=False)
print(f"\nSaved E12 results to {out_path}")
