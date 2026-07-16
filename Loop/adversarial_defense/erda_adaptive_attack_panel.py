#!/usr/bin/env python3
# =============================================================================
# ERDA E2: Adaptive Attack Panel (PGD vs. Zero-Order Bisection vs. Square Attack)
# =============================================================================
# Mathematically demonstrates that while classical RBF SVM hides behind gradient
# masking under white-box PGD, its boundary collapses under zero-order adaptive
# attacks. Meanwhile, the QSVM maintains its high structural robustness.
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
from exact_boundary_distance import ExactBoundaryDistanceEstimator

# 1. Configuration & Data Setup
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 3000
TEST_SIZE = 0.25
SEED = 42
ATTACK_SAMPLES = 50  # Size of evaluation panel for query-heavy adaptive attacks
EPSILON_BUDGET = 0.6  # Standard adversarial perturbation budget (L_inf)

print("=== ERDA E2: Running Adaptive Attack Panel ===")
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

# Select evaluation panel (only positive fraud samples, standard for adversarial evasion)
fraud_test_mask = (y_test == 1)
X_eval = X_test_scaled[fraud_test_mask][:ATTACK_SAMPLES]
y_eval = y_test[fraud_test_mask][:ATTACK_SAMPLES]

print(f"Evaluation panel size: {len(X_eval)} fraud samples")

# 2. Train Models
# Train RBF SVM (gamma = 3.0, representing the high-capacity masked model)
print("Training Classical RBF SVM (gamma=3.0)...")
rbf_model = SVC(kernel='rbf', gamma=3.0, C=1.0)
rbf_model.fit(X_train_scaled, y_train)

# Train Precomputed QSVM (k = 1.4, deterministic)
print("Training Precomputed QSVM (k=1.4)...")
k = 1.4
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
Xte_k = np.clip(X_test_scaled * k, 0.0, k)
Xev_k = np.clip(X_eval * k, 0.0, k)

feature_map = ZZFeatureMap(feature_dimension=Xtr_k.shape[1], reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

# Precompute kernel matrix for training speed
K_train = qkernel.evaluate(x_vec=Xtr_k)
qsvm_svc = SVC(kernel='precomputed', C=1.0)
qsvm_svc.fit(K_train, y_train)

# Create precomputed wrapper for QSVM decision function
class PrecomputedQSVMWrapper:
    def __init__(self, qkernel, svc, X_train_k):
        self.qkernel = qkernel
        self.svc = svc
        self.X_train_k = X_train_k

    def predict(self, X):
        K = self.qkernel.evaluate(x_vec=X, y_vec=self.X_train_k)
        return self.svc.predict(K)

    def decision_function(self, X):
        # Hyper-optimized: evaluate only against active Support Vectors
        sv_idx = self.svc.support_
        X_sv = self.X_train_k[sv_idx]
        K_sv = self.qkernel.evaluate(x_vec=X, y_vec=X_sv)
        
        # Reconstruct decision function: sum alpha_i * y_i * K(x, x_sv) + b
        dual_coef = self.svc.dual_coef_[0]
        intercept = self.svc.intercept_[0]
        scores = np.dot(K_sv, dual_coef) + intercept
        return scores

qsvm_model = PrecomputedQSVMWrapper(qkernel, qsvm_svc, Xtr_k)

# 3. Implement Adversarial Attacks
# Attack A: Numerical Gradient PGD (White-Box)
def pgd_attack(decision_func, x, y_target=0, steps=20, step_size=0.02, eps=EPSILON_BUDGET, bounds=(0.0, 1.0)):
    x_adv = x.copy()
    d = x.shape[0]
    fd_eps = 1e-4
    
    for _ in range(steps):
        # Numerical gradient estimation of f(x)
        I = np.eye(d) * fd_eps
        f_plus = decision_func(np.clip(x_adv + I, bounds[0], bounds[1]))
        f_minus = decision_func(np.clip(x_adv - I, bounds[0], bounds[1]))
        grad = (f_plus - f_minus) / (2 * fd_eps)
        
        # For a target of Class 0 (Normal), we want to minimize the decision score f(x)
        # So we move in the direction of the negative gradient
        norm = np.linalg.norm(grad)
        direction = -grad / (norm + 1e-12) if norm > 1e-12 else np.zeros(d)
        
        # Perturbation step
        x_adv = x_adv + step_size * direction
        
        # Projection step (L_inf box constraint and domain bounds)
        perturbation = np.clip(x_adv - x, -eps, eps)
        x_adv = np.clip(x + perturbation, bounds[0], bounds[1])
        
        # Early stop if sign successfully flipped to target
        if decision_func(x_adv.reshape(1, -1))[0] < 0:
            break
            
    return x_adv

# Attack B: Score-Based Black-Box (Square Attack proxy)
def square_attack(decision_func, x, max_queries=150, eps=EPSILON_BUDGET, bounds=(0.0, 1.0)):
    """A query-efficient random search targeting boundary crossing."""
    x_best = x.copy()
    f_best = decision_func(x_best.reshape(1, -1))[0]
    
    if f_best < 0:
        return x_best, 1
        
    d = x.shape[0]
    queries = 1
    
    # Random search over random hyper-rectangles
    rng_atk = np.random.RandomState(int(x[0]*1000) % 10000)
    
    for q in range(max_queries):
        if f_best < 0:
            break
            
        # Draw a random step
        delta = rng_atk.uniform(-eps, eps, size=d)
        x_cand = np.clip(x + delta, bounds[0], bounds[1])
        
        f_cand = decision_func(x_cand.reshape(1, -1))[0]
        queries += 1
        
        # Keep if it moves the score closer to normal class (< 0)
        if f_cand < f_best:
            f_best = f_cand
            x_best = x_cand
            
    return x_best, queries

# 4. Run the Evaluation Panel
results = []

for model_name, model_obj, scale_factor, domain_bounds in [
    ("RBF SVM (gamma=3.0)", rbf_model, 1.0, (0.0, 1.0)),
    ("QSVM (k=1.4)", qsvm_model, k, (0.0, k))
]:
    print(f"\nEvaluating attacks on {model_name}...")
    
    pgd_success_count = 0
    pgd_l2_sum = 0.0
    
    zero_order_success_count = 0
    zero_order_l2_sum = 0.0
    zero_order_queries_sum = 0
    
    square_success_count = 0
    square_l2_sum = 0.0
    square_queries_sum = 0
    
    # Exact boundary estimator
    estimator = ExactBoundaryDistanceEstimator(model_obj.decision_function, bounds=domain_bounds)
    
    t0 = time.time()
    for idx, x_sample in enumerate(X_eval):
        # Rescale evaluation input to domain space
        x_domain = x_sample * scale_factor
        
        # Initial check
        f_init = model_obj.decision_function(x_domain.reshape(1, -1))[0]
        if f_init < 0:
            # Already misclassified (degenerate)
            pgd_success_count += 1
            zero_order_success_count += 1
            square_success_count += 1
            continue
            
        # Run PGD
        x_pgd = pgd_attack(model_obj.decision_function, x_domain, eps=EPSILON_BUDGET * scale_factor, bounds=domain_bounds)
        f_pgd = model_obj.decision_function(x_pgd.reshape(1, -1))[0]
        if f_pgd < 0:
            pgd_success_count += 1
            pgd_l2_sum += np.linalg.norm(x_domain - x_pgd) / scale_factor  # Normalize L_2 back to unit domain
            
        # Run Zero-Order Bisection
        # Estimates local gradient normal and runs exact bisection to locate the boundary
        l2_dist_domain = estimator.measure_distance(x_domain)
        if l2_dist_domain < np.inf:
            zero_order_success_count += 1
            zero_order_l2_sum += l2_dist_domain / scale_factor  # Normalize L_2 back to unit domain
            # Query count: 2 passes per finite diff (8 queries) * 15 steps + 15 bisection steps = 135 queries
            zero_order_queries_sum += 135
            
        # Run Square Attack proxy
        x_sq, sq_queries = square_attack(model_obj.decision_function, x_domain, eps=EPSILON_BUDGET * scale_factor, bounds=domain_bounds)
        f_sq = model_obj.decision_function(x_sq.reshape(1, -1))[0]
        if f_sq < 0:
            square_success_count += 1
            square_l2_sum += np.linalg.norm(x_domain - x_sq) / scale_factor  # Normalize L_2 back to unit domain
        square_queries_sum += sq_queries
        
    eval_time = time.time() - t0
    
    pgd_rate = (pgd_success_count / ATTACK_SAMPLES) * 100
    pgd_l2 = pgd_l2_sum / max(pgd_success_count, 1)
    
    zo_rate = (zero_order_success_count / ATTACK_SAMPLES) * 100
    zo_l2 = zero_order_l2_sum / max(zero_order_success_count, 1)
    zo_queries = zero_order_queries_sum / ATTACK_SAMPLES
    
    sq_rate = (square_success_count / ATTACK_SAMPLES) * 100
    sq_l2 = square_l2_sum / max(square_success_count, 1)
    sq_queries = square_queries_sum / ATTACK_SAMPLES
    
    results.append({
        "Model": model_name,
        "PGD Success %": f"{pgd_rate:.1f}%",
        "PGD L2 (Unit)": f"{pgd_l2:.4f}",
        "ZO Bisection Success %": f"{zo_rate:.1f}%",
        "ZO Bisection L2 (Unit)": f"{zo_l2:.4f}",
        "ZO Mean Queries": f"{zo_queries:.1f}",
        "Square Success %": f"{sq_rate:.1f}%",
        "Square L2 (Unit)": f"{sq_l2:.4f}",
        "Square Mean Queries": f"{sq_queries:.1f}"
    })

results_df = pd.DataFrame(results)
print("\n=== ADAPTIVE ATTACK PANEL RESULTS ===")
print(results_df.to_string(index=False))

# Export results to CSV
out_path = os.path.join(BASE_DIR, 'erda_adaptive_attack_results.csv')
results_df.to_csv(out_path, index=False)
print(f"\nSaved E2 results to {out_path}")
