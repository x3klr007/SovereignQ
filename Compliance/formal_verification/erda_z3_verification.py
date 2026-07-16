#!/usr/bin/env python3
# =============================================================================
# ERDA Z3 Neuro-Symbolic Verifier Experiment
# =============================================================================
# Implements a real, working formal verification solver using Z3 to verify
# SVM decision boundaries on credit card fraud features.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from z3 import Solver, Real, RealVal, And, unsat, sat

print("=== ERDA Z3 Neuro-Symbolic Verifier ===")

# 1. Load Data Subset
DATASET_PATH = 'creditcard.csv'
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 3000
TEST_SIZE = 0.25
SEED = 42

print("Loading data...")
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

# Scale features to [0, 1] for neat numerical representation in Z3
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# 2. Train a Polynomial Kernel SVM
# A polynomial kernel is chosen because K(x, y) = (gamma * x^T y + coef0)^degree
# is a polynomial, allowing exact, decidable non-linear real arithmetic in Z3!
print("\nTraining Polynomial Kernel SVM (degree=2)...")
gamma = 0.5
coef0 = 1.0
degree = 2
C_val = 1.0

model = SVC(kernel='poly', degree=degree, gamma=gamma, coef0=coef0, C=C_val)
model.fit(X_train, y_train)

# Calculate training and testing scores
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Model trained. Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Extract model weights and support vectors
support_vectors = model.support_vectors_
dual_coefs = model.dual_coef_[0]  # Shape: (n_support_vectors,)
intercept = model.intercept_[0]

n_sv = len(support_vectors)
print(f"Number of Support Vectors: {n_sv}")

def z3_to_float(val):
    """Safely converts a Z3 real/rational value to a Python float."""
    if val is None:
        return 0.0
    try:
        return float(val.as_fraction())
    except:
        pass
    try:
        return float(val.as_double())
    except:
        pass
    try:
        s = str(val).replace('?', '')
        return float(s)
    except:
        return 0.0

# 3. Z3 Verification Function
def verify_local_robustness(sample_idx, delta=0.05, silent=False):
    """
    Formally verifies local L-infinity robustness of the SVM model
    around X_test[sample_idx] within radius delta.
    """
    x0 = X_test[sample_idx]
    y_true = y_test[sample_idx]
    
    # Calculate model's empirical decision function score and prediction
    emp_score = model.decision_function(x0.reshape(1, -1))[0]
    y_pred = 1 if emp_score >= 0 else 0
    
    if not silent:
        print(f"\n--- Verifying Sample #{sample_idx} ---")
        print(f"True Class: {y_true}, Pred Class: {y_pred}")
        print(f"Empirical Decision Score: {emp_score:.6f}")
        print(f"Adversarial Tolerance Radius (delta): {delta}")
    
    # Initialize Z3 Solver
    solver = Solver()
    
    # Create Z3 real variables for the input coordinates
    d = x0.shape[0]
    x_vars = [Real(f'x_{j}') for j in range(d)]
    
    # Add input space bounds: 0.0 <= x_j <= 1.0
    for j in range(d):
        solver.add(x_vars[j] >= 0.0)
        solver.add(x_vars[j] <= 1.0)
    
    # Add adversarial perturbation constraints: x0_j - delta <= x_vars_j <= x0_j + delta
    for j in range(d):
        solver.add(x_vars[j] >= float(max(0.0, x0[j] - delta)))
        solver.add(x_vars[j] <= float(min(1.0, x0[j] + delta)))
    
    # Encode the polynomial decision function: f(x) = sum_i alpha_i * y_i * (gamma * s_i^T x + coef0)^2 + b
    f_x = RealVal(float(intercept))
    
    for i in range(n_sv):
        alpha_y_i = float(dual_coefs[i])
        s_i = support_vectors[i]
        
        # Calculate symbolic dot product s_i^T x
        dot_product = RealVal(0.0)
        for j in range(d):
            dot_product = dot_product + RealVal(float(s_i[j])) * x_vars[j]
        
        # (gamma * dot_product + coef0)
        kernel_term = RealVal(float(gamma)) * dot_product + RealVal(float(coef0))
        
        # Raise to the power (degree=2)
        kernel_pow = kernel_term * kernel_term
        
        # Add to the decision score
        f_x = f_x + RealVal(alpha_y_i) * kernel_pow
        
    # Assert adversarial violation: we seek a point x such that its prediction sign flips!
    if y_pred == 1:
        # If model predicts Class 1 (Fraud), search for a counter-example where it predicts Class 0 (Normal)
        solver.add(f_x < 0.0)
    else:
        # If model predicts Class 0 (Normal), search for a counter-example where it predicts Class 1 (Fraud)
        solver.add(f_x >= 0.0)
        
    # Solve the constraints
    status = solver.check()
    
    if status == unsat:
        if not silent:
            print("Verification Result: UNSAT (100% Robust!)")
            print(f"Formal Proof: No adversarial counter-example exists within L_inf distance {delta}.")
        return "verified_robust"
    elif status == sat:
        if not silent:
            print("Verification Result: SAT (Vulnerable!)")
            print("Adversarial Counter-example Found by Z3 Solver:")
        m = solver.model()
        x_adv = np.zeros(d)
        for j in range(d):
            val = m[x_vars[j]]
            x_adv[j] = z3_to_float(val) if val is not None else x0[j]
        
        adv_score = model.decision_function(x_adv.reshape(1, -1))[0]
        y_adv_pred = 1 if adv_score >= 0 else 0
        if not silent:
            print(f"Original Input:  {x0}")
            print(f"Adversarial Input: {x_adv}")
            print(f"Perturbation L_inf: {np.max(np.abs(x0 - x_adv)):.6f}")
            print(f"Perturbation L_2:   {np.linalg.norm(x0 - x_adv):.6f}")
            print(f"New Decision Score: {adv_score:.6f} (New Prediction: {y_adv_pred})")
        return "vulnerable"
    else:
        if not silent:
            print("Verification Result: UNKNOWN (Timeout or Insoluble)")
        return "unknown"

# Let's find a sample that is close to the decision boundary (low absolute decision score)
scores = model.decision_function(X_test)
abs_scores = np.abs(scores)
closest_indices = np.argsort(abs_scores)[:5]

print("\nClosest samples to decision boundary:")
for rank, idx in enumerate(closest_indices):
    print(f"Rank {rank+1}: Sample #{idx}, Score={scores[idx]:.6f}, True Class={y_test[idx]}")

# Let's run a sweep over delta for the closest sample to show the robust-to-vulnerable transition!
target_idx = closest_indices[0]
print(f"\nRunning L_inf perturbation sweep over delta for Sample #{target_idx} (closest to boundary)...")
deltas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
for d_val in deltas:
    res = verify_local_robustness(target_idx, delta=d_val, silent=False)
    print(f"Delta = {d_val:.3f} -> Result: {res}")

# Let's verify a highly robust sample (far from the boundary)
far_idx = closest_indices[-1]
# Let's find the absolute furthest sample
furthest_idx = np.argmax(abs_scores)
print(f"\nRunning L_inf verification for Sample #{furthest_idx} (furthest from boundary, score={scores[furthest_idx]:.6f})...")
verify_local_robustness(furthest_idx, delta=0.05, silent=False)
verify_local_robustness(furthest_idx, delta=0.20, silent=False)
