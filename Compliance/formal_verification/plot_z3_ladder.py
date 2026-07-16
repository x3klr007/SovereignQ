#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from z3 import Solver, Real, RealVal, And, unsat, sat

# Configuration
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 1500  # Optimized for execution speed
TEST_SIZE = 0.25
SEED = 42

print("Loading dataset for Z3 Robustness Ladder...")
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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Train Poly SVM (degree 2) for exact SMT verification
gamma = 0.5
coef0 = 1.0
degree = 2
model = SVC(kernel='poly', degree=degree, gamma=gamma, coef0=coef0, C=1.0)
model.fit(X_train, y_train)

support_vectors = model.support_vectors_
dual_coefs = model.dual_coef_[0]
intercept = model.intercept_[0]
n_sv = len(support_vectors)
d = X_test.shape[1]

def z3_to_float(val):
    if val is None:
        return 0.0
    try:
        return float(val.as_fraction())
    except:
        pass
    try:
        return float(val.as_double())
    except:
        return float(str(val).replace('?', ''))

def verify_local_robustness(x0, y_pred, delta):
    solver = Solver()
    x_vars = [Real(f'x_{j}') for j in range(d)]
    
    # Input bounds
    for j in range(d):
        solver.add(x_vars[j] >= 0.0)
        solver.add(x_vars[j] <= 1.0)
        solver.add(x_vars[j] >= float(max(0.0, x0[j] - delta)))
        solver.add(x_vars[j] <= float(min(1.0, x0[j] + delta)))
        
    f_x = RealVal(float(intercept))
    for i in range(n_sv):
        alpha_y_i = float(dual_coefs[i])
        s_i = support_vectors[i]
        dot_product = RealVal(0.0)
        for j in range(d):
            dot_product = dot_product + RealVal(float(s_i[j])) * x_vars[j]
        kernel_term = RealVal(float(gamma)) * dot_product + RealVal(float(coef0))
        f_x = f_x + RealVal(alpha_y_i) * (kernel_term * kernel_term)
        
    if y_pred == 1:
        solver.add(f_x < 0.0)
    else:
        solver.add(f_x >= 0.0)
        
    t0 = time.time()
    status = solver.check()
    solve_time = time.time() - t0
    
    if status == unsat:
        return "UNSAT", solve_time, None
    elif status == sat:
        m = solver.model()
        x_adv = np.zeros(d)
        for j in range(d):
            val = m[x_vars[j]]
            x_adv[j] = z3_to_float(val) if val is not None else x0[j]
        return "SAT", solve_time, x_adv
    return "UNKNOWN", solve_time, None

# Find close (boundary) and far (interior) test samples
scores = model.decision_function(X_test)
abs_scores = np.abs(scores)
closest_idx = np.argmin(abs_scores)
furthest_idx = np.argmax(abs_scores)

# Explicitly choose Sample #496 (boundary proximity) and Sample #724 (interior)
# If index matches test size, use closest_idx/furthest_idx respectively.
idx_boundary = target_idx = closest_idx
idx_interior = far_idx = furthest_idx

# We will sweep delta
deltas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

print(f"Sweeping boundary sample (Sample #{idx_boundary}, score={scores[idx_boundary]:.6f})...")
boundary_results = []
x_boundary = X_test[idx_boundary]
y_boundary_pred = 1 if scores[idx_boundary] >= 0 else 0
x_adv_boundary = None

for delta in deltas:
    status, t, x_adv = verify_local_robustness(x_boundary, y_boundary_pred, delta)
    boundary_results.append((delta, status, t))
    if status == "SAT" and x_adv_boundary is None:
        x_adv_boundary = x_adv

print(f"Sweeping interior sample (Sample #{idx_interior}, score={scores[idx_interior]:.6f})...")
interior_results = []
x_interior = X_test[idx_interior]
y_interior_pred = 1 if scores[idx_interior] >= 0 else 0

for delta in deltas:
    status, t, _ = verify_local_robustness(x_interior, y_interior_pred, delta)
    interior_results.append((delta, status, t))

# Plotting 2-panel figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel 1: SMT Verification Time & Status
ax1 = axes[0]
d_vals = [r[0] for r in boundary_results]
times_b = [r[2] for r in boundary_results]
status_b = [r[1] for r in boundary_results]
times_i = [r[2] for r in interior_results]
status_i = [r[1] for r in interior_results]

# Plot boundary sample
line1 = ax1.plot(d_vals, times_b, color="#E74C3C", linewidth=2.0, linestyle="--", label="Boundary Sample (Close)")
# Scatter markers colored by status: green for UNSAT, red for SAT
colors_b = ["#2ECC71" if s == "UNSAT" else "#E74C3C" for s in status_b]
ax1.scatter(d_vals, times_b, color=colors_b, zorder=5, s=60, edgecolors="black")

# Plot interior sample
line2 = ax1.plot(d_vals, times_i, color="#2ECC71", linewidth=2.0, linestyle="-", label="Interior Sample (Far)")
colors_i = ["#2ECC71" if s == "UNSAT" else "#E74C3C" for s in status_i]
ax1.scatter(d_vals, times_i, color=colors_i, zorder=5, s=60, edgecolors="black")

ax1.set_xlabel("Adversarial Perturbation Bound ($\delta$)", fontweight="bold", fontsize=11)
ax1.set_ylabel("Z3 Solver Execution Time (s)", fontweight="bold", fontsize=11)
ax1.set_title("SMT Verification Execution Time vs. Perturbation Radius", fontweight="bold", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(loc="upper left")

# Add a text box explaining the color markers
ax1.text(0.1, 0.75, "Green = UNSAT (100% Robust)\nRed = SAT (Counterexample Found)", 
         transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="#FBFCFC", ec="gray", lw=1),
         fontsize=9, fontweight="bold")

# Panel 2: 4-Feature Radar comparison for Boundary Sample SAT counterexample
ax2 = axes[1]
categories = QUANTUM_FEATURES
N_features = len(categories)

# Radar Plot setup
angles = [n / float(N_features) * 2 * np.pi for n in range(N_features)]
angles += angles[:1]

# Set projection as polar
ax2.remove()
ax2 = fig.add_subplot(1, 2, 2, projection="polar")

# Original features
orig_vals = list(x_boundary)
orig_vals += orig_vals[:1]

# Adversarial features (Z3 model output)
if x_adv_boundary is not None:
    adv_vals = list(x_adv_boundary)
    adv_vals += adv_vals[:1]
else:
    adv_vals = orig_vals

# Draw radar elements
plt.xticks(angles[:-1], categories, fontweight="bold", fontsize=10)
ax2.plot(angles, orig_vals, linewidth=2, linestyle="solid", color="#3498DB", label="Original Sample")
ax2.fill(angles, orig_vals, color="#3498DB", alpha=0.25)

ax2.plot(angles, adv_vals, linewidth=2, linestyle="solid", color="#E74C3C", label="Z3 Adversarial Counterexample")
ax2.fill(angles, adv_vals, color="#E74C3C", alpha=0.2)

ax2.set_title("Z3-Synthesized Counterexample Comparison", fontweight="bold", fontsize=12, pad=15)
ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.suptitle("Z3 SMT Verification: Boundary Proximity vs. Interior Sample Robustness", fontweight="bold", fontsize=14, y=0.98)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "erda_z3_robustness_ladder.png")
plt.savefig(out_path, dpi=300)
print(f"Successfully generated Z3 formal verification plot and saved to {out_path}")
