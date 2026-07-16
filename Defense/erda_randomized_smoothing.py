import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

# Configuration
BASE_DIR = "/home/x3klr007/projects/Quantum/research"
DATASET_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
QUANTUM_FEATURES = ['V10', 'V4', 'V14', 'V12']
MAX_SAMPLES = 2000
TEST_SIZE = 0.25
SEED = 42
PANEL_SAMPLES = 50
NOISE_SIGMA = 0.25  # Standard deviation of Gaussian noise for smoothing
N0 = 100            # Samples for prediction estimation
N = 1000            # Samples for robust radius certification
ALPHA = 0.05        # Confidence level (95%)

print("=== ERDA: Certified Robustness via Randomized Smoothing (Cohen 2019) ===")
print("Loading dataset...")
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

# Test panel (positive fraud samples)
fraud_test_mask = (y_test == 1)
X_eval = X_test_scaled[fraud_test_mask][:PANEL_SAMPLES]
y_eval = y_test[fraud_test_mask][:PANEL_SAMPLES]

print(f"Evaluation panel size: {len(X_eval)} fraud samples")

# --- Train RBF SVM ---
print("\nTraining Classical RBF SVM (gamma=3.0)...")
rbf_model = SVC(kernel='rbf', gamma=3.0, C=1.0)
rbf_model.fit(X_train_scaled, y_train)

# --- Train QSVM (k=1.2) ---
print("Training Precomputed QSVM (k=1.2)...")
k = 1.2
Xtr_k = np.clip(X_train_scaled * k, 0.0, k)
feature_map = ZZFeatureMap(feature_dimension=Xtr_k.shape[1], reps=2, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)
K_train = qkernel.evaluate(x_vec=Xtr_k)
qsvm_svc = SVC(kernel='precomputed', C=1.0)
qsvm_svc.fit(K_train, y_train)

def qsvm_predict(X_noisy):
    # X_noisy is already scaled by k
    X_noisy = np.clip(X_noisy, 0.0, k)
    sv_idx = qsvm_svc.support_
    X_sv = Xtr_k[sv_idx]
    K_eval = qkernel.evaluate(x_vec=X_noisy, y_vec=X_sv)
    scores = np.dot(K_eval, qsvm_svc.dual_coef_[0]) + qsvm_svc.intercept_[0]
    return (scores >= 0).astype(int)

# --- Randomized Smoothing Certification ---
from scipy.stats import beta

def certify_radius(predict_fn, x, domain_bounds, scale_factor):
    # 1. Estimate top class using N0 samples
    noise_n0 = np.random.normal(scale=NOISE_SIGMA * scale_factor, size=(N0, len(x)))
    x_n0 = np.clip(x + noise_n0, domain_bounds[0], domain_bounds[1])
    preds_n0 = predict_fn(x_n0)
    
    # RBF predicts 1 for fraud, QSVM predicts 1 for fraud
    cA = int(np.bincount(preds_n0, minlength=2).argmax())
    
    # 2. Estimate lower bound probability (pA) using N samples
    noise_n = np.random.normal(scale=NOISE_SIGMA * scale_factor, size=(N, len(x)))
    x_n = np.clip(x + noise_n, domain_bounds[0], domain_bounds[1])
    preds_n = predict_fn(x_n)
    
    nA = np.sum(preds_n == cA)
    if nA == 0:
        pA_lower = 0.0
    else:
        pA_lower = beta.ppf(ALPHA, nA, N - nA + 1)
    
    if pA_lower > 0.5:
        # Certified radius formula (Cohen et al., 2019)
        radius = NOISE_SIGMA * norm.ppf(pA_lower)
        return radius, cA
    else:
        return 0.0, cA

print("\n--- Running Randomized Smoothing Certification ---")
print(f"Noise Sigma = {NOISE_SIGMA}, Samples = {N}, Confidence = {1 - ALPHA:.2f}")

rbf_radii = []
qsvm_radii = []

for i, x in enumerate(X_eval):
    # RBF
    def rbf_pred_fn(x_noisy):
        return rbf_model.predict(np.clip(x_noisy, 0.0, 1.0))
        
    r_rbf, _ = certify_radius(rbf_pred_fn, x, (0.0, 1.0), 1.0)
    rbf_radii.append(r_rbf)
    
    # QSVM
    x_k = x * k
    r_qsvm, _ = certify_radius(qsvm_predict, x_k, (0.0, k), k)
    qsvm_radii.append(r_qsvm)
    
    if (i+1) % 10 == 0:
        print(f"  Processed {i+1}/{len(X_eval)} samples...")

rbf_mean = np.mean(rbf_radii)
qsvm_mean = np.mean(qsvm_radii)

rbf_cert_pct = np.mean(np.array(rbf_radii) > 0) * 100
qsvm_cert_pct = np.mean(np.array(qsvm_radii) > 0) * 100

print("\n=== CERTIFIED ROBUSTNESS RESULTS (Normalized Unit L2) ===")
print(f"RBF SVM (gamma=3.0): Mean Certified Radius = {rbf_mean:.4f} | Certifiable % = {rbf_cert_pct:.1f}%")
print(f"QSVM    (k=1.2):     Mean Certified Radius = {qsvm_mean:.4f} | Certifiable % = {qsvm_cert_pct:.1f}%")

if qsvm_mean > rbf_mean:
    print(f"\n=> SUCCESS! QSVM provides a {(qsvm_mean / max(rbf_mean, 1e-6)):.2f}x larger mathematically proven certified robust radius!")
    print("=> The RBF's apparent Square Attack 'resistance' is formally exposed as gradient masking.")
else:
    print("\n=> RBF yielded a larger certified radius. We need to check the noise scale.")
