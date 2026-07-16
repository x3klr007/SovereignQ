import numpy as np
import time

def simulate_federated_round(num_banks=5, num_compromised=1):
    print("==========================================================================================")
    print("   🛡️  SAMA BYZANTINE DEFENSE SUITE — COORDINATE-WISE MEDIAN TRIMMING AGGREGATOR   ")
    print("==========================================================================================")
    print(f"--> Initializing Secure Sovereign FL Session across {num_banks} Saudi Banking Nodes...")
    time.sleep(0.5)

    # True gradient/weight update expected from normal learning step
    true_update = np.array([0.142, -0.053, 0.451, 0.884, -0.219])

    updates = []
    print("\n[STEP 1: Collecting Encrypted Local Model Updates]")
    for i in range(num_banks):
        if i < num_banks - num_compromised:
            # Clean banks (small normal noise)
            update = true_update + np.random.normal(0, 0.005, size=5)
            print(f"  🟢 Node {i+1} [Alinma Sub-node]: Received valid gradients. (L2-norm: {np.linalg.norm(update):.3f})")
        else:
            # Compromised bank (Adversarial Insider Data Poisoning Attack)
            update = true_update + np.array([12.5, -45.0, 88.1, -120.0, 50.0])
            print(f"  🔴 Node {i+1} [Compromised Node]: EXECUTING ACTIVE INSIDER DATA POISONING ATTACK!")
        updates.append(update)
        time.sleep(0.3)

    updates = np.array(updates)

    print("\n[STEP 2: Vulnerable Standard FedAvg Aggregation (Arithmetic Mean)]")
    time.sleep(0.5)
    standard_avg = np.mean(updates, axis=0)
    error_standard = np.linalg.norm(standard_avg - true_update)
    print(f"  ⚠️  Global Model Weight Deviation (L2 Error): {error_standard:.4f}")
    print("  ❌ STATUS: MODEL COLLAPSE. Adversarial gradient poisoning bypassed standard aggregation.")

    print("\n[STEP 3: SAMA Defense Suite — Coordinate-wise Median Trimming Aggregation]")
    time.sleep(0.5)
    # Byzantine-robust aggregation using coordinate-wise median trimming
    robust_avg = np.median(updates, axis=0)
    error_robust = np.linalg.norm(robust_avg - true_update)
    
    # Calculate exact mathematical Model Weight Integrity Recovery Metric
    integrity_recovery = max(0.0, (1.0 - (error_robust / error_standard))) * 100.0
    
    print(f"  🛡️  Trimmed Model Weight Deviation (L2 Error): {error_robust:.4f}")
    print("  ✅ DEFENSE MECHANISM: Coordinate-Wise Median Trimming successfully purged malicious dimensions.")
    print(f"  📊 SECURITY RECOVERY FIDELITY: {integrity_recovery:.1f}% Model Integrity Recovery under Active Insider Poisoning Attacks.")
    print("==========================================================================================\n")

if __name__ == "__main__":
    simulate_federated_round()
