#!/usr/bin/env python3
# =============================================================================
# Unit Tests for Z3 Formal Verifiers
# =============================================================================

import sys
import os
import unittest

# Add tools directory to path
sys.path.append(os.path.dirname(__file__))

from orthogonality_z3_verification import verify_orthogonality_bounds
from blockade_z3_verification import verify_blockade_bounds
from orbach_z3_verification import verify_orbach_bounds
from cce_z3_verification import verify_cce_bounds
from tsv_z3_verification import verify_tsv_routing
from jitter_z3_verification import verify_jitter_bounds


class TestZ3Verifiers(unittest.TestCase):
    """Unit tests for Z3-based physical parameter formal verifiers."""

    def test_orthogonality_robust_regime(self):
        """Test that a tight target (e.g. 50 Hz) is verified robust (UNSAT) for small errors."""
        res = verify_orthogonality_bounds(
            delta_theta_max_deg=0.5,
            delta_epsilon_max_pct=0.01,
            E_gate_max_Vcm=10000.0,
            gamma_target_Hz=50.0,
            silent=True
        )
        self.assertEqual(res, "verified_robust")

    def test_orthogonality_vulnerable_regime(self):
        """Test that a very strict target (e.g. 10 Hz) is flagged as vulnerable (SAT)."""
        res = verify_orthogonality_bounds(
            delta_theta_max_deg=0.5,
            delta_epsilon_max_pct=0.01,
            E_gate_max_Vcm=10000.0,
            gamma_target_Hz=10.0,
            silent=True
        )
        self.assertEqual(res, "vulnerable")

    def test_blockade_robust_regime(self):
        """Test that standard Rydberg gate fluctuations are verified robust (UNSAT)."""
        res = verify_blockade_bounds(
            n_index=10,
            Omega0_MHz=600.0,
            delta_Omega_max_pct=1.0,
            delta_U_max_pct=3.0,
            L_target=0.005,  # 0.5%
            silent=True
        )
        self.assertEqual(res, "verified_robust")

    def test_blockade_vulnerable_regime(self):
        """Test that very tight leakage targets are flagged as vulnerable (SAT) under fluctuations."""
        res = verify_blockade_bounds(
            n_index=10,
            Omega0_MHz=600.0,
            delta_Omega_max_pct=1.0,
            delta_U_max_pct=3.0,
            L_target=0.001,  # 0.1%
            silent=True
        )
        self.assertEqual(res, "vulnerable")

    def test_orbach_robust_regime(self):
        """Test that T1 is robustly above 1.0 ms."""
        res = verify_orbach_bounds(
            T0_K=300.0,
            delta_T_K=5.0,
            epsilon0=0.028,
            delta_epsilon=0.001,
            A0_s1=1.5e9,
            delta_A_s1=0.5e9,
            T1_target_ms=1.0,
            silent=True
        )
        self.assertEqual(res, "verified_robust")

    def test_orbach_vulnerable_regime(self):
        """Test that T1 target of 2.0 ms is vulnerable to fluctuations."""
        res = verify_orbach_bounds(
            T0_K=300.0,
            delta_T_K=5.0,
            epsilon0=0.028,
            delta_epsilon=0.001,
            A0_s1=1.5e9,
            delta_A_s1=0.5e9,
            T1_target_ms=2.0,
            silent=True
        )
        self.assertEqual(res, "vulnerable")

    def test_cce_robust_regime(self):
        """Test that T2 is robustly above 9.0 ms."""
        res = verify_cce_bounds(
            rho0_ppb=10.0,
            delta_rho_ppb=2.0,
            f0=0.047,
            delta_f=0.003,
            delta_triplet0=0.10,
            delta_triplet_max=0.05,
            T2_target_ms=9.0,
            silent=True
        )
        self.assertEqual(res, "verified_robust")

    def test_cce_vulnerable_regime(self):
        """Test that T2 target of 10.0 ms is vulnerable to fluctuations."""
        res = verify_cce_bounds(
            rho0_ppb=10.0,
            delta_rho_ppb=2.0,
            f0=0.047,
            delta_f=0.003,
            delta_triplet0=0.10,
            delta_triplet_max=0.05,
            T2_target_ms=10.0,
            silent=True
        )
        self.assertEqual(res, "vulnerable")

    def test_tsv_robust_regime(self):
        """Test that standard TSV layout parameters are robustly verified."""
        res = verify_tsv_routing(
            pitch_tsv0=35.0,
            delta_pitch=5.0,
            donor_dist0=22.3,
            delta_donor=2.0,
            v_crosstalk_max_meV=0.1,
            silent=True
        )
        self.assertEqual(res, "verified_robust")

    def test_tsv_vulnerable_regime(self):
        """Test that too narrow spacing parameters are flagged as vulnerable."""
        res = verify_tsv_routing(
            pitch_tsv0=18.0,
            delta_pitch=5.0,
            donor_dist0=22.3,
            delta_donor=2.0,
            v_crosstalk_max_meV=0.1,
            silent=True
        )
        self.assertEqual(res, "vulnerable")

    def test_jitter_robust_regime(self):
        """Test that tight spacing and low jitter is verified robust."""
        res = verify_jitter_bounds(
            r_spacing0=12.4322,
            delta_r=0.5,
            bohr_radius=2.5,
            j_floor_MHz=150.0,
            silent=True
        )
        self.assertEqual(res, "verified_robust")

    def test_jitter_vulnerable_regime(self):
        """Test that large jitter is flagged as vulnerable."""
        res = verify_jitter_bounds(
            r_spacing0=12.4322,
            delta_r=3.0,
            bohr_radius=2.5,
            j_floor_MHz=150.0,
            silent=True
        )
        self.assertEqual(res, "vulnerable")


if __name__ == "__main__":
    unittest.main()
