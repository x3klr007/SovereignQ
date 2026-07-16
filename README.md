# SovereignQ

## Project Overview

SovereignQ is a sovereign quantum research project focused on building a SAMA-compliant financial fraud and AML defense stack using quantum machine learning, federated privacy, and topological assurance.

The repository contains a hybrid quantum-classical platform for: 
- adversarial fraud detection using quantum kernel methods,
- secure temporal transaction analysis with quantum-enhanced recurrent models,
- SAMA regulatory compliance via conformal prediction and data residency safeguards,
- topological verification and explainability for financial decision boundaries.

## Table of Contents
- [AMAD Hackathon Submission](#amad-hackathon-submission)
- [What’s Included](#whats-included)
- [High-Level Capabilities](#high-level-capabilities)
- [Generating Figures and Charts](#generating-figures-and-charts)
- [Installation](#installation)
- [Usage](#usage)
- [Key Results](#key-results)
- [Notes](#notes)
- [Contact](#contact)

## AMAD Hackathon Submission

This repository supports the AMAD Fintech Hackathon submission titled:
**SAMA-Compliant Sovereign Quantum Fraud Defense (SQFD)**.

Key artifacts:
- `Highlights/AMAD_HACKATHON_SUBMISSION.docx` — submission brief and executive summary
- `Compliance/visualization_and_auditing/` — compliance and assurance plots
- `Injection/sama_compliance_suite/` — SAMA compliance suite and TDA monitoring
- `Matrix/qml_engine/` — quantum kernel alignment and QML analysis
- `Reduction/` — adversarial training and robustness experiments

## What’s Included

### Core directories
- `Compliance/` — formal verification, assurance metrics, and visualization tools
- `Defense/` — adversarial smoothing and quantum robustness experiments
- `Injection/` — SAMA compliance suite and chaos/TDA monitoring
- `Loop/` — adversarial defense and sequential model pipelines
- `Matrix/` — quantum kernel alignment, ablations, and QML engine code
- `Reduction/` — adversarial training and robust kernel workflows
- `Highlights/` — presentation and hackathon briefing artifacts
- `docs/` — supporting documentation and regulatory notes

### Selected scripts
- `Compliance/formal_verification/erda_z3_verification.py` — symbolic verification for quantum classifier margins
- `Compliance/visualization_and_auditing/plot_tda_persistence.py` — visualize topological invariants across quantum kernel regimes
- `Compliance/visualization_and_auditing/plot_conformal_histogram.py` — conformal-bound histograms for prediction uncertainty
- `Compliance/visualization_and_auditing/verify_assurance_metrics.py` — assurance-first metrics for quantum risk controls
- `Matrix/qml_engine/run_kernel_alignment.py` — classical vs quantum kernel alignment analysis
- `Injection/sama_compliance_suite/erda_tda_chaos.py` — chaos / TDA analysis for quantum kernel manifolds
- `Reduction/run_kernel_adversarial_training.py` — robust training with adversarial kernel perturbations

## High-Level Capabilities

### Quantum Adversarial Defense
- Demonstrates the ERDA (Encoding Range Dependent Advantage) phenomenon,
- Reduces adversarial fraud attack success from ~96% to ~10% in benchmark settings,
- Uses quantum kernel scaling to produce a stable decision boundary with low gradient masking.

### Temporal AML and Quantum RNNs
- Implements lightweight quantum LSTM-inspired models for transaction sequence analysis,
- Achieves strong convergence and parameter compression relative to classical LSTMs,
- Targets anti-money laundering (AML) detection across temporal transaction flows.

### SAMA-Compliant Federated Privacy
- Enables horizontal federated learning with secure local training,
- Never transfers raw transaction data outside domestic bank servers,
- Uses robust aggregation to resist Byzantine poisoning and adversarial model attacks.

### Explainability and Assurance
- Generates topological summaries of quantum decision manifolds,
- Integrates conformal prediction for calibrated uncertainty guarantees,
- Supports regulator-facing proof artifacts for explainable finance AI.

## Generating Figures and Charts

The repository includes scripts that produce key charts for the project.
Run them locally to generate visual summaries used in documentation and reports.

Example scripts:

- `python Compliance/visualization_and_auditing/plot_tda_persistence.py`
  - Produces Betti persistence diagrams and topological stability traces.
- `python Matrix/qml_engine/run_kernel_alignment.py`
  - Plots quantum-kernel vs classical RBF alignment across encoding scale `k`.
- `python Compliance/visualization_and_auditing/plot_conformal_histogram.py`
  - Visualizes conformal prediction intervals and uncertainty coverage.
- `python Injection/sama_compliance_suite/erda_tda_chaos.py`
  - Generates topological chaos metrics for quantum kernel manifolds.

> Note: Some scripts expect an external dataset path or may need a local copy of the credit card fraud dataset.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/x3klr007/SovereignQ.git
   cd SovereignQ
   ```
2. Create and activate a Python environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib ripser qiskit qiskit-machine-learning pennylane
   ```
4. Set up data paths in scripts if the dataset is stored outside the repository.

## Usage

1. Activate the Python environment:
   ```bash
   source .venv/bin/activate
   ```
2. Run a visual analysis script, for example:
   ```bash
   python Compliance/visualization_and_auditing/plot_tda_persistence.py
   ```
3. Inspect generated figures and verify output.

## Key Results

- **Adversarial fraud attack success reduced from ~96% to ~10%** using ERDA-optimized quantum kernel scaling.
- **98.6% conformal coverage** for uncertainty-bounded fraud predictions.
- **74.8% reduction in MSE** for quantum-enhanced temporal transaction models versus classical LSTMs.
- **Byzantine-robust federated aggregation** for secure model updates across domestic bank nodes.

## Notes

- The repository is structured around a sovereign deployment strategy that avoids foreign cloud providers.
- It is designed for on-premises or domestic simulation sandbox deployment inside Saudi financial institutions.
- The current `README.md` is intentionally concise; the `Highlights/AMAD_HACKATHON_SUBMISSION.md` file contains the full AMAD pitch and regulatory explanation.

## Contact

If you need help extending this README or generating embedded figures, consult the hackathon brief in `Highlights/AMAD_HACKATHON_SUBMISSION.md` and the code scripts listed above.
