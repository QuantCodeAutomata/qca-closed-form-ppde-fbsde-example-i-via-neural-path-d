# Path-Dependent PPDE/FBSDE via Neural Picard Iteration

Reproduction of the numerical experiments from the paper on path-dependent
BSDE/PPDE solving via a neural Picard iteration scheme (Section 6–9).

## Research Overview

This repository reproduces eight experiments:

| Exp | Description | Key Result |
|-----|-------------|------------|
| 1 | Linear BSDE (Example I) | Y(0)→1.0, Z(0)→0.3 |
| 2 | Nonlinear BSDE (Example II) | Y(0)→cos(1)≈0.5403, Z(0)→-σsin(1)≈-0.2524 |
| 3 | Floating-strike lookback call | Y(0)≈0.151 (Euler MC), exact≈0.162986 |
| 4 | Straddle with path-dependent dividends | Y≈94.35, Z≈44.72 |
| 5 | Duration-dependent CEV European call | Apple/Amazon pricing tables |
| 6 | Duration-dependent CEV barrier digital | Stronger duration sensitivity |
| 7 | Discretization-rate sharpness (Remark 3.10) | E[max\|ΔW\|²]~h·log(1/h) |
| 8 | MC estimator concentration (Theorem 5.4) | Exponential tail decay in L |

## Repository Structure

```
.
├── data/               # No external data required (all paths simulated)
├── src/                # Core implementation
│   ├── forward.py      # Brownian path simulation, Euler solvers, concatenation
│   ├── labels.py       # Monte Carlo label computation for Picard updates
│   ├── models.py       # LSTM + MLP neural network (Section 6 architecture)
│   ├── train.py        # Picard iteration training loop (Algorithm 6.1)
│   └── evaluate.py     # Out-of-sample evaluation on 100k test paths
├── exp/                # Experiment scripts
│   ├── exp1_example1.py    # Table 1: linear BSDE
│   ├── exp2_example2.py    # Table 2: nonlinear BSDE
│   ├── exp3_lookback.py    # Table 3: lookback call
│   ├── exp4_straddle.py    # Section 8: straddle
│   ├── exp5_cev_european.py# Tables 5-6: CEV European call
│   ├── exp6_cev_barrier.py # Tables 7-8: CEV barrier digital
│   ├── exp7_discretization_rate.py  # Remark 3.10
│   └── exp8_concentration.py        # Theorem 5.4
├── tests/              # pytest test suite
│   ├── test_forward.py
│   ├── test_models.py
│   ├── test_labels.py
│   └── test_methodology.py
├── results/            # Output: tables, plots, RESULTS.md
└── README.md
```

## Method

The implementation follows the paper's **path-dependent Picard iteration** (eq. 3.1):

- **Zero initialization**: Y⁰ = Z⁰ = 0
- **Path concatenation** (ω ⊕_s ω̃): the exact operator from the paper
- **Label computation**: conditional Monte Carlo over inner continuation paths
- **Neural approximation**: LSTM encoder + dual MLP heads for u^{m+1}, v^{m+1}
- **Loss**: J_{n,L}^{m+1} = (1/(nL)) ΣᵢΣₗ [|U−u|² + |V−v|²]

### Architecture (external assumption, documented per paper)
- LSTM: hidden=64, layers=2
- MLP heads: hidden=64, tanh activation
- Optimizer: Adam (lr=0.01, decay 0.98 every 50 epochs)

## Experiments 5–6 (CEV model)

Use direct Monte Carlo Picard (no neural network), with:
- Duration-dependent CEV volatility σ_CEV(s,X) = σ·h_R(X(s))·(η^{δ,k}(s,ω)∨ε₂)^β
- Absorbing boundary at X=0
- Regularized trailing average for η

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run individual experiments (reduced parameters for speed)
python exp/exp1_example1.py
python exp/exp7_discretization_rate.py   # fast, no neural network
python exp/exp8_concentration.py         # fast, no neural network

# Run all experiments
for i in 1 2 3 4 5 6 7 8; do python exp/exp${i}_*.py; done
```

## Implementation Assumptions

Hyperparameters not fully specified in the paper are documented as external assumptions:

| Parameter | Assumed Value | Source |
|-----------|--------------|--------|
| LSTM hidden size | 64 | External assumption |
| LSTM layers | 2 | External assumption |
| MLP hidden | 64, tanh | External assumption |
| Optimizer | Adam | External assumption |
| Batch size | Full batch | External assumption |
| r (Exp 3) | 0.05 | Inferred from exact benchmark |

## Dependencies

- numpy, scipy, pandas, matplotlib, seaborn
- torch (PyTorch CPU)
- pytest
