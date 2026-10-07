"""
exp1_example1.py – Reproduce Section 7 Example I (Table 1).

Exact solution: Y(0) = 1.0,  Z(0) = sigma = 0.3.

Forward SDE:  dX = sigma dW,  X(0) = gamma(0) = 1.
Terminal:     g(omega) = X(T) + integral_0^T X(s) ds
Driver:       f(s, X_s, Y, Z) = -X(s)  (no Y/Z dependence)

Paper parameters (documented for reference):
  n=70, M=10, K=70, L=100, bar_L=15000, epochs=5000, lr=0.01, decay=0.98/50ep

REDUCED parameters used here for tractability:
  n=10, M=5, K=5, L=20, bar_L=200, epochs=150, test_paths=2000
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train import run_picard_neural

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

# ── Exact benchmark ──────────────────────────────────────────────────────────
Y_EXACT = 1.0
Z_EXACT = 0.3  # = sigma

# ── Reduced config (paper values in comments) ────────────────────────────────
config = dict(
    n_steps=10,          # paper: 70
    dt=0.1,              # = T / n_steps = 1 / 10
    L=20,                # paper: 100
    K=5,                 # paper: 70
    bar_L=200,           # paper: 15000
    n_epochs=150,        # paper: 5000
    lr=0.01,
    lr_decay_factor=0.98,
    lr_decay_every=50,
    sigma=0.3,
    x0=1.0,              # gamma(0) = 1
    test_paths=2000,     # paper: 100000
    seed_outer=42,
    seed_inner=123,
    seed_test=999,
)

print("=" * 60)
print("Experiment 1: Example I (linear path-dependent BSDE)")
print(f"Exact: Y(0)={Y_EXACT:.4f}, Z(0)={Z_EXACT:.4f}")
print("=" * 60)

results = run_picard_neural(example_id=1, config=config, M=5)

# ── Tabulate ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df["err_Y"] = (df["Y0_hat"] - Y_EXACT).abs() / Y_EXACT
df["err_Z"] = (df["Z0_hat"] - Z_EXACT).abs() / abs(Z_EXACT)
print("\n=== Results Table ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS_DIR, "exp1_results.csv"), index=False)

# ── Convergence plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(df["picard_step"], df["Y0_hat"], "b-o", label="Ŷ(0)")
axes[0].axhline(Y_EXACT, color="r", linestyle="--", label=f"Y_exact={Y_EXACT}")
axes[0].set_xlabel("Picard Step m")
axes[0].set_ylabel("Y(0)")
axes[0].legend()
axes[0].set_title("Example I: Y(0) Convergence")

axes[1].plot(df["picard_step"], df["Z0_hat"], "g-o", label="Ẑ(0)")
axes[1].axhline(Z_EXACT, color="r", linestyle="--", label=f"Z_exact={Z_EXACT:.3f}")
axes[1].set_xlabel("Picard Step m")
axes[1].set_ylabel("Z(0)")
axes[1].legend()
axes[1].set_title("Example I: Z(0) Convergence")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp1_convergence.png"), dpi=100)
plt.close()

# ── Loss plot ─────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogy(df["picard_step"], df["loss_Y"], "b-o", label="Loss_Y")
ax2.semilogy(df["picard_step"], df["loss_Z"], "r-s", label="Loss_Z")
ax2.set_xlabel("Picard Step m")
ax2.set_ylabel("Loss (log scale)")
ax2.legend()
ax2.set_title("Example I: Training Losses per Picard Step")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp1_losses.png"), dpi=100)
plt.close()

print(f"\nResults saved to {RESULTS_DIR}")
print("Done.")
