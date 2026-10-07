"""
exp7_discretization_rate.py – Remark 3.10 sharpness: |pi| log(1/|pi|) rate.

The path functional g(omega) = sup_{0<=s<=T} |omega(s) - omega(s-)| equals 0 a.s.
for Brownian motion (continuous paths), so the true BSDE solution is Y=Z=0.

Under piecewise-constant approximation, the discretized functional becomes:
    g(X^pi) = max_i |Delta W_i|

Mean-square discretization error:  E[(max_i |Delta W_i|)^2]

For Brownian increments Delta W_i ~ N(0, h) with h = 1/n:
    E[(max_i |Delta W_i|)^2] ~ h * log(1/h)  as h -> 0

This empirically validates that path dependence changes the Markovian O(h) rate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Grid of mesh sizes ─────────────────────────────────────────────────────────
N_GRID = [10, 20, 40, 80, 160, 320, 640]
N_PATHS = 200_000  # paths per mesh size

print("=" * 60)
print("Experiment 7: Sharpness of |pi|*log(1/|pi|) discretization rate")
print(f"Paths per mesh: {N_PATHS:,}")
print("=" * 60)

rows = []

for n in N_GRID:
    h = 1.0 / n
    rng = np.random.default_rng(12345 + n)
    # Shape (N_PATHS, n): all increments Delta W_i ~ N(0, h)
    dW = rng.normal(0.0, np.sqrt(h), size=(N_PATHS, n))

    # Discretized path-dependent functional: max_i |Delta W_i|
    max_abs_dW = np.max(np.abs(dW), axis=1)  # (N_PATHS,)

    # Mean-square error = E[(max_i |Delta W_i|)^2]
    mse = float(np.mean(max_abs_dW ** 2))

    # Analytical scaling factors
    ratio_to_h = mse / h
    ratio_to_h_log = mse / (h * np.log(1.0 / h))

    rows.append({
        "n": n,
        "h": h,
        "log(1/h)": float(np.log(1.0 / h)),
        "E[(max|DW|)^2]": mse,
        "ratio_to_h": ratio_to_h,
        "ratio_to_h_log": ratio_to_h_log,
    })
    print(f"  n={n:3d}, h={h:.4f}: E[max|DW|^2]={mse:.6f}, "
          f"ratio/h={ratio_to_h:.4f}, ratio/(h*log(1/h))={ratio_to_h_log:.4f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "exp7_results.csv"), index=False)
print("\n=== Discretization Rate Table ===")
print(df.to_string(index=False))

# ── Regression analysis ────────────────────────────────────────────────────────
h_vals = df["h"].values
mse_vals = df["E[(max|DW|)^2]"].values

# Fit 1: E ~ c1 * h
c1_fit, _ = curve_fit(lambda h, c: c * h, h_vals, mse_vals, p0=[1.0])
mse_fit_h = c1_fit[0] * h_vals

# Fit 2: E ~ c2 * h * log(1/h)
c2_fit, _ = curve_fit(
    lambda h, c: c * h * np.log(1.0 / h), h_vals, mse_vals, p0=[1.0]
)
mse_fit_hlog = c2_fit[0] * h_vals * np.log(1.0 / h_vals)

# Residuals
ss_res_h = np.sum((mse_vals - mse_fit_h) ** 2)
ss_res_hlog = np.sum((mse_vals - mse_fit_hlog) ** 2)
ss_tot = np.sum((mse_vals - mse_vals.mean()) ** 2)
r2_h = 1 - ss_res_h / ss_tot
r2_hlog = 1 - ss_res_hlog / ss_tot

print(f"\nRegression fit R^2:  O(h) model: {r2_h:.6f},  O(h*log(1/h)) model: {r2_hlog:.6f}")
print(f"O(h*log(1/h)) model constant: c = {c2_fit[0]:.4f}")
print("Interpretation: R^2 closer to 1.0 for h*log(1/h) confirms sharper rate.")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Log-log plot of E vs h
axes[0].loglog(h_vals, mse_vals, "ko-", label="Empirical E[max|DW|^2]")
axes[0].loglog(h_vals, mse_fit_h, "b--", label=f"c*h (R²={r2_h:.4f})")
axes[0].loglog(h_vals, mse_fit_hlog, "r-.", label=f"c*h*log(1/h) (R²={r2_hlog:.4f})")
axes[0].set_xlabel("Mesh size h = 1/n")
axes[0].set_ylabel("E[(max|ΔW|)²]")
axes[0].legend(fontsize=8)
axes[0].set_title("Log-log scaling comparison")
axes[0].grid(True)

# Panel 2: Ratio E / h  (should grow slowly if rate is h*log(1/h))
axes[1].plot(df["n"], df["ratio_to_h"], "b-o")
axes[1].set_xlabel("n = 1/h")
axes[1].set_ylabel("E[(max|ΔW|)²] / h")
axes[1].set_title("Ratio to h (should grow slowly)")
axes[1].grid(True)

# Panel 3: Ratio E / (h*log(1/h))  (should stabilize)
axes[2].plot(df["n"], df["ratio_to_h_log"], "r-o")
axes[2].set_xlabel("n = 1/h")
axes[2].set_ylabel("E[(max|ΔW|)²] / (h·log(1/h))")
axes[2].set_title("Ratio to h·log(1/h) (should stabilize)")
axes[2].axhline(c2_fit[0], color="gray", linestyle="--", label=f"c={c2_fit[0]:.3f}")
axes[2].legend()
axes[2].grid(True)

plt.suptitle("Exp 7: Sharpness of |π|·log(1/|π|) Discretization Rate")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp7_discretization_rate.png"), dpi=100)
plt.close()

# ── Save regression summary ────────────────────────────────────────────────────
reg_summary = pd.DataFrame({
    "model": ["O(h)", "O(h*log(1/h))"],
    "constant": [float(c1_fit[0]), float(c2_fit[0])],
    "R_squared": [r2_h, r2_hlog],
})
reg_summary.to_csv(os.path.join(RESULTS_DIR, "exp7_regression.csv"), index=False)
print(f"\nResults saved to {RESULTS_DIR}")
