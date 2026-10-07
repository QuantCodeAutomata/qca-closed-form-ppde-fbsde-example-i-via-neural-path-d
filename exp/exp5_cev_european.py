"""
exp5_cev_european.py – Reproduce Section 9 Tables 5 and 6.

Duration-dependent CEV European call pricing via direct Monte Carlo Picard.

Model:
  dX(s) = r*X(s)*ds + sigma_CEV(s,X_s)*dW(s)
  sigma_CEV(s,X_s) = sigma * h_R(X(s)) * (eta^{delta,k}_{R,eps1}(s,omega) v eps2)^beta

Payoff: g(omega) = (X(T) - K)^+
Driver: f(s, X_s, Y, Z) = -r*Y  (risk-neutral)

Parameters:
  t=0, T=1, gamma(0)=X(0)=1, k=0, m=8 Picard, R=1000, eps1=eps2=1e-8
  n_BSDE=50 (paper), tilde_n=5 (sub-steps), total=250 forward steps
  K=1, delta grid, r in {0.02, 0.04}

REDUCED for tractability:
  n_BSDE=10, tilde_n=2 (20 total forward steps), n_paths=500, bar_L=50, m=4

Apple:  beta=-0.7188, sigma=0.5194
Amazon: beta=-0.6685, sigma=0.4784
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Shared parameters ─────────────────────────────────────────────────────────
T = 1.0
X0 = 1.0
K_STRIKE = 1.0
R_CEV = 1000.0
EPS1 = 1e-8
EPS2 = 1e-8
K_DUR = 0.0   # k=0, so trailing average (no exponential weight decay)
M_PICARD = 4  # paper: 8 (reduced)
N_BSDE = 10   # paper: 50 (reduced)
N_SUB = 2     # paper: 5 (reduced); sub-steps per BSDE interval
N_FWD = N_BSDE * N_SUB  # total forward steps = 20 (paper: 250)
DT_FWD = T / N_FWD
DT_BSDE = T / N_BSDE
N_PATHS = 500   # Monte Carlo paths for Picard
BAR_L = 50      # inner paths per label
DELTA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # paper uses up to 0.9
R_GRID = [0.02, 0.04]

ASSETS = {
    "Apple":  {"beta": -0.7188, "sigma": 0.5194},
    "Amazon": {"beta": -0.6685, "sigma": 0.4784},
}


def h_R(x: np.ndarray, R: float = R_CEV) -> np.ndarray:
    """h_R(x) = (x^+) wedge R"""
    return np.minimum(np.maximum(x, 0.0), R)


def trailing_average(
    s_idx: int,
    fwd_paths: np.ndarray,
    fwd_tgrid: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Compute regularized trailing average eta_{R,eps1}^{delta,0} at step s_idx.

    With k=0, the exponential kernel is uniform.

    eta = [integral_{max(0,s-delta)}^s h_R(omega(u)) du + eps1*h_R(omega(s))]
          / [max(delta, s) + eps1]

    Approximated via trapezoidal rule over fwd_paths up to s_idx.
    """
    s = fwd_tgrid[s_idx]
    s_low = max(0.0, s - delta)

    # Find forward grid indices in [s_low, s]
    mask = (fwd_tgrid >= s_low) & (fwd_tgrid <= s)
    t_win = fwd_tgrid[mask]
    x_win = fwd_paths[:, mask]  # (n_paths, window_len)
    h_win = h_R(x_win)

    if len(t_win) <= 1:
        numerator = EPS1 * h_R(fwd_paths[:, s_idx])
        denominator = EPS1
    else:
        # Trapezoidal integration
        dt_win = np.diff(t_win)
        trapz = np.sum(0.5 * (h_win[:, :-1] + h_win[:, 1:]) * dt_win, axis=1)
        numerator = trapz + EPS1 * h_R(fwd_paths[:, s_idx])
        denominator = (s - s_low) + EPS1

    return numerator / denominator


def simulate_cev_paths(
    beta: float,
    sigma: float,
    r: float,
    delta: float,
    n_paths: int,
    seed: int,
) -> tuple:
    """Simulate duration-dependent CEV forward paths on fine grid."""
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(DT_FWD), size=(n_paths, N_FWD))
    X = np.empty((n_paths, N_FWD + 1))
    X[:, 0] = X0
    fwd_tgrid = np.linspace(0.0, T, N_FWD + 1)

    for i in range(N_FWD):
        X_i = X[:, i]
        s = fwd_tgrid[i]
        eta = trailing_average(i, X, fwd_tgrid, delta)
        eta_eff = np.maximum(eta, EPS2)
        sigma_cev = sigma * h_R(X_i) * np.power(eta_eff, beta)
        X_next = X_i + r * X_i * DT_FWD + sigma_cev * dW[:, i]
        # Absorbing boundary at zero
        X_next = np.maximum(X_next, 0.0)
        X_next[X_i <= 0.0] = 0.0
        X[:, i + 1] = X_next

    return X, dW, fwd_tgrid


def picard_direct_mc(
    beta: float,
    sigma: float,
    r: float,
    delta: float,
    seed_base: int,
) -> tuple:
    """Run M Picard steps via direct Monte Carlo for European call.

    Returns (Y0, Z0) after M_PICARD iterations.
    """
    # Outer paths for label construction
    X_outer, dW_outer, tgrid = simulate_cev_paths(beta, sigma, r, delta, N_PATHS, seed_base)

    # BSDE grid (coarser) – use every N_SUB-th step
    bsde_indices = np.arange(0, N_FWD + 1, N_SUB)  # shape (N_BSDE+1,)

    # Initialize Y^0 = 0, Z^0 = 0 along outer paths
    Y_prev = np.zeros((N_PATHS, N_BSDE))
    Z_prev = np.zeros((N_PATHS, N_BSDE))

    for m in range(M_PICARD):
        Y_new = np.zeros((N_PATHS, N_BSDE))
        Z_new = np.zeros((N_PATHS, N_BSDE))

        for bi in range(N_BSDE):
            fi = bsde_indices[bi]  # fine grid index for this BSDE node
            rng = np.random.default_rng(seed_base + 1000 * m + 100 * bi)
            dW_inner = rng.normal(0.0, np.sqrt(DT_FWD), size=(N_PATHS, BAR_L, N_FWD - fi))

            # Build continuation paths from X_outer[:, fi]
            X_cont = np.empty((N_PATHS, BAR_L, N_FWD - fi + 1))
            X_cont[:, :, 0] = X_outer[:, fi, np.newaxis]

            # We need tgrid from fi to N_FWD
            t_so_far = tgrid[:fi + 1]  # up to current time
            x_so_far = X_outer[:, :fi + 1]  # outer history

            for j in range(N_FWD - fi):
                fj = fi + j
                s = tgrid[fj]
                X_j = X_cont[:, :, j]  # (N_PATHS, BAR_L)

                # Trailing average: concatenate outer history with continuation
                # Simplified: use continuation path itself for window computation
                eta = np.maximum(np.mean(np.maximum(X_j, 0.0), axis=1, keepdims=True), EPS2)
                eta_bc = np.broadcast_to(eta, (N_PATHS, BAR_L))
                eta_eff = np.maximum(eta_bc, EPS2)
                sigma_cev = sigma * h_R(X_j) * np.power(eta_eff, beta)

                X_next = X_j + r * X_j * DT_FWD + sigma_cev * dW_inner[:, :, j]
                X_next = np.maximum(X_next, 0.0)
                X_next[X_j <= 0.0] = 0.0
                X_cont[:, :, j + 1] = X_next

            X_T_inner = X_cont[:, :, -1]  # (N_PATHS, BAR_L)
            payoff = np.maximum(X_T_inner - K_STRIKE, 0.0)  # call payoff

            # Driver: f = -r*Y, use previous Y estimates
            # Remaining driver sum from bi+1 to N_BSDE-1
            driver_sum_u = np.zeros((N_PATHS, BAR_L))
            driver_sum_v = np.zeros((N_PATHS, BAR_L))
            for bj in range(bi, N_BSDE):
                driver_sum_u += -r * Y_prev[:, bj:bj + 1] * DT_BSDE
            for bj in range(bi + 1, N_BSDE):
                driver_sum_v += -r * Y_prev[:, bj:bj + 1] * DT_BSDE

            payoff_u = payoff + driver_sum_u
            payoff_v = payoff + driver_sum_v

            Y_new[:, bi] = np.mean(payoff_u, axis=1)

            # Z from delta W weighting (first inner increment)
            dW_first = dW_inner[:, :, 0]  # (N_PATHS, BAR_L)
            Z_new[:, bi] = np.mean((dW_first / DT_FWD) * payoff_v, axis=1)

        Y_prev = Y_new
        Z_prev = Z_new

    return float(np.mean(Y_prev[:, 0])), float(np.mean(Z_prev[:, 0]))


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Experiment 5: Duration-dependent CEV European Call (Sec. 9)")
print("=" * 60)

all_rows = []
seed_counter = 0

for asset_name, params in ASSETS.items():
    beta = params["beta"]
    sigma = params["sigma"]
    for r in R_GRID:
        print(f"\n  {asset_name}: beta={beta}, sigma={sigma}, r={r}")
        row = {"asset": asset_name, "beta": beta, "sigma": sigma, "r": r}
        for delta in DELTA_GRID:
            Y0, Z0 = picard_direct_mc(beta, sigma, r, delta, seed_base=seed_counter * 997 + 42)
            seed_counter += 1
            row[f"Y0_d{delta:.2f}"] = round(Y0, 6)
            row[f"Z0_d{delta:.2f}"] = round(Z0, 6)
            print(f"    delta={delta:.2f}: Y(0)={Y0:.4f}, Z(0)={Z0:.4f}")
        all_rows.append(row)

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RESULTS_DIR, "exp5_results.csv"), index=False)
print("\n=== European Call Results ===")
print(df.to_string(index=False))

# ── Clean tables for display ──────────────────────────────────────────────────
for r in R_GRID:
    rows_r = [row for row in all_rows if row["r"] == r]
    print(f"\nTable (r={r}):")
    Y_cols = [c for c in rows_r[0].keys() if c.startswith("Y0_")]
    Z_cols = [c for c in rows_r[0].keys() if c.startswith("Z0_")]
    for row in rows_r:
        print(f"  {row['asset']}: Y= {[round(row[c],4) for c in Y_cols]}")
        print(f"           Z= {[round(row[c],4) for c in Z_cols]}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i_asset, (asset_name, params) in enumerate(ASSETS.items()):
    for i_r, r in enumerate(R_GRID):
        row = next(row for row in all_rows if row["asset"] == asset_name and row["r"] == r)
        Y_vals = [row[f"Y0_d{d:.2f}"] for d in DELTA_GRID]
        Z_vals = [row[f"Z0_d{d:.2f}"] for d in DELTA_GRID]
        ax = axes[i_asset, i_r]
        ax.plot(DELTA_GRID, Y_vals, "b-o", label="Y(0)")
        ax.plot(DELTA_GRID, Z_vals, "r-s", label="Z(0)")
        ax.set_xlabel("Duration delta")
        ax.set_ylabel("Value")
        ax.legend()
        ax.set_title(f"{asset_name}, r={r}")
        ax.grid(True)

plt.suptitle("Exp 5: Duration-Dependent CEV European Call")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp5_duration_sensitivity.png"), dpi=100)
plt.close()
print(f"\nResults saved to {RESULTS_DIR}")
