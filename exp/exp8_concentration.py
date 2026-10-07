"""
exp8_concentration.py – Theorem 5.4 concentration test (empirical).

Tests whether the Monte Carlo estimators for the Y-label (Phi) and Z-label
(Psi^k) exhibit exponential concentration as inner sample size L grows.

Uses Example I as the base model (simple dynamics, known exact values).
Fixes a specific outer path omega and time index i, then generates R repeated
independent estimations for each inner sample size L.

Note: the theorem statement in the paper appears typographically malformed, so
we target qualitative exponential tail decay rather than exact coefficient matching.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model parameters (Example I) ──────────────────────────────────────────────
SIGMA = 0.3
X0 = 1.0
N_STEPS = 70    # fine grid as in paper
DT = 1.0 / N_STEPS

# Fix the outer path and time index
T_I = 35        # midpoint of grid
SEED_OUTER = 42

# Inner MC parameters
L_GRID = [50, 100, 200, 500, 1000, 2000, 5000]
N_REPS = 3000   # independent repetitions for tail estimation
EPS_GRID = [0.02, 0.05, 0.10]   # threshold values for tail probability

# Reference (high-precision) estimator uses L_REF samples
L_REF = 100_000


def example1_terminal(X: np.ndarray) -> np.ndarray:
    """g(omega) = X(T) + sum_{j<n} X(t_j) * dt"""
    return X[:, -1] + np.sum(X[:, :-1], axis=1) * DT


def driver_sum_example1(X_path: np.ndarray, t_start: int) -> np.ndarray:
    """sum_{j=t_start}^{n-1} dt * f(t_j, X_j) = -dt * sum_{j=t_start}^{n-1} X_j"""
    return -DT * np.sum(X_path[:, t_start:N_STEPS], axis=1)


def build_continuation(outer_X_ti: np.ndarray, inner_dW: np.ndarray) -> np.ndarray:
    """Build continuation paths from outer_X at time t_i.

    Parameters
    ----------
    outer_X_ti : outer path value at t_i, shape (n_paths,)
    inner_dW   : inner increments from t_i onward, shape (n_paths, n_future)

    Returns X_cont: shape (n_paths, n_future+1)
    """
    n_paths, n_future = inner_dW.shape
    X_cont = np.empty((n_paths, n_future + 1))
    X_cont[:, 0] = outer_X_ti
    for j in range(n_future):
        X_cont[:, j + 1] = X_cont[:, j] + SIGMA * inner_dW[:, j]
    return X_cont


def single_phi_estimate(
    outer_val_at_ti: float,
    L: int,
    rng: np.random.Generator,
) -> float:
    """Compute one hat_Phi estimate from L inner paths."""
    n_future = N_STEPS - T_I
    dW_inner = rng.normal(0.0, np.sqrt(DT), size=(L, n_future))
    X_cont = build_continuation(
        np.full(L, outer_val_at_ti),
        dW_inner,
    )
    # Full path: prepend fixed outer prefix (just the last value for terminal calculation)
    # We approximate the full path functional using only continuation
    g_vals = example1_terminal(X_cont)
    driver_vals = driver_sum_example1(X_cont, t_start=0)  # driver from T_I onward
    return float(np.mean(g_vals + driver_vals))


def single_psi_estimate(
    outer_val_at_ti: float,
    L: int,
    rng: np.random.Generator,
) -> float:
    """Compute one hat_Psi estimate (Z-label, weighted by dW_i/h_i)."""
    n_future = N_STEPS - T_I
    dW_inner = rng.normal(0.0, np.sqrt(DT), size=(L, n_future))
    X_cont = build_continuation(
        np.full(L, outer_val_at_ti),
        dW_inner,
    )
    g_vals = example1_terminal(X_cont)
    driver_vals = driver_sum_example1(X_cont, t_start=1)  # j=i+1 for Z
    payoff_v = g_vals + driver_vals
    # Weight by first increment delta W_i / h
    return float(np.mean((dW_inner[:, 0] / DT) * payoff_v))


# ── Generate fixed outer path ─────────────────────────────────────────────────
rng_outer = np.random.default_rng(SEED_OUTER)
dW_outer_full = rng_outer.normal(0.0, np.sqrt(DT), size=N_STEPS)
X_outer = np.zeros(N_STEPS + 1)
X_outer[0] = X0
for i in range(N_STEPS):
    X_outer[i + 1] = X_outer[i] + SIGMA * dW_outer_full[i]
outer_val_at_ti = X_outer[T_I]
print(f"Fixed outer path: X(t_{T_I}) = {outer_val_at_ti:.4f}")

# ── Compute reference (high-precision) estimates ───────────────────────────────
rng_ref = np.random.default_rng(9999)
phi_ref = single_phi_estimate(outer_val_at_ti, L_REF, rng_ref)
psi_ref = single_psi_estimate(outer_val_at_ti, L_REF, rng_ref)
print(f"Reference Phi (Y-type): {phi_ref:.6f}")
print(f"Reference Psi (Z-type): {psi_ref:.6f}")

# ── Main concentration experiment ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Experiment 8: Concentration of MC Estimators (Theorem 5.4)")
print("=" * 60)

results = []

for L in L_GRID:
    print(f"\n  L={L}: generating {N_REPS} reps...")
    rng_main = np.random.default_rng(42 + L)

    phi_reps = np.array([
        single_phi_estimate(outer_val_at_ti, L, rng_main)
        for _ in range(N_REPS)
    ])
    psi_reps = np.array([
        single_psi_estimate(outer_val_at_ti, L, rng_main)
        for _ in range(N_REPS)
    ])

    phi_err = np.abs(phi_reps - phi_ref)
    psi_err = np.abs(psi_reps - psi_ref)

    var_phi = float(np.var(phi_reps))
    var_psi = float(np.var(psi_reps))

    row = {
        "L": L,
        "var_Phi": var_phi,
        "var_Psi": var_psi,
    }
    for eps in EPS_GRID:
        p_phi = float(np.mean(phi_err > eps))
        p_psi = float(np.mean(psi_err > eps))
        row[f"P(Phi,eps={eps:.2f})"] = p_phi
        row[f"P(Psi,eps={eps:.2f})"] = p_psi

    results.append(row)
    print(f"    Var(Phi)={var_phi:.4e}, Var(Psi)={var_psi:.4e}")
    for eps in EPS_GRID:
        print(f"    eps={eps:.2f}: P(Phi>eps)={row[f'P(Phi,eps={eps:.2f})']:.4f}, "
              f"P(Psi>eps)={row[f'P(Psi,eps={eps:.2f})']:.4f}")

df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, "exp8_results.csv"), index=False)
print("\n=== Concentration Table ===")
print(df.to_string(index=False))

# ── Semilog plot ────────────────────────────────────────────────────────────────
L_vals = df["L"].values
n_eps = len(EPS_GRID)
n_cols = n_eps
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
if n_cols == 1:
    axes = [axes]

for j, eps in enumerate(EPS_GRID):
    ax = axes[j]
    p_phi_col = f"P(Phi,eps={eps:.2f})"
    p_psi_col = f"P(Psi,eps={eps:.2f})"
    p_phi_vals = df[p_phi_col].clip(1e-4, 1.0)
    p_psi_vals = df[p_psi_col].clip(1e-4, 1.0)

    ax.semilogy(L_vals, p_phi_vals, "b-o", label="P(|Phi-ref|>ε)")
    ax.semilogy(L_vals, p_psi_vals, "r-s", label="P(|Psi-ref|>ε)")
    ax.set_xlabel("Inner sample size L")
    ax.set_ylabel("Tail probability (log scale)")
    ax.legend(fontsize=8)
    ax.set_title(f"Concentration: ε={eps}")
    ax.grid(True)

plt.suptitle("Exp 8: Empirical Concentration of MC Estimators")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp8_concentration.png"), dpi=100)
plt.close()

# ── Variance decay plot ────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.loglog(df["L"], df["var_Phi"], "b-o", label="Var(Phi)")
ax2.loglog(df["L"], df["var_Psi"], "r-s", label="Var(Psi)")
# O(1/L) reference line
ref_line = df["var_Phi"].iloc[0] * (L_vals[0] / L_vals)
ax2.loglog(L_vals, ref_line, "k--", label="O(1/L)")
ax2.set_xlabel("Inner sample size L")
ax2.set_ylabel("Estimator variance")
ax2.legend()
ax2.set_title("Variance decay: Phi (Y) vs Psi (Z)")
ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp8_variance_decay.png"), dpi=100)
plt.close()

print(f"\nResults saved to {RESULTS_DIR}")
print("Qualitative expectation: semilog curves should be roughly linear downward,")
print("indicating exponential concentration. Z-type (Psi) should be noisier than Y-type (Phi).")
