"""
exp3_lookback.py – Reproduce Section 7.3 (Table 3): Floating-strike lookback call.

Payoff: g(omega) = X(T) - inf_{0<=u<=T} X(u)
Driver: f(u, X_u, Y, Z) = -r * Y  (discount)

Exact continuous-time benchmark (paper, r inferred to match 0.162986):
  Y(0) = gamma(0)*Phi(a1) - m(0)*e^{-rT}*Phi(a2)
         - gamma(0)*(sigma^2/(2r))*[Phi(-a1) - e^{-rT}*(m(0)/gamma(0))^{2r/sigma^2}*Phi(-a3)]
  Reported exact value: 0.162986

Euler-discretized Monte Carlo benchmark: reported 0.151638

Since r is not stated in the paper, we assume r=0.05 and document this as an
implementation ambiguity.  We also report results for r=0.04 and r=0.06 so the
reader can see sensitivity.

REDUCED parameters for tractability:
  n=10, M=5, K=5, L=20, bar_L=200, test_paths=2000
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.forward import simulate_brownian_increments, euler_forward_additive
from src.models import build_model
from src.evaluate import evaluate_on_test

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
SIGMA = 0.2   # paper specifies sigma=0.2 for Section 7.3
R_INFERRED = 0.05  # ambiguous in paper; documented assumption
X0 = 1.0
T = 1.0
N_STEPS = 10   # paper: 70 (reduced)
DT = T / N_STEPS
M_PICARD = 5   # paper: 10 (reduced)
K = 5          # paper: 70 (reduced)
L = 20         # paper: 100 (reduced)
BAR_L = 200    # paper: 15000 (reduced)
N_EPOCHS = 150 # paper: not specified for lookback; following Exp1 pattern
N_TEST = 2000  # paper: 100000 (reduced)
N_BENCHMARK = 5000  # paper: 200000 (reduced)


def exact_lookback(sigma: float, r: float, x0: float = 1.0, T: float = 1.0) -> float:
    """Exact continuous-time floating-strike lookback call price.

    Formula: Y(0) = x0*Phi(a1) - m0*exp(-r*T)*Phi(a2)
               - x0*(sigma^2/(2r))*[Phi(-a1) - exp(-r*T)*(m0/x0)^{2r/sigma^2}*Phi(-a3)]
    where m0 = x0 (running min at time 0 equals x0).
    """
    m0 = x0  # inf_{0<=u<=0} gamma(u) = gamma(0) = x0
    a1 = (np.log(x0 / m0) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    a2 = a1 - sigma * np.sqrt(T)
    a3 = a1 - 2 * r * np.sqrt(T) / sigma
    val = (
        x0 * norm.cdf(a1)
        - m0 * np.exp(-r * T) * norm.cdf(a2)
        - x0 * (sigma ** 2 / (2 * r)) * (
            norm.cdf(-a1)
            - np.exp(-r * T) * (m0 / x0) ** (2 * r / sigma ** 2) * norm.cdf(-a3)
        )
    )
    return val


def euler_mc_benchmark(
    sigma: float, r: float, n_paths: int, n_steps: int, dt: float, x0: float = 1.0, seed: int = 5555
) -> float:
    """Euler-discretized Monte Carlo lookback call price."""
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        X[:, i + 1] = X[:, i] + sigma * dW[:, i]
    # running min
    X_min = np.min(X, axis=1)
    payoff = np.exp(-r * T) * (X[:, -1] - X_min)
    return float(np.mean(payoff))


# ── Label computation for lookback ────────────────────────────────────────────

def compute_labels_lookback(
    outer_X: np.ndarray,
    t_i: int,
    n_steps: int,
    dt: float,
    sigma: float,
    r: float,
    bar_L: int,
    prev_u: np.ndarray,
    prev_v: np.ndarray,
    seed_base: int,
) -> tuple:
    """Compute Y/Z labels for lookback call at time node t_i."""
    L_out = outer_X.shape[0]
    n_future = n_steps - t_i
    rng = np.random.default_rng(seed_base + t_i * 1000)
    dW_inner = rng.normal(0.0, np.sqrt(dt), size=(L_out, bar_L, n_future))

    # Continuation paths shape (L_out, bar_L, n_future+1)
    X_cont = np.empty((L_out, bar_L, n_future + 1))
    X_cont[:, :, 0] = outer_X[:, t_i, np.newaxis]
    for j in range(n_future):
        X_cont[:, :, j + 1] = X_cont[:, :, j] + sigma * dW_inner[:, :, j]

    # Full path (L_out, bar_L, n_steps+1)
    X_outer_left = outer_X[:, :t_i + 1][:, np.newaxis, :]
    X_full = np.concatenate([
        np.broadcast_to(X_outer_left, (L_out, bar_L, t_i + 1)),
        X_cont[:, :, 1:],
    ], axis=-1)

    # Running min of full path (important: include history from t=0)
    X_min_full = np.min(X_full, axis=-1)  # (L_out, bar_L)
    X_T = X_full[:, :, -1]

    # Payoff = X(T) - min over whole path (no discount, driver handles discounting)
    # For Picard labels with driver f=-rY, we approximate the driver using prev_u
    # U-label: sum_{j=i}^{n-1} dt * (-r) * prev_u[l, j]  (driver contribution)
    # Using prev Picard Y values along outer path
    # For first iterate, prev_u=0, so pure terminal payoff
    payoff_terminal = X_T - X_min_full  # (L_out, bar_L)

    # Driver sum from j=i to n-1: f(t_j,...)=-r*Y^m(t_j)
    # Approximated using outer path prev_u values (simplified)
    driver_sum_u = np.zeros((L_out, bar_L))
    driver_sum_v = np.zeros((L_out, bar_L))
    for j in range(t_i, n_steps):
        prev_Y_j = prev_u[:, j][:, np.newaxis]  # (L_out, 1)
        driver_sum_u += -r * prev_Y_j * dt
    for j in range(t_i + 1, n_steps):
        prev_Y_j = prev_u[:, j][:, np.newaxis]
        driver_sum_v += -r * prev_Y_j * dt

    payoff_u = payoff_terminal + driver_sum_u
    payoff_v = payoff_terminal + driver_sum_v

    U_labels = np.mean(payoff_u, axis=1)  # (L_out,)

    dW_first = dW_inner[:, :, 0]  # (L_out, bar_L)
    V_labels = np.mean((dW_first / dt) * payoff_v, axis=1)  # (L_out,)

    return U_labels, V_labels


def run_lookback_picard(r: float) -> dict:
    """Run M Picard steps for the lookback call with interest rate r."""
    device = torch.device("cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.98)

    prev_u_np = np.zeros((K * L, N_STEPS))
    prev_v_np = np.zeros((K * L, N_STEPS))

    results = {"picard_step": [], "Y0_hat": [], "Z0_hat": [], "loss_Y": [], "loss_Z": []}

    for m in range(1, M_PICARD + 1):
        print(f"  Picard m={m}")
        # Generate outer paths
        rng = np.random.default_rng(42 + m * 37)
        dW_outer = rng.normal(0.0, np.sqrt(DT), size=(K * L, N_STEPS))
        X_outer = np.empty((K * L, N_STEPS + 1))
        X_outer[:, 0] = X0
        for i in range(N_STEPS):
            X_outer[:, i + 1] = X_outer[:, i] + SIGMA * dW_outer[:, i]

        # Compute labels
        U_all = np.zeros((K * L, N_STEPS))
        V_all = np.zeros((K * L, N_STEPS))
        for t_i in range(N_STEPS):
            U_all[:, t_i], V_all[:, t_i] = compute_labels_lookback(
                outer_X=X_outer,
                t_i=t_i,
                n_steps=N_STEPS,
                dt=DT,
                sigma=SIGMA,
                r=r,
                bar_L=BAR_L,
                prev_u=prev_u_np,
                prev_v=prev_v_np,
                seed_base=123 + m * 53,
            )

        dW_tensor = torch.tensor(dW_outer[:, :, np.newaxis], dtype=torch.float32)
        U_tensor = torch.tensor(U_all, dtype=torch.float32)
        V_tensor = torch.tensor(V_all, dtype=torch.float32)

        model.train()
        losses_Y, losses_Z = [], []
        for epoch in range(1, N_EPOCHS + 1):
            optimizer.zero_grad()
            u_pred, v_pred = model(dW_tensor)
            ly = torch.mean((U_tensor - u_pred) ** 2)
            lz = torch.mean((V_tensor - v_pred) ** 2)
            (ly + lz).backward()
            optimizer.step()
            scheduler.step()
            losses_Y.append(ly.item())
            losses_Z.append(lz.item())

        # Update prev
        model.eval()
        with torch.no_grad():
            u_np, v_np = model(dW_tensor)
            prev_u_np = u_np.numpy()
            prev_v_np = v_np.numpy()

        # Evaluate on test
        Y0_hat, Z0_hat = evaluate_on_test(model, N_TEST, N_STEPS, DT, SIGMA, X0, seed=999 + m * 71)
        avg_ly = float(np.mean(losses_Y[-max(1, N_EPOCHS // 5):]))
        avg_lz = float(np.mean(losses_Z[-max(1, N_EPOCHS // 5):]))
        results["picard_step"].append(m)
        results["Y0_hat"].append(Y0_hat)
        results["Z0_hat"].append(Z0_hat)
        results["loss_Y"].append(avg_ly)
        results["loss_Z"].append(avg_lz)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Experiment 3: Floating-strike Lookback Call (Section 7.3)")
print(f"sigma={SIGMA}, r={R_INFERRED} (inferred; ambiguous in paper)")
print("=" * 60)

# Benchmarks
exact_val = exact_lookback(SIGMA, R_INFERRED, X0, T)
mc_val = euler_mc_benchmark(SIGMA, R_INFERRED, N_BENCHMARK, N_STEPS, DT, X0)
print(f"Exact continuous benchmark (r={R_INFERRED}): {exact_val:.6f}")
print(f"  Paper-reported exact value: 0.162986")
print(f"Euler MC benchmark (r={R_INFERRED}): {mc_val:.6f}")
print(f"  Paper-reported MC benchmark: 0.151638")

results = run_lookback_picard(r=R_INFERRED)

df = pd.DataFrame(results)
df["mc_benchmark"] = mc_val
df["exact_benchmark"] = exact_val
print("\n=== Lookback Results ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS_DIR, "exp3_results.csv"), index=False)

# Save benchmark sensitivity table
bench_rows = []
for r_test in [0.03, 0.04, 0.05, 0.06]:
    exact_r = exact_lookback(SIGMA, r_test, X0, T)
    mc_r = euler_mc_benchmark(SIGMA, r_test, N_BENCHMARK, N_STEPS, DT, X0, seed=5555 + int(r_test * 100))
    bench_rows.append({"r": r_test, "exact_continuous": exact_r, "mc_euler": mc_r})
df_bench = pd.DataFrame(bench_rows)
print("\nBenchmark sensitivity (r ambiguity):")
print(df_bench.to_string(index=False))
df_bench.to_csv(os.path.join(RESULTS_DIR, "exp3_benchmark_sensitivity.csv"), index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(df["picard_step"], df["Y0_hat"], "b-o", label="Ŷ(0) neural")
axes[0].axhline(mc_val, color="orange", linestyle="--", label=f"MC bench={mc_val:.4f}")
axes[0].axhline(exact_val, color="r", linestyle="--", label=f"Exact={exact_val:.4f}")
axes[0].set_xlabel("Picard Step m")
axes[0].set_ylabel("Y(0)")
axes[0].legend(fontsize=8)
axes[0].set_title("Lookback Call: Y(0)")
axes[1].plot(df["picard_step"], df["Z0_hat"], "g-o", label="Ẑ(0) neural")
axes[1].set_xlabel("Picard Step m")
axes[1].set_ylabel("Z(0)")
axes[1].legend()
axes[1].set_title("Lookback Call: Z(0)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp3_convergence.png"), dpi=100)
plt.close()
print(f"\nResults saved to {RESULTS_DIR}")
