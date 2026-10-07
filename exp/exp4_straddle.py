"""
exp4_straddle.py – Reproduce Section 8: Truncated path-dependent BSDE.

Black-Scholes straddle with path-dependent dividend yield.

Parameters (exact from paper Section 8):
  mu=0.1, sigma=0.4, r=0.03, t=0.1, T=1, n0=100, n=40, c=0.01, q=0.015
  eta_1=-1000, eta_2=1000, gamma(s)=400*(1-s)^2, K=324 (strike)
  D = q/(2*gamma(t)), n_delta=5, m=10 Picard iterations

REDUCED for tractability:
  m=5 Picard steps, n=10 time steps, L=10, bar_L=100, epochs=100

Expected: Y ~ 94.3490, Z ~ 44.7177
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models import build_model
from src.evaluate import evaluate_on_test

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Exact paper parameters ────────────────────────────────────────────────────
MU = 0.1
SIGMA_BS = 0.4
R = 0.03
T_START = 0.1   # t = 0.1 (historical initial time)
T_END = 1.0     # T = 1
N_STEPS = 10    # paper: 40 (reduced)
DT = (T_END - T_START) / N_STEPS
C = 0.01
Q = 0.015
ETA_1 = -1000.0
ETA_2 = 1000.0
STRIKE = 324.0
N_DELTA = 5     # integration sub-steps for dividend average
M_PICARD = 5    # paper: 10 (reduced)
L = 10          # paper: not specified (reduced)
BAR_L = 100     # paper: not specified (reduced)
N_EPOCHS = 100  # paper: not specified (reduced)

# Historical path: gamma(s) = 400*(1-s)^2 for s in [0, t=0.1]
# Initial value X(t=0.1) = gamma(0.1) = 400*(1-0.1)^2
def gamma_hist(s: float) -> float:
    return 400.0 * (1.0 - s) ** 2

X0 = gamma_hist(T_START)    # initial value at t=0.1
D = Q / (2.0 * gamma_hist(T_START))  # D = q / (2*gamma(t))

print(f"X0 = gamma(t=0.1) = {X0:.4f}")
print(f"D  = q / (2*gamma(t)) = {D:.6f}")

# Historical time grid [0, t]  for dividend computation
N_HIST = 10  # points for historical path representation
T_HIST = np.linspace(0.0, T_START, N_HIST + 1)
GAMMA_HIST = np.array([gamma_hist(s) for s in T_HIST])


def h_trunc(x: np.ndarray, eta1: float = ETA_1, eta2: float = ETA_2) -> np.ndarray:
    """Z-truncation: h^{eta1,eta2}(x) = clip(x, eta1, eta2)."""
    return np.clip(x, eta1, eta2)


def dividend_rate(u: float, path_u: np.ndarray, t_grid_u: np.ndarray) -> np.ndarray:
    """Compute dividend rate delta(u, X_u) for each path.

    delta(u, X_u) = min( (D/c) * sum_{i=1}^{n_delta} X(u-c+(i-1)*c/n_delta) * c/n_delta, q )

    Parameters
    ----------
    u         : current time
    path_u    : full concatenated path values up to u, shape (n_paths, len_hist)
    t_grid_u  : time grid up to u, shape (len_hist,)

    Returns
    -------
    delta_vals : shape (n_paths,)
    """
    n_paths = path_u.shape[0]
    sub_times = np.array([
        max(0.0, u - C + (i - 1) * C / N_DELTA)
        for i in range(1, N_DELTA + 1)
    ])
    # Interpolate path values at sub-times
    interp_vals = np.zeros((n_paths, N_DELTA))
    for k, ts in enumerate(sub_times):
        interp_vals[:, k] = np.interp(ts, t_grid_u, path_u.T)
    avg = (D / C) * np.sum(interp_vals, axis=1) * (C / N_DELTA)
    return np.minimum(avg, Q)


def simulate_gbm_with_history(n_paths: int, seed: int) -> tuple:
    """Simulate GBM paths from T_START to T_END and build full history.

    Returns
    -------
    X         : GBM forward paths, shape (n_paths, N_STEPS+1)
    dW        : increments, shape (n_paths, N_STEPS)
    t_grid    : time grid, shape (N_STEPS+1,)
    """
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(DT), size=(n_paths, N_STEPS))
    X = np.empty((n_paths, N_STEPS + 1))
    X[:, 0] = X0
    t_grid = np.linspace(T_START, T_END, N_STEPS + 1)

    for i in range(N_STEPS):
        u = t_grid[i]
        # Build concatenated path history up to current time
        # Combine historical gamma + current X up to index i
        t_so_far = np.concatenate([T_HIST, t_grid[1:i + 1]])
        x_so_far = np.concatenate([
            np.broadcast_to(GAMMA_HIST, (n_paths, len(T_HIST))),
            X[:, 1:i + 1],
        ], axis=1)

        # GBM step with path-dependent dividend in drift (driver only, not forward SDE)
        # Forward SDE: dX = mu X dt + sigma X dW  (dividend only enters BSDE driver)
        X_i = X[:, i]
        X[:, i + 1] = X_i + MU * X_i * DT + SIGMA_BS * X_i * dW[:, i]

    return X, dW, t_grid


def compute_labels_straddle(
    X: np.ndarray,
    dW: np.ndarray,
    t_grid: np.ndarray,
    t_i: int,
    prev_u: np.ndarray,
    prev_v: np.ndarray,
    seed_base: int,
) -> tuple:
    """U/V labels for straddle BSDE at time node t_i."""
    L_out = X.shape[0]
    n_future = N_STEPS - t_i
    rng = np.random.default_rng(seed_base + t_i * 1000)
    dW_inner = rng.normal(0.0, np.sqrt(DT), size=(L_out, BAR_L, n_future))

    # Build continuation X paths
    X_cont = np.empty((L_out, BAR_L, n_future + 1))
    X_cont[:, :, 0] = X[:, t_i, np.newaxis]
    for j in range(n_future):
        X_j = X_cont[:, :, j]
        X_cont[:, :, j + 1] = X_j + MU * X_j * DT + SIGMA_BS * X_j * dW_inner[:, :, j]

    X_T = X_cont[:, :, -1]  # (L_out, BAR_L)

    # Terminal payoff: |X(T) - K|
    payoff_terminal = np.abs(X_T - STRIKE)  # (L_out, BAR_L)

    # Driver sum: f = r*Y + ((mu+delta-r)/sigma)*h^{eta}(Z)
    # Using previous Picard estimates for simplification
    driver_sum_u = np.zeros((L_out, BAR_L))
    for j in range(t_i, N_STEPS):
        prev_Y_j = prev_u[:, j][:, np.newaxis]
        prev_Z_j = h_trunc(prev_v[:, j])[:, np.newaxis]
        # Estimate dividend at midpoint using X values
        # Approximate delta ~ D (constant average over short window c=0.01)
        delta_approx = Q / 2  # simplified approximation
        theta_j = (MU + delta_approx - R) / SIGMA_BS * prev_Z_j
        driver_sum_u += (R * prev_Y_j + theta_j) * DT

    driver_sum_v = np.zeros((L_out, BAR_L))
    for j in range(t_i + 1, N_STEPS):
        prev_Y_j = prev_u[:, j][:, np.newaxis]
        prev_Z_j = h_trunc(prev_v[:, j])[:, np.newaxis]
        delta_approx = Q / 2
        theta_j = (MU + delta_approx - R) / SIGMA_BS * prev_Z_j
        driver_sum_v += (R * prev_Y_j + theta_j) * DT

    payoff_u = payoff_terminal + driver_sum_u
    payoff_v = payoff_terminal + driver_sum_v

    U_labels = np.mean(payoff_u, axis=1)
    dW_first = dW_inner[:, :, 0]
    V_labels = np.mean((dW_first / DT) * payoff_v, axis=1)

    return U_labels, V_labels


# ── Picard loop ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Experiment 4: Straddle with Path-Dependent Dividends (Sec. 8)")
print(f"Target: Y(0.1)~94.35, Z(0.1)~44.72")
print("=" * 60)

device = torch.device("cpu")
model = build_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.98)

prev_u_np = np.zeros((L, N_STEPS))
prev_v_np = np.zeros((L, N_STEPS))

results = {"picard_step": [], "Y0_hat": [], "Z0_hat": [], "loss_Y": [], "loss_Z": []}

for m in range(1, M_PICARD + 1):
    print(f"  Picard m={m}/{M_PICARD}")
    X, dW, t_grid = simulate_gbm_with_history(n_paths=L, seed=42 + m * 37)

    U_all = np.zeros((L, N_STEPS))
    V_all = np.zeros((L, N_STEPS))
    for t_i in range(N_STEPS):
        U_all[:, t_i], V_all[:, t_i] = compute_labels_straddle(
            X, dW, t_grid, t_i, prev_u_np, prev_v_np, seed_base=123 + m * 53
        )

    dW_tensor = torch.tensor(dW[:, :, np.newaxis], dtype=torch.float32)
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

    model.eval()
    with torch.no_grad():
        u_np, v_np = model(dW_tensor)
        prev_u_np = u_np.numpy()
        prev_v_np = v_np.numpy()

    # Test evaluation (use more paths for stability)
    X_test, dW_test, _ = simulate_gbm_with_history(n_paths=500, seed=999 + m * 71)
    dW_test_t = torch.tensor(dW_test[:, :, np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        u_test, v_test = model(dW_test_t)
    Y0_hat = float(u_test[:, 0].mean().item())
    Z0_hat = float(v_test[:, 0].mean().item())

    avg_ly = float(np.mean(losses_Y[-max(1, N_EPOCHS // 5):]))
    avg_lz = float(np.mean(losses_Z[-max(1, N_EPOCHS // 5):]))
    results["picard_step"].append(m)
    results["Y0_hat"].append(Y0_hat)
    results["Z0_hat"].append(Z0_hat)
    results["loss_Y"].append(avg_ly)
    results["loss_Z"].append(avg_lz)
    print(f"    Y0_hat={Y0_hat:.4f}, Z0_hat={Z0_hat:.4f}")

df = pd.DataFrame(results)
df["Y_target"] = 94.3490
df["Z_target"] = 44.7177
print("\n=== Straddle Results ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS_DIR, "exp4_results.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(df["picard_step"], df["Y0_hat"], "b-o", label="Ŷ(t)")
axes[0].axhline(94.3490, color="r", linestyle="--", label="Target Y≈94.35")
axes[0].set_xlabel("Picard Step m")
axes[0].set_ylabel("Y(t)")
axes[0].legend()
axes[0].set_title("Straddle: Y(0.1)")
axes[1].plot(df["picard_step"], df["Z0_hat"], "g-o", label="Ẑ(t)")
axes[1].axhline(44.7177, color="r", linestyle="--", label="Target Z≈44.72")
axes[1].set_xlabel("Picard Step m")
axes[1].set_ylabel("Z(t)")
axes[1].legend()
axes[1].set_title("Straddle: Z(0.1)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp4_convergence.png"), dpi=100)
plt.close()
print(f"\nResults saved to {RESULTS_DIR}")
