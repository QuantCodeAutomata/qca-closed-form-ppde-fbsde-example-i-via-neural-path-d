"""
labels.py – Monte Carlo label computation for the path-dependent Picard updates.

For each Picard level m+1 and each time node t_i, the U-label (Y) and V-label (Z)
are the conditional-expectation estimates over bar_L inner Brownian continuations
concatenated with the outer path history.

Equation references follow the paper's notation:
  U^{m+1}(t_i, omega) = E[ g(X_T^pi(omega ⊕_{t_i} W~)) +
      sum_{j=i}^{n-1} h_j * f(t_j, X_j^pi(concat), Y^m_j, Z^m_j) | F_{t_i} ]

  V^{m+1}(t_i, omega) = E[ (Delta W_i / h_i) *
      ( g(X_T^pi(concat)) +
        sum_{j=i+1}^{n-1} h_j * f(t_j, X_j^pi(concat), Y^m_j, Z^m_j) ) | F_{t_i} ]
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Example I helpers
# ---------------------------------------------------------------------------

def _terminal_example1(X: np.ndarray, dt: float) -> np.ndarray:
    """Terminal functional for Example I: g(omega) = X(T) + sum_j X(t_j)*dt.

    Parameters
    ----------
    X  : ndarray of shape (..., n_steps+1), full path
    dt : time step

    Returns
    -------
    g  : ndarray of shape (...)
    """
    # X[:, -1] is X(T); X[:, :-1] is X(t_0),...,X(t_{n-1})
    return X[..., -1] + np.sum(X[..., :-1], axis=-1) * dt


def _driver_example1(X: np.ndarray, Y_prev: np.ndarray, Z_prev: np.ndarray) -> np.ndarray:
    """Driver for Example I: f(s, X_s, Y, Z) = -X(s).

    The driver does not depend on Y or Z for this example.
    Y_prev and Z_prev arguments are accepted for API consistency.

    Parameters
    ----------
    X      : ndarray of shape (...,) – value of X at a single time step
    Y_prev : previous Picard Y estimate at this time step (unused here)
    Z_prev : previous Picard Z estimate at this time step (unused here)

    Returns
    -------
    f : ndarray same shape as X
    """
    return -X


def compute_labels_example1(
    outer_dW: np.ndarray,
    outer_X: np.ndarray,
    t_i: int,
    n_steps: int,
    dt: float,
    sigma: float,
    bar_L: int,
    prev_u: np.ndarray,
    prev_v: np.ndarray,
    seed_base: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute U/V labels for Example I at time node t_i.

    Parameters
    ----------
    outer_dW : increments of outer paths, shape (L, t_i)
    outer_X  : outer path values,        shape (L, t_i + 1)
    t_i      : current time index (0-indexed)
    n_steps  : total grid points n
    dt       : step size
    sigma    : diffusion coefficient
    bar_L    : inner Monte Carlo sample size
    prev_u   : previous Picard Y at outer paths, shape (L, n_steps)
    prev_v   : previous Picard Z at outer paths, shape (L, n_steps)
    seed_base: base seed for inner RNG

    Returns
    -------
    U_labels : shape (L,)
    V_labels : shape (L,)
    """
    L = outer_X.shape[0]
    n_future = n_steps - t_i  # number of remaining steps

    rng = np.random.default_rng(seed_base + t_i * 1000)
    # Inner Brownian increments: shape (L, bar_L, n_future)
    dW_inner = rng.normal(0.0, np.sqrt(dt), size=(L, bar_L, n_future))

    # Build concatenated X paths shape (L, bar_L, n_future + 1)
    # Continuation starts from outer_X[:, t_i]
    X_start = outer_X[:, t_i]  # (L,)
    X_cont = np.empty((L, bar_L, n_future + 1))
    X_cont[:, :, 0] = X_start[:, np.newaxis]
    for j in range(n_future):
        X_cont[:, :, j + 1] = X_cont[:, :, j] + sigma * dW_inner[:, :, j]

    # Full concatenated X: prepend the outer history (broadcast)
    # Shape: (L, bar_L, n_steps + 1)
    # Left part: outer_X[:, :t_i+1] broadcast over bar_L
    X_outer_left = outer_X[:, :t_i + 1][:, np.newaxis, :]  # (L,1,t_i+1)
    # Right part from continuation (skip the duplicated start)
    X_full = np.concatenate([
        np.broadcast_to(X_outer_left, (L, bar_L, t_i + 1)),
        X_cont[:, :, 1:],  # (L, bar_L, n_future)
    ], axis=-1)  # (L, bar_L, n_steps + 1)

    # Terminal functional g(X^pi)
    g_vals = _terminal_example1(X_full, dt)  # (L, bar_L)

    # Driver sum from j=i to n-1 using prev_u/prev_v (for Example I, f=-X, no Y/Z dep.)
    # We use X_full values at each future step for driver: f(t_j, X_j) = -X_j
    # sum_{j=i}^{n-1} dt * f(t_j, X_j) = -dt * sum_{j=i}^{n-1} X_full[:,:,j]
    driver_sum_u = -dt * np.sum(X_full[:, :, t_i:n_steps], axis=-1)  # (L, bar_L)

    # V-label driver sum starts at j=i+1
    if t_i + 1 < n_steps:
        driver_sum_v = -dt * np.sum(X_full[:, :, t_i + 1:n_steps], axis=-1)
    else:
        driver_sum_v = np.zeros((L, bar_L))

    payoff_u = g_vals + driver_sum_u  # (L, bar_L)
    payoff_v = g_vals + driver_sum_v  # (L, bar_L)

    U_labels = np.mean(payoff_u, axis=1)  # (L,)

    # V-label: weight by Delta W_i / dt for first inner increment
    dW_first = dW_inner[:, :, 0]  # (L, bar_L)
    V_labels = np.mean((dW_first / dt) * payoff_v, axis=1)  # (L,)

    return U_labels, V_labels


# ---------------------------------------------------------------------------
# Example II helpers
# ---------------------------------------------------------------------------

def _running_integral(X: np.ndarray, dt: float) -> np.ndarray:
    """Compute A(t_i) = X(t_i) + sum_{j=0}^{i-1} X(t_j)*dt for each i.

    Returns array of same shape as X containing A values at each time node.
    """
    # cumulative sum over j=0..i-1 (exclusive), approximating integral
    # A_i = X_i + int_0^{t_i} X(s)ds ~ X_i + sum_{j<i} X_j * dt
    integral = np.concatenate([
        np.zeros(X.shape[:-1] + (1,)),
        np.cumsum(X[..., :-1], axis=-1) * dt,
    ], axis=-1)
    return X + integral


def _terminal_example2(X: np.ndarray, dt: float) -> np.ndarray:
    """Terminal functional for Example II: g(omega) = cos(X(T) + int_0^T X(s)ds)."""
    A_T = X[..., -1] + np.sum(X[..., :-1], axis=-1) * dt
    return np.cos(A_T)


def _driver_example2(X_path: np.ndarray, dt: float) -> np.ndarray:
    """Compute driver sum contributions for Example II.

    Driver: f(s, X_path, Y, Z) = X(s)*sin(A(s)) + (sigma^2/2)*cos(A(s))
    where A(s) = X(s) + integral_0^s X(u)du.

    Derived from Ito's lemma on u(t,gamma)=cos(gamma(t)+int_0^t gamma(s)ds).

    Parameters
    ----------
    X_path : shape (..., n_steps+1)
    dt     : step size

    Returns
    -------
    f_vals : shape (..., n_steps) – driver values at t_0,...,t_{n-1}
    """
    A = _running_integral(X_path, dt)  # (..., n_steps+1)
    A_steps = A[..., :-1]             # (..., n_steps) – values at t_i
    X_steps = X_path[..., :-1]        # (..., n_steps)
    return X_steps * np.sin(A_steps)  # sigma^2/2 * cos(A) is handled via sigma param


def compute_labels_example2(
    outer_dW: np.ndarray,
    outer_X: np.ndarray,
    t_i: int,
    n_steps: int,
    dt: float,
    sigma: float,
    bar_L: int,
    prev_u: np.ndarray,
    prev_v: np.ndarray,
    seed_base: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute U/V labels for Example II at time node t_i.

    The driver (derived via Ito's lemma) is:
        f(s, X_path, Y, Z) = X(s)*sin(A(s)) + (sigma^2/2)*cos(A(s))
    where A(s) = X(s) + integral_0^s X(u)du.  This does not depend on Y or Z.

    Parameters mirror compute_labels_example1.
    """
    L = outer_X.shape[0]
    n_future = n_steps - t_i

    rng = np.random.default_rng(seed_base + t_i * 1000)
    dW_inner = rng.normal(0.0, np.sqrt(dt), size=(L, bar_L, n_future))

    # Build continuation X paths
    X_start = outer_X[:, t_i]
    X_cont = np.empty((L, bar_L, n_future + 1))
    X_cont[:, :, 0] = X_start[:, np.newaxis]
    for j in range(n_future):
        X_cont[:, :, j + 1] = X_cont[:, :, j] + sigma * dW_inner[:, :, j]

    # Full path (L, bar_L, n_steps+1)
    X_outer_left = outer_X[:, :t_i + 1][:, np.newaxis, :]
    X_full = np.concatenate([
        np.broadcast_to(X_outer_left, (L, bar_L, t_i + 1)),
        X_cont[:, :, 1:],
    ], axis=-1)

    # Terminal
    g_vals = _terminal_example2(X_full, dt)  # (L, bar_L)

    # Driver: f(s, X) = X(s)*sin(A(s)) + (sigma^2/2)*cos(A(s))
    A_full = _running_integral(X_full, dt)  # (L, bar_L, n_steps+1)
    A_steps = A_full[:, :, :-1]            # (L, bar_L, n_steps)
    X_steps = X_full[:, :, :-1]            # (L, bar_L, n_steps)
    f_all = X_steps * np.sin(A_steps) + (sigma ** 2 / 2) * np.cos(A_steps)

    # Sum from j=i to n-1
    driver_sum_u = dt * np.sum(f_all[:, :, t_i:n_steps], axis=-1)
    if t_i + 1 < n_steps:
        driver_sum_v = dt * np.sum(f_all[:, :, t_i + 1:n_steps], axis=-1)
    else:
        driver_sum_v = np.zeros((L, bar_L))

    payoff_u = g_vals + driver_sum_u
    payoff_v = g_vals + driver_sum_v

    U_labels = np.mean(payoff_u, axis=1)

    dW_first = dW_inner[:, :, 0]
    V_labels = np.mean((dW_first / dt) * payoff_v, axis=1)

    return U_labels, V_labels
