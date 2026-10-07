"""
forward.py – Brownian path simulation and Euler SDE solvers.

All functions use numpy for array operations. Paths are stored as full
histories because the PPDE/FBSDE targets are path functionals.
"""

import numpy as np
from typing import Optional


def simulate_brownian_increments(
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate i.i.d. Brownian increments Delta W_i ~ N(0, dt).

    Parameters
    ----------
    n_paths : number of Monte Carlo paths
    n_steps : number of time steps
    dt      : step size (mesh)
    seed    : RNG seed for reproducibility

    Returns
    -------
    dW : ndarray of shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))


def simulate_brownian_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate cumulative Brownian paths W(t_i), i=0,...,n_steps.

    Returns
    -------
    W : ndarray of shape (n_paths, n_steps + 1), W[:, 0] = 0
    """
    dW = simulate_brownian_increments(n_paths, n_steps, dt, seed)
    W = np.zeros((n_paths, n_steps + 1))
    np.cumsum(dW, axis=1, out=W[:, 1:])
    return W


def euler_forward_additive(
    dW: np.ndarray,
    x0: float,
    sigma: float,
) -> np.ndarray:
    """Euler scheme for dX = sigma dW (additive noise, zero drift).

    X_{i+1} = X_i + sigma * dW_i,  X_0 = x0.

    Parameters
    ----------
    dW    : increments, shape (n_paths, n_steps)
    x0    : initial value
    sigma : diffusion coefficient

    Returns
    -------
    X : ndarray of shape (n_paths, n_steps + 1)
    """
    n_paths, n_steps = dW.shape
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        X[:, i + 1] = X[:, i] + sigma * dW[:, i]
    return X


def euler_forward_gbm(
    dW: np.ndarray,
    x0: float,
    mu: float,
    sigma: float,
    dt: float,
    absorb_at_zero: bool = True,
) -> np.ndarray:
    """Euler scheme for dX = mu X dt + sigma X dW (geometric Brownian motion).

    X_{i+1} = X_i + mu*X_i*dt + sigma*X_i*dW_i.
    If absorb_at_zero=True, once X drops to or below 0 it stays there.

    Parameters
    ----------
    dW              : increments, shape (n_paths, n_steps)
    x0              : initial value
    mu              : drift coefficient
    sigma           : diffusion coefficient
    dt              : step size
    absorb_at_zero  : enforce absorbing boundary at 0

    Returns
    -------
    X : ndarray of shape (n_paths, n_steps + 1)
    """
    n_paths, n_steps = dW.shape
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        X_i = X[:, i]
        X_next = X_i + mu * X_i * dt + sigma * X_i * dW[:, i]
        if absorb_at_zero:
            X_next = np.maximum(X_next, 0.0)
            # paths already at zero stay there
            X_next[X_i <= 0.0] = 0.0
        X[:, i + 1] = X_next
    return X


def concatenate_paths(
    omega: np.ndarray,
    omega_tilde: np.ndarray,
    s_idx: int,
) -> np.ndarray:
    """Path concatenation operator (omega ⊕_s omega_tilde)(u).

    Definition (cadlag extension):
      (omega ⊕_s omega_tilde)(u) =
        omega(u)                                           for u <= s
        omega_tilde(u) - omega_tilde(s) + omega(s)        for u in (s, T]

    Parameters
    ----------
    omega       : outer path, shape (n_paths, s_idx + 1) (values at t_0,...,t_s)
    omega_tilde : inner continuation, shape (n_paths, n_steps + 1) (values at t_0,...,t_n)
    s_idx       : index of the splicing time t_s

    Returns
    -------
    concat : ndarray of shape (n_paths, n_steps + 1)
    """
    n_paths, n_steps_p1 = omega_tilde.shape
    concat = np.empty((n_paths, n_steps_p1))
    # Left part: copy outer path up to s
    concat[:, : s_idx + 1] = omega[:, : s_idx + 1]
    # Right part: shifted continuation
    concat[:, s_idx + 1 :] = (
        omega_tilde[:, s_idx + 1 :]
        - omega_tilde[:, s_idx : s_idx + 1]
        + omega[:, s_idx : s_idx + 1]
    )
    return concat


def running_min(X: np.ndarray) -> np.ndarray:
    """Compute running minimum along time axis.

    Parameters
    ----------
    X : ndarray of shape (n_paths, n_steps + 1)

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1), entry [l, i] = min(X[l, 0:i+1])
    """
    return np.minimum.accumulate(X, axis=1)


def running_max(X: np.ndarray) -> np.ndarray:
    """Compute running maximum along time axis."""
    return np.maximum.accumulate(X, axis=1)
