"""
evaluate.py – Out-of-sample evaluation of trained Picard-step models.

Evaluates the trained neural network on n_test independent Brownian paths
and returns the estimated initial BSDE values hat Y^m(0) and hat Z^m(0).
"""

import numpy as np
import torch
from typing import Tuple
from src.forward import simulate_brownian_increments, euler_forward_additive
from src.models import LSTMPicard


def evaluate_on_test(
    model: LSTMPicard,
    n_test: int,
    n_steps: int,
    dt: float,
    sigma: float,
    x0: float,
    seed: int = 999,
) -> Tuple[float, float]:
    """Evaluate model on independent test paths.

    Parameters
    ----------
    model   : trained LSTMPicard model (set to eval mode internally)
    n_test  : number of test paths
    n_steps : time-grid steps
    dt      : step size
    sigma   : diffusion coefficient for additive Brownian noise
    x0      : initial state
    seed    : RNG seed for test paths (must be different from training seeds)

    Returns
    -------
    Y0_hat : estimated initial value of Y, scalar
    Z0_hat : estimated initial value of Z, scalar
    """
    model.eval()
    device = next(model.parameters()).device

    dW_test = simulate_brownian_increments(n_test, n_steps, dt, seed=seed)
    dW_tensor = torch.tensor(dW_test[:, :, np.newaxis], dtype=torch.float32).to(device)

    with torch.no_grad():
        u_pred, v_pred = model(dW_tensor)  # (n_test, n_steps)

    # Initial-time estimates: first time step output
    Y0_hat = float(u_pred[:, 0].mean().item())
    Z0_hat = float(v_pred[:, 0].mean().item())
    return Y0_hat, Z0_hat
