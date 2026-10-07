"""
train.py – Neural Picard iteration training loop.

Implements Algorithm 6.1 from the paper:
  For each Picard step m = 1, ..., M:
    1. Generate K rounds of L outer training paths.
    2. Compute U/V labels for all time nodes using the previous Picard
       approximations u^m and v^m.
    3. Train the neural network (initialized from previous weights) to
       minimise J_{n,L}^{m+1} = (1/(n L)) sum_i sum_l [|U - u|^2 + |V - v|^2].
    4. Evaluate on 100 000 independent test paths.

Implementation notes (external assumptions, not fully specified in the paper):
  - Optimizer: Adam
  - Batch: full-batch gradient descent over all K*L paths per Picard step
  - LSTM: hidden_size=64, num_layers=2; MLP head hidden=64, tanh activation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List
from src.forward import simulate_brownian_increments, euler_forward_additive
from src.models import build_model
from src.labels import compute_labels_example1, compute_labels_example2
from src.evaluate import evaluate_on_test


def _compute_all_labels(
    example_id: int,
    outer_dW: np.ndarray,
    outer_X: np.ndarray,
    n_steps: int,
    dt: float,
    sigma: float,
    bar_L: int,
    prev_u_np: np.ndarray,
    prev_v_np: np.ndarray,
    seed_base: int,
) -> Dict[str, np.ndarray]:
    """Compute U/V labels for every time node and every outer path.

    Returns dict with keys 'U_labels' and 'V_labels', each of shape (L, n_steps).
    """
    L = outer_X.shape[0]
    U_labels = np.zeros((L, n_steps))
    V_labels = np.zeros((L, n_steps))

    label_fn = (
        compute_labels_example1 if example_id == 1 else compute_labels_example2
    )

    for t_i in range(n_steps):
        U_i, V_i = label_fn(
            outer_dW=outer_dW[:, :t_i] if t_i > 0 else np.zeros((L, 0)),
            outer_X=outer_X[:, : t_i + 1],
            t_i=t_i,
            n_steps=n_steps,
            dt=dt,
            sigma=sigma,
            bar_L=bar_L,
            prev_u=prev_u_np,
            prev_v=prev_v_np,
            seed_base=seed_base,
        )
        U_labels[:, t_i] = U_i
        V_labels[:, t_i] = V_i

    return {"U_labels": U_labels, "V_labels": V_labels}


def run_picard_neural(
    example_id: int,
    config: Dict[str, Any],
    M: int,
) -> Dict[str, List]:
    """Run the full neural Picard iteration.

    Parameters
    ----------
    example_id : 1 or 2 (selects driver and terminal condition)
    config     : hyperparameter dict with keys:
                   n_steps, dt, L, K, bar_L, n_epochs, lr,
                   lr_decay_factor, lr_decay_every, sigma, x0,
                   test_paths, seed_outer, seed_inner, seed_test
    M          : number of Picard iterations

    Returns
    -------
    results dict with lists of length M:
      picard_step, Y0_hat, Z0_hat, loss_Y, loss_Z
    """
    n_steps = config["n_steps"]
    dt = config["dt"]
    L = config["L"]
    K = config["K"]
    bar_L = config["bar_L"]
    n_epochs = config["n_epochs"]
    lr = config["lr"]
    decay_factor = config["lr_decay_factor"]
    decay_every = config["lr_decay_every"]
    sigma = config["sigma"]
    x0 = config["x0"]
    test_paths = config["test_paths"]
    seed_outer = config["seed_outer"]
    seed_inner = config["seed_inner"]
    seed_test = config["seed_test"]

    device = torch.device("cpu")

    # Build initial model (theta^0 = random)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=decay_factor)

    # Previous Picard estimates (initialized to zero: Y^0=0, Z^0=0)
    # prev_u, prev_v: not actually used in label computation when driver is independent of Y,Z
    # but kept for API consistency and possible future extension
    prev_u_np = np.zeros((K * L, n_steps))
    prev_v_np = np.zeros((K * L, n_steps))

    results = {
        "picard_step": [],
        "Y0_hat": [],
        "Z0_hat": [],
        "loss_Y": [],
        "loss_Z": [],
    }

    for m in range(1, M + 1):
        print(f"\n=== Picard step m={m}/{M} ===")

        # ---- Step 1: Generate K*L outer training paths ----
        outer_dW = simulate_brownian_increments(
            n_paths=K * L,
            n_steps=n_steps,
            dt=dt,
            seed=seed_outer + m * 37,
        )
        outer_X = euler_forward_additive(outer_dW, x0=x0, sigma=sigma)

        # ---- Step 2: Compute labels ----
        print("  Computing labels...")
        labels = _compute_all_labels(
            example_id=example_id,
            outer_dW=outer_dW,
            outer_X=outer_X,
            n_steps=n_steps,
            dt=dt,
            sigma=sigma,
            bar_L=bar_L,
            prev_u_np=prev_u_np,
            prev_v_np=prev_v_np,
            seed_base=seed_inner + m * 53,
        )
        U_labels = labels["U_labels"]  # (K*L, n_steps)
        V_labels = labels["V_labels"]

        # Convert to tensors
        # Network input: dW sequences, shape (K*L, n_steps, 1)
        dW_tensor = torch.tensor(outer_dW[:, :, np.newaxis], dtype=torch.float32).to(device)
        U_tensor = torch.tensor(U_labels, dtype=torch.float32).to(device)
        V_tensor = torch.tensor(V_labels, dtype=torch.float32).to(device)

        # ---- Step 3: Train neural network ----
        print(f"  Training for {n_epochs} epochs...")
        model.train()
        loss_Y_hist: List[float] = []
        loss_Z_hist: List[float] = []

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            u_pred, v_pred = model(dW_tensor)  # (K*L, n_steps)
            loss_Y_raw = torch.mean((U_tensor - u_pred) ** 2)
            loss_Z_raw = torch.mean((V_tensor - v_pred) ** 2)
            loss = loss_Y_raw + loss_Z_raw
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_Y_hist.append(loss_Y_raw.item())
            loss_Z_hist.append(loss_Z_raw.item())
            if epoch % max(1, n_epochs // 10) == 0:
                print(f"    Epoch {epoch:4d}: loss_Y={loss_Y_raw.item():.4e}, "
                      f"loss_Z={loss_Z_raw.item():.4e}")

        # ---- Update prev estimates for next Picard step ----
        model.eval()
        with torch.no_grad():
            u_np, v_np = model(dW_tensor)
            prev_u_np = u_np.cpu().numpy()
            prev_v_np = v_np.cpu().numpy()

        # ---- Step 4: Evaluate on test paths ----
        Y0_hat, Z0_hat = evaluate_on_test(
            model=model,
            n_test=test_paths,
            n_steps=n_steps,
            dt=dt,
            sigma=sigma,
            x0=x0,
            seed=seed_test + m * 71,
        )

        # Record results
        avg_loss_Y = float(np.mean(loss_Y_hist[-max(1, n_epochs // 5):]))
        avg_loss_Z = float(np.mean(loss_Z_hist[-max(1, n_epochs // 5):]))
        results["picard_step"].append(m)
        results["Y0_hat"].append(Y0_hat)
        results["Z0_hat"].append(Z0_hat)
        results["loss_Y"].append(avg_loss_Y)
        results["loss_Z"].append(avg_loss_Z)

        print(f"  Y0_hat={Y0_hat:.6f}, Z0_hat={Z0_hat:.6f}, "
              f"loss_Y={avg_loss_Y:.4e}, loss_Z={avg_loss_Z:.4e}")

    return results
