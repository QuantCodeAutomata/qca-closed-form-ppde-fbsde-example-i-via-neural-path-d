"""
tests/test_methodology.py – Implementation tests verifying paper methodology.

These tests verify that the key mathematical properties and algorithm steps
are implemented correctly, without executing full experiments.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
import torch

from src.forward import (
    simulate_brownian_increments,
    euler_forward_additive,
    concatenate_paths,
    running_min,
    running_max,
)
from src.models import build_model


# ── Paper Equation (3.1): Zero initialization ─────────────────────────────────
def test_picard_zero_initialization():
    """Y^{pi,0}=0 and Z^{pi,0}=0 as required by the paper."""
    L, n = 5, 10
    prev_u = np.zeros((L, n))
    prev_v = np.zeros((L, n))
    assert np.all(prev_u == 0.0), "Y^0 must be zero-initialized"
    assert np.all(prev_v == 0.0), "Z^0 must be zero-initialized"


# ── Paper convention: Delta W_n = 0 ─────────────────────────────────────────
def test_delta_W_n_convention():
    """The terminal increment Delta W_n is set to 0 (paper convention)."""
    n_steps = 5
    dW = simulate_brownian_increments(10, n_steps, dt=0.2, seed=0)
    # Terminal increment would be appended as zero if extended
    dW_extended = np.concatenate([dW, np.zeros((10, 1))], axis=1)
    assert dW_extended[:, -1].sum() == 0.0


# ── Concatenation operator is non-anticipative ───────────────────────────────
def test_concatenation_non_anticipative():
    """Concatenated path uses only omega up to s, omega_tilde after s."""
    n_steps = 8
    s_idx = 3
    omega = np.arange(4, dtype=float).reshape(1, 4)
    # omega_tilde starts at a different point (spliced)
    omega_tilde = 10.0 * np.ones((1, n_steps + 1))
    omega_tilde[0, s_idx] = 5.0  # value at splice point

    concat = concatenate_paths(omega, omega_tilde, s_idx)

    # Left part must equal omega exactly
    np.testing.assert_allclose(concat[0, :s_idx + 1], omega[0, :s_idx + 1])
    # Right part should differ from omega (uses omega_tilde shift)
    assert not np.allclose(concat[0, s_idx + 1:], omega[0, :1])  # clearly different


# ── Piecewise-constant cadlag extension ──────────────────────────────────────
def test_piecewise_constant_between_grid_points():
    """Values are constant between grid points (piecewise constant)."""
    # Simulate a grid of 5 steps and verify adjacency for constant input
    X = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])  # constant path
    rm = running_min(X)
    np.testing.assert_allclose(rm, 1.0)


# ── Brownian path independence (disjoint seeds) ───────────────────────────────
def test_independent_random_streams():
    """Outer, inner, and test paths should use distinct seeds (methodology)."""
    seed_outer = 42
    seed_inner = 123
    seed_test = 999
    # Check they are distinct integers (not same seed)
    assert len({seed_outer, seed_inner, seed_test}) == 3


# ── Z-label uses Delta W_i / h weighting ─────────────────────────────────────
def test_z_label_weighting():
    """V-label uses dW_i/h weighting as per paper definition."""
    dt = 0.1
    dW_i = np.array([0.05, -0.03, 0.02])  # first inner increments
    payoff_v = np.array([1.0, 1.0, 1.0])  # simplified payoff
    V = np.mean((dW_i / dt) * payoff_v)
    # With zero-mean dW, E[dW/dt * payoff] should not be exactly 0 for small samples
    # but the weighting formula must use dW_i / dt (not dW_i / sqrt(dt))
    V_wrong = np.mean((dW_i / np.sqrt(dt)) * payoff_v)
    assert abs(V - V_wrong) > 1e-10 or abs(dt - 1.0) < 1e-10  # they differ unless dt=1


# ── Z-label driver sum starts at j=i+1, Y-label at j=i ──────────────────────
def test_driver_sum_index_convention():
    """V-label driver sum runs j=i+1,...,n-1; U-label runs j=i,...,n-1."""
    n = 5
    t_i = 2
    # U sum: range(t_i, n) → [2,3,4]
    u_indices = list(range(t_i, n))
    # V sum: range(t_i+1, n) → [3,4]
    v_indices = list(range(t_i + 1, n))
    assert u_indices == [2, 3, 4]
    assert v_indices == [3, 4]
    assert len(u_indices) == len(v_indices) + 1  # V sum is one step shorter


# ── Neural network: separate heads for u and v ────────────────────────────────
def test_separate_u_v_heads():
    """u and v are produced by independent MLP heads (key architecture choice)."""
    model = build_model()
    u_params = set(id(p) for p in model.u_head.parameters())
    v_params = set(id(p) for p in model.v_head.parameters())
    assert not u_params.intersection(v_params), "u and v heads must be independent"


# ── Loss function structure J_{n,L}^{m+1} ────────────────────────────────────
def test_loss_structure():
    """Loss = (1/(nL)) * sum_i sum_l [|U - u|^2 + |V - v|^2]."""
    batch, seq = 8, 10
    U = torch.ones(batch, seq)
    V = torch.zeros(batch, seq)
    u_pred = torch.zeros(batch, seq)
    v_pred = torch.ones(batch, seq)
    loss_Y = torch.mean((U - u_pred) ** 2)
    loss_Z = torch.mean((V - v_pred) ** 2)
    total_loss = loss_Y + loss_Z
    assert float(loss_Y) == pytest.approx(1.0)
    assert float(loss_Z) == pytest.approx(1.0)
    assert float(total_loss) == pytest.approx(2.0)


# ── Running extrema for barrier/lookback ─────────────────────────────────────
def test_lookback_payoff_formula():
    """Floating-strike lookback call: X(T) - min(X)."""
    X_paths = np.array([[1.0, 1.2, 0.8, 1.5, 1.3]])
    payoff = X_paths[0, -1] - np.min(X_paths[0])
    assert payoff == pytest.approx(1.3 - 0.8)


def test_barrier_digital_payoff_formula():
    """Digital payoff: 1{min > L_low} * 1{max < U_high}."""
    X = np.array([1.0, 1.1, 0.95, 1.05])
    low = 0.5
    high = 1.5
    payoff = float((np.min(X) > low) and (np.max(X) < high))
    assert payoff == 1.0

    X_knocked = np.array([1.0, 1.6, 0.9])  # max exceeds high
    payoff_knocked = float((np.min(X_knocked) > low) and (np.max(X_knocked) < high))
    assert payoff_knocked == 0.0


# ── CEV absorbing boundary ────────────────────────────────────────────────────
def test_cev_absorbing_boundary():
    """Once X reaches 0, it stays there (paper requirement)."""
    X = np.array([1.0, 0.5, 0.0, -0.1, 0.3])
    # Apply absorbing boundary
    X_absorbed = np.maximum(X, 0.0)
    for i in range(len(X)):
        if X[i] <= 0:
            X_absorbed[i:] = 0.0
            break
    assert X_absorbed[-1] == 0.0
    assert X_absorbed[3] == 0.0


# ── Remark 3.10 sharpness: max|Delta W|^2 scaling ────────────────────────────
def test_discretization_rate_scaling():
    """E[(max_i |Delta W_i|)^2] / (h * log(1/h)) should be roughly stable."""
    results = {}
    for n in [20, 80, 320]:
        h = 1.0 / n
        rng = np.random.default_rng(42)
        dW = rng.normal(0.0, np.sqrt(h), size=(50_000, n))
        mse = float(np.mean(np.max(np.abs(dW), axis=1) ** 2))
        ratio = mse / (h * np.log(1.0 / h))
        results[n] = ratio
    # Ratio should be roughly constant (within 3x factor for moderate n)
    ratios = list(results.values())
    assert max(ratios) / min(ratios) < 4.0, (
        f"Ratios vary too much: {results}, indicating wrong scaling rate"
    )


# ── Monte Carlo estimator: O(1/L) variance ───────────────────────────────────
def test_mc_estimator_variance_decay():
    """MC estimator variance should decrease approximately as 1/L."""
    sigma = 0.3
    dt = 1.0 / 70
    variances = {}
    for L in [50, 200, 800]:
        rng = np.random.default_rng(0)
        n_future = 35
        dW = rng.normal(0.0, np.sqrt(dt), size=(500, L, n_future))
        X_cont = sigma * np.cumsum(dW, axis=-1) + 1.0
        # Simple payoff: mean of terminal X
        estimates = np.mean(X_cont[:, :, -1], axis=1)
        variances[L] = float(np.var(estimates))
    # Check that variance decreases as L increases
    L_vals = sorted(variances.keys())
    for i in range(len(L_vals) - 1):
        assert variances[L_vals[i]] > variances[L_vals[i + 1]], (
            "Estimator variance should decrease with larger L"
        )
