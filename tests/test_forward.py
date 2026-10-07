"""
tests/test_forward.py – Unit tests for src/forward.py
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.forward import (
    simulate_brownian_increments,
    simulate_brownian_paths,
    euler_forward_additive,
    euler_forward_gbm,
    concatenate_paths,
    running_min,
    running_max,
)


def test_brownian_increments_shape():
    dW = simulate_brownian_increments(n_paths=100, n_steps=50, dt=0.1, seed=0)
    assert dW.shape == (100, 50)


def test_brownian_increments_variance():
    """Var(Delta W_i) = dt."""
    dt = 0.01
    dW = simulate_brownian_increments(n_paths=100_000, n_steps=1, dt=dt, seed=1)
    assert abs(float(np.var(dW)) - dt) < 0.001


def test_brownian_paths_starts_at_zero():
    W = simulate_brownian_paths(n_paths=10, n_steps=20, dt=0.05, seed=2)
    assert W.shape == (10, 21)
    np.testing.assert_array_equal(W[:, 0], 0.0)


def test_brownian_paths_increments():
    W = simulate_brownian_paths(n_paths=5, n_steps=10, dt=0.1, seed=3)
    dW = np.diff(W, axis=1)
    assert dW.shape == (5, 10)


def test_euler_additive_initial_value():
    dW = np.zeros((3, 10))
    X = euler_forward_additive(dW, x0=2.0, sigma=0.5)
    np.testing.assert_allclose(X[:, 0], 2.0)


def test_euler_additive_zero_noise():
    dW = np.zeros((5, 20))
    X = euler_forward_additive(dW, x0=1.0, sigma=0.3)
    np.testing.assert_allclose(X, 1.0)


def test_euler_additive_shape():
    dW = simulate_brownian_increments(50, 30, 0.1, seed=4)
    X = euler_forward_additive(dW, x0=1.0, sigma=0.3)
    assert X.shape == (50, 31)


def test_euler_gbm_initial_value():
    dW = np.zeros((4, 15))
    X = euler_forward_gbm(dW, x0=100.0, mu=0.05, sigma=0.2, dt=0.1)
    np.testing.assert_allclose(X[:, 0], 100.0)


def test_euler_gbm_absorb_at_zero():
    """Once path hits 0, it stays there."""
    n_steps = 5
    dW = np.zeros((1, n_steps))
    # Force large negative jump
    dW[0, 1] = -1e6
    X = euler_forward_gbm(dW, x0=1.0, mu=0.0, sigma=1.0, dt=0.01, absorb_at_zero=True)
    assert X[0, 2] == pytest.approx(0.0, abs=1e-10)
    assert X[0, -1] == pytest.approx(0.0, abs=1e-10)


def test_concatenate_paths_left_part():
    """Concatenated path matches omega on [0, s]."""
    n_steps = 10
    s_idx = 4
    omega = np.arange(5, dtype=float).reshape(1, 5)  # values at t_0,...,t_4
    omega_tilde = np.ones((1, n_steps + 1))
    concat = concatenate_paths(omega, omega_tilde, s_idx)
    np.testing.assert_allclose(concat[0, :s_idx + 1], omega[0])


def test_concatenate_paths_right_shift():
    """Right part is shifted so concat is continuous at splice point."""
    n_steps = 6
    s_idx = 2
    omega = np.array([[0.0, 1.0, 2.0]])       # shape (1, 3)
    omega_tilde = np.zeros((1, n_steps + 1))   # all zeros
    omega_tilde[0, s_idx] = 0.5               # omega_tilde(s) = 0.5
    concat = concatenate_paths(omega, omega_tilde, s_idx)
    # At s_idx: concat = omega(s) = 2.0
    assert concat[0, s_idx] == pytest.approx(2.0)
    # Right part: omega_tilde(u) - omega_tilde(s) + omega(s) = 0 - 0.5 + 2.0 = 1.5
    assert concat[0, s_idx + 1] == pytest.approx(1.5)


def test_running_min_monotone():
    X = np.array([[3.0, 1.0, 2.0, 0.5]])
    rm = running_min(X)
    assert rm.shape == (1, 4)
    for i in range(3):
        assert rm[0, i + 1] <= rm[0, i]


def test_running_max_monotone():
    X = np.array([[1.0, 3.0, 2.0, 4.0]])
    rmax = running_max(X)
    for i in range(3):
        assert rmax[0, i + 1] >= rmax[0, i]


def test_running_min_matches_numpy():
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, size=(10, 20))
    rm = running_min(X)
    for i in range(X.shape[1]):
        np.testing.assert_allclose(rm[:, i], np.min(X[:, :i + 1], axis=1))
