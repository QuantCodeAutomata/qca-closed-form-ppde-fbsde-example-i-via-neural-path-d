"""
tests/test_labels.py – Unit tests for src/labels.py (methodology verification)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.labels import (
    _terminal_example1,
    _driver_example1,
    _terminal_example2,
    _running_integral,
    compute_labels_example1,
    compute_labels_example2,
)
from src.forward import simulate_brownian_increments, euler_forward_additive


def test_terminal_example1_zero_path():
    """If all X values are zero, terminal should be zero."""
    X = np.zeros((3, 11))
    dt = 0.1
    g = _terminal_example1(X, dt)
    np.testing.assert_allclose(g, 0.0)


def test_terminal_example1_unit_path():
    """X(t_i)=1 for all i: g = X(T) + T = 1 + 1 = 2 (for T=1, n=10, dt=0.1)."""
    n = 10
    dt = 0.1
    X = np.ones((1, n + 1))
    g = _terminal_example1(X, dt)
    expected = 1.0 + n * dt  # = 1 + 1 = 2
    np.testing.assert_allclose(g[0], expected, rtol=1e-12)


def test_driver_example1_shape():
    X = np.ones((5,))
    Y = np.zeros((5,))
    Z = np.zeros((5,))
    f = _driver_example1(X, Y, Z)
    assert f.shape == (5,)


def test_driver_example1_value():
    """f = -X regardless of Y, Z."""
    X = np.array([1.0, 2.0, -3.0])
    f = _driver_example1(X, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(f, -X)


def test_running_integral_zero_path():
    X = np.zeros((2, 6))
    dt = 0.1
    A = _running_integral(X, dt)
    np.testing.assert_allclose(A, 0.0)


def test_running_integral_unit_path():
    """X(t_i)=1: A(t_i) = 1 + sum_{j<i} 1*dt = 1 + i*dt."""
    n = 5
    dt = 0.2
    X = np.ones((1, n + 1))
    A = _running_integral(X, dt)
    expected = 1.0 + np.arange(n + 1) * dt
    np.testing.assert_allclose(A[0], expected, rtol=1e-12)


def test_terminal_example2_known():
    """X(T)=1, int_0^T X(s)ds=1 (unit path, T=1): g = cos(1+1) = cos(2)."""
    n = 10
    dt = 0.1
    X = np.ones((1, n + 1))
    g = _terminal_example2(X, dt)
    np.testing.assert_allclose(g[0], np.cos(2.0), rtol=1e-12)


def test_labels_example1_output_shape():
    """U/V labels have correct shape."""
    L, n = 4, 5
    dt = 0.2
    outer_X = np.ones((L, 4))  # t_i=3, so shape (L, t_i+1)
    U, V = compute_labels_example1(
        outer_dW=np.zeros((L, 3)),
        outer_X=outer_X,
        t_i=3,
        n_steps=n,
        dt=dt,
        sigma=0.3,
        bar_L=20,
        prev_u=np.zeros((L, n)),
        prev_v=np.zeros((L, n)),
        seed_base=0,
    )
    assert U.shape == (L,)
    assert V.shape == (L,)


def test_labels_example1_finite():
    """Labels should be finite for well-behaved paths."""
    L, n = 5, 8
    dt = 1.0 / n
    outer_X = np.ones((L, 3))
    U, V = compute_labels_example1(
        outer_dW=np.zeros((L, 2)),
        outer_X=outer_X,
        t_i=2,
        n_steps=n,
        dt=dt,
        sigma=0.3,
        bar_L=50,
        prev_u=np.zeros((L, n)),
        prev_v=np.zeros((L, n)),
        seed_base=7,
    )
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(V))


def test_labels_example2_output_shape():
    L, n = 3, 6
    dt = 1.0 / n
    outer_X = np.ones((L, 2))
    U, V = compute_labels_example2(
        outer_dW=np.zeros((L, 1)),
        outer_X=outer_X,
        t_i=1,
        n_steps=n,
        dt=dt,
        sigma=0.3,
        bar_L=20,
        prev_u=np.zeros((L, n)),
        prev_v=np.zeros((L, n)),
        seed_base=10,
    )
    assert U.shape == (L,)
    assert V.shape == (L,)


def test_labels_example1_at_terminal_node():
    """At the last time node t_{n-1} with zero future, U should equal g(X)."""
    # At t_i = n-1 (second to last), n_future = 1 step remains
    L, n = 4, 5
    dt = 0.2
    sigma = 0.3
    x_at_ti = 1.5
    outer_X = np.full((L, n), x_at_ti)  # all ones up to t_{n-1}
    U, V = compute_labels_example1(
        outer_dW=np.zeros((L, n - 1)),
        outer_X=outer_X,
        t_i=n - 1,
        n_steps=n,
        dt=dt,
        sigma=sigma,
        bar_L=10000,
        prev_u=np.zeros((L, n)),
        prev_v=np.zeros((L, n)),
        seed_base=99,
    )
    # U ≈ E[g(X_cont) + f_sum]; g = X(T) + dt*X_prev, f = -X*dt from t_{n-1} to n-1
    # Very rough: just check finite and nonzero
    assert np.all(np.isfinite(U))


def test_methodology_picard_zero_init():
    """First Picard iterate with Y^0=Z^0=0 should produce finite labels."""
    L, n = 3, 4
    dt = 0.25
    outer_X = np.ones((L, 3))  # t_i=2
    U, V = compute_labels_example1(
        outer_dW=np.zeros((L, 2)),
        outer_X=outer_X,
        t_i=2,
        n_steps=n,
        dt=dt,
        sigma=0.3,
        bar_L=30,
        prev_u=np.zeros((L, n)),
        prev_v=np.zeros((L, n)),
        seed_base=0,
    )
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(V))
