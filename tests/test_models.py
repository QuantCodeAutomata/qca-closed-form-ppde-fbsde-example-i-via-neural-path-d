"""
tests/test_models.py – Unit tests for src/models.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest

from src.models import build_model, LSTMPicard


def test_model_output_shape():
    model = build_model()
    batch, seq = 8, 10
    dW = torch.zeros(batch, seq, 1)
    u, v = model(dW)
    assert u.shape == (batch, seq)
    assert v.shape == (batch, seq)


def test_model_forward_deterministic():
    """Same input produces same output (no stochasticity at inference)."""
    model = build_model()
    model.eval()
    x = torch.ones(4, 5, 1)
    with torch.no_grad():
        u1, v1 = model(x)
        u2, v2 = model(x)
    torch.testing.assert_close(u1, u2)
    torch.testing.assert_close(v1, v2)


def test_model_trainable_parameters():
    model = build_model()
    params = list(model.parameters())
    assert len(params) > 0
    total = sum(p.numel() for p in params)
    assert total > 0


def test_model_gradients_computed():
    model = build_model()
    dW = torch.randn(4, 5, 1)
    u, v = model(dW)
    loss = (u ** 2 + v ** 2).mean()
    loss.backward()
    # Check at least one parameter has a gradient
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_model_lstm_hidden_size():
    model = build_model(lstm_hidden=32, mlp_hidden=16)
    assert model.lstm.hidden_size == 32


def test_model_single_path():
    """Works with batch size 1."""
    model = build_model()
    dW = torch.randn(1, 10, 1)
    u, v = model(dW)
    assert u.shape == (1, 10)


def test_model_variable_seq_length():
    """Different sequence lengths should work."""
    model = build_model()
    for seq_len in [5, 10, 20]:
        dW = torch.zeros(3, seq_len, 1)
        u, v = model(dW)
        assert u.shape == (3, seq_len)


def test_model_double_head_independence():
    """u_head and v_head have separate parameters."""
    model = build_model()
    u_params = set(id(p) for p in model.u_head.parameters())
    v_params = set(id(p) for p in model.v_head.parameters())
    assert len(u_params.intersection(v_params)) == 0
