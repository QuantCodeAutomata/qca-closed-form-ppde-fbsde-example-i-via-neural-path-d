"""
models.py – Neural network architecture for path-dependent Picard iterates.

Section 6 of the paper uses an LSTM encoder for path history combined with
feed-forward heads for u^{m+1} and v^{m+1}.

Implementation assumption (not fully specified in the paper):
  - LSTM hidden size: 64, 2 layers
  - MLP head: one hidden layer with 64 units, tanh activation
  - Optimizer: Adam (documented as external assumption per paper)
  - Batch size: all training paths in one batch (small-scale experiments)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMPicard(nn.Module):
    """LSTM path encoder + dual MLP heads for (u, v) Picard approximation.

    The model processes the sequence of Brownian increments dW_0, ..., dW_{n-1}
    through an LSTM, producing hidden states at each time step.  Two separate
    linear heads then map each hidden state to a scalar, giving u^{m+1}(t_i)
    and v^{m+1}(t_i) for every i.

    Parameters
    ----------
    input_size   : feature dimension per time step (default 1 for scalar dW)
    lstm_hidden  : LSTM hidden-state dimension
    lstm_layers  : number of LSTM layers
    mlp_hidden   : hidden units in each MLP head
    output_size  : output dimension per head (1 for scalar Y and Z)
    """

    def __init__(
        self,
        input_size: int = 1,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        mlp_hidden: int = 64,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # Head for u^{m+1} (Y approximation)
        self.u_head = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, output_size),
        )
        # Head for v^{m+1} (Z approximation)
        self.v_head = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, output_size),
        )

    def forward(self, dW_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        dW_seq : Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        u_pred : Tensor of shape (batch, seq_len)
        v_pred : Tensor of shape (batch, seq_len)
        """
        # LSTM output: (batch, seq_len, lstm_hidden)
        lstm_out, _ = self.lstm(dW_seq)
        u_pred = self.u_head(lstm_out).squeeze(-1)  # (batch, seq_len)
        v_pred = self.v_head(lstm_out).squeeze(-1)  # (batch, seq_len)
        return u_pred, v_pred


def build_model(
    input_size: int = 1,
    lstm_hidden: int = 64,
    lstm_layers: int = 2,
    mlp_hidden: int = 64,
) -> LSTMPicard:
    """Construct an LSTMPicard model with given hyperparameters."""
    return LSTMPicard(
        input_size=input_size,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        mlp_hidden=mlp_hidden,
    )
