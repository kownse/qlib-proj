"""
TCN with FiLM (Feature-wise Linear Modulation) for Macro Conditioning.

Architecture:
    Stock (batch, d_feat, step_len) → TCN → tcn_out (batch, n_chans)
    Macro (batch, n_macro) → MLP → γ, β (batch, n_chans each)
    Output = γ * tcn_out + β → Linear → prediction
"""

import torch
import torch.nn as nn
from qlib.contrib.model.tcn import TemporalConvNet


class TCNFiLM(nn.Module):
    """
    TCN with FiLM conditioning from macro features.

    FiLM: y = γ * tcn_out + β
    - γ > 1 amplifies features in high-volatility regimes
    - γ < 1 dampens features in low-volatility regimes
    """

    def __init__(
        self,
        d_feat: int = 5,
        n_macro: int = 6,
        n_chans: int = 32,
        num_layers: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.5,
        film_hidden: int = 32,
    ):
        super().__init__()

        # TCN backbone
        self.tcn = TemporalConvNet(
            d_feat,
            [n_chans] * num_layers,
            kernel_size,
            dropout=dropout
        )

        # FiLM generator: macro → (γ, β)
        self.film_generator = nn.Sequential(
            nn.Linear(n_macro, film_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden, n_chans * 2)
        )

        # Initialize γ=1, β=0
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.film_generator[-1].bias.data[:n_chans] = 1.0

        # Prediction head
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(n_chans, 1)

    def forward(self, x_stock, x_macro):
        """
        Args:
            x_stock: (batch, d_feat, step_len)
            x_macro: (batch, n_macro)
        Returns:
            predictions: (batch,)
        """
        # TCN encoding
        tcn_out = self.tcn(x_stock)[:, :, -1]  # (batch, n_chans)

        # FiLM modulation
        film = self.film_generator(x_macro)
        gamma, beta = film.chunk(2, dim=1)
        modulated = gamma * tcn_out + beta

        # Prediction
        out = self.dropout_layer(modulated)
        return self.fc(out).squeeze(-1)
