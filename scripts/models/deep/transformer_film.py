"""
Transformer with FiLM (Feature-wise Linear Modulation) for Macro Conditioning.

Architecture:
    Stock (batch, seq_len, d_feat) -> Transformer Encoder -> (batch, d_model)
    Macro (batch, n_macro) -> MLP -> gamma, beta (batch, d_model each)
    Output = gamma * transformer_out + beta -> MLP -> prediction

FiLM modulates the transformer output based on macro conditions (VIX, yields, etc.)
enabling market-aware predictions.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFiLM(nn.Module):
    """
    Transformer with FiLM conditioning from macro features.

    FiLM: y = gamma * transformer_out + beta
    - gamma > 1 amplifies features in high-volatility regimes
    - gamma < 1 dampens features in low-volatility regimes
    """

    def __init__(
        self,
        d_feat: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 60,
        n_macro: int = 23,
        film_hidden: int = 32,
    ):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(d_feat, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # FiLM generator: macro -> (gamma, beta)
        self.film_generator = nn.Sequential(
            nn.Linear(n_macro, film_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden, d_model * 2)
        )

        # Initialize gamma=1, beta=0 for identity transform at start
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.film_generator[-1].bias.data[:d_model] = 1.0

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights (except FiLM which has special init)."""
        for name, p in self.named_parameters():
            if 'film_generator' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_stock, x_macro):
        """
        Args:
            x_stock: (batch, seq_len, d_feat) or (batch, seq_len * d_feat)
            x_macro: (batch, n_macro)
        Returns:
            predictions: (batch,)
        """
        batch_size = x_stock.size(0)

        # Reshape if flattened input
        if x_stock.dim() == 2:
            x_stock = x_stock.view(batch_size, self.seq_len, self.d_feat)

        # 1. Transformer encoding
        x = self.input_proj(x_stock)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # 2. FiLM modulation
        film = self.film_generator(x_macro)
        gamma, beta = film.chunk(2, dim=1)
        x = gamma * x + beta

        # 3. Predict
        return self.output_mlp(x).squeeze(-1)
