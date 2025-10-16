"""File with an example supervised forecasting model."""

import torch
from torch import nn

from ..layers.mlp import MLP
from .base import BaseForecasting


class GRUForecaster(BaseForecasting):
    """A supervised forecasting model that uses GRU-based encoder-decoder architecture.

    This class implements a forecasting model that encodes past context and observations
    using a GRU encoder, then decodes future predictions using a GRU decoder cell.
    The model outputs probabilistic forecasts with mean and standard deviation predictions.

    The encoder processes the concatenation of context and observation sequences,
    while the decoder iteratively generates future predictions conditioned on the
    encoded state and future context information.

    Attributes:
        encoder (nn.GRU): GRU encoder that processes past context and observations.
        decoder (nn.GRUCell): GRU cell decoder for generating future predictions.
        mu_mlp (MLP): Multi-layer perceptron for predicting the mean of the distribution.
        std_mlp (MLP): Multi-layer perceptron for predicting the standard deviation.

    Args:
        **base_kwargs: Keyword arguments passed to the BaseForecasting parent class.

    """

    def __init__(
        self, num_layers: int, dropout: float, hidden_dim: int, lr: float, **base_kwargs
    ):
        """Initialize internal state."""
        super().__init__(**base_kwargs)

        self.encoder = nn.GRU(
            self.target_dim + self.time_feat_dim,
            hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True,
        )

        self.decoder = nn.GRU(
            self.time_feat_dim,
            hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mu_mlp = MLP(hidden_dim, hidden_dim, self.target_dim)
        self.std_mlp = MLP(hidden_dim, hidden_dim, self.target_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr

    def forward(self, time_feats, x):
        """Run the forward pass of the model.

        Args:
            ctx (Tensor): Context tensor of shape (batch_size, ctx_len, ctx_dim).
            obs (Tensor): Observation tensor of shape (batch_size, obs_len, obs_dim).
            T (int): Number of timesteps to forecast.

        Returns:
            tuple: A tuple containing the mean and standard deviation tensors of shape
                (batch_size, T, tgt_dim) for the forecasted distribution.

        """
        L = x.shape[1]
        T = time_feats.shape[1] - L

        # We predict using future dt value
        past = torch.cat([time_feats[:, :L], x], dim=-1)
        h = self.encoder(past)[1]

        hiddens = []
        for i in range(L, L + T):
            out, h = self.decoder(time_feats[:, None, i], h)
            hiddens.append(out.squeeze(1))

        hiddens = torch.stack(hiddens, dim=1)
        means = self.mu_mlp(hiddens)
        scales = self.std_mlp(hiddens).exp()
        return means, scales

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
