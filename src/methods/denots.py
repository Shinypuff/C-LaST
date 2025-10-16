import torch
from torch import Tensor

from ..layers.denots import DeNOTS
from ..layers.mlp import MLP
from ..layers.node import NeuralODE
from .base import BaseForecasting


class DeNOTSForecaster(BaseForecasting):
    def __init__(self, timescale: int, hidden_dim: int, lr: float, **kwargs):
        super().__init__(**kwargs)
        self.encoder = DeNOTS(1 + self.target_dim, hidden_dim, timescale)
        self.decoder = NeuralODE(hidden_dim, timescale)
        self.mu_mlp = MLP(hidden_dim, hidden_dim, self.target_dim)
        self.sigma_mlp = MLP(
            hidden_dim, hidden_dim, self.target_dim, final_act=torch.exp
        )

        self.hidden_dim = hidden_dim
        self.timescale = timescale
        self.lr = lr

    def forward(self, dt: Tensor, x: Tensor) -> Tensor:
        B, L, D = x.shape
        T = dt.shape[1] - L
        t = torch.cumsum(dt, dim=1)
        t_x, t_y = torch.split(t, [L, T], dim=1)

        # Compress observations into embedding
        x_obs = torch.cat([dt[:, :L].unsqueeze(-1), x], dim=-1)
        h = self.encoder(x_obs, t_x)

        # Guide the new trajectory with context.
        preds = self.decoder(h, t_y)
        return self.mu_mlp(preds), self.sigma_mlp(preds)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
