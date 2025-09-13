import torch
from torch import Tensor

from ..layers.denots import DeNOTS
from ..layers.mlp import MLP
from .base import BaseForecasting


class DeNOTSForecaster(BaseForecasting):
    def __init__(self, timescale: int, hidden_dim: int, lr: float, **kwargs):
        super().__init__(**kwargs)
        self.encoder = DeNOTS(self.ctx_dim + self.tgt_dim, hidden_dim, timescale)
        self.decoder = DeNOTS(self.ctx_dim, hidden_dim, timescale)
        self.mu_mlp = MLP(hidden_dim, hidden_dim, self.tgt_dim)
        self.sigma_mlp = MLP(hidden_dim, hidden_dim, self.tgt_dim, final_act=torch.exp)

        self.hidden_dim = hidden_dim
        self.timescale = timescale
        self.lr = lr

    def forward(self, ctx: Tensor, obs: Tensor, T: int) -> Tensor:
        B, L, D = obs.shape
        t = torch.cumsum(ctx[..., 0], dim=1)
        ctx_obs, ctx_tgt = torch.split(ctx, [L, T], dim=1)
        t_obs, t_tgt = torch.split(t, [L, T], dim=1)

        # Compress observations into embedding
        x_obs = torch.cat([ctx_obs, obs], dim=-1)
        h = self.encoder(x_obs, t_obs)

        # Guide the new trajectory with context.
        preds = self.decoder(ctx_tgt, t_tgt, y0=h, return_intermediate=True)
        return self.mu_mlp(preds), self.sigma_mlp(preds)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
