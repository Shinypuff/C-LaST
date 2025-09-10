import torch
from torch import Tensor, nn

from ..nn.layers.mlp import MLP
from .base import BaseForecasting


class SupervisedForecasting(BaseForecasting):
    def __init__(
        self,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        self.encoder = nn.GRU(
            self.tgt_dim + self.ctx_dim, self.hidden_dim, batch_first=True
        )
        self.decoder = nn.GRUCell(self.hidden_dim + self.ctx_dim, self.hidden_dim)

        self.mu_mlp = MLP(self.hidden_dim, self.hidden_dim, self.tgt_dim)
        self.std_mlp = MLP(self.hidden_dim, self.hidden_dim, self.tgt_dim)

    def forward(self, ctx: Tensor, obs: Tensor, T: int):
        L = obs.shape[1]
        past_emb = torch.cat([ctx[:, :L], obs], dim=-1)
        h0 = self.encoder(past_emb)[1].squeeze(0)
        h = h0
        pred_embs = []
        for i in range(T):
            x = torch.cat([h0, ctx[:, L + i]], dim=-1)
            h = self.decoder(x, h)
            pred_embs.append(h)

        pred_embs = torch.stack(pred_embs, dim=1)
        mu = self.mu_mlp(pred_embs)
        std = self.std_mlp(pred_embs).exp()

        return mu, std
