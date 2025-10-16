import torch
from torch import Tensor

from ..layers.denots import DeNOTS
from ..layers.mlp import MLP
from ..layers.node import NeuralODE
from ..losses.kldiv import gaussian_kldiv_loss
from ..losses.nll import gaussian_nll_loss
from .base import BaseForecasting


class LatentODE(BaseForecasting):
    def __init__(self, hidden_dim: int, timescale: float, lr: float, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr

        self.encoder = DeNOTS(self.target_dim + 1, hidden_dim, timescale)
        self.z_scale_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.decoder = NeuralODE(hidden_dim, timescale)

        self.z_mean_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.z_scale_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, final_act=torch.exp)

        self.x_mean_mlp = MLP(hidden_dim, hidden_dim, self.target_dim)
        self.x_scale_mlp = MLP(
            hidden_dim, hidden_dim, self.target_dim, final_act=torch.exp
        )

    def encode(self, dt: Tensor, x: Tensor, return_intermediate: bool = False):
        L = x.shape[1]

        time_inv = dt[:, :L].roll(-1, 1).flip(1).cumsum(dim=1)
        x_ = torch.cat([dt[:, :L, None], x], dim=-1)
        embedding = self.encoder(
            x_.flip(1), time_inv, return_intermediate=return_intermediate
        )
        return embedding

    def latent_distribution(self, embedding: Tensor):
        mean = self.z_mean_mlp(embedding)
        scale = self.z_scale_mlp(embedding)
        return mean, scale

    def decode(self, sample: Tensor, time: Tensor):
        embedded_seq = self.decoder(sample, time)
        return embedded_seq

    def forward(self, dt: Tensor, x: Tensor):
        L = x.shape[1]
        embedding = self.encode(dt, x)
        mean, scale = self.latent_distribution(embedding)
        sample = torch.randn_like(mean) * scale + mean
        decoded = self.decode(sample, dt.cumsum(1))
        mean = self.x_mean_mlp(decoded)
        scale = self.x_scale_mlp(decoded)
        return mean[:, L:], scale[:, L:]

    def training_step(self, batch, *args, **kwargs):
        dt, x, y = batch
        embedding = self.encode(dt, x)
        zmean, zscale = self.latent_distribution(embedding)
        z0 = torch.randn_like(zmean) * zscale + zmean
        z = self.decode(z0, dt.cumsum(1))

        xmean = self.x_mean_mlp(z)
        xscale = self.x_scale_mlp(z)
        reconstruction_loss = gaussian_nll_loss(torch.cat([x, y], dim=1), xmean, xscale)
        regularization_loss = gaussian_kldiv_loss(zmean, zscale)

        self.log(
            "train_reconstruction_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_regularization_loss",
            regularization_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return reconstruction_loss + regularization_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)
