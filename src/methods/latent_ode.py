import torch
from torch import Tensor
from torch.nn import functional as F

from ..layers.denots import DeNOTS
from ..layers.mlp import MLP
from .base import BaseForecasting


class LatentODE(BaseForecasting):
    def __init__(self, hidden_dim: int, timescale: float, lr: float, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr

        self.encoder = DeNOTS(
            self.target_dim + self.time_feat_dim, hidden_dim, timescale, sigma=0
        )

        self.z_mean_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.z_lvar_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)

        self.decoder = DeNOTS(self.time_feat_dim, hidden_dim, timescale, sigma=0)

        self.x_mean_mlp = MLP(hidden_dim, hidden_dim, self.target_dim)
        self.x_lvar_mlp = MLP(hidden_dim, hidden_dim, self.target_dim)

    def encode(self, time_feats: Tensor, x: Tensor):
        dt = time_feats[..., 0]
        time_inv = dt.roll(-1, 1).flip(1).cumsum(dim=1)
        x_ = torch.cat([time_feats, x], dim=-1)
        embedding = self.encoder(x_.flip(1), time_inv)
        return embedding

    def latent_distribution(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.z_mean_mlp(embedding)
        lvar = self.z_lvar_mlp(embedding)
        return mean, lvar

    def decode(self, sample: Tensor, time_feats: Tensor):
        time = time_feats[..., 0].cumsum(1)
        embedded_seq = self.decoder(
            time_feats, time, y0=sample, return_intermediate=True
        )
        return embedded_seq

    def sample(
        self,
        time_feats: Tensor,
        x: Tensor,
        num_samples: int,
    ):
        embedding = self.encode(time_feats[:, : x.shape[1]], x)
        meanz, lvarz = self.latent_distribution(embedding)

        preds = []

        for _ in range(num_samples):
            sample = torch.randn_like(meanz) * (lvarz / 2).exp() + meanz
            decoded = self.decode(sample, time_feats)[:, self.context :]
            mean = self.x_mean_mlp(decoded)
            lvar = self.x_lvar_mlp(decoded)
            scale = (lvar / 2).exp()
            pred: Tensor = torch.randn_like(mean) * scale + mean
            preds.append(pred.cpu())

        return torch.stack(preds, dim=-1)

    def training_step(self, batch, *args, **kwargs):
        time_feats, x, y = batch
        embedding = self.encode(time_feats[:, : x.shape[1]], x)
        meanz, lvarz = self.latent_distribution(embedding)
        sample = torch.randn_like(meanz) * (lvarz / 2).exp() + meanz
        decoded = self.decode(sample, time_feats)

        meanxy: Tensor = self.x_mean_mlp(decoded)
        lvarxy: Tensor = self.x_lvar_mlp(decoded)
        varxy = lvarxy.exp()
        varz = lvarz.exp()

        xy = torch.cat([x, y], dim=1)

        reconstruction_loss = ((xy - meanxy) ** 2 / varxy + lvarxy).mean() / 2

        regularization_loss = (varz + meanz**2 - lvarz).mean() / 2

        self.log(
            "train_nll_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_kld_loss",
            regularization_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return reconstruction_loss + regularization_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)
