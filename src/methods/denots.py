import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm.auto import trange

from ..layers.denots import DeNOTS
from ..layers.mlp import MLP
from .base import BaseForecasting


class DeNOTSForecaster(BaseForecasting):
    def __init__(
        self, timescale: int, hidden_dim: int, lr: float, sigma: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = DeNOTS(
            self.time_feat_dim + self.target_dim,
            hidden_dim,
            timescale,
            sigma=0,
        )
        self.decoder = DeNOTS(self.time_feat_dim, hidden_dim, timescale, sigma=sigma)

        self.pred_mlp = MLP(hidden_dim, hidden_dim, self.target_dim)
        self.hidden_dim = hidden_dim
        self.timescale = timescale
        self.lr = lr

    def sample(self, time_feats: Tensor, x: Tensor, num_samples: int) -> Tensor:
        B, L, D = x.shape
        T = time_feats.shape[1] - L
        t = torch.cumsum(time_feats[..., 0], dim=1)
        t_x, t_y = torch.split(t, [L, T], dim=1)

        # Compress observations into embedding
        x_obs = torch.cat([time_feats[:, :L], x], dim=-1)
        h = self.encoder(x_obs, t_x)

        preds = []
        for _ in trange(num_samples, desc="Sampling", leave=False):
            hidden = self.decoder(
                time_feats[:, L:], t_y, y0=h, return_intermediate=True
            )

            pred = self.pred_mlp(hidden).to("cpu", non_blocking=True)
            preds.append(pred)

        torch.cuda.synchronize()

        return torch.stack(preds, dim=-1)

    def training_step(self, batch, *args, **kwargs):
        time_feats, x, y_true = batch

        t = torch.cumsum(time_feats[..., 0], dim=1)
        t_x, t_y = torch.split(t, [self.context, self.horizon], dim=1)
        tf_x, tf_y = torch.split(time_feats, [self.context, self.horizon], dim=1)

        # Compress observations into embedding
        x_obs = torch.cat([tf_x, x], dim=-1)
        h = self.encoder(x_obs, t_x)
        hidden = self.decoder(tf_y, t_y, y0=h, return_intermediate=True)
        y_pred = self.pred_mlp(hidden)
        loss = F.mse_loss(y_pred, y_true)

        self.log(
            "train_mse_loss",
            loss.cpu().detach().item(),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
