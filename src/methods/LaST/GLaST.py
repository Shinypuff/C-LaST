import torch
from torch.nn import functional as F
from tqdm.auto import trange

from ...layers.mlp import MLP
from .last_components import LaSTBlock, SNet, TNet
from ..base import BaseForecasting


class GLaSTForecaster(BaseForecasting):
    def __init__(
        self,
        var_num=1,
        hidden_dim=64,
        dropout=0.1,
        lr=1e-3,
        loss_args=None,
        backbone="feednet",
        backbone_args=None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        self.in_dim = self.target_dim + self.time_feat_dim  # all channels + dt

        self.v_num = var_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr

        self.LaSTLayer = LaSTBlock(
            self.in_dim,
            self.target_dim,
            self.context,
            self.horizon,
            SNet,
            self.hidden_dim,
            TNet,
            self.hidden_dim,
            dropout=dropout,
            backbone=backbone,
            backbone_args=backbone_args,
        )

        self.fusion_net = MLP(self.context, hidden_dim, self.horizon)

        self.mean_net = MLP(hidden_dim * 2, hidden_dim, self.target_dim)
        self.scale_net = MLP(
            hidden_dim * 2, hidden_dim, self.target_dim, final_act=torch.exp,
        )

    def sample(self, dt, x, num_samples):
        L = x.shape[1]

        dt = dt[:, :L, :]
        x_his = torch.cat([x, dt], dim=-1)

        z_s, x_s, z_t, x_t, _, _, _ = self.LaSTLayer(x_his) # Z: B x T x hid_dim

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1)  # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(
            0, 2, 1
        )  # Z: B x 2*hid_dim x pred_len --> B x pred_len x 2*hid_dim

        mean = x_s + x_t #self.mean_net(z)
        scale = self.scale_net(z)

        trajectories = torch.stack(
            [
                (mean + torch.randn_like(mean) * scale).cpu()
                for _ in trange(num_samples, desc="Sampling...", leave=False)
            ],
            dim=-1,
        )

        return trajectories

    def training_step(self, batch, *args, **kwargs):
        dt, x, y = batch

        L = x.shape[1]

        dt = dt[:, :L, :]
        x_his = torch.cat([x, dt], dim=-1)

        z_s, x_s, z_t, x_t, elbo, mlbo, mubo = self.LaSTLayer(x_his)  # Z: B x T x hid_dim

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1)  # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(
            0, 2, 1
        )  # Z: B x 2*hid_dim x pred_len --> B x pred_len x2*hid_dim

        mean = x_s + x_t
        scale = self.scale_net(z)

        loss = F.gaussian_nll_loss(mean, y, scale) - elbo - mlbo + mubo
        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
