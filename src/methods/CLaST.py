import torch

from .LaST.last_components import LaSTBlock, SNet, TNet
from ..losses.nll import gaussian_nll_loss
from ..losses.depts2vec_loss import DepTS2Vec_loss
from ..layers.mlp import MLP

from .base import BaseForecasting

class CLaSTForecaster(BaseForecasting):
    def __init__(self, var_num=1, hidden_dim=64, dropout=0.1, lr=1e-3, loss_args=None, backbone="feednet", backbone_args=None, **base_kwargs):
        super().__init__(**base_kwargs)

        self.in_dim = self.target_dim + self.time_feat_dim # all channels + dt

        self.v_num = var_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.contrastive_loss = DepTS2Vec_loss(**loss_args)

        self.LaSTLayer = LaSTBlock(
            self.in_dim, self.target_dim, 
            self.context, self.horizon, 
            SNet, self.hidden_dim, 
            TNet, self.hidden_dim, 
            dropout=dropout,
            backbone=backbone,
            backbone_args=backbone_args,
            )

        self.fusion_net = MLP(self.context, hidden_dim, self.horizon)

        self.mean_net = MLP(hidden_dim * 2, hidden_dim, self.target_dim)
        self.scale_net = MLP(hidden_dim * 2, hidden_dim, self.target_dim)

    def forward(self, dt, x):
        L = x.shape[1]

        dt = dt[:, :L,:]
        x_his = torch.cat([x, dt], dim=-1)

        z_s, z_t, _ = self.LaSTLayer.get_emb(x_his) # Z: B x T x hid_dim

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1) # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(0, 2, 1) # Z: B x 2*hid_dim x pred_len --> B x pred_len x 2*hid_dim

        mean = self.mean_net(z)
        scale = self.scale_net(z).exp()

        return mean, scale

    def training_step(self, batch, *args, **kwargs):
        dt, x, y = batch

        L = x.shape[1]

        dt = dt[:, :L,:]
        x_his = torch.cat([x, dt], dim=-1)

        z_s, z_t, elbo = self.LaSTLayer.get_emb(x_his) # Z: B x T x hid_dim
        z_s2, z_t2, elbo2 = self.LaSTLayer.get_emb(x_his)

        loss_s = self.contrastive_loss(z_s, z_s2)
        loss_t = self.contrastive_loss(z_t, z_t2)

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1) # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(0, 2, 1) # Z: B x 2*hid_dim x pred_len --> B x pred_len x2*hid_dim

        mean = self.mean_net(z)
        scale = self.scale_net(z).exp()

        loss = gaussian_nll_loss(y, mean, scale) + (loss_s + loss_t) - (elbo + elbo2)/2
        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)