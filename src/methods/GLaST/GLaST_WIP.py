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


class GLaSTForecaster(BaseForecasting):
    def __init__(self, input_len, output_len, var_num=1, hidden_dim=64, dropout=0.1, lr=1e-3, **base_kwargs):
        super().__init__(**base_kwargs)

        self.in_dim = self.target_dim + 1 # all channels + dt
        self.out_dim = self.target_dim
        self.seq_len = input_len # L
        self.pred_len = output_len # T

        self.v_num = var_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr

        self.LaSTLayer = LaSTBlock(
            self.in_dim, self.out_dim, 
            input_len, 
            output_len, 
            SNet, self.hidden_dim, 
            TNet, self.hidden_dim, 
            dropout=dropout
            )

        self.fusion_net = MLP(self.seq_len, hidden_dim, self.pred_len)

        self.mean_net = MLP(hidden_dim * 2, hidden_dim, self.target_dim)
        self.scale_net = nn.Sequential(
            MLP(hidden_dim * 2, hidden_dim, self.target_dim),
            nn.Softplus()
        )

    def forward(self, dt, x):
        L = x.shape[1]

        dt = dt.unsqueeze(-1)[:, :L,:]
        x_his = torch.cat([x, dt], dim=-1)

        z_s, z_t = self.LaSTLayer(x_his) # Z: B x T x hid_dim

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1) # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(0, 2, 1) # Z: B x 2*hid_dim x pred_len --> B x pred_len x2*hid_dim

        mean = self.mean_net(z)
        scale = self.scale_net(z)

        return mean, scale

    def training_step(self, batch, *args, **kwargs):
        dt, x, y = batch

        L = x.shape[1]

        dt = dt.unsqueeze(-1)[:, :L,:]
        x_his = torch.cat([x, dt], dim=-1)

        z_s, z_t, elbo, mlbo, mubo = self.LaSTLayer.get_losses(x_his) # Z: B x T x hid_dim

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1) # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(0, 2, 1) # Z: B x 2*hid_dim x pred_len --> B x pred_len x2*hid_dim

        mean = self.mean_net(z)
        scale = self.scale_net(z)

        loss = gaussian_nll(y, mean, scale) - elbo - mlbo + mubo
        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)