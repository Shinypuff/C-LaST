import torch
from torch import nn

from .last_components import LaSTBlock, SNet, TNet
from ..base import BaseForecasting


class LaSTForecaster(BaseForecasting):
    def __init__(self, var_num=1, hidden_dim=64, dropout=0.1, lr=1e-3, backbone="feednet", backbone_args=None, **base_kwargs):
        super().__init__(**base_kwargs)

        self.in_dim = self.target_dim + self.time_feat_dim # all channels + dt

        self.v_num = var_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.criterion = nn.MSELoss()
        self.lr = lr

        self._para_mode = 0

        self.LaSTLayer = LaSTBlock(
            self.in_dim, self.target_dim, 
            self.context, self.horizon, 
            SNet, self.hidden_dim, 
            TNet, self.hidden_dim, 
            dropout=dropout,
            backbone=backbone,
            backbone_args=backbone_args,
            )

    def forward(self, dt, x):
        L = x.shape[1]
        T = dt.shape[1] - L

        dt = dt[:, :L,:]

        x_his = torch.cat([x, dt], dim=-1)
        x_s, x_t, _, _, _ = self.LaSTLayer(x_his)
        
        mean = x_s + x_t
        scale = torch.zeros_like(mean)

        return mean, scale

    def training_step(self, batch, *args, **kwargs):
        dt, x, y = batch

        L = x.shape[1]
        T = dt.shape[1] - L

        dt = dt[:, :L,:]
        x_his = torch.cat([x, dt], dim=-1)

        model_optim = self.optimizers()

        for para_mode in range(2):
            model_optim.zero_grad()

            if para_mode == 0:
                for para in self.parameters():
                    para.requires_grad = True

                for para in self.LaSTLayer.MuboNet.parameters():
                    para.requires_grad = False
                for para in self.LaSTLayer.SNet.VarUnit_s.critic_xz.parameters():
                    para.requires_grad = False
                for para in self.LaSTLayer.TNet.VarUnit_t.critic_xz.parameters():
                    para.requires_grad = False

            elif para_mode == 1:
                for para in self.parameters():
                    para.requires_grad = False

                for para in self.LaSTLayer.MuboNet.parameters():
                    para.requires_grad = True
                for para in self.LaSTLayer.SNet.VarUnit_s.critic_xz.parameters():
                    para.requires_grad = True
                for para in self.LaSTLayer.TNet.VarUnit_t.critic_xz.parameters():
                    para.requires_grad = True

            x_s, x_t, elbo, mlbo, mubo = self.LaSTLayer(x_his)

            mean = x_s + x_t
            loss = self.criterion(mean, y) - elbo - mlbo + mubo if para_mode == 0 else mubo - mlbo

            loss.backward()
            model_optim.step()
            
        self.log("train_last_loss", self.criterion(mean, y) - elbo - mlbo + mubo, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @property
    def automatic_optimization(self):
        return False