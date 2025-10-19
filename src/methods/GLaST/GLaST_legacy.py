import math
import torch
from torch import nn
from torch.autograd import Variable

from ...losses.nll import gaussian_nll_loss
from ...layers.mlp import MLP
from ..base import BaseForecasting
from ..LaST.units import *
from ..LaST.last_components import LaSTBlock, SNet, TNet

class GLaSTForecaster(BaseForecasting):
    def __init__(self, input_len=24, output_len=24, var_num=1, hidden_dim=64, dropout=0.1, lr=1e-3, backbone="feednet", backbone_args=None, **base_kwargs):
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
            dropout=dropout,
            backbone=backbone,
            backbone_args=backbone_args,
        )

        self.fusion_net = MLP(self.seq_len, hidden_dim, self.pred_len)

        self.mean_net = MLP(hidden_dim * 2, hidden_dim, self.target_dim)
        
        self.scale_net = nn.Sequential(
            MLP(hidden_dim * 2, hidden_dim, self.target_dim),
            nn.Softplus()
        )

    def forward(self, dt, x):
        #print(f"x shape: {x.shape}")  
        #print(f"dt shape: {dt.shape}") 
        L = x.shape[1]

        dt_simple = dt[:, :L, 0:1] 
        #print(f"dt after processing: {dt_simple.shape}")
        
        x_his = torch.cat([x, dt_simple], dim=-1)
        #print(f"x_his after concat: {x_his.shape}")

        #print("Calling LaSTLayer...")
        x_s, x_t, _ = self.LaSTLayer.get_emb(x_his)
        #print(f"x_s shape: {x_s.shape}, x_t shape: {x_t.shape}")

        z = torch.cat([x_s, x_t], dim=-1)
        #print(f"z after concat: {z.shape}")
        
        z = z.permute(0, 2, 1)  # [batch, features, seq_len]
       # print(f"z after permute: {z.shape}")
        
        #print(f"Fusion net input: {z.shape}, expecting seq_len={self.seq_len}")
        z = self.fusion_net(z) 
        #print(f"z after fusion_net: {z.shape}")
        
        z = z.permute(0, 2, 1)  # [batch, pred_len, features]
        #print(f"z after final permute: {z.shape}")

        mean = self.mean_net(z)
        #print(f"mean shape: {mean.shape}")
        scale = self.scale_net(z)
        #print(f"scale shape: {scale.shape}")

        return mean, scale  

    def training_step(self, batch, *args, **kwargs):
        dt, x, y = batch
        L = x.shape[1]

        dt_simple = dt[:, :L, 0:1]  # [batch, seq_len, 1]
        #print(f"dt_simple shape: {dt_simple.shape}")
        
        x_his = torch.cat([x, dt_simple], dim=-1)
        #print(f"x_his shape: {x_his.shape}")

        z_s, z_t, elbo, mlbo, mubo = self.LaSTLayer.get_losses(x_his)

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1)
        z = self.fusion_net(z).permute(0, 2, 1)

        mean = self.mean_net(z)
        scale = self.scale_net(z)

        #print(f"mean shape: {mean.shape}, scale shape: {scale.shape}")
        #print(f"elbo: {elbo}, mlbo: {mlbo}, mubo: {mubo}")

        loss = gaussian_nll_loss(y, mean, scale) - elbo - mlbo + mubo
        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)