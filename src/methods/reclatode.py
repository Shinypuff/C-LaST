import numpy as np
import torch
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import nn
from torch.nn.functional import mse_loss

from ..nn.decoders.ode_decoder import ODEDecoder
from ..nn.encoders.denots import DeNOTS


class LatentODE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder = DeNOTS(input_dim, hidden_dim)
        self.decoder = ODEDecoder(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.encoder(x.flip(1), t.flip(1))
        x_pred = self.decoder(h, t)
        return h, x_pred


class RecLatODEForecasting(LightningModule):
    def __init__(
        self,
        target: str,
        feats: list[str],
        dim_inflation: int,
        time_compression: float,
        patch_size: int,
        patch_stride: int,
        depth: int,
        learning_rate: float,
    ):
        super().__init__()

        self.target = target
        self.feats = feats
        self.dim_inflation = dim_inflation
        self.time_compression = time_compression
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.learning_rate = learning_rate

        self.layers = nn.ModuleList()

        for d in depth:
            layer = LatentODE(d, d * self.dim_inflation)
            self.layers.append(layer)

    def _step(self, x: torch.Tensor, t: torch.Tensor):
        loss = 0
        B, L, C = x.shape
        Lp = self.patch_size
        Sp = self.patch_stride
        Np = L // Sp

        for layer in self.layers:
            x_patched = x.unfold(dimension=1, size=Lp, step=Sp)  # (B, Np, C, Lp)
            t_patched = t.unfold(dimension=1, size=Lp, step=Sp)  # (B, Np, Lp)

            x = x_patched.permute(0, 3, 1, 2).reshape(B * Np, Lp, C)

            h, x_pred = layer(x, t)
            loss += mse_loss(x, x_pred)
            x = x_pred

    def _step(self, batch):
        feats = [batch[f] for f in self.feats]
        feats.append(batch[self.target])
        x = torch.stack(feats, dim=-1)
        y = batch[self.target]
        t = batch["timestamp"].to(torch.int32)
        L = t.shape[1]

        L_obs = (L * self.obs_frac).int()
        t_obs = t[:L_obs]
        t_pred = t[L_obs - 1 :]  # To include time since last observed timestamp
        x_obs = x[:, :L_obs]
        y_true = y[:, L_obs:]

        y_pred = self(x_obs, t_obs, t_pred)
        loss = mse_loss(y_true, y_pred)
        return loss

    def training_step(self, batch: TensorDict) -> torch.Tensor:
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        loss = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        loss = self._step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
