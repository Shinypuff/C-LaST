from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import nn
from torch.nn.functional import mse_loss

from ..nn.decoders.gru_decoder import GRUDecoder


class SimpleForecasting(LightningModule):
    def __init__(
        self,
        target: str,
        feats: list[str],
        backbone: nn.Module,
        decoder: Literal["GRU", "ODE"],
        hidden_dim: int,
        obs_frac: float,
        learning_rate: float,
    ):
        super().__init__()

        self.target = target
        self.feats = feats
        self.backbone = backbone
        self.obs_frac = obs_frac
        self.learning_rate = learning_rate

        match decoder:
            case "GRU":
                self.decoder = GRUDecoder(hidden_dim)
            case _:
                raise ValueError(f"Unknown decoder! {decoder}.")

    def forward(self, x_obs, t_obs, t_pred) -> torch.Tensor:
        embedding = self.backbone(x_obs, t_obs)
        prediction = self.decoder(embedding, t_pred)
        return prediction

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
