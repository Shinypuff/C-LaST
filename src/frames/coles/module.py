import numpy as np
import torch
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import nn

from ...encoder import EvSEncoder
from .loss import ContrastiveLoss
from .metric import RecallTopK


class ColesModule(LightningModule):
    def __init__(
        self,
        neg_samples: int,
        margin: float,
        encoder: EvSEncoder,
        backbone: nn.Module,
        learning_rate: float,
    ):
        super().__init__()

        self.encoder = encoder
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.neg_samples = neg_samples
        self.margin = margin

        self.loss = ContrastiveLoss(neg_samples, margin)
        metric = RecallTopK(k=4, metric="cosine")
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

    def forward(self, batch: TensorDict) -> torch.Tensor:
        time = batch["time"].to(torch.float32)
        mask = batch["mask"]

        x = self.encoder(batch)
        h = self.backbone(x, time, mask)

        return h

    def training_step(self, batch: TensorDict) -> torch.Tensor:
        user_tags = batch["target"]
        h = self(batch)
        loss = self.loss(h, user_tags)
        self.log(
            "train_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        user_tags = batch["target"]
        h = self(batch)
        loss = self.loss(h, user_tags)
        self.val_metric(h, user_tags)

        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_metric_unsup",
            self.val_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        user_tags = batch["target"]
        h = self(batch)
        loss = self.loss(h, user_tags)
        self.test_metric(h, user_tags)

        self.log(
            "test_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_metric_unsup",
            self.test_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
