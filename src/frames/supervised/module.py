from itertools import chain
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import nn
from torchmetrics import AUROC, Accuracy, Metric, R2Score

from ...encoder import EvSEncoder, TSEncoder
from ...nn.losses import RelaxedBCELogitLoss, RelaxedCrossEntropyLoss


def choose_criterion(
    criterion_name: Literal["binary", "multiclass", "regression"],
    hidden_dim,
    num_classes,
):
    match criterion_name:
        case "binary":
            loss_fn = RelaxedBCELogitLoss()
            metric_fn = AUROC("binary")
            head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Flatten(0, -1))
            activation = nn.Sigmoid()
        case "multiclass":
            loss_fn = RelaxedCrossEntropyLoss()
            metric_fn = Accuracy("multiclass", num_classes=num_classes)
            head = nn.Linear(hidden_dim, num_classes)
            activation = nn.Softmax(dim=1)
        case "regression":
            loss_fn = nn.MSELoss()
            metric_fn = R2Score()
            head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Flatten(0, -1))
            activation = nn.Identity()
        case unknown_criterion:
            raise ValueError(f"Unknown criterion: {unknown_criterion}")
    return loss_fn, metric_fn, head, activation


class SupervisedModule(LightningModule):
    def __init__(
        self,
        encoder: EvSEncoder | TSEncoder,
        backbone: nn.Module,
        head: nn.Module,
        activation: nn.Module,
        loss_fn: nn.Module,
        metric: Metric,
        learning_rate: float,
    ):
        super().__init__()

        self.encoder = encoder
        self.backbone = backbone

        for param in chain(self.encoder.parameters(), self.backbone.parameters()):
            param.requires_grad = False

        self.learning_rate = learning_rate
        self.head = head
        self.activation = activation
        self.loss_fn = loss_fn

        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

    def train(self, mode: bool = True):
        for module in self.children():
            module.train(mode)

        self.encoder.train(False)
        self.backbone.train(False)
        return self

    def forward(self, batch: TensorDict) -> torch.Tensor:
        time = batch["time"].to(torch.float32)
        mask = batch["mask"]

        x = self.encoder(batch)
        h = self.backbone(x, time, mask)
        logits = self.head(h)
        return logits

    def training_step(self, batch: TensorDict) -> torch.Tensor:
        y_true = batch["target"]
        logits = self(batch)
        loss = self.loss_fn(logits, y_true)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        y_true = batch["target"]
        logits = self(batch)
        y_pred = self.activation(logits)
        self.val_metric(y_pred, y_true)
        self.log(
            "val_metric", self.val_metric, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        y_true = batch["target"]
        logits = self(batch)
        y_pred = self.activation(logits)
        self.test_metric(y_pred, y_true)
        self.log(
            "test_metric", self.test_metric, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
