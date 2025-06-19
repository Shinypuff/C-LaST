import os
from functools import partial
from typing import Callable

import mlflow
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from tensordict import TensorDict
from torch import nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm

from ..data import Dataset, collate_td
from ..encoder import Encoder
from ..nn.losses import RelaxedBCELogitLoss, RelaxedCrossEntropyLoss


def choose_criterion(criterion_name, hidden_dim, num_classes):
    match criterion_name:
        case "binary":
            loss_fn = RelaxedBCELogitLoss()
            metric_fn = roc_auc_score
            head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Flatten(0, -1))
            activation = nn.Sigmoid()
        case "multiclass":
            loss_fn = RelaxedCrossEntropyLoss()
            metric_fn = accuracy_score
            head = nn.Linear(hidden_dim, num_classes)
            activation = nn.Softmax(dim=1)
        case "regression":
            loss_fn = nn.MSELoss()
            metric_fn = r2_score
            head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Flatten(0, -1))
            activation = nn.Identity()
        case unknown_criterion:
            raise ValueError(f"Unknown criterion: {unknown_criterion}")
    return loss_fn, metric_fn, head, activation


class SupervisedTrainer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        backbone: nn.Module,
        head: nn.Module,
        activation: nn.Module,
        loss_fn: nn.Module,
        metric_fn: Callable,
        learning_rate: float,
        batch_size: int,
        samples_per_epoch: int,
        patience: int,
        balance: bool,
        device: str,
    ):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.head = head
        self.activation = activation
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.patience = patience
        self.balance = balance
        self.device = device

        self.collate_fn = partial(
            collate_td, seq_feats=encoder.cat_feats + encoder.num_feats + ["time"]
        )
        self._step = 0
        self._epoch = 0

    @classmethod
    def from_config(cls, cfg: DictConfig):
        cat_feats = cfg["cat_feats"]
        num_feats = cfg["num_feats"]
        emb_dim = cfg["emb_dim"]
        hidden_dim = cfg["hidden_dim"]
        learning_rate = cfg["learning_rate"]
        batch_size = cfg["batch_size"]
        patience = cfg["patience"]
        balance = cfg["balance"]
        samples_per_epoch = cfg["samples_per_epoch"]
        num_classes = cfg.get("num_classes")
        device = cfg["device"]

        encoder = Encoder(
            cat_feats=cat_feats,
            num_feats=num_feats,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )

        backbone = instantiate(cfg["backbone"], _partial_=True)(hidden_dim=hidden_dim)

        loss_fn, metric_fn, head, activation = choose_criterion(
            cfg["criterion"], hidden_dim, num_classes
        )

        return cls(
            encoder=encoder,
            backbone=backbone,
            head=head,
            activation=activation,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            learning_rate=learning_rate,
            batch_size=batch_size,
            samples_per_epoch=samples_per_epoch,
            patience=patience,
            balance=balance,
            device=device,
        )

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
        return self.loss_fn(logits, y_true)

    @torch.no_grad()
    def eval_step(self, batch: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        y_true = batch["target"]
        logits = self(batch)
        y_pred = self.activation(logits)
        return y_true.cpu().numpy(), y_pred.cpu().numpy()

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        pbar_desc: str = "",
    ):
        loss_sum = 0
        self.train()
        pbar = tqdm(total=len(dataloader), leave=False, desc=pbar_desc)
        for batch in dataloader:
            batch = batch.to(self.device)
            loss = self.training_step(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update()
            loss_sum += loss.detach().item()
            self._step += 1
        return loss_sum / len(dataloader)

    def eval_epoch(self, dataloader: DataLoader):
        self.eval()
        Y_true = []
        Y_pred = []
        pbar = tqdm(total=len(dataloader), leave=False, desc="Evaluating")
        for batch in dataloader:
            batch = batch.to(self.device)
            y_true, y_pred = self.eval_step(batch)
            Y_true.append(y_true)
            Y_pred.append(y_pred)
            pbar.update()

        Y_true_np = np.concatenate(Y_true)
        Y_pred_np = np.concatenate(Y_pred)
        return self.metric_fn(Y_true_np, Y_pred_np)

    def fit(self, train_dataset: Dataset, val_dataset: Dataset):
        logger.info(
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}; {self.device=}"
        )
        self.to(self.device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        best_val_metric = -1e9

        if self.balance:
            targets = np.array([el["target"].item() for el in train_dataset])
            _, counts = np.unique(targets, return_counts=True)
            sampler = WeightedRandomSampler(
                weights=1 / counts[targets.astype(np.int32)],
                num_samples=self.samples_per_epoch,
            )
        else:
            sampler = RandomSampler(
                train_dataset,
                replacement=self.samples_per_epoch > len(train_dataset),
                num_samples=self.samples_per_epoch,
            )

        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

        patience = self.patience
        pbar_desc = "Epoch 0"
        run_id = mlflow.active_run().info.run_id
        weights_path = f"/tmp/{run_id}_best_model.pt"

        while patience >= 0:
            train_loss = self.train_epoch(
                train_dataloader, optimizer, pbar_desc=pbar_desc
            )
            val_metric = self.eval_epoch(val_dataloader)

            pbar_desc = f"Epoch {self._epoch}: {train_loss=:.3f}, {val_metric=:.2f}"

            mlflow.log_metric("train_loss", train_loss, step=self._epoch)
            mlflow.log_metric("val_metric", val_metric, step=self._epoch)

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience = self.patience
                torch.save(self.state_dict(), weights_path)
            else:
                patience -= 1
            self._epoch += 1

        logger.info(f"Best val metric: {best_val_metric}")
        self.load_state_dict(torch.load(weights_path))
        return best_val_metric

    def test(self, test_dataset: Dataset):
        test_metric = self.eval_epoch(
            DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )
        )

        mlflow.log_metric("test_metric", test_metric)
        logger.info(f"Test metric: {test_metric}")
        return test_metric
