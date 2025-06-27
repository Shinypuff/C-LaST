from functools import partial

import lightgbm
import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from ...train_val_test import train_val_test
from ...utils.collate_td import collate_td
from ...utils.predict_dataset import PredictDataset
from .instantiate import from_config


def coles(cfg: DictConfig):
    module, datamodule = from_config(cfg)
    run_id = train_val_test(module, datamodule, cfg, monitor="val_metric_unsup")

    clf = lightgbm.LGBMClassifier(
        n_estimators=500,
        boosting_type="gbdt",
        subsample=0.5,
        subsample_freq=1,
        learning_rate=0.02,
        feature_fraction=0.75,
        max_depth=6,
        lambda_l1=1,
        lambda_l2=1,
        min_data_in_leaf=50,
        n_jobs=8,
        verbosity=-1,
    )

    pred_trainer = Trainer(devices=1, logger=False, enable_checkpointing=False)
    train_ds = PredictDataset(datamodule.train_dataset.datasource)
    test_ds = PredictDataset(datamodule.test_dataset.datasource)

    collate_fn = partial(collate_td, seq_feats=module.encoder.feats)
    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_dl = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    train_embs, test_embs = pred_trainer.predict(module, (train_dl, test_dl))

    train_embs = torch.cat(train_embs).cpu().numpy()
    test_embs = torch.cat(test_embs).cpu().numpy()

    train_lbls = train_ds.labels
    test_lbls = test_ds.labels

    clf = clf.fit(train_embs, train_lbls)
    preds = clf.predict_proba(test_embs)

    match cfg["criterion"]:
        case "binary":
            metric = roc_auc_score(test_lbls, preds[:, 1])
        case "multiclass":
            metric = accuracy_score(test_lbls, preds.argmax(axis=1))
    mlflow.log_metric("test_metric", metric, run_id=run_id)
    logger.info(f"Metric: {metric}")
