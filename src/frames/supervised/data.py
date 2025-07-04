import os
from functools import partial

import numpy as np
import polars as pl
from pytorch_lightning import LightningDataModule
from tensordict import from_struct_array
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torch.utils.data import Dataset as Dataset_

from ...utils.collate_td import collate_td


class Dataset(Dataset_):
    def __init__(self, dataframe: pl.DataFrame):
        self.data = dataframe.to_numpy(structured=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return from_struct_array(self.data[index].copy(), batch_size=())


class DataModule(LightningDataModule):
    def __init__(
        self,
        datasource: str,
        batch_size: int,
        balance: bool,
        samples_per_epoch: int,
        seq_feats: list[str],
    ):
        super().__init__()
        self.balance = balance
        self.datasource = datasource
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch

        if "time" not in seq_feats:
            seq_feats = seq_feats + ["time"]
        self.collate_fn = partial(collate_td, seq_feats=seq_feats)

        data_path = os.path.join(
            os.environ["DATA_DIR"], "preprocessed", datasource + ".parquet"
        )

        df = pl.read_parquet(data_path)
        splits = df.partition_by("split", as_dict=True, include_key=False)

        self.train_dataset = Dataset(splits[("train",)])
        self.val_dataset = Dataset(splits[("val",)])
        self.test_dataset = Dataset(splits[("test",)])

    def train_dataloader(self):
        if self.balance:
            targets = np.array([el["target"].item() for el in self.train_dataset])
            _, counts = np.unique(targets, return_counts=True)
            sampler = WeightedRandomSampler(
                weights=1 / counts[targets.astype(np.int32)],
                num_samples=self.samples_per_epoch,
            )
        else:
            sampler = RandomSampler(
                self.train_dataset,
                replacement=self.samples_per_epoch > len(self.train_dataset),
                num_samples=self.samples_per_epoch,
            )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
