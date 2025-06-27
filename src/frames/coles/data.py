import os
from copy import deepcopy
from functools import partial

import numpy as np
import polars as pl
import torch
from pytorch_lightning import LightningDataModule
from tensordict import TensorDict, from_struct_array, merge_tensordicts
from torch.utils.data import DataLoader, IterableDataset

from ...utils.collate_td import collate_td


class ColesDataset(IterableDataset):
    def __init__(self, datasource: str, window_size: int, n_slices: int, seed: int):
        self.datasource = datasource
        self.data = pl.read_parquet(datasource).to_numpy(structured=True)
        self.window_size = window_size
        self.n_slices = n_slices
        self.rng = np.random.default_rng(seed=seed)

    def __iter__(self):
        for index in self.rng.permutation(len(self.data)):
            row_td: TensorDict = from_struct_array(
                deepcopy(self.data[index]), batch_size=()
            )

            seq_keys = [k for k in row_td.keys() if len(row_td.get_item_shape(k)) == 1]
            seq_td: TensorDict = row_td.select(*seq_keys)
            seq_td.auto_batch_size_()

            stat_td = row_td.exclude(*seq_keys)
            stat_td["target"] = index

            starts = torch.randint(0, len(seq_td) - self.window_size, (self.n_slices,))
            ends = starts + self.window_size

            for s, e in zip(starts, ends):
                slice_td: TensorDict = seq_td[s:e]
                slice_td.batch_size = ()
                yield merge_tensordicts(slice_td, stat_td)

    def __len__(self):
        return len(self.data) * self.n_slices


class ColesDataModule(LightningDataModule):
    def __init__(
        self,
        datasource: str,
        window_size: int,
        n_slices: int,
        seq_feats: list[str],
        batch_size: int,
        seed: int,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_slices = n_slices
        self.batch_size = batch_size
        self.data_dir = datasource
        self.batch_size = batch_size
        self.collate_fn = partial(collate_td, seq_feats=seq_feats)
        fpath: str = (
            os.environ["DATA_DIR"] + "/preprocessed/" + datasource + "_{}.parquet"
        )

        ds_args = dict(n_slices=n_slices, window_size=window_size, seed=seed)

        self.train_dataset = ColesDataset(fpath.format("train"), **ds_args)
        self.val_dataset = ColesDataset(fpath.format("val"), **ds_args)
        self.test_dataset = ColesDataset(fpath.format("test"), **ds_args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=1,
        )
