"""File with all data-related tools."""

import polars as pl
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .preprocess import process_split


class DataModule(LightningDataModule):
    def __init__(
        self,
        datasource: str,
        context: int,
        horizon: int,
        target_cols: list[str] | None,
        time_col: str | None,
        has_time: bool,
        has_date: bool,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.datasource = datasource
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.context = context
        self.horizon = horizon

        df = pl.read_csv(datasource, try_parse_dates=True)

        if time_col is None:
            df = df.with_row_index("time")
            time_col = "time"

        if target_cols is None:
            target_cols = [c for c in df.columns if c != time_col]

        df = df.with_columns(
            time_norm=(pl.col(time_col) - pl.col(time_col).min())
            / (pl.col(time_col).max() - pl.col(time_col).min())
        )

        # 6:2:2 split
        train_df: pl.DataFrame
        train_df, valtest_df = train_test_split(df, test_size=0.4, shuffle=False)
        val_df, test_df = train_test_split(valtest_df, test_size=0.5, shuffle=False)

        # Calculate train statistics
        target_mean = train_df.select(target_cols).mean().to_torch().float()
        target_scale = train_df.select(target_cols).std().to_torch().float()
        dt_mean = train_df.select(pl.col(time_col).diff()).mean().item()

        prepr_args = dict(
            context=context,
            horizon=horizon,
            target_cols=target_cols,
            time_col=time_col,
            target_mean=target_mean,
            target_scale=target_scale,
            dt_mean=dt_mean,
            has_time=has_time,
            has_date=has_date,
        )

        # TODO: add group-by ids for multi-sequence datasets
        t_train, x_train, y_train = process_split(train_df, **prepr_args)
        t_val, x_val, y_val = process_split(val_df, **prepr_args)
        t_test, x_test, y_test = process_split(test_df, **prepr_args)

        self.train_ds = TensorDataset(t_train, x_train, y_train)
        self.val_ds = TensorDataset(t_val, x_val, y_val)
        self.test_ds = TensorDataset(t_test, x_test, y_test)

        self.target_dim = x_train.shape[-1]
        self.time_feat_dim = t_train.shape[-1]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )