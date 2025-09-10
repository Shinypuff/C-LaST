import polars as pl
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset

from .utils import split_history_horizon


class DataModule(LightningDataModule):
    def __init__(
        self,
        datasource: str,
        context: int,
        horizon: int,
        batch_size: int = 32,
        num_workers: int = 16,
        tgt_cols: list[str] | None = None,
        time_col: str | None = None,
    ):
        super().__init__()
        self.datasource = datasource
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context = context
        self.horizon = horizon

        df = pl.read_csv(datasource, try_parse_dates=True)

        # 6:2:2 split
        train_df, valtest_df = train_test_split(df, test_size=0.4, shuffle=False)
        val_df, test_df = train_test_split(valtest_df, test_size=0.5, shuffle=False)

        # TODO: add group-by ids for multi-sequence datasets
        ctx_train, obs_train, tgt_train = split_history_horizon(
            train_df, context, horizon, tgt_cols, time_col
        )
        ctx_val, obs_val, tgt_val = split_history_horizon(
            val_df, context, horizon, tgt_cols, time_col
        )
        ctx_test, obs_test, tgt_test = split_history_horizon(
            test_df, context, horizon, tgt_cols, time_col
        )

        self.ctx_mean = ctx_train.mean(dim=(0, 1))
        self.tgt_mean = obs_train.mean(dim=(0, 1))
        self.ctx_scale = ctx_train.std(dim=(0, 1))
        self.tgt_scale = obs_train.std(dim=(0, 1))

        # TODO: check if tensors:
        self.train_ds = TensorDataset(ctx_train, obs_train, tgt_train)
        self.val_ds = TensorDataset(ctx_val, obs_val, tgt_val)
        self.test_ds = TensorDataset(ctx_test, obs_test, tgt_test)

        self.ctx_dim = ctx_train.shape[-1]
        self.tgt_dim = tgt_train.shape[-1]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
