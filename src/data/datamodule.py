import polars as pl
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

        ctx_scaler = StandardScaler().fit(ctx_train)
        tgt_scaler = StandardScaler().fit(obs_train)

        ctx_train = ctx_scaler.transform(ctx_train)
        ctx_val = ctx_scaler.transform(ctx_val)
        ctx_test = ctx_scaler.transform(ctx_test)
        obs_train = tgt_scaler.transform(obs_train)
        obs_val = tgt_scaler.transform(obs_val)
        obs_test = tgt_scaler.transform(obs_test)
        tgt_train = tgt_scaler.transform(tgt_train)
        tgt_val = tgt_scaler.transform(tgt_val)
        tgt_test = tgt_scaler.transform(tgt_test)

        self.train_ds = TensorDataset(ctx_train, obs_train, tgt_train)
        self.val_ds = TensorDataset(ctx_val, obs_val, tgt_val)
        self.test_ds = TensorDataset(ctx_test, obs_test, tgt_test)

        self.context_dim = ctx_train.shape[-1]
        self.target_dim = tgt_train.shape[-1]

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
