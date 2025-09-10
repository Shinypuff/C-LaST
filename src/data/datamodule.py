"""File with all data-related tools."""

import polars as pl
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .utils import split_history_horizon


class DataModule(LightningDataModule):
    """A PyTorch Lightning DataModule for handling time series data.

    This class encapsulates the data loading and preprocessing logic for time series datasets.
    It performs a 60/20/20 train/validation/test split, preprocesses the data into context, observation, and target tensors,
    and provides DataLoaders for each split.

    Attributes:
        datasource (str): Path to the CSV file containing the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        context (int): Number of time steps to use as context.
        horizon (int): Number of future time steps to predict.
        ctx_mean (torch.Tensor): Mean of the context data.
        tgt_mean (torch.Tensor): Mean of the target data.
        ctx_scale (torch.Tensor): Standard deviation of the context data.
        tgt_scale (torch.Tensor): Standard deviation of the target data.
        train_ds (TensorDataset): Training dataset.
        val_ds (TensorDataset): Validation dataset.
        test_ds (TensorDataset): Test dataset.
        ctx_dim (int): Dimension of the context data.
        tgt_dim (int): Dimension of the target data.

    Args:
        datasource (str): Path to the CSV file containing the dataset.
        context (int): Number of time steps to use as context.
        horizon (int): Number of future time steps to predict.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 16.
        tgt_cols (list[str] | None, optional): List of target column names. If None, all columns except time_col are used. Defaults to None.
        time_col (str | None, optional): Name of the time column. If None, the index is used. Defaults to None.

    """

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
        """Initialize the DataModule with the provided parameters and prepare the dataset.

        This method loads the data from a CSV file, performs a 6:2:2 train/validation/test split,
        and processes the data into context, observation, and target tensors for each split.
        It also computes the mean and standard deviation for normalization and stores the datasets
        as TensorDatasets. The dimensions of the context and target tensors are also stored.

        Args:
            datasource (str): Path to the CSV file containing the data.
            context (int): The number of time steps to use as context for the model.
            horizon (int): The number of time steps to predict into the future.
            batch_size (int, optional): The batch size for data loading. Defaults to 32.
            num_workers (int, optional): The number of workers for data loading. Defaults to 16.
            tgt_cols (list[str] | None, optional): List of target column names. If None, all columns are used. Defaults to None.
            time_col (str | None, optional): The name of the time column. If None, the row number is used. Defaults to None.

        """
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
        """Return a DataLoader for the training dataset.

        This method creates and returns a DataLoader instance configured with the training dataset,
        batch size, shuffle setting, and number of worker processes.

        Returns:
            DataLoader: A DataLoader instance for the training dataset.

        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Return a DataLoader for the validation dataset.

        This method creates and returns a DataLoader instance configured for the validation dataset.
        The DataLoader is set up with the specified batch size, no shuffling, and the specified number of worker processes.

        Returns:
            DataLoader: A DataLoader instance for the validation dataset.

        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Create and return a DataLoader for the test dataset.

        This method initializes a DataLoader instance using the test dataset (`self.test_ds`) with the specified
        batch size, shuffle setting (set to False for testing), and number of worker threads.

        Returns:
            DataLoader: A DataLoader instance for the test dataset.

        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
