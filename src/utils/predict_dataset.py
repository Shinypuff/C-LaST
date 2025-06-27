from copy import deepcopy

import polars as pl
from tensordict import from_struct_array
from torch.utils.data import Dataset


class PredictDataset(Dataset):
    def __init__(self, datasource: str):
        self.data = pl.read_parquet(datasource).to_numpy(structured=True)

    def __getitem__(self, index):
        return from_struct_array(deepcopy(self.data[index]), batch_size=())

    @property
    def labels(self):
        return self.data["target"]

    def __len__(self):
        return len(self.data)
