import polars as pl
import torch
from funcy import first
from tensordict import TensorDict, from_struct_array, pad_sequence
from torch.utils.data import Dataset as Dataset_


class Dataset(Dataset_):
    def __init__(self, datapath: str):
        self.data = pl.read_parquet(datapath).to_numpy(structured=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return from_struct_array(self.data[index].copy(), batch_size=())


def collate_td(batch: tuple[TensorDict], seq_feats: list[str]):
    td: TensorDict = pad_sequence(
        [item.select(*seq_feats) for item in batch],
        return_mask=True,
    )

    td["mask"] = first(td["masks"].values())
    td = td.exclude("masks")

    targets = torch.stack([item["target"] for item in batch])
    td["target"] = targets
    return td
