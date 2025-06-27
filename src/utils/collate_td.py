import torch
from funcy import first
from tensordict import TensorDict, pad_sequence


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
