import torch
from torchmetrics import Metric


class NMAE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "absolute_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("y_true", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.absolute_error += torch.sum(torch.abs(preds - target))
        self.y_true += torch.sum(target)

    def compute(self):
        return self.absolute_error / self.y_true
