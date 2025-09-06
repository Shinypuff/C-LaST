import torch
from torch import Tensor, nn


class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gru = nn.GRUCell(hidden_dim + 1, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.timenorm = nn.BatchNorm1d(1)

    def forward(self, h0: Tensor, t_pred: Tensor):
        hx = torch.zeros_like(h0)
        dt = t_pred.diff(dim=1)
        dt = self.timenorm(dt.unsqueeze(1)).squeeze(1)
        outputs = []
        for dti in dt.unbind(1):
            hx = torch.cat([hx, dti.unsqueeze(-1)], dim=-1)
            hx = self.gru(h0, hx)
            outputs.append(hx)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.head(outputs).squeeze(-1)
        return outputs
