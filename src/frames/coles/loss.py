import torch
from torch import Tensor
from torch.nn import functional as F


def calc_contrastive_loss(
    h: Tensor, tags: Tensor, neg_samples: int, margin: float
) -> Tensor:
    h = h / torch.linalg.norm(h, dim=-1, keepdim=True)

    # B, B
    sims = torch.linalg.vecdot(h[:, None], h[None, :], dim=-1)
    labels = tags[:, None] == tags[None, :]

    sims = torch.triu(sims, diagonal=1)
    dists = 2 * (1 - sims)
    labels = torch.triu(labels, diagonal=1)

    pos_dists = dists[labels]
    pos_loss = torch.mean(pos_dists)

    neg_dists = dists[~labels]

    # Get the smallest negative distances
    hard_neg_dists = torch.topk(neg_dists, k=neg_samples, dim=0, largest=False)[0]
    neg_loss = F.relu(margin - hard_neg_dists).mean()
    return pos_loss + neg_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, neg_samples: int, margin: float):
        super().__init__()
        self.neg_samples = neg_samples
        self.margin = margin

    def forward(self, h: Tensor, tags: Tensor) -> Tensor:
        return calc_contrastive_loss(h, tags, self.neg_samples, self.margin)
