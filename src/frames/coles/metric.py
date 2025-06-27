"""Module with functional & class metrics for MetricLearning tasks."""

from typing import Literal

import torch
import torchmetrics
from torch import Tensor
from torch.nn import functional as F


def metric_recall_topk(
    embeddings: Tensor,
    labels: Tensor,
    k: int,
    metric: Literal["cosine", "euclidean"] = "cosine",
):
    """Calculate the recall@k metric for a given set of embeddings and labels.

    Args:
    ----
        embeddings (Tensor): A tensor of embeddings.
        labels (Tensor): A tensor of labels.
        k (int): The number of top predictions to consider.
        metric (Literal["cosine", "euclidean"], optional): The distance metric to use. Defaults to "cosine".

    Returns:
    -------
        float: The recall@k metric.

    """
    if metric == "cosine":
        pdist = -1 * F.cosine_similarity(
            embeddings[None, :], embeddings[:, None], dim=-1
        )  # B x B
    elif metric == "euclidean":
        pdist = torch.cdist(embeddings, embeddings)  # B x B
    else:
        raise ValueError(f'wrong metric "{metric}"')

    indices = pdist.topk(k + 1, dim=0, largest=False).indices  # K + 1 x B
    labels_pred = labels[indices[1:]]  # K x B
    return (labels_pred == labels[None]).float().mean()


class RecallTopK(torchmetrics.MeanMetric):
    """See docs for metric_recall_topk."""

    def __init__(self, k, metric="cosine"):
        super().__init__()

        self.k = k
        self.metric = metric

    def update(self, embeddings, labels):
        new_val = metric_recall_topk(embeddings, labels, self.k, self.metric)
        super().update(new_val.to(self.device))
