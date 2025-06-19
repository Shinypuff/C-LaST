"""The file with Relaxed supervised losses.

This is necessary, because BCE and CE losses have incompatible target types.
"""

from torch import nn


class RelaxedBCELogitLoss(nn.BCEWithLogitsLoss):
    """The BCELogitLoss class compatible with int targets."""

    def forward(self, pred, target):
        """Calculate the loss, casting the targets to float."""
        return super().forward(pred, target.float())


class RelaxedCrossEntropyLoss(nn.CrossEntropyLoss):
    """The CrossEntropy loss class, compatible with int targets."""

    def forward(self, pred, target):
        """Calculate the cross entropy loss, clipping targets."""
        return super().forward(pred, target.long())
    