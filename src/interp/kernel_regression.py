import torch
from torch import Tensor

from .base import BaseInterp


class KernelRegression(BaseInterp):
    def __init__(self, bandwidth: float, temperature: float):
        super().__init__()
        self.bandwidth = bandwidth
        self.temperature = temperature

    def fit(self, t_obs: Tensor, x_obs: Tensor, weights: Tensor):
        """Fit the kernel regression model with observed data.

        Args:
            t_obs: Observation time points tensor with shape (B, L).
            x_obs: Observation values tensor which is transposed to shape (B, M, L, H1).
            weights: Weights tensor for the observations with shape (B, M, L).

        Shapes:
            B -- batch size
            L -- number of observations
            M -- number of heads
            H1 -- number of features per head

        Note:
            The input tensors are stored as instance variables for later use in the model.

        """
        self.t_obs = t_obs  # (B, L)
        self.x_obs_t = x_obs.transpose(2, 3)  # (B, M, H, L)
        self.weights = weights  # (B, M, L)

    def forward(self, t_eval: Tensor):
        # Time differences (B, M, L)
        time_diff = t_eval.unsqueeze(-1) - self.t_obs.unsqueeze(1)

        # Weights
        logit_weights = (
            self.weights / self.temperature - time_diff.abs() / self.bandwidth
        )
        weights = torch.softmax(logit_weights, dim=-1)

        # Softmax derivative (B, M, L)
        d_weights = weights * (1 - weights) * (-torch.sign(time_diff) / self.bandwidth)

        # Apply the weights to the observations
        dx = torch.linalg.vecdot(self.x_obs_t, d_weights.unsqueeze(2))  # (B, M, H)
        return dx
