import torch
from torch import Tensor


class KernelRegression(torch.nn.Module):
    def __init__(self, bandwidth: float, temperature: float):
        super().__init__()
        self.bandwidth = bandwidth
        self.temperature = temperature

    def fit(self, x_obs: Tensor, t_obs: Tensor, weights: Tensor):
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
