import torch
from torch import Tensor


class KernelRegression(torch.nn.Module):
    def __init__(
        self, past_window: int, future_window: int, bandwidth: float, temperature: float
    ):
        super().__init__()
        self.pw = past_window
        self.fw = future_window
        self.bandwidth = bandwidth
        self.temperature = temperature

    def fit(self, x_obs: Tensor, t_obs: Tensor, weights: Tensor):
        self.t_obs = t_obs  # (B, L)
        self.x_obs = x_obs  # (B, L, M, H)
        self.weights = weights  # (B, L, M)
        B = x_obs.shape[0]
        d = x_obs.device
        self.B_arange = torch.arange(B, device=d)
        self.w_arange = torch.arange(-self.pw, self.fw + 1, device=d)

    def forward(self, t_eval: Tensor):
        L = self.t_obs.size(1)
        closest_idx = torch.searchsorted(self.t_obs, t_eval.unsqueeze(-1)).squeeze(
            -1
        )  # (B,)

        # Past and future indices
        obs_idx = closest_idx[:, None] + self.w_arange[None, :]  # (B, W)

        # Handle out-of-bounds
        oob = (obs_idx < 0) | (obs_idx >= L)
        obs_idx = torch.clamp(obs_idx, 0, L - 1)

        t_obs = self.t_obs[self.B_arange[:, None], obs_idx]
        # Time differences
        time_diff = (t_eval[:, None] - t_obs).unsqueeze(-1)  # (B, W, 1)

        # Weights
        attention_weights = self.weights[self.B_arange[:, None], obs_idx]  # (B, W, M)
        logit_weights = (
            attention_weights / self.temperature - time_diff.abs() / self.bandwidth
        )
        logit_weights[oob] = -torch.inf
        weights = torch.softmax(logit_weights, dim=1)

        # Softmax derivative
        d_weights = weights * (1 - weights) * (-torch.sign(time_diff) / self.bandwidth)

        # Observations
        x_obs = self.x_obs[self.B_arange[:, None], obs_idx]  # (B, W, M, H)

        # Apply the weights to the observations
        dx = torch.einsum("bwmh,bwm->bmh", x_obs, d_weights)  # (B, W, M, H))
        return dx
