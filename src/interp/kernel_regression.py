import torch
from torch import Tensor, nn


class KernelRegression(nn.Module):
    """A PyTorch module for performing differentiable kernel regression.

    This class implements a kernel regression model that can be fitted with observed data
    and evaluated at specific time points. It uses a Gaussian-like kernel weighted by
    temperature and bandwidth parameters to compute smooth interpolations.

    Attributes:
        bandwidth (float): The bandwidth parameter for the kernel.
        temperature (float): The temperature parameter for the softmax weights.
        max_window (int): The maximum window size for efficient computation.
        w_arange (Tensor): A buffer tensor containing the range of indices for the window.

    Shapes:
        B -- batch size
        L -- number of observations
        M -- number of heads
        H1 -- number of features per head

    """

    def __init__(
        self,
        bandwidth: float,
        temperature: float,
        window_bandwidth_multiplier: float,
    ):
        """Initialize the KernelRegression module.

        Args:
            bandwidth (float): The bandwidth parameter for the kernel regression.
            temperature (float): The temperature parameter for the kernel regression.
            window_bandwidth_multiplier (float): Multiplier to determine the maximum window size based on the bandwidth.

        """
        super().__init__()
        self.bandwidth = bandwidth
        self.temperature = temperature

        # At edge of window, the effect is ~e^{-wbm}
        # for wbm=4 this equals 0.018
        self.max_window = int(bandwidth * window_bandwidth_multiplier)
        w_arange = torch.arange(-self.max_window // 2, self.max_window // 2)
        self.register_buffer("w_arange", w_arange)

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
        self.t_obs = t_obs.unsqueeze(1).expand_as(weights).contiguous()  # (B, M, L)
        self.x_obs = x_obs  # (B, M, L, H1)
        self.weights = weights  # (B, M, L)

    def forward(self, t_eval: Tensor):
        """Evaluate the kernel at time t_eval.

        Args:
            t_eval (B, M): time to evaluate the smoothing at.

        """
        L = self.t_obs.size(2)
        W = self.max_window

        if L > W:
            # (B, M)
            idx = torch.searchsorted(self.t_obs, t_eval.unsqueeze(-1)).squeeze(-1)

            # (B, M, L)
            idx_window = idx.unsqueeze(-1) + self.w_arange
            oob = (idx_window < 0) | (idx_window >= L)
            idx_window = torch.clamp(idx_window, 0, L - 1)

            t_win = self.t_obs.gather(dim=2, index=idx_window)
            w_win = self.weights.gather(dim=2, index=idx_window)

            idx_window_x = idx_window.unsqueeze(-1).expand(
                *idx_window.shape, self.x_obs.size(-1)
            )
            x_win = self.x_obs.gather(dim=2, index=idx_window_x)

            time_diff = t_eval.unsqueeze(-1) - t_win
            time_diff[oob] = torch.inf
        else:
            time_diff = t_eval.unsqueeze(-1) - self.t_obs.unsqueeze(1)
            x_win = self.x_obs
            w_win = self.weights

        # Weights
        logit_weights = w_win / self.temperature - time_diff.abs() / self.bandwidth
        weights = torch.softmax(logit_weights, dim=-1)

        # Softmax derivative (B, M, L)
        d_weights = weights * (1 - weights) * (-torch.sign(time_diff) / self.bandwidth)

        # Apply the weights to the observations
        dx = torch.linalg.vecdot(x_win, d_weights.unsqueeze(-1), dim=2)  # (B, M, H)
        return dx
