from torch import Tensor, nn


class BaseInterp(nn.Module):
    def fit(self, x_obs: Tensor, t_obs: Tensor, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, t_eval: Tensor):
        raise NotImplementedError()
