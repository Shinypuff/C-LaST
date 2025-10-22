"""File with an example supervised forecasting model."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import trange

from .base import BaseForecasting
from ..layers.mlp import MLP


class DeepAR(BaseForecasting):
    """A supervised forecasting model that uses RNN-based encoder-decoder architecture.

    This class implements a forecasting model that encodes past context and observations
    using a GRU encoder, then decodes future predictions using a GRU decoder cell.
    The model outputs probabilistic forecasts with mean and standard deviation predictions.

    The encoder processes the concatenation of context and observation sequences,
    while the decoder iteratively generates future predictions conditioned on the
    encoded state and future context information.

    Attributes:
        encoder (nn.GRU): GRU encoder that processes past context and observations.
        decoder (nn.GRUCell): GRU cell decoder for generating future predictions.
        mu_mlp (MLP): Multi-layer perceptron for predicting the mean of the distribution.
        std_mlp (MLP): Multi-layer perceptron for predicting the standard deviation.

    Args:
        **base_kwargs: Keyword arguments passed to the BaseForecasting parent class.

    """

    def __init__(
        self,
        num_layers: int,
        dropout: float,
        hidden_dim: int,
        lr: float,
        **base_kwargs,
    ):
        """Initialize internal state."""
        super().__init__(**base_kwargs)

        self.rnn = nn.LSTM(
            input_size=self.target_dim + self.time_feat_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True,
        )

        self.distr_mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=self.target_dim * 2,
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr

    def encode(self, time_feats_shifted, x):
        """Encode history into a hidden state."""
        past = torch.cat([time_feats_shifted, x], dim=-1)
        out, state = self.rnn(past)
        mean, lvar = self.distr_mlp(out).chunk(2, -1)
        return mean, lvar, state

    def decode(self, time_feats, state_last, xlast):
        """Decode the hidden state into predictions & sample from them."""
        preds = []

        state = state_last
        x = xlast.unsqueeze(1)
        for i in range(self.context, self.context + self.horizon):
            inp = torch.cat([time_feats[:, None, i], x], dim=-1)
            out, state = self.rnn(inp, state)
            mean, lvar = self.distr_mlp(out).chunk(2, -1)
            scale = (0.5 * lvar).exp()
            x = torch.randn_like(mean) * scale + mean
            preds.append(x)

        preds = torch.cat(preds, dim=1)
        return preds

    def training_step(self, batch, *args, **kwargs):
        time_feats, x, y = batch

        xy = torch.cat([x, y], dim=1)
        xy_means, xy_lvars, _ = self.encode(time_feats[:, 1:], xy[:, :-1])

        loss = F.gaussian_nll_loss(xy_means, xy[:, 1:], xy_lvars.exp())

        self.log(
            "train_nll_loss",
            loss.cpu().detach().item(),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def sample(self, time_feats: Tensor, x: Tensor, num_samples: int):
        """Sample the given amount of samples from the learned distribution."""
        state = self.encode(time_feats[:, 1 : x.shape[1]], x[:, :-1])[-1]
        xlast = x[:, -1]
        trajectories = [
            self.decode(time_feats, state, xlast).to("cpu", non_blocking=True)
            for _ in trange(num_samples, desc="Sampling", leave=False)
        ]

        torch.cuda.synchronize()

        return torch.stack(trajectories, dim=-1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
