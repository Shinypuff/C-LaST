import torch
import torch.nn.functional as F
from ..layers.k2vae.k2vae import k2VAE
from torch import Tensor
from einops import rearrange

from .base import BaseForecasting


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class K2VAEForecaster(BaseForecasting):
    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        n_heads: int = 4,
        factor: int = 3,
        patch_len: int = 12,
        multistep: bool = True,
        dynamic_dim: int = 128,
        hidden_layers: int = 3,
        hidden_dim: int = 256,
        weight_beta: float = 0.01,
        sample_schedule: int = 20,
        init_kalman: str = "dynamic",
        init_koopman: str = "both",
        lr: float = 0.001,
        threshold: float = 1e2,
        **base_kwargs,
    ):
        """K2VAE: Koopman–Kalman Enhanced VAE for probabilistic time-series forecasting.

        Linearizes nonlinear dynamics in a learned latent space via KoopmanNet, then
        refines predictions and uncertainty using KalmanNet (data-driven Kalman filter).
        Provides Q(Z|X) for VAE training; decoder maps latent samples to P(Y|Z,X).

        Args:
            d_model: Base hidden size for MLP/linear blocks in KoopmanNet/KalmanNet/Decoder.
            d_ff: Feed-forward dimension for Integrator's transformer blocks.
            e_layers: Number of encoder layers in Integrator (residual-fusion for KalmanNet).
            dropout: Dropout rate across MLP/attention modules.
            activation: Activation function for MLPs ("relu", "gelu", etc.).
            n_heads: Attention heads in Integrator's transformer encoder.
            factor: Attention speedup/top-k factor (prob-sparse style).
            patch_len: Patch length (token size) for patchifying multivariate sequences.
            multistep: If True, roll Koopman operator for multiple tokens to cover horizon.
            dynamic_dim: Latent measurement/state dimension (size of Koopman/Kalman states).
            hidden_layers: MLP depth for KoopmanNet/Decoder.
            hidden_dim: MLP width for KoopmanNet/Decoder.
            weight_beta: β weight for KL term in ELBO.
            sample_schedule: Number of sampling iterations during inference for diversity.
            init_kalman: Initialization strategy for KalmanNet matrices ("identity" or "dynamic").
            init_koopman: Koopman operator mode ("static"=eDMD only, "dynamic"=learnable only, "both"=combined).
            lr: Optimizer learning rate.
            threshold: Threshold for posterior loss; if exceeded, posterior term is excluded from training loss.
            **base_kwargs: Additional arguments passed to BaseForecasting parent class.

        """
        super().__init__(**base_kwargs)

        config = ConvertedParams(base_kwargs)
        config.d_model = d_model
        config.d_ff = d_ff
        config.hidden_layers = hidden_layers
        config.dropout = dropout
        config.activation = activation
        config.e_layers = e_layers
        config.n_heads = n_heads
        config.factor = factor
        config.patch_len = patch_len
        config.multistep = multistep
        config.dynamic_dim = dynamic_dim
        config.hidden_dim = hidden_dim

        config.n_vars = self.target_dim
        config.seq_len = self.context
        config.pred_len = self.horizon
        self.weight_beta = weight_beta
        self.lr = lr
        self.threshold = threshold

        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman

        self.model = k2VAE(config)

    def sample(self, t: Tensor, x: Tensor, num_samples: int):
        """Run the forward pass of the model.

        Args:
            ctx (Tensor): Context tensor of shape (batch_size, ctx_len, ctx_dim).
            obs (Tensor): Observation tensor of shape (batch_size, obs_len, obs_dim).
            T (int): Number of timesteps to forecast.

        Returns:
            tuple: A tuple containing the mean and standard deviation tensors of shape
                (batch_size, T, tgt_dim) for the forecasted distribution.

        """
        samples = self.model.sample(x, num_samples)
        samples = rearrange(samples, "n b l c -> b l c n")
        return samples

    def kld_loss(self, dist):
        # Extract the mean and covariance matrix from the distribution
        mu_q = dist.loc  # Shape: (B, P, hidden)
        cov_q = dist.covariance_matrix  # Shape: (B, P, hidden, hidden)
        # Compute the log determinant of the covariance matrix
        log_det_q = torch.linalg.slogdet(cov_q)[1]  # Shape: (B, P)

        # Compute the trace of the covariance matrix (sum of diagonal elements)
        trace_q = torch.einsum("btii->bt", cov_q)  # Shape: (B, P)

        # Compute the squared norm of the mean for each time step
        mu_term = torch.sum(mu_q**2, dim=-1)  # Shape: (B, P)

        # Get the latent space dimension (256 in this case)
        latent_dim = mu_q.size(-1)

        # Compute the KL divergence for each time step
        kld = 0.5 * (trace_q + mu_term - latent_dim - log_det_q)  # Shape: (B, P)

        # Take the mean over the time steps, resulting in one KL value per batch sample
        kld = kld.mean(dim=-1)  # Shape: (B,)

        # Take the mean over the batch to get the final average KL divergence
        kld = kld.mean()  # Scalar

        return kld

    def training_step(self, batch, *args, **kwargs):
        _, x, y = batch
        rec_x, prior_dist, post_dist = self.model(x)
        rec_loss = F.mse_loss(rec_x, x) + F.mse_loss(post_dist.loc, y)
        post_loss = -post_dist.log_prob(y).mean()
        kld_loss = self.kld_loss(prior_dist)
        weight_alpha = 1 if post_loss < self.threshold else 0
        train_loss = rec_loss + weight_alpha * post_loss + self.weight_beta * kld_loss

        self.log(
            "train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return train_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
