import torch
import torch.nn.functional as F
from torch import Tensor

from src.methods.TSDiff.utils.s4 import BackboneModel
from src.methods.base import BaseForecasting

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)

def repeat(tensor: torch.Tensor, n: int, dim: int = 0):
    return tensor.repeat_interleave(repeats=n, dim=dim)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class TSDiffCond(BaseForecasting):
    def __init__(
        self,
        hidden_dim: int,
        step_emb: int,
        timesteps: int,
        num_residual_blocks: int,
        dropout: float = 0,
        init_skip=True,
        mode="diag",
        measure="diag",
        lr=1e-3,
        num_samples=10,
        noise_observed = False,
        **base_kwargs
    ):
        print(base_kwargs)
        super().__init__(**base_kwargs)
        backbone_parameters = {
            "input_dim": self.target_dim,
            "hidden_dim": hidden_dim,
            "output_dim": self.target_dim,
            "step_emb": step_emb,
            "num_residual_blocks": num_residual_blocks,
            "residual_block": "s4",
            "mode": mode,
            'measure': measure,
        }

        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.context_leng = self.context
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.backbone = BackboneModel(
            **backbone_parameters,
            num_features=self.time_feat_dim, #dt dim
            init_skip=init_skip,
            dropout=dropout,
        )

        self.lr = lr
        self.num_samples = num_samples
        self.noise_observed = noise_observed

    def _get_obs_mask(self, x, y):
        observation_mask = torch.cat([
            torch.ones_like(x),
            torch.zeros_like(y)
        ], dim=1)

        input_seq = torch.cat([x, y], dim=1)
        
        return input_seq, observation_mask

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l2",
        reduction="none",
        time_features=None,
    ):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.backbone(x_noisy, t, features=time_features)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(
                noise, predicted_noise, reduction=reduction
            )
        else:
            raise NotImplementedError()

        return loss, x_noisy, predicted_noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index, time_features=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        predicted_noise = self.backbone(x, t, features=time_features)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def _compute_loss(self, x, t, loss_mask, time_features=None):
        noise = torch.randn_like(x)
        if not self.noise_observed:
            noise = (1 - loss_mask) * x + noise * loss_mask

        num_eval = loss_mask.sum()
        sq_err, _, _ = self.p_losses(
            x,
            t,
            loss_type="l2",
            reduction="none",
            noise=noise,
            time_features=time_features,
        )

        if self.noise_observed:
            elbo_loss = sq_err.mean()
        else:
            sq_err = sq_err * loss_mask
            elbo_loss = sq_err.sum() / (num_eval if num_eval else 1)
        return elbo_loss


    def forecast(self, obs, num_samples, T, time_features=None):
        b, _, d = obs.shape
        future = torch.zeros(b, T, d, device=obs.device)

        input_seq, observation_mask = self._get_obs_mask(obs, future)

        pred = self.get_sample(
            observation=input_seq,
            time_features=time_features,
            observation_mask=observation_mask,
            n_samples=num_samples,
        ).permute(0, 2, 3, 1) 

        return pred[:,-T:,:,:]

    @torch.no_grad()
    def get_sample(self, observation, observation_mask, n_samples, time_features=None):

        repeated_observation = repeat(observation, n_samples)
        repeated_observation_mask = repeat(observation_mask, n_samples)
        repeated_time_features = repeat(time_features, n_samples) if time_features is not None  else None
        
        batch_size, length, ch = repeated_observation.shape
        seq = torch.randn_like(repeated_observation)

        for i in reversed(range(0, self.timesteps)):
            if not self.noise_observed:
                seq = repeated_observation_mask * repeated_observation + seq * (1 - repeated_observation_mask)

            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=repeated_observation.device, dtype=torch.long),
                i,
                time_features=repeated_time_features 
            )

        seq = seq.reshape(-1, n_samples, length, ch)
        return seq
    

    def sample(self, dt: Tensor, x: Tensor, num_samples: int):
        L = x.shape[1]
        T = dt.shape[1] - L

        samples = self.forecast(x, num_samples, T, time_features=dt)

        return samples

    def training_step(self, batch):
        dt, x, y = batch

        input_seq, observation_mask = self._get_obs_mask(x, y)
        loss_mask = 1 - observation_mask

        t = torch.randint(
            0, self.timesteps, [input_seq.shape[0]], device=input_seq.device
        ).long()
        
        loss = self._compute_loss(input_seq, t, loss_mask, time_features=dt)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)