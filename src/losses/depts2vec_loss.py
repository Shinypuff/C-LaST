import torch
import torch.nn.functional as F


class DepTS2Vec_loss(torch.nn.Module):
    
    def __init__(self, alpha=0.5, temporal_unit=0, tau=0.1, G_type="ma", decay_type="gaussian", k=1):
        super().__init__()
        
        assert G_type in ["ma", "ar"]
        assert decay_type in ["gaussian", "laplace"]

        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.tau = tau

        self.G_type = G_type
        self.decay_type = decay_type
        self.k = k

    def instance_contrastive_loss(self, z1, z2):
        
        B, T = z1.size(0), z1.size(1)
        
        if B == 1:
            return z1.new_tensor(0.)
        
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:] # Removing similarities between same samples in the batch, like (B[1], B[1])
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z.device)
        
        # Here we choose only similarities between augmentations like (B[1], B'[1]), (B[2], B'[2]), ... (first term)
        # and averaging first term across batch and time dimension
        # Likewise for the second term where we choose another pairs like (B'[1], B[1]), (B'[2], B[2])
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss        

    def calc_G(self, T):
        if self.G_type == "ma":
            G = torch.ones((T, T))
            G = torch.tril(torch.triu(G, diagonal=1), diagonal=1) 
            return G + G.T

        if self.decay_type == "gaussian":
            G = torch.exp(-(torch.arange(T) - torch.arange(T)[:, None])**2 / self.k)
        else:
            G = torch.exp(-abs(torch.arange(T) - torch.arange(T)[:, None]) / self.k)

        G = torch.triu(G, diagonal=1)
        # G = G / G.sum(dim = 1, keepdim=True).clamp(min=1e-9)
        G = G + G.T
        return G

    def log_softmax_temporal(self, logits, epsilon=1e-5):
        device = logits.device
        dtype = logits.dtype
        M = logits.shape[1]
        # Upper triangular mask (ind_y > ind_x)
        mask = torch.triu(torch.ones(M, M, device=device, dtype=dtype), diagonal=1)

        # Compute max per instance: shape (regions, 1, 1)
        const = (logits / self.tau).amax(dim=(1, 2), keepdim=True)
        scaled = (logits / self.tau) - const  # Numerically stable

        # Compute denominators for each row
        exp_scaled_masked = torch.exp(scaled) * mask.unsqueeze(0)  # Zero out lower triangle
        denominators = exp_scaled_masked.sum(dim=2, keepdim=True)  # Sum over columns

        # Compute log_softmax values for upper triangle
        log_denominators = torch.log(denominators + epsilon)
        loss_upper = (scaled - log_denominators) * mask.unsqueeze(0)

        # Symmetrize the loss matrix
        loss_symmetric = loss_upper + loss_upper.transpose(1, 2)
        
        return loss_symmetric

    def temporal_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)

        logits = torch.cat([z1, z2], dim=0)  # 2B x T x C
        sim = logits @ logits.transpose(1, 2)
        sim = -self.log_softmax_temporal(sim)
        
        G = self.calc_G(T).to(sim.device)
        loss = torch.triu(G * sim, diagonal=1).sum(dim=2) + torch.tril(G * sim, diagonal=-1).sum(dim=1)
        return loss.mean()

    def forward(self, z_orig, z_augs):
        
        loss = torch.tensor(0., device=z_orig.device)
        d = 0
        while z_orig.size(1) > 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z_orig, z_augs)
            if d >= self.temporal_unit:
                if 1 - self.alpha != 0:
                    loss += (1 - self.alpha) * self.temporal_contrastive_loss(z_orig, z_augs)
            d += 1
            z_orig = F.max_pool1d(z_orig.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z_augs = F.max_pool1d(z_augs.transpose(1, 2), kernel_size=2).transpose(1, 2)

        if z_orig.size(1) == 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z_orig, z_augs)
            d += 1

        return loss / d
