import torch
from tensordict import TensorDict
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        cat_feats: dict[str, int],
        num_feats: list[str],
        emb_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.cat_enc = nn.ModuleDict()
        self.num_enc = nn.ModuleDict()

        # Sort for deterministic hidden order!
        self.cat_feats = sorted(cat_feats.keys())
        self.num_feats = sorted(num_feats)

        for f in cat_feats:
            self.cat_enc[f] = nn.Embedding(cat_feats[f] + 1, emb_dim, padding_idx=0)

        for f in num_feats:
            self.num_enc[f] = nn.Linear(1, emb_dim)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim * (len(cat_feats) + len(num_feats)), hidden_dim),
            nn.LayerNorm(
                elementwise_affine=False, bias=False, normalized_shape=hidden_dim
            ),
        )

    def forward(self, x: TensorDict) -> TensorDict:
        cat_emb = [self.cat_enc[f](x[f]) for f in self.cat_feats]
        num_emb = [
            self.num_enc[f](x[f].unsqueeze(-1).to(torch.float32))
            for f in self.num_feats
        ]

        x = torch.cat(cat_emb + num_emb, dim=-1)
        x = self.proj(x)
        return x
