from torch import nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings

from ..utils.mask_utils import masklast


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.ffd_norm = nn.RMSNorm(hidden_dim)
        ffd = hidden_dim * 8 // 3
        self.w1 = nn.Linear(hidden_dim, ffd)
        self.w2 = nn.Linear(hidden_dim, ffd)
        self.w3 = nn.Linear(ffd, hidden_dim)

    def forward(self, x, mask):
        """Pass x through the layer.

        Args:
        ----
            x (torch.Tensor): Input tensor, (B, L, H).
            mask (torch.Tensor): Mask tensor (B, L).

        """
        x = self.attn_norm(x)
        x = self.attn(x, x, x, key_padding_mask=~mask)[0] + x
        x = self.ffd_norm(x)
        x = self.w3(F.silu(self.w1(x)) * self.w2(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.headdim = hidden_dim // num_heads
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.embedding = RotaryPositionalEmbeddings(
            self.headdim, max_seq_len=max_seq_len
        )

        self.layers = nn.ModuleList(
            [TransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, time, mask):
        x = self.proj(x)

        B, L, H = x.shape
        x = x.reshape(B, L, self.num_heads, self.headdim)
        x = self.embedding(x)
        x = x.reshape(B, L, H)

        for layer in self.layers:
            x = layer(x, mask)

        return masklast(x, mask, dim=1)
