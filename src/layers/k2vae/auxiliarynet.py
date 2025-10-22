# The implementation of K2VAE building blocks from https://github.com/decisionintelligence/K2VAE.

from einops import rearrange
import torch
from torch import nn
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Transformer_EncDec import Encoder, EncoderLayer


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=False,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )
        # Adaption Head
        self.adaptation = nn.Linear(config.dynamic_dim, config.d_model)

        # Prediction Head
        self.head = nn.Linear(
            (self.seq_len // config.patch_len) * config.d_model,
            (self.pred_len // config.patch_len) * config.dynamic_dim,
        )

        self.config = config

    def forward(self, x_enc):
        # Encoder
        # z: [B L C]
        x_enc = self.adaptation(x_enc)
        enc_out, attns = self.encoder(x_enc)

        # Decoder
        enc_out = rearrange(enc_out, "b l c -> b (l c)")
        dec_out = self.head(enc_out)  # z: [B (C P)]
        dec_out = rearrange(dec_out, "b (p c) -> b p c", c=self.config.dynamic_dim)

        return dec_out
