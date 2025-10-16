import torch.nn as nn
import torch
import math

class FeedNet(nn.Module):
    def __init__(self, in_dim, out_dim, type="mlp", n_layers=1, inner_dim=None, activaion=None, dropout=0.1, **kwargs):
        super(FeedNet, self).__init__()
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in = in_dim if i == 0 else inner_dim[i - 1]
            layer_out = out_dim if i == n_layers - 1 else inner_dim[i]
            if type == "mlp":
                self.layers.append(nn.Linear(layer_in, layer_out))
            else:
                raise Exception("KeyError: Feedward Net keyword error. Please use word in ['mlp']")
            if i != n_layers - 1 and activaion is not None:
                self.layers.append(activaion)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, out_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(out_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(out_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, out_dim),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + self.dropout(attn_out))
        
        ff_out = self.ff(x)
        x = self.ff_norm(x + self.dropout(ff_out))
        
        return x

class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, d_ff=128, n_layers=1, n_heads=1, dropout=0.1, **kwargs):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, out_dim)
        
        self.pos_encoding = PositionalEncoding(out_dim)
        
        self.layers = nn.Sequential(*[
            TransformerBlock(
                out_dim,
                out_dim, 
                n_heads, 
                d_ff, 
                dropout
            )
            for i in range(n_layers)
        ])

    def forward(self, x):
        # x: [batch_size, seq_len, in_dim]
        x = self.input_proj(x)
        
        # позиционное кодирование
        x = self.pos_encoding(x)
        
        # трансформерные слои
        x = self.layers(x)
        return x
