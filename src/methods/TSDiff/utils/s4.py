import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange

from .model_components import LinearActivation, Activation, DropoutNd, SinusoidalPositionEmbeddings
from .kernels.kernel_utils import contract
from .kernels import SSKernel
from .logging import log

def Conv1dKaiming(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class S4(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=None,
        channels=1,
        mode="nplr",
        measure="legs",
        bidirectional=False,
        # Arguments for position-wise feedforward components
        activation="gelu",
        postact="glu",
        hyper_act=None,
        dropout=0.0,
        tie_dropout=False,
        bottleneck=None,
        gate=None,
        transposed=True,
        verbose=False,
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF
        hyper_act: use a "hypernetwork" multiplication (experimental)
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        gate: add gated activation (GSS)
        bottleneck: reduce SSM dimension (GSS)

        See the class SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            log.info(
                f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})"
            )

        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.H = self.H // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.H,
                transposed=self.transposed,
                activation=activation,
                activate=True,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=self.transposed,
                activation=activation,
                activate=True,
            )
            self.output_gate = LinearActivation(
                self.d_model * gate,
                self.d_model,
                transposed=self.transposed,
                activation=None,
                activate=False,
            )

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = SSKernel(
            self.H,
            N=self.N,
            L=self.L,
            channels=channels,
            verbose=verbose,
            mode=mode,
            measure=measure,
            **kernel_args,
        )

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.H * self.channels,
            self.d_model * (1 if self.gate is None else self.gate),
            transposed=self.transposed,
            activation=postact,
            activate=True,
        )

    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs):
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Mask out padding tokens
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(
                    lengths, dtype=torch.long, device=u.device
                )
            else:
                lengths = None
        if lengths is not None:
            assert (
                isinstance(lengths, torch.Tensor)
                and lengths.ndim == 1
                and lengths.size(0) in [1, u.size(0)]
            )
            mask = torch.where(
                torch.arange(L, device=lengths.device)
                < lengths[:, None, None],
                1.0,
                0.0,
            )
            u = u * mask

        if self.gate is not None:
            v = self.input_gate(u)
        if self.bottleneck is not None:
            u = self.input_linear(u)

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(
            L=L_kernel, rate=rate, state=state
        )  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=L_kernel + L)  # (C H L)
        u_f = torch.fft.rfft(u, n=L_kernel + L)  # (B H L)
        y_f = contract("bhl,chl->bchl", u_f, k_f)
        y = torch.fft.irfft(y_f, n=L_kernel + L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract("bhl,ch->bchl", u, self.D)

        # Compute state update
        if state is not None:
            assert (
                not self.bidirectional
            ), "Bidirectional not supported with state forwarding"
            y = y + k_state  #
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, "b (s c) h l -> s b c h l", s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, "... c h l -> ... (c h) l")

        y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        y = self.output_linear(y)

        if self.gate is not None:
            y = self.output_gate(y * v)

        return y, next_state

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, "b c h -> b (c h)")
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model


class S4Layer(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        mode="nplr",
        l_max=None,
        measure="legs"
    ):
        super().__init__()
        self.layer = S4(
            d_model=d_model,
            d_state=128,
            bidirectional=True,
            dropout=dropout,
            transposed=True,
            postact=None,
            mode=mode,
            l_max=l_max,
            measure=measure,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = (
            nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer: we ignore the state input and output for training
        z, _ = self.layer(z)
        # Dropout on the output of the layer
        z = self.dropout(z)
        # Residual connection
        x = z + x
        return x, None

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)
        # Residual connection
        x = z + x
        return x, state


class S4Block(nn.Module):
    def __init__(self, d_model, dropout=0.0, expand=2, num_features=0,mode="nplr",l_max=None,measure="legs"):
        super().__init__()
        self.s4block = S4Layer(d_model, dropout=dropout,mode=mode,l_max=l_max,measure=measure)

        self.time_linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.out_linear1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.out_linear2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.feature_encoder = nn.Conv1d(num_features, d_model, kernel_size=1)

    def forward(self, x, t, features=None):
        t = self.time_linear(t)[:, None, :].repeat(1, x.shape[2], 1)
        t = t.transpose(-1, -2)
        out, _ = self.s4block(x + t)
        if features is not None:
            out = out + self.feature_encoder(features)
        out = self.tanh(out) * self.sigm(out)
        out1 = self.out_linear1(out)
        out2 = self.out_linear2(out)
        return out1 + x, out2


class BackboneModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        step_emb,
        num_residual_blocks,
        num_features,
        residual_block="s4",
        mode="nplr",
        measure="legs",
        l_max=None,
        dropout=0.0,
        init_skip=True,
    ):
        super().__init__()
        if residual_block == "s4":
            residual_block = S4Block
        else:
            raise ValueError(f"Unknown residual block {residual_block}")
        self.input_init = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.time_init = nn.Sequential(
            nn.Linear(step_emb, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        residual_blocks = []
        for i in range(num_residual_blocks):
            residual_blocks.append(
                residual_block(
                    hidden_dim, 
                    num_features=num_features, 
                    dropout=dropout, 
                    mode=mode,l_max=l_max,
                    measure=measure,
                )
            )
        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.step_embedding = SinusoidalPositionEmbeddings(step_emb)
        self.init_skip = init_skip

    def forward(self, input, t, features=None):
        x = self.input_init(input)  # B, L ,C
        t = self.time_init(self.step_embedding(t))
        x = x.transpose(-1, -2)
        if features is not None:
            features = features.transpose(-1, -2)
        skips = []
        for layer in self.residual_blocks:
            x, skip = layer(x, t, features)
            skips.append(skip)

        skip = torch.stack(skips).sum(0)
        skip = skip.transpose(-1, -2)
        out = self.out_linear(skip)
        if self.init_skip:
            out = out + input
        return out