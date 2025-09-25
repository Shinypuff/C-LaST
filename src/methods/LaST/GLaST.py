import math
import torch
from torch import nn
from torch.autograd import Variable

from ...losses.nll import gaussian_nll
from ...layers.mlp import MLP
from ..base import BaseForecasting
from .units import *


class FeedNet(nn.Module):
    def __init__(self, in_dim, out_dim, type="mlp", n_layers=1, inner_dim=None, activaion=None, dropout=0.1):
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


class VarUnit(nn.Module):
    def __init__(self, in_dim, z_dim, VampPrior=False, pseudo_dim=201, device="cuda:0"):
        super(VarUnit, self).__init__()

        self.in_dim = in_dim
        self.z_dim = z_dim
        self.prior = VampPrior
        # self.device = device

        self.loc_net = FeedNet(in_dim, z_dim, type="mlp", n_layers=1)
        self.var_net = nn.Sequential(
            FeedNet(in_dim, z_dim, type="mlp", n_layers=1),
            nn.Softplus()
        )

        if self.prior:
            self.pseudo_dim = pseudo_dim
            self.pseudo_mean = 0
            self.pseudo_std = 0.01
            self.add_pseudoinputs()

        self.critic_xz = CriticFunc(z_dim, in_dim)

    def add_pseudoinputs(self):
        self.idle_input = Variable(torch.eye(self.pseudo_dim, self.pseudo_dim, dtype=torch.float64, device=self.device),
                                   requires_grad=False).cuda()

        nonlinearity = nn.ReLU()
        self.means = NonLinear(self.pseudo_dim, self.in_dim, bias=False, activation=nonlinearity)
        self.normal_init(self.means.linear, self.pseudo_mean, self.pseudo_std)

    def normal_init(self, m, mean=0., std=0.01):
        m.weight.data.normal_(mean, std)

    def log_p_z(self, z):
        if self.prior:
            C = self.pseudo_dim
            X = self.means(self.idle_input).unsqueeze(dim=0)

            z_p_mean = self.loc_net(X)
            z_p_var = self.var_net(X)

            # expand z
            z_expand = z.unsqueeze(1)
            means = z_p_mean.unsqueeze(0)
            vars = z_p_var.unsqueeze(0)

            if len(z.shape) > 3:
                means = means.unsqueeze(-2).repeat(1, 1, 1, z.shape[-2], 1)
                vars = vars.unsqueeze(-2).repeat(1, 1, 1, z.shape[-2], 1)

            a = log_Normal_diag(z_expand, means, vars, dim=2) - math.log(C)
            a_max, _ = torch.max(a, 1)
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        else:
            log_prior = log_Normal_standard(z, dim=1)
        return log_prior

    def compute_KL(self, z_q, z_q_mean, z_q_var):
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_var, dim=1)
        KL = -(log_p_z - log_q_z)

        return KL.mean()

    def compute_MLBO(self, x, z_q, method="our"):
        idx = torch.randperm(z_q.shape[0])
        z_q_shuffle = z_q[idx].view(z_q.size())
        if method == "MINE":
            mlbo = self.critic_xz(x, z_q).mean() - torch.log(
                torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean(dim=-1)).mean()
        else:
            point = 1 / torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean()
            point = point.detach()

            if len(x.shape) == 3:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(
                    self.critic_xz(x, z_q_shuffle))  # + 1 + torch.log(point)
            else:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(self.critic_xz(x, z_q_shuffle))

        return mlbo.mean()

    def forward(self, x, return_para=True):
        mean, var = self.loc_net(x), self.var_net(x)
        var += 1e-6
        qz_gaussian = torch.distributions.Normal(loc=mean, scale=var)
        qz = qz_gaussian.rsample()  # mu+sigma*epsilon
        return (qz, mean, var) if return_para else qz


class CalculateMubo(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super().__init__()
        self.critic_st = CriticFunc(x_dim, y_dim, dropout)

    def forward(self, x_his, var_net_t, var_net_s):
        zs, zt = var_net_s(x_his, return_para=False), var_net_t(x_his, return_para=False)
        idx = torch.randperm(zt.shape[0])
        zt_shuffle = zt[idx].view(zt.size())
        f_st = self.critic_st(zs, zt)
        f_s_t = self.critic_st(zs, zt_shuffle)

        mubo = f_st - f_s_t
        pos_mask = torch.zeros_like(f_st)
        pos_mask[mubo < 0] = 1
        mubo_musk = mubo * pos_mask
        reg = (mubo_musk ** 2).mean()

        return mubo.mean() + reg


class SNet(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, inner_s, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_s = inner_s
        
        """ VAE Net """
        self.VarUnit_s = VarUnit(in_dim, inner_s)
        self.rec_s = nn.Linear(inner_s, in_dim)
        self.RecUnit_s = FeedNet(inner_s, in_dim, type="mlp", n_layers=1, dropout=dropout)

    def forward(self, x_his):
        return self.VarUnit_s(x_his)

    def get_losses(self, x_his):
        qz_s, mean_qz_s, var_qz_s = self(x_his)
        xs_rec = self.RecUnit_s(qz_s)
        
        elbo_s = period_sim(xs_rec, x_his) - self.VarUnit_s.compute_KL(
            qz_s, mean_qz_s, var_qz_s)
        
        mlbo_s = self.VarUnit_s.compute_MLBO(x_his, qz_s)

        return qz_s, xs_rec, elbo_s, mlbo_s


class TNet(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, inner_t, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_t = inner_t

        self.VarUnit_t = VarUnit(in_dim, inner_t)
        self.RecUnit_t = FeedNet(inner_t, in_dim, type="mlp", n_layers=1)

    def forward(self, x_his):
        return self.VarUnit_t(x_his)
    
    def get_losses(self, x_his):
        qz_t, mean_qz_t, var_qz_t = self(x_his)
        xt_rec = self.RecUnit_t(qz_t)
        
        elbo_t = trend_sim(xt_rec, x_his) - self.VarUnit_t.compute_KL(qz_t, mean_qz_t, var_qz_t)
        
        mlbo_t = self.VarUnit_t.compute_MLBO(x_his, qz_t)

        return qz_t, xt_rec, elbo_t, mlbo_t


class LaSTBlock(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, s_func, inner_s, t_func, inner_t, dropout=0.1):
        super().__init__()
        self.input_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.SNet = s_func(in_dim, out_dim, seq_len, pred_len, inner_s, dropout=dropout)
        self.TNet = t_func(in_dim, out_dim, seq_len, pred_len, inner_t, dropout=dropout)
        self.MuboNet = CalculateMubo(inner_s, inner_t, dropout=dropout)

    def forward(self, x_his):
        z_s, _, _ = self.SNet(x_his)
        z_t, _, _ = self.TNet(x_his)
        return z_s, z_t 
    
    def get_losses(self, x_his):
        z_s, xs_rec, elbo_s, mlbo_s = self.SNet.get_losses(x_his)
        z_t, xt_rec, elbo_t, mlbo_t = self.TNet.get_losses(x_his)

        rec_err = ((xs_rec + xt_rec - x_his) ** 2).mean()

        elbo = elbo_t + elbo_s - rec_err
        mlbo = mlbo_t + mlbo_s
        mubo = self.MuboNet(x_his, self.SNet.VarUnit_s, self.TNet.VarUnit_t)

        return z_s, z_t, elbo, mlbo, mubo


class GLaSTForecaster(BaseForecasting):
    def __init__(self, input_len, output_len, input_dim, out_dim, var_num=1, hidden_dim=64, dropout=0.1, lr=1e-3, **base_kwargs):
        super().__init__(**base_kwargs)

        self.in_dim = input_dim # self.tgt_dim + self.ctx_dim
        self.out_dim = out_dim # tgt_dim
        self.seq_len = input_len # L
        self.pred_len = output_len # T

        self.v_num = var_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr

        self.LaSTLayer = LaSTBlock(
            self.in_dim, self.out_dim, 
            input_len, 
            output_len, 
            SNet, self.hidden_dim, 
            TNet, self.hidden_dim, 
            dropout=dropout
            )

        self.fusion_net = MLP(self.seq_len, hidden_dim, self.pred_len)

        self.mean_net = MLP(hidden_dim * 2, hidden_dim, self.tgt_dim)
        self.scale_net = nn.Sequential(
            MLP(hidden_dim * 2, hidden_dim, self.tgt_dim),
            nn.Softplus()
        )

    def forward(self, ctx, obs, T=None):
        b, t, _ = obs.shape
        ctx, obs = self.scale(ctx, obs)

        ctx = ctx[:, :self.seq_len, :]
        x_his = torch.cat([obs, ctx], dim=-1)

        z_s, z_t = self.LaSTLayer(x_his) # Z: B x T x hid_dim

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1) # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(0, 2, 1) # Z: B x 2*hid_dim x pred_len --> B x pred_len x2*hid_dim

        mean = self.mean_net(z)
        scale = self.scale_net(z)

        return mean, scale

    def training_step(self, batch, *args, **kwargs):
        ctx, obs, tgt = batch
        ctx, obs, tgt = self.scale(ctx, obs, tgt)
        
        ctx = ctx[:, :self.seq_len, :]
        x_his = torch.cat([obs, ctx], dim=-1)

        z_s, z_t, elbo, mlbo, mubo = self.LaSTLayer.get_losses(x_his) # Z: B x T x hid_dim

        z = torch.cat([z_s, z_t], dim=-1).permute(0, 2, 1) # Z: B x 2*hid_dim x T
        z = self.fusion_net(z).permute(0, 2, 1) # Z: B x 2*hid_dim x pred_len --> B x pred_len x2*hid_dim

        mean = self.mean_net(z)
        scale = self.scale_net(z)
        
        loss = gaussian_nll(tgt, mean, scale) - elbo - mlbo + mubo

        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)