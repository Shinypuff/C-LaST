import math
import torch
from torch import nn
from torch.autograd import Variable

from .backbones import FeedNet, Transformer
from .units import *

class VarUnit(nn.Module):
    def __init__(self, in_dim, z_dim, backbone=None, backbone_args=None, VampPrior=False, pseudo_dim=201, contrast=False):
        super(VarUnit, self).__init__()

        self.in_dim = in_dim
        self.z_dim = z_dim
        self.prior = VampPrior

        backbone_args = backbone_args or {}

        if backbone == "feednet":
            self.loc_net = FeedNet(in_dim, z_dim, type="mlp", **backbone_args)
            self.var_net = nn.Sequential(
                FeedNet(in_dim, z_dim, type="mlp", **backbone_args),
                nn.Softplus()
            )
        elif backbone == "transformer":
            self.loc_net = Transformer(in_dim, z_dim, **backbone_args)
            self.var_net = nn.Sequential(
                Transformer(in_dim, z_dim, **backbone_args),
                nn.Softplus()
            )
        else:
            raise ValueError(f"Unknown backbone {backbone}")

        if self.prior:
            self.pseudo_dim = pseudo_dim
            self.pseudo_mean = 0
            self.pseudo_std = 0.01
            self.add_pseudoinputs()

        if not contrast:
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
    def __init__(self, in_dim, out_dim, seq_len, pred_len, inner_s, dropout=0.1, backbone=None, backbone_args=None, contrast=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_s = inner_s
        """ VAE Net """
        self.VarUnit_s = VarUnit(in_dim, inner_s, backbone, backbone_args, contrast=contrast)
        self.RecUnit_s = FeedNet(inner_s, in_dim, type="mlp", n_layers=1, dropout=dropout)
        """ Fourier """
        self.FourierNet = NeuralFourierLayer(inner_s, out_dim, seq_len, pred_len)
        self.pred = FeedNet(self.inner_s, self.out_dim, type="mlp", n_layers=1)

        self.contrast = contrast

    def forward(self, x_his):
        qz_s, xs_rec, elbo_s = self.get_emb(x_his)

        """ Fourier """
        xs_pred = self.pred(self.FourierNet(qz_s)[:, -self.pred_len:])

        if not self.contrast:
            mlbo_s = self.VarUnit_s.compute_MLBO(x_his, qz_s)
            return xs_pred, xs_rec, elbo_s, mlbo_s

        return qz_s, xs_pred, xs_rec, elbo_s

    
    def get_emb(self, x_his):
        qz_s, mean_qz_s, var_qz_s = self.VarUnit_s(x_his)
        xs_rec = self.RecUnit_s(qz_s)
        elbo_s = period_sim(xs_rec, x_his) - self.VarUnit_s.compute_KL(
            qz_s, mean_qz_s, var_qz_s)
        return qz_s, xs_rec, elbo_s


class TNet(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, inner_t, dropout=0.1, backbone=None, backbone_args=None, contrast=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_t = inner_t

        self.VarUnit_t = VarUnit(in_dim, inner_t, backbone, backbone_args, contrast=contrast)
        self.RecUnit_t = FeedNet(inner_t, in_dim, type="mlp", n_layers=1, dropout=dropout)

        self.t_pred_1 = FeedNet(self.seq_len, self.pred_len, type="mlp", n_layers=1)
        self.t_pred_2 = FeedNet(self.inner_t, self.out_dim, type="mlp", n_layers=1)

        self.contrast = contrast

    def forward(self, x_his):
        qz_t, xt_rec, elbo_t = self.get_emb(x_his)

        # mlp
        if len(x_his.shape) == 3:
            xt_pred = self.t_pred_2(self.t_pred_1(qz_t.permute(0, 2, 1)).permute(0, 2, 1))
        else:
            xt_pred = self.t_pred_2(self.t_pred_1(qz_t.permute(0, 3, 2, 1)).permute(0, 3, 2, 1))

        if not self.contrast:
            mlbo_t = self.VarUnit_t.compute_MLBO(x_his, qz_t)
            return xt_pred, xt_rec, elbo_t, mlbo_t

        return qz_t, xt_pred, xt_rec, elbo_t
    
    def get_emb(self, x_his):
        qz_t, mean_qz_t, var_qz_t = self.VarUnit_t(x_his)
        xt_rec = self.RecUnit_t(qz_t)
        elbo_t = trend_sim(xt_rec, x_his) - self.VarUnit_t.compute_KL(qz_t, mean_qz_t, var_qz_t)
        return qz_t, xt_rec, elbo_t


class LaSTBlock(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, s_func, inner_s, t_func, inner_t, dropout=0.1, backbone=None, backbone_args=None, contrast=False):
        super().__init__()
        self.input_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.SNet = s_func(in_dim, out_dim, seq_len, pred_len, inner_s, dropout=dropout, backbone=backbone, backbone_args=backbone_args, contrast=contrast)
        self.TNet = t_func(in_dim, out_dim, seq_len, pred_len, inner_t, dropout=dropout, backbone=backbone, backbone_args=backbone_args, contrast=contrast)
        self.contrast = contrast

        if not contrast:
            self.MuboNet = CalculateMubo(inner_s, inner_t, dropout=dropout)

    def forward(self, x_his):

        if not self.contrast:
            x_s, xs_rec, elbo_s, mlbo_s = self.SNet(x_his)
            x_t, xt_rec, elbo_t, mlbo_t = self.TNet(x_his)

            mlbo = mlbo_t + mlbo_s
            mubo = self.MuboNet(x_his, self.SNet.VarUnit_s, self.TNet.VarUnit_t)

        else:
            z_s, x_s, xs_rec, elbo_s = self.SNet(x_his)
            z_t, x_t, xt_rec, elbo_t = self.TNet(x_his)

        rec_err = ((xs_rec + xt_rec - x_his) ** 2).mean()
        elbo = elbo_t + elbo_s - rec_err

        if not self.contrast:
            return x_s, x_t, elbo, mlbo, mubo
        else:
            return z_s, z_t, x_s, x_t, elbo
        

    def get_emb(self, x_his):
        z_s, xs_rec, elbo_s = self.SNet.get_emb(x_his)
        z_t, xt_rec, elbo_t = self.TNet.get_emb(x_his)

        rec_err = ((xs_rec + xt_rec - x_his) ** 2).mean()
        elbo = elbo_t + elbo_s - rec_err

        return z_s, z_t, elbo
