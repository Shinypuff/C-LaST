import torchode as to
from torch import Tensor, nn

from .mlpvf import MLPVF


class NeuralODE(nn.Module):
    def __init__(self, hidden_dim: int, timescale: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.timescale = timescale

        f = to.ODETerm(MLPVF(hidden_dim))

        self.solver = to.AutoDiffAdjoint(
            to.Dopri5(term=f),
            to.IntegralController(atol=1e-3, rtol=1e-3, term=f),
        )

    def forward(self, h: Tensor, t: Tensor):
        T = t.shape[1]
        t = (t - t[:, 0, None]) * (self.timescale / T)
        ivp = to.InitialValueProblem(h, t_start=t[:, 0], t_end=t[:, -1], t_eval=t)

        solution = self.solver.solve(ivp)
        return solution.ys
