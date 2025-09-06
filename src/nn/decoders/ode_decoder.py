import torchode as to
from torch import Tensor, nn

from ..nn.vf import GeneratorVF


class ODEDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        net = GeneratorVF(hidden_dim, output_dim)

        term = to.ODETerm(net, with_args=True)
        step_method = to.Tsit5(term=term)
        step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        self.solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, h: Tensor, t: Tensor):
        tnorm = self.norm(t.unsqueeze(1)).squeeze(1)
        ivp = to.InitialValueProblem(h, t_eval=tnorm)
        solution = self.solver.solve(ivp)
        return solution.ys
