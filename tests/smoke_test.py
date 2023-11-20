import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path[0] = sys.path[0][:-6]  # Adding parent directory to path for imports below

from ProbabilisticDiffusion import Diffusion
from ProbabilisticDiffusion.utils import generate_circular


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = nn.Linear(128, 2)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)


# Test Data
n = 100
n_steps = 50
x_orig, y_orig = generate_circular(n, 2)

x = x_orig + np.random.normal(loc=0.0, scale=0.3, size=n)
y = y_orig + np.random.normal(loc=0.0, scale=0.3, size=n)

data = torch.stack([torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)], dim=1)


def test_defining_model():
    model = ConditionalModel(n_steps)
    loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diffusion_model = Diffusion(data, n_steps,
                          1e-5, 1e-2, 'linear',
                                model, loss, optimizer)
    return diffusion_model


def test_forward_sample():
    diffusion = test_defining_model()
    noised = diffusion.forward(1, s=5, plot=False)


def test_training_sampling():
    diffusion = test_defining_model()
    diffusion.train(50, 2, plot_loss=False)
    last_10 = diffusion.sample(10, keep='last')
    assert last_10.shape == (10, 2), "shape of keeping 'last' samples only should be (10,2) not {}".format(last_10.shape)
    all_10 = diffusion.sample(n, keep='all')
    assert len(all_10) == n_steps+1, 'len of keeping all timestep samples should be {}, not {}'.format(n_steps, len(all_10))