import torch
import torch.nn.functional as F
from torch import nn


def orthogonal_weight_init(m: nn.Module) -> None:
    """Orthogonal weight initialization for neural networks."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        use_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.use_normalize = use_normalize
        self.apply(orthogonal_weight_init)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> None:
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        use_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        self.use_normalize = use_normalize
        self.apply(orthogonal_weight_init)

    def forward(self, x: torch.Tensor) -> tuple:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple:
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        action = torch.clone(action)
        action[:, 2:] = action[:, 2:] * 0.5 + 0.5
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean
