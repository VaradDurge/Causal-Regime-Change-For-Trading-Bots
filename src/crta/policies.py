"""Policy heads for the daily long/flat task.

Every head maps a position logit z to a position in [0, 1]:

  clip01  Gaussian on z, executed position clip(z, 0, 1)
  tanh01  Gaussian on z, executed position (tanh(z) + 1) / 2
  beta    Beta(m k, (1 - m) k) with mean m = sigmoid(z) and a learned concentration k,
          which can rest anywhere inside (0, 1) (Chou et al., 2017)

The monotone head writes z = base(features, previous position) + sum_g w_g p_g. The base
network never sees the regime block, and the weights are forced to fall along a risk
order fixed from the training fold, so moving probability towards a less defensive
regime can never lower the position. It is faithful by construction; a verifier then
confirms a constraint rather than discovers behaviour.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.nn import functional as F

from crta.regime import REGIMES

HEADS = ("clip01", "tanh01", "beta")
HIDDEN = 64
# Beta log-densities are clamped this far inside (0, 1); a sampled 0 or 1 would be -inf.
ACTION_EPS = 1e-6
MEAN_EPS = 1e-4
KAPPA_MIN = 2.0
# softplus(-2) = 0.13: the regime weights start nearly equal, so the order is learned, not imposed in size.
GAP_INIT = -2.0


class BetaDistribution(Distribution):
    """Beta distribution on [0, 1] for Stable-Baselines3."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, alpha: th.Tensor, beta: th.Tensor) -> "BetaDistribution":
        self.distribution = th.distributions.Beta(alpha, beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions.clamp(ACTION_EPS, 1.0 - ACTION_EPS)).sum(dim=-1)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        # The mean, not the mode: it is sigmoid(z) exactly, so the monotone head stays monotone.
        return self.distribution.mean

    def actions_from_params(self, alpha: th.Tensor, beta: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(alpha, beta)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, alpha: th.Tensor, beta: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(alpha, beta)
        return actions, self.log_prob(actions)


def monotone_weights(w_top: th.Tensor, gaps: th.Tensor, rank: th.Tensor) -> th.Tensor:
    """Regime weights that never rise with rank. ``rank[g]`` is regime g's rank, 0 least defensive."""
    by_rank = th.cat([w_top.reshape(1), w_top.reshape(1) - th.cumsum(F.softplus(gaps), dim=0)])
    return by_rank[rank]


def regime_rank_vector(order: dict) -> list[int]:
    """Rank per regime in REGIMES order. Unranked regimes go after the ranked ones, as most defensive."""
    ranks = dict(order["rank"])
    nxt = len(order["ranked"])
    out = []
    for name in REGIMES:
        if ranks.get(name) is None:
            out.append(nxt)
            nxt += 1
        else:
            out.append(int(ranks[name]))
    return out


def _mlp(dim_in: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim_in, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh())


class PositionPolicy(ActorCriticPolicy):
    """Actor-critic with a position-logit head; the regime block is the last ``n_regime`` inputs."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        head: str = "beta",
        n_regime: int = 0,
        monotone: bool = False,
        regime_rank: list[int] | None = None,
        **kwargs,
    ):
        if head not in HEADS:
            raise ValueError(f"unknown head {head}, expected one of {HEADS}")
        if monotone and (n_regime < 2 or regime_rank is None or len(regime_rank) != n_regime):
            raise ValueError("a monotone head needs the regime block and one rank per regime")
        self.head = head
        self.n_regime = int(n_regime)
        self.monotone = bool(monotone)
        self.regime_rank = [int(r) for r in (regime_rank or [])]
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        obs_dim = int(np.prod(self.observation_space.shape))
        self.pi_net = _mlp(obs_dim - self.n_regime if self.monotone else obs_dim)
        self.vf_net = _mlp(obs_dim)
        self.value_net = nn.Linear(HIDDEN, 1)
        self.z_head = nn.Linear(HIDDEN, 1)
        if self.monotone:
            self.w_top = nn.Parameter(th.zeros(1))
            self.w_gaps = nn.Parameter(th.full((self.n_regime - 1,), GAP_INIT))
            self.register_buffer("rank_index", th.as_tensor(self.regime_rank, dtype=th.long))
        if self.head == "beta":
            self.kappa_head = nn.Linear(HIDDEN, 1)
            self.action_dist = BetaDistribution(1)
        else:
            self.log_std = nn.Parameter(th.zeros(1))
            self.action_dist = DiagGaussianDistribution(1)
        for net in (self.pi_net, self.vf_net):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        for layer, gain in ((self.z_head, 0.01), (self.value_net, 1.0)):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        if self.head == "beta":
            nn.init.orthogonal_(self.kappa_head.weight, gain=0.01)
            nn.init.zeros_(self.kappa_head.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def regime_weights(self) -> th.Tensor | None:
        if not self.monotone:
            return None
        return monotone_weights(self.w_top, self.w_gaps, self.rank_index)

    def _logit_latent(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        obs = obs.float()
        base = obs[:, : obs.shape[1] - self.n_regime] if self.monotone else obs
        latent = self.pi_net(base)
        z = self.z_head(latent).squeeze(-1)
        if self.monotone:
            z = z + obs[:, -self.n_regime :] @ self.regime_weights()
        return z, latent

    def position_logit(self, obs: th.Tensor) -> th.Tensor:
        return self._logit_latent(obs)[0]

    def position(self, obs: th.Tensor) -> th.Tensor:
        """Deterministic executed position in [0, 1], non-decreasing in the logit."""
        z = self.position_logit(obs)
        if self.head == "beta":
            return th.sigmoid(z).clamp(MEAN_EPS, 1.0 - MEAN_EPS)
        if self.head == "tanh01":
            return (th.tanh(z) + 1.0) / 2.0
        return z.clamp(0.0, 1.0)

    def _dist(self, obs: th.Tensor) -> Distribution:
        z, latent = self._logit_latent(obs)
        if self.head == "beta":
            m = th.sigmoid(z).clamp(MEAN_EPS, 1.0 - MEAN_EPS)
            kappa = KAPPA_MIN + F.softplus(self.kappa_head(latent).squeeze(-1))
            return self.action_dist.proba_distribution((m * kappa).unsqueeze(-1), ((1.0 - m) * kappa).unsqueeze(-1))
        return self.action_dist.proba_distribution(z.unsqueeze(-1), self.log_std)

    def _value(self, obs: th.Tensor) -> th.Tensor:
        return self.value_net(self.vf_net(obs.float()))

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        dist = self._dist(obs)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self._value(obs), dist.log_prob(actions)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        dist = self._dist(obs)
        return self._value(obs), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        return self._dist(obs)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self._value(obs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._dist(observation).get_actions(deterministic=deterministic)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(head=self.head, n_regime=self.n_regime, monotone=self.monotone, regime_rank=self.regime_rank)
        return data
