"""Beta, tanh and clipped heads, and a monotone regime head that is faithful by construction."""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from crta.policies import BetaDistribution, PositionPolicy, monotone_weights, regime_rank_vector


def _policy(head="beta", monotone=True, n_regime=4, rank=(2, 0, 3, 1), obs_dim=13):
    obs_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32) if head != "tanh01" else spaces.Box(-1e3, 1e3, shape=(1,), dtype=np.float32)
    return PositionPolicy(obs_space, act_space, lambda _: 3e-4, head=head, n_regime=n_regime, monotone=monotone, regime_rank=list(rank))


def test_beta_distribution_matches_torch():
    dist = BetaDistribution(1).proba_distribution(th.tensor([[2.0]]), th.tensor([[3.0]]))
    a = th.tensor([[0.3]])
    assert dist.log_prob(a).item() == pytest.approx(th.distributions.Beta(2.0, 3.0).log_prob(th.tensor(0.3)).item())
    assert dist.mode().item() == pytest.approx(0.4)
    samples = th.stack([dist.sample() for _ in range(50)])
    assert ((samples > 0) & (samples < 1)).all()


def test_monotone_weights_never_rise_with_rank():
    rng = np.random.default_rng(0)
    rank = th.tensor([2, 0, 3, 1])
    for _ in range(20):
        w = monotone_weights(th.tensor(rng.normal(size=1)), th.tensor(rng.normal(size=3)), rank).detach().numpy()
        by_rank = w[np.argsort(rank.numpy())]
        assert np.all(np.diff(by_rank) <= 1e-9)


def test_rank_vector_puts_unranked_regimes_last():
    order = {"ranked": ["calm", "bull", "bear"], "rank": {"bull": 1, "bear": 2, "calm": 0, "crisis": None}}
    assert regime_rank_vector(order) == [1, 2, 0, 3]


@pytest.mark.parametrize("head", ["beta", "tanh01", "clip01"])
def test_monotone_policy_never_holds_less_in_a_less_defensive_regime(head):
    th.manual_seed(0)
    policy = _policy(head=head)
    with th.no_grad():
        for p in policy.parameters():
            p.add_(th.randn_like(p) * 0.5)
    rank = [2, 0, 3, 1]  # bull, bear, calm, crisis: bear least defensive, crisis most
    base = th.randn(64, 9)
    order = np.argsort(rank)
    for lo, hi in zip(order[:-1], order[1:]):
        p_less = th.zeros(64, 4)
        p_less[:, lo] = 1.0
        p_more = th.zeros(64, 4)
        p_more[:, hi] = 1.0
        a_less = policy.position(th.cat([base, p_less], dim=1))
        a_more = policy.position(th.cat([base, p_more], dim=1))
        assert th.all(a_less >= a_more - 1e-6)


def test_monotone_base_net_does_not_see_the_regime_block():
    policy = _policy()
    obs = th.randn(8, 13)
    other = obs.clone()
    other[:, -4:] = th.rand(8, 4)
    w = policy.regime_weights()
    dz = policy.position_logit(other) - policy.position_logit(obs)
    assert th.allclose(dz, (other[:, -4:] - obs[:, -4:]) @ w, atol=1e-5)


def test_blind_policy_has_no_regime_weights():
    policy = _policy(monotone=False, n_regime=0, rank=(), obs_dim=9)
    assert policy.regime_weights() is None
    actions, values, log_prob = policy(th.randn(5, 9))
    assert actions.shape == (5, 1) and values.shape == (5, 1) and log_prob.shape == (5,)


def test_evaluate_actions_gives_finite_log_probs_at_the_edges():
    policy = _policy(head="beta")
    obs = th.randn(4, 13)
    values, log_prob, entropy = policy.evaluate_actions(obs, th.tensor([[0.0], [1.0], [0.5], [1e-9]]))
    assert th.isfinite(log_prob).all() and th.isfinite(entropy).all()
