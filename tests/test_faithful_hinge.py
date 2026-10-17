"""The faithfulness hinge must keep a gradient while the policy mean sits beyond the limits."""

import numpy as np
import pytest
import torch as th

from crta.faithful import faith_hinge_loss, hinge_position


def test_gaussian_clip_hinge_sees_means_beyond_the_limit():
    actual = th.tensor([5.0], requires_grad=True)
    crisis = th.tensor([6.0], requires_grad=True)
    loss = faith_hinge_loss(actual, crisis, None, "gaussian_clip")
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert crisis.grad.item() > 0.0
    assert actual.grad.item() < 0.0


def test_clamping_first_would_have_hidden_that_violation():
    actual = th.tensor([5.0])
    crisis = th.tensor([6.0])
    assert th.relu(crisis.clamp(-1, 1) - actual.clamp(-1, 1)).item() == 0.0
    assert faith_hinge_loss(actual, crisis, None, "gaussian_clip").item() > 0.0


def test_tanh_mode_scores_the_executed_position():
    assert hinge_position(th.tensor([1.0]), "tanh_env").item() == pytest.approx(np.tanh(1.0))
    loss = faith_hinge_loss(th.tensor([0.0]), th.tensor([1.0]), None, "tanh_env")
    assert loss.item() == pytest.approx(np.tanh(1.0))


def test_a_faithful_ordering_costs_nothing():
    loss = faith_hinge_loss(th.tensor([0.5]), th.tensor([0.2]), th.tensor([0.9]), "gaussian_clip")
    assert loss.item() == 0.0


def test_unknown_action_mode_is_rejected():
    with pytest.raises(ValueError):
        hinge_position(th.tensor([0.0]), "discrete5")
