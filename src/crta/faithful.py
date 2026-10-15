"""PPO with an in-distribution faithfulness penalty on the conditioned twin only.

The auxiliary term is added to the PPO loss inside ``train``, which is Stable-Baselines3
2.9.0 ``PPO.train`` plus that term. It is off unless ``faithfulness_regularizer`` is set.
A result trained with it is regularized: the verifier is no longer an independent audit.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


def faithfulness_hinge(action, action_crisis, action_bull):
    """Penalize a crisis shift that raises the position and a bull shift that lowers it."""
    terms = []
    if action_crisis is not None:
        terms.append(np.maximum(0.0, np.asarray(action_crisis) - np.asarray(action)))
    if action_bull is not None:
        terms.append(np.maximum(0.0, np.asarray(action) - np.asarray(action_bull)))
    if not terms:
        return np.zeros_like(np.asarray(action, dtype=float))
    return np.sum(terms, axis=0)


class FaithfulPPO(PPO):
    """Conditioned PPO. ``crta_crisis_bank`` and ``crta_bull_bank`` are training-set rows."""

    def _position_mean(self, obs: th.Tensor) -> th.Tensor:
        mean = self.policy.get_distribution(obs).distribution.mean
        return mean.squeeze(-1).clamp(-1.0, 1.0)

    def _nearest(self, probs: th.Tensor, bank: np.ndarray) -> th.Tensor:
        table = th.as_tensor(bank, device=probs.device, dtype=probs.dtype)
        dist = (table[None, :, :] - probs[:, None, :]).abs().sum(dim=-1)
        return table[dist.argmin(dim=1)]

    def _faith_loss(self, obs: th.Tensor) -> th.Tensor:
        coef = float(getattr(self, "crta_faith_coef", 0.0))
        if coef <= 0.0 or obs.shape[-1] < 10:
            return obs.new_zeros(())
        probs = obs[:, -4:]
        action = self._position_mean(obs)
        parts = []
        crisis = getattr(self, "crta_crisis_bank", None)
        bull = getattr(self, "crta_bull_bank", None)
        if crisis is not None and len(crisis):
            shifted = obs.clone()
            shifted[:, -4:] = self._nearest(probs, crisis)
            parts.append(th.relu(self._position_mean(shifted) - action))
        if bull is not None and len(bull):
            shifted = obs.clone()
            shifted[:, -4:] = self._nearest(probs, bull)
            parts.append(th.relu(action - self._position_mean(shifted)))
        if not parts:
            return obs.new_zeros(())
        return coef * th.stack(parts, dim=0).mean()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        faith_losses = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                faith = self._faith_loss(rollout_data.observations)
                faith_losses.append(float(faith.detach().cpu()))
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + faith

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/faithfulness_loss", float(np.mean(faith_losses)) if faith_losses else 0.0)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
