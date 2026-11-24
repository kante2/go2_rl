"""Conditional PPO-Lagrangian with an episode-start discounted cost budget."""

import torch

from ..go2_quietpaw.step_03_training.ppo import unitree_go2_ppo_runner_cfg
from ..go2_quietpaw.step_03_training.ppo_lagrangian import QuietPawPPO
from ..go2_quietpaw.step_03_training.storage import generalized_advantage


class EpisodeCostReturn:
  """Carry the prefix from each environment's last reset across rollout boundaries."""

  def __init__(self, num_envs, device, gamma):
    self.prefix = torch.zeros(num_envs, device=device)
    self.discount = torch.ones_like(self.prefix)
    self.gamma = gamma

  def estimate(self, costs, dones, final_values):
    # costs already include the terminal-observation bootstrap for timeouts.
    completed = torch.zeros_like(self.prefix)
    count = torch.zeros_like(self.prefix)
    for cost, done in zip(costs, dones):
      self.prefix.add_(self.discount * cost)
      self.discount.mul_(self.gamma)
      completed += self.prefix * done
      count += done
      self.prefix.masked_fill_(done, 0.0)
      self.discount.masked_fill_(done, 1.0)
    # Include the current episode once if the final transition did not reset.
    partial = ~dones[-1]
    completed += (self.prefix + self.discount * final_values) * partial
    return completed / (count + partial)


class QuietWalkPPO(QuietPawPPO):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.episode_cost = EpisodeCostReturn(self.storage.num_envs, self.device, self.gamma)

  @torch.no_grad()
  def compute_returns(self, obs):
    st = self.storage
    if st.step != st.num_transitions_per_env:
      raise ValueError("Expected a complete rollout.")
    limits = st.cost_limits[0, :, 0]
    if not torch.allclose(st.cost_limits[..., 0], limits.expand_as(st.cost_limits[..., 0])):
      raise ValueError("Each environment must keep its constraint fixed during training.")
    final_values = self.critic(obs)
    returns, advantages = generalized_advantage(
      torch.cat((st.rewards, st.costs), dim=-1), st.values, st.dones, final_values, self.gamma, self.lam
    )
    st.returns.copy_(returns)
    multipliers = self.pid.multipliers
    combined = (advantages[..., 0] - multipliers * advantages[..., 1]) / (1.0 + multipliers)
    st.advantages.copy_(combined.unsqueeze(-1))
    if not self.normalize_advantage_per_mini_batch:
      st.advantages.sub_(st.advantages.mean()).div_(st.advantages.std(unbiased=False) + 1e-8)
    # Algorithm 1 updates the multipliers after constructing this rollout's advantages.
    cost_return = self.episode_cost.estimate(st.costs[..., 0], st.dones[..., 0].bool(), final_values[:, 1])
    self.pid.update(cost_return, limits)
    self.constraint_metrics = {
      "noise_cost": st.raw_costs.mean().item(),
      "discounted_cost_return": cost_return.mean().item(),
      "cost_violation": (cost_return - limits).clamp_min(0).mean().item(),
      "lagrange_multiplier": multipliers.mean().item(),
    }


def runner_cfg():
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_quietwalk_1124"
  cfg.algorithm.class_name = f"{__name__}:QuietWalkPPO"
  cfg.algorithm.use_clipped_value_loss = False  # Eq. (7)
  cfg.max_iterations = 10_000
  return cfg
