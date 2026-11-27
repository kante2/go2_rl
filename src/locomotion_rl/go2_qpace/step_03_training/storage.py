"""기존 RSL-RL rollout 구조에 cost와 두 종류의 value/return을 추가한다."""

import torch
from rsl_rl.storage import RolloutStorage


def generalized_advantage(signals, values, dones, last_values, gamma, lam):
  """[T, B, K] 신호의 GAE. K=2이면 reward/cost를 같은 종료 경계로 처리한다."""
  advantage = torch.zeros_like(last_values)
  returns = torch.empty_like(values)
  for step in reversed(range(signals.shape[0])):
    next_value = last_values if step == signals.shape[0] - 1 else values[step + 1]
    alive = 1.0 - dones[step].float()
    delta = signals[step] + gamma * alive * next_value - values[step]
    advantage = delta + gamma * lam * alive * advantage
    returns[step] = values[step] + advantage
  return returns, returns - values


class QuietPawStorage(RolloutStorage):
  def __init__(self, num_envs, num_steps, obs, actions_shape, device):
    super().__init__("rl", num_envs, num_steps, obs, actions_shape, device)
    pair_shape = (num_steps, num_envs, 2)
    self.values = torch.zeros(pair_shape, device=device)
    self.returns = torch.zeros_like(self.values)
    self.costs = torch.zeros_like(self.rewards)
    self.raw_costs = torch.zeros_like(self.rewards)  # PID에는 bootstrap을 더하기 전 비용을 전달.
    self.cost_limits = torch.zeros_like(self.rewards)

  def add_cost(self, cost, raw_cost, cost_limit):
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("QuietPaw rollout buffer is full.")
    self.costs[self.step].copy_(cost.view(-1, 1))
    self.raw_costs[self.step].copy_(raw_cost.view(-1, 1))
    self.cost_limits[self.step].copy_(cost_limit.view(-1, 1))

  def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    # tensor_split은 배치 크기가 나누어떨어지지 않아도 마지막 샘플을 버리지 않는다.
    size = self.num_envs * self.num_transitions_per_env
    if not 1 <= num_mini_batches <= size:
      raise ValueError("num_mini_batches must be between 1 and rollout size.")
    observations = self.observations.flatten(0, 1)
    for _ in range(num_epochs):
      indices = torch.randperm(size, device=self.device)
      for ids in indices.tensor_split(num_mini_batches):
        yield RolloutStorage.Batch(
          observations=observations[ids],
          actions=self.actions.flatten(0, 1)[ids],
          values=self.values.flatten(0, 1)[ids],
          returns=self.returns.flatten(0, 1)[ids],
          advantages=self.advantages.flatten(0, 1)[ids],
          old_actions_log_prob=self.actions_log_prob.flatten(0, 1)[ids],
          old_distribution_params=tuple(p.flatten(0, 1)[ids] for p in self.distribution_params),
        )
