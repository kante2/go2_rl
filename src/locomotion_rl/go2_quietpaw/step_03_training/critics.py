"""Reward와 Cost의 조건 분해 Critic: V(s, ε) = ξ(s)ᵀ w(ε)."""

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization, MLP


class DecomposedCritic(nn.Module):
  def __init__(
    self, state_dim: int, hidden_dims=(512, 256, 128), feature_dim=64,
    condition_hidden_dims=(64, 64), activation="elu", obs_normalization=True,
  ):
    super().__init__()
    self.normalizer = EmpiricalNormalization(state_dim) if obs_normalization else nn.Identity()
    self.state_features = MLP(state_dim, feature_dim, hidden_dims, activation)
    self.condition_weights = MLP(1, feature_dim, condition_hidden_dims, activation)

  def forward(self, state, condition):
    # ε는 상태 인코더·상태 정규화에 들어가지 않는다. 조건 경로에만 전달한다.
    features = self.state_features(self.normalizer(state))
    weights = self.condition_weights(condition)
    return (features * weights).sum(dim=-1, keepdim=True)

  def update_normalization(self, state):
    if isinstance(self.normalizer, EmpiricalNormalization):
      self.normalizer.update(state)


class QuietPawCritics(nn.Module):
  """독립적인 두 critic을 묶어 [B, 2] = [reward value, cost value]로 반환한다."""

  is_recurrent = False

  def __init__(self, obs, obs_groups, obs_set, output_dim=2, **kwargs):
    super().__init__()
    if output_dim != 2 or obs_groups[obs_set] != ["critic"]:
      raise ValueError("QuietPaw critics require the single 'critic' observation group and two outputs.")
    state_dim = obs["critic"].shape[-1] - 1
    self.reward = DecomposedCritic(state_dim, **kwargs)
    self.cost = DecomposedCritic(state_dim, **kwargs)

  def forward(self, obs, **kwargs):
    state, condition = obs["critic"][..., :-1], obs["critic"][..., -1:]
    return torch.cat((self.reward(state, condition), self.cost(state, condition)), dim=-1)

  def update_normalization(self, obs):
    state = obs["critic"][..., :-1]
    self.reward.update_normalization(state)
    self.cost.update_normalization(state)

  def get_hidden_state(self):
    return None

  def reset(self, dones=None):
    pass  # Feed-forward 모델에는 episode별 hidden state가 없다.
