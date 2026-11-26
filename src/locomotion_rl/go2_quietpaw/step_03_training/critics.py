"""Reward와 Cost의 조건 분해 Critic: V(s, ε) = ξ(s)ᵀ w(ε)."""

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization, MLP


class DecomposedCritic(nn.Module):
  # Decompose : 가치 함수의 계산을 두 부분으로 분해했다는 뜻
  #   state feature : 상태 s을 받아 특징벡터 epsilon(s) 을 생성
  #   condition_weights : 소음한계 (epsilon)을 받아 가중치벡터 w(epsilon) 을 생성
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
    features = self.state_features(self.normalizer(state)) # s 에서 만든 64 차원의 특징 벡터
    weights = self.condition_weights(condition) # 소음한게 (epsilon) 에서 만든 64차원 가중치 벡터 
    # V_r(s, ε) = ξ_r(s)ᵀw_r(ε) 이 return 이후의 수식에 해당한다.
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
    # reward와 cost를 독립적으로 계산하는 두 critic을 생성한다.
    self.reward = DecomposedCritic(state_dim, **kwargs) # 1.reward critic : V_r(s, ε) = ξ_r(s)ᵀw_r(ε)
    self.cost = DecomposedCritic(state_dim, **kwargs) # 2.cost critic : V_c(s, ε) = ξ_c(s)ᵀw_c(ε)

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
