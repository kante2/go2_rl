"""CNCP: 조건부 PPO-Lagrangian (논문 식 (2)~(10), Algorithm 1).

- 제약 J_c(s₀, ε) ≤ ε (식 2~4). J_c는 에피소드 시작부터의 할인 비용 합에 (1−γ)를 곱한 값이다.
  비용 c_t ∈ [0, 1]이므로 J_c도 [0, 1]이 되어 ε ∈ [0, 1]과 같은 척도가 된다 (정규화는 구현 선택).
- 환경 i마다 λ_i를 두고 PID로 갱신한다 (식 8, [53]).
- Â_i = (A_r − λ_i A_c)/(1 + λ_i) (식 8). advantage를 만든 뒤 λ를 갱신한다 (Algorithm 1 4→5행).
- 두 critic은 각각 제곱오차로 학습한다 (식 7, value clipping 없음).
- 실패 종료(넘어짐·불법 접촉) 뒤에는 비용 failure_cost가 영원히 이어진다고 보고 bootstrap한다.
  넘어져서 비용 제약을 피하는 해를 막는 구현 선택이다 (논문에 명시 없음).
"""

from copy import deepcopy

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_callable

from .pid_lagrangian import PidLagrangian, PidLagrangianCfg
from .storage import QuietPawStorage, generalized_advantage


class EpisodeCostReturn:
  """환경별 에피소드 시작부터의 할인 비용 합. rollout 경계를 넘어 누적하고 reset에서 끊는다."""

  def __init__(self, num_envs, device, gamma):
    self.prefix = torch.zeros(num_envs, device=device)
    self.discount = torch.ones_like(self.prefix)
    self.gamma = gamma

  def reset(self):
    self.prefix.zero_()
    self.discount.fill_(1.0)

  def estimate(self, costs, dones, final_values):
    """costs [T, B]에는 timeout·실패 bootstrap이 이미 들어 있다. 반환 [B]: 끝난 에피소드 + 진행 중 추정의 평균."""
    completed = torch.zeros_like(self.prefix)
    count = torch.zeros_like(self.prefix)
    for cost, done in zip(costs, dones):
      self.prefix.add_(self.discount * cost)
      self.discount.mul_(self.gamma)
      completed += self.prefix * done
      count += done
      self.prefix.masked_fill_(done, 0.0)
      self.discount.masked_fill_(done, 1.0)
    # 마지막에 끝나지 않은 에피소드는 남은 비용을 cost critic으로 추정해 한 번 포함한다.
    partial = ~dones[-1]
    completed += (self.prefix + self.discount * final_values) * partial
    return completed / (count + partial)


class QuietPawPPO(PPO):
  storage: QuietPawStorage

  def __init__(self, actor, critic, storage, pid, cost_value_loss_coef=1.0, **kwargs):
    if kwargs.get("multi_gpu_cfg"):
      raise ValueError("CNCP currently supports a single training process/GPU.")
    if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
      raise ValueError("CNCP does not support RND or symmetry augmentation.")
    super().__init__(actor, critic, storage, **kwargs)
    self.pid = PidLagrangian(storage.num_envs, PidLagrangianCfg(**pid), self.device)
    self.cost_value_loss_coef = cost_value_loss_coef
    self.episode_cost = EpisodeCostReturn(storage.num_envs, self.device, self.gamma)
    self.failure_cost = 1.0  # Runner가 환경 설정값으로 맞춘다.
    self.constraint_metrics = {}

  @property
  def failure_value(self):
    # 실패 뒤 매 스텝 failure_cost가 영원히 이어질 때의 할인 합.
    return self.failure_cost / (1.0 - self.gamma)

  @classmethod
  def construct_algorithm(cls, obs, env, cfg, device):
    actor_cfg = deepcopy(cfg["actor"])
    critic_cfg = deepcopy(cfg["critic"])
    algorithm_cfg = deepcopy(cfg["algorithm"])
    algorithm_cfg.pop("class_name")
    if algorithm_cfg.pop("share_cnn_encoders", False):
      raise ValueError("CNCP uses separate MLP critics, not shared CNN encoders.")
    groups = {key: list(value) for key, value in cfg["obs_groups"].items()}
    if groups != {"actor": ["actor"], "critic": ["critic"]}:
      raise ValueError("CNCP expects the configured actor and critic observation groups.")
    for model_cfg in (actor_cfg, critic_cfg):
      for name in ("cnn_cfg", "distribution_cfg"):
        if model_cfg.get(name) is None:
          model_cfg.pop(name, None)
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    critic_class = resolve_callable(critic_cfg.pop("class_name"))
    actor = actor_class(obs, groups, "actor", env.num_actions, **actor_cfg)
    critic = critic_class(obs, groups, "critic", 2, **critic_cfg)
    storage = QuietPawStorage(env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    cfg["algorithm"].setdefault("rnd_cfg", None)  # 기본 Runner가 로깅 중 조회한다.
    return cls(actor, critic, storage, device=device, multi_gpu_cfg=cfg.get("multi_gpu"), **algorithm_cfg)

  def process_env_step(self, obs, rewards, dones, extras):
    if "noise_cost" not in extras or "cost_limit" not in extras:
      raise ValueError("CNCP requires noise_cost and cost_limit from the environment.")
    raw_cost = extras["noise_cost"].to(self.device)
    cost_limit = extras["cost_limit"].to(self.device)
    if not torch.isfinite(raw_cost).all() or not ((raw_cost >= 0) & (raw_cost <= 1)).all():
      raise ValueError("noise_cost must be finite and in [0, 1].")
    reward, cost = rewards.clone(), raw_cost.clone()
    failures = extras.get("true_terminations")
    failures = (torch.zeros_like(raw_cost, dtype=torch.bool) if failures is None
                else failures.to(self.device).bool())
    timeouts = extras.get("time_outs")
    if timeouts is not None:
      # 시간 제한과 실패가 동시에 일어나면 실패를 우선한다.
      timeouts = timeouts.to(self.device).bool() & ~failures
      if timeouts.any():
        terminal = extras.get("terminal_observations")
        if terminal is None:
          raise ValueError("Timeout bootstrap requires observations from before reset.")
        terminal = TensorDict(terminal, batch_size=[self.storage.num_envs]).to(self.device)
        values = self.critic(terminal).detach()
        reward += self.gamma * values[:, 0] * timeouts
        cost += self.gamma * values[:, 1] * timeouts
    # 실패: reward는 bootstrap 없음(종료 페널티만), cost는 최대 비용이 이어지는 것으로 bootstrap.
    cost += self.gamma * self.failure_value * failures

    self.transition.rewards = reward
    self.transition.dones = dones
    self.storage.add_cost(cost, raw_cost, cost_limit)
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    self.actor.reset(dones)
    self.critic.reset(dones)

  @torch.no_grad()
  def compute_returns(self, obs):
    st = self.storage
    if st.step != st.num_transitions_per_env:
      raise ValueError("Compute returns only after filling the entire rollout.")
    limits = st.cost_limits[0, :, 0]
    if not torch.allclose(st.cost_limits[..., 0], limits.expand_as(st.cost_limits[..., 0])):
      raise ValueError("Each environment must keep its ε fixed during training.")
    final_values = self.critic(obs)
    returns, advantages = generalized_advantage(
      torch.cat((st.rewards, st.costs), dim=-1), st.values, st.dones, final_values, self.gamma, self.lam
    )
    st.returns.copy_(returns)
    # 식 (8): 현재 λ로 결합한 뒤 PPO 표준화를 적용한다 (A_r, A_c를 따로 표준화하지 않는다).
    multipliers = self.pid.multipliers.clone()
    combined = (advantages[..., 0] - multipliers * advantages[..., 1]) / (1.0 + multipliers)
    st.advantages.copy_(combined.unsqueeze(-1))
    if not self.normalize_advantage_per_mini_batch:
      st.advantages.sub_(st.advantages.mean()).div_(st.advantages.std(unbiased=False) + 1e-8)
    # Algorithm 1 5행: λ_i 갱신. J_c = (1−γ)·할인 비용 합.
    discounted = self.episode_cost.estimate(st.costs[..., 0], st.dones[..., 0].bool(), final_values[:, 1])
    normalized = (1.0 - self.gamma) * discounted
    self.pid.update(normalized, limits)
    self.constraint_metrics = {
      "noise_cost": st.raw_costs.mean().item(),
      "normalized_cost_return": normalized.mean().item(),
      "cost_violation": (normalized - limits).clamp_min(0).mean().item(),
      "lagrange_multiplier": multipliers.mean().item(),
      "lagrange_multiplier_max": multipliers.max().item(),
    }

  def _value_loss(self, predicted, old, target):
    loss = (predicted - target).square()
    if self.use_clipped_value_loss:
      clipped = old + (predicted - old).clamp(-self.clip_param, self.clip_param)
      loss = torch.maximum(loss, (clipped - target).square())
    return loss.mean(dim=0)  # [reward critic loss, cost critic loss]

  def update(self):
    totals = {"reward_value": 0.0, "cost_value": 0.0, "surrogate": 0.0, "entropy": 0.0}
    updates = 0
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
      advantage = batch.advantages.squeeze(-1)
      if self.normalize_advantage_per_mini_batch:
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      entropy = self.actor.output_entropy.mean()
      values = self.critic(batch.observations)

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.no_grad():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, self.actor.output_distribution_params
          ).mean().item()
        if kl > 2.0 * self.desired_kl:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0.0 < kl < 0.5 * self.desired_kl:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
          group["lr"] = self.learning_rate

      # 식 (9)(10): 모든 환경(ε_i)의 클립 surrogate 평균.
      ratio = (log_prob - batch.old_actions_log_prob.squeeze(-1)).exp()
      surrogate = torch.maximum(
        -advantage * ratio,
        -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
      ).mean()
      reward_loss, cost_loss = self._value_loss(values, batch.values, batch.returns)
      loss = (surrogate + self.value_loss_coef * reward_loss
              + self.cost_value_loss_coef * cost_loss - self.entropy_coef * entropy)
      if not torch.isfinite(loss):
        raise FloatingPointError("CNCP loss is not finite.")
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      for key, value in zip(totals, (reward_loss, cost_loss, surrogate, entropy)):
        totals[key] += value.item()
      updates += 1
    self.storage.clear()
    return {**{key: value / updates for key, value in totals.items()}, **self.constraint_metrics}

  def save(self):
    state = super().save()
    state["cncp_pid"] = self.pid.state_dict()
    state["cncp_learning_rate"] = self.learning_rate
    return state

  def load(self, loaded_dict, load_cfg=None, strict=True):
    resume = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("optimizer", False):
      # 환경 수가 다르면 PID 대응이 깨지므로 strict shape 검사로 중단한다.
      self.pid.load_state_dict(loaded_dict["cncp_pid"], strict=True)
      self.learning_rate = loaded_dict["cncp_learning_rate"]
      self.episode_cost.reset()  # 재개 시 환경은 새 에피소드에서 시작한다.
    return resume
