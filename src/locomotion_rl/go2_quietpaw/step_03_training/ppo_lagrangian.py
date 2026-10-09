"""조건별 PID multiplier와 Reward/Cost GAE를 사용하는 PPO.

일반 actor, optimizer, KL 조정 방식은 baseline의 RSL-RL을 따른다.
제약은 할인 누적 비용이 아니라 rollout의 스텝 평균 raw cost에 적용한다.
Cost critic 자체는 할인 return을 예측한다. 이는 논문의 누적 단위 모호성을
피하기 위한 명시적인 구현 선택이며, 엄밀한 제약 만족 보장은 아니다.
"""

from copy import deepcopy

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_callable

from .pid_lagrangian import PidLagrangian, PidLagrangianCfg
from .storage import QuietPawStorage, generalized_advantage


class QuietPawPPO(PPO):
  storage: QuietPawStorage

  def __init__(self, actor, critic, storage, pid, cost_value_loss_coef=1.0, **kwargs):
    if kwargs.get("multi_gpu_cfg"):
      raise ValueError("QuietPaw currently supports a single training process/GPU.")
    if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
      raise ValueError("QuietPaw does not yet support RND or symmetry augmentation.")
    super().__init__(actor, critic, storage, **kwargs)
    self.pid = PidLagrangian(storage.num_envs, PidLagrangianCfg(**pid), self.device)
    self.cost_value_loss_coef = cost_value_loss_coef

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    # Runner 설정 원본을 보존한다. checkpoint와 YAML에 class_name 등이 남는다.
    actor_cfg = deepcopy(cfg["actor"])
    critic_cfg = deepcopy(cfg["critic"])
    algorithm_cfg = deepcopy(cfg["algorithm"])
    algorithm_cfg.pop("class_name")
    if algorithm_cfg.pop("share_cnn_encoders", False):
      raise ValueError("QuietPaw uses separate MLP critics, not shared CNN encoders.")
    groups = {key: list(value) for key, value in cfg["obs_groups"].items()}
    if groups != {"actor": ["actor"], "critic": ["critic"]}:
      raise ValueError("QuietPaw expects the configured actor and critic observation groups.")
    for model_cfg in (actor_cfg, critic_cfg):
      for name in ("cnn_cfg", "distribution_cfg"):
        if model_cfg.get(name) is None:
          model_cfg.pop(name, None)
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    critic_class = resolve_callable(critic_cfg.pop("class_name"))
    actor = actor_class(obs, groups, "actor", env.num_actions, **actor_cfg)
    critic = critic_class(obs, groups, "critic", 2, **critic_cfg)
    storage = QuietPawStorage(env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    # 기본 Runner는 로깅 중 rnd_cfg 키를 조회한다.
    cfg["algorithm"].setdefault("rnd_cfg", None)
    return QuietPawPPO(
      actor, critic, storage, device=device, multi_gpu_cfg=cfg.get("multi_gpu"), **algorithm_cfg
    )

  def process_env_step(self, obs, rewards, dones, extras):
    if "noise_cost" not in extras or "cost_limit" not in extras:
      raise ValueError("QuietPawPPO requires QuietPawEnv noise_cost and cost_limit outputs.")
    raw_cost = extras["noise_cost"].to(self.device)
    cost_limit = extras["cost_limit"].to(self.device)
    if not torch.isfinite(raw_cost).all() or not ((raw_cost >= 0) & (raw_cost <= 1)).all():
      raise ValueError("noise_cost must be finite and in [0, 1].")
    reward, cost = rewards.clone(), raw_cost.clone()
    timeouts = extras.get("time_outs")
    if timeouts is not None:
      # 시간 제한과 넘어짐이 동시에 발생하면 실제 종료를 우선한다.
      timeouts = timeouts.to(self.device).bool() & ~extras["true_terminations"].to(self.device).bool()
    if timeouts is not None and timeouts.any():
      terminal = extras.get("terminal_observations")
      if terminal is None:
        raise ValueError("Timeout bootstrap requires observations from before reset.")
      terminal = TensorDict(terminal, batch_size=[self.storage.num_envs]).to(self.device)
      terminal_values = self.critic(terminal).detach()
      mask = timeouts.to(self.device)
      reward += self.gamma * terminal_values[:, 0] * mask
      cost += self.gamma * terminal_values[:, 1] * mask

    self.transition.rewards = reward
    self.transition.dones = dones
    self.storage.add_cost(cost, raw_cost, cost_limit)
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs):
    st = self.storage
    if st.step != st.num_transitions_per_env:
      raise ValueError("Compute returns only after filling the entire rollout.")
    signals = torch.cat((st.rewards, st.costs), dim=-1)
    returns, advantages = generalized_advantage(
      signals, st.values, st.dones, self.critic(obs).detach(), self.gamma, self.lam
    )
    st.returns.copy_(returns)
    # 환경별 조건은 rollout 중 고정되어야 PID가 같은 제약을 조절한다.
    if not torch.allclose(st.cost_limits, st.cost_limits[0:1].expand_as(st.cost_limits)):
      raise ValueError("Noise conditions must remain fixed throughout a training rollout.")
    mean_cost = st.raw_costs.mean(dim=0).squeeze(-1)
    limit = st.cost_limits[0, :, 0]
    multipliers = self.pid.update(mean_cost, limit)
    # Reward/Cost advantage를 따로 표준화하면 λ의 상대 의미가 바뀐다.
    # 원래 척도로 먼저 결합하고, 결합 결과에만 PPO 표준화를 적용한다.
    combined = (advantages[..., 0] - multipliers * advantages[..., 1]) / (1.0 + multipliers)
    st.advantages.copy_(combined.unsqueeze(-1))
    if not self.normalize_advantage_per_mini_batch:
      st.advantages.sub_(st.advantages.mean()).div_(st.advantages.std(unbiased=False) + 1e-8)
    self.constraint_metrics = {
      "noise_cost": mean_cost.mean().item(),
      "cost_violation": (mean_cost - limit).clamp_min(0).mean().item(),
      "lagrange_multiplier": multipliers.mean().item(),
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

      ratio = (log_prob - batch.old_actions_log_prob.squeeze(-1)).exp()
      surrogate = torch.maximum(
        -advantage * ratio,
        -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
      ).mean()
      reward_loss, cost_loss = self._value_loss(values, batch.values, batch.returns)
      loss = (surrogate + self.value_loss_coef * reward_loss
              + self.cost_value_loss_coef * cost_loss - self.entropy_coef * entropy)
      if not torch.isfinite(loss):
        raise FloatingPointError("QuietPaw loss is not finite.")
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
    state["quietpaw_pid"] = self.pid.state_dict()
    state["quietpaw_learning_rate"] = self.learning_rate
    return state

  def load(self, loaded_dict, load_cfg=None, strict=True):
    resume = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("optimizer", False):
      # 환경 수를 바꾼 resume은 PID 매핑이 달라지므로 strict shape 오류로 중단한다.
      self.pid.load_state_dict(loaded_dict["quietpaw_pid"], strict=True)
      self.learning_rate = loaded_dict["quietpaw_learning_rate"]
    return resume
