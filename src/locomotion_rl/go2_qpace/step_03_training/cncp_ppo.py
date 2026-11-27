"""QPace: 목표 dB 조건부 PPO-Lagrangian (QuietPaw CNCP 구조, 소음 비용 = DiffImpact 소음 모델).

- 비용 c_t ≥ 0: 로봇 전체 착지 에너지 / (Δt · P_ref)  (cncp_env.py). 착지가 없는 스텝은 0.
- 제약: L_robot = 10·log10((1−γ)·Σγᵗ c_t) + reference_db ≤ 목표 dB.
  에너지 비율 J_c = (1−γ)·Σγᵗ c_t 와 ε = 10^((목표 − reference)/10) 의 비교와 같다.
- λ_i 는 dB 차이 (L_robot − 목표 dB)를 오차로 하는 PID로 갱신한다. 목표가 수십 배 범위로 달라도
  "1 dB 초과"가 같은 크기의 오차가 된다.
- Â_i = (A_r − λ_i A_c)/(1 + λ_i) (식 8). advantage를 만든 뒤 λ를 갱신한다 (Algorithm 1).
- 두 critic은 각각 제곱오차로 학습한다 (식 7).
- 실패 종료 뒤에는 failure_cost(학습 범위 상단 + margin dB 파워)가 영원히 이어진다고 보고 bootstrap한다.
- [C 방식] baseline 앵커: actor 손실에 anchor_coef·Σ(μ − μ_base)² 를 더한다 (μ_base: warm start 시점 actor 의
  고정 복사본). 제약이 느슨하면 baseline 보행에 머물고, λ 가 미는 만큼만 벗어난다.
- 행동 표준편차는 max_std 로 위를 자른다.
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

  def __init__(self, actor, critic, storage, pid, cost_value_loss_coef=1.0, anchor_coef=0.0, max_std=None,
               **kwargs):
    if kwargs.get("multi_gpu_cfg"):
      raise ValueError("CNCP currently supports a single training process/GPU.")
    if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
      raise ValueError("CNCP does not support RND or symmetry augmentation.")
    super().__init__(actor, critic, storage, **kwargs)
    self.pid = PidLagrangian(storage.num_envs, PidLagrangianCfg(**pid), self.device)
    self.cost_value_loss_coef = cost_value_loss_coef
    if anchor_coef < 0 or (max_std is not None and not max_std > 0):
      raise ValueError("anchor_coef must be nonnegative and max_std positive or None.")
    self.anchor_coef = anchor_coef
    self.max_std = max_std
    self.anchor_actor = None  # set_anchor() 로 만든다 (warm start 직후 또는 체크포인트 재개)
    # 목표 커리큘럼 게이트를 통과한 iteration 수 (runner.py). 체크포인트에 저장해 재개 시 이어 간다.
    self.target_progress = 0
    self.episode_cost = EpisodeCostReturn(storage.num_envs, self.device, self.gamma)
    self.failure_cost = 1.0  # Runner가 환경 설정값으로 맞춘다.
    self.curriculum = None  # QPaceCurriculumCfg. Runner가 환경 설정에서 넣는다 (None: 단계 없음)
    self.iteration = 0
    # Runner가 넣는 콜백: rollout이 끝날 때 다음 iteration 번호로 환경 목표 범위를 갱신하고 현재 하한 [dB]을 돌려준다.
    # rollout 사이에만 바뀌므로 한 rollout 안의 ε 는 고정이다 (compute_returns 검사).
    self.on_iteration = None
    self.constraint_metrics = {}

  def set_anchor(self, state_dict=None):
    """현재 actor(또는 state_dict)의 고정 복사본을 baseline 앵커로 둔다. anchor_coef = 0 이면 만들지 않는다."""
    if self.anchor_coef == 0:
      return
    anchor = deepcopy(self.actor)
    if state_dict is not None:
      anchor.load_state_dict(state_dict)
    anchor.eval()
    for param in anchor.parameters():
      param.requires_grad_(False)
    self.anchor_actor = anchor

  @torch.no_grad()
  def _clamp_std(self):
    dist = getattr(self.actor, "distribution", None)
    if self.max_std is None or dist is None:
      return
    if hasattr(dist, "std_param"):
      dist.std_param.clamp_(max=self.max_std)
    elif hasattr(dist, "log_std_param"):
      dist.log_std_param.clamp_(max=float(torch.log(torch.tensor(self.max_std))))

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
    if not torch.isfinite(raw_cost).all() or not (raw_cost >= 0).all():
      raise ValueError("noise_cost must be finite and nonnegative.")
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
    # Algorithm 1 5행: λ_i 갱신. 오차 = L_robot − 목표 dB.
    discounted = self.episode_cost.estimate(st.costs[..., 0], st.dones[..., 0].bool(), final_values[:, 1])
    ratio = ((1.0 - self.gamma) * discounted).clamp_min(1e-6)  # J_c (-60 dB 아래는 같은 값으로 본다)
    excess_db = 10.0 * torch.log10(ratio) - 10.0 * torch.log10(limits)
    cap = self._lambda_cap()
    if cap > 0:
      self.pid.update(excess_db, torch.zeros_like(excess_db))
      # 커리큘럼 상한. 적분항도 함께 묶어 상한이 오를 때 λ가 튀지 않게 한다.
      self.pid.integral.clamp_(max=cap)
      self.pid.multipliers.clamp_(max=cap)
    else:  # 소음 제약을 켜기 전: λ = 0
      self.pid.integral.zero_()
      self.pid.multipliers.zero_()
    self.iteration += 1
    target_min = self.on_iteration(self.iteration) if self.on_iteration is not None else float("nan")
    self.constraint_metrics = {
      "target_min_db": target_min,  # 목표 범위 커리큘럼의 현재 가장 엄격한 목표
      "target_progress": float(self.target_progress),  # 게이트를 통과한 목표 ramp 진행 iteration 수
      "noise_cost": st.raw_costs.mean().item(),
      "noise_excess_db": excess_db.mean().item(),  # L_robot − 목표 (환경 평균)
      "noise_violation_db": excess_db.clamp_min(0).mean().item(),
      "noise_satisfied_frac": (excess_db <= 0).float().mean().item(),
      "lagrange_multiplier": multipliers.mean().item(),
      "lagrange_multiplier_max": multipliers.max().item(),
      "lambda_cap": cap,
      "actor_frozen": float(self._actor_frozen()),
    }

  def _actor_frozen(self) -> bool:
    return self.curriculum is not None and self.curriculum.actor_frozen(self.iteration)

  def _lambda_cap(self) -> float:
    if self.curriculum is None:
      return self.pid.cfg.max_multiplier
    return min(self.curriculum.lambda_cap(self.iteration), self.pid.cfg.max_multiplier)

  def _value_loss(self, predicted, old, target):
    loss = (predicted - target).square()
    if self.use_clipped_value_loss:
      clipped = old + (predicted - old).clamp(-self.clip_param, self.clip_param)
      loss = torch.maximum(loss, (clipped - target).square())
    return loss.mean(dim=0)  # [reward critic loss, cost critic loss]

  def update(self):
    totals = {"reward_value": 0.0, "cost_value": 0.0, "surrogate": 0.0, "entropy": 0.0, "anchor_mse": 0.0}
    updates = 0
    # compute_returns가 iteration을 이미 올렸으므로 이번 업데이트는 iteration − 1 기준이다.
    frozen = self.curriculum is not None and self.curriculum.actor_frozen(self.iteration - 1)
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
      advantage = batch.advantages.squeeze(-1)
      if self.normalize_advantage_per_mini_batch:
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      entropy = self.actor.output_entropy.mean()
      values = self.critic(batch.observations)

      if not frozen and self.desired_kl is not None and self.schedule == "adaptive":
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
      if self.anchor_actor is not None:
        with torch.no_grad():
          base_mean = self.anchor_actor(batch.observations)
        anchor = (self.actor.output_mean - base_mean).square().sum(-1).mean()
      else:
        anchor = torch.zeros((), device=self.device)
      reward_loss, cost_loss = self._value_loss(values, batch.values, batch.returns)
      value_loss = self.value_loss_coef * reward_loss + self.cost_value_loss_coef * cost_loss
      # critic 워밍업: actor 손실을 빼서 actor 가중치를 고정한다 (gradient 없음 → optimizer가 건너뜀).
      loss = value_loss if frozen else (surrogate + value_loss - self.entropy_coef * entropy
                                        + self.anchor_coef * anchor)
      if not torch.isfinite(loss):
        raise FloatingPointError("CNCP loss is not finite.")
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if not frozen:
        self._clamp_std()
      for key, value in zip(totals, (reward_loss, cost_loss, surrogate, entropy, anchor)):
        totals[key] += value.item()
      updates += 1
    self.storage.clear()
    return {**{key: value / updates for key, value in totals.items()}, **self.constraint_metrics}

  def save(self):
    state = super().save()
    state["cncp_pid"] = self.pid.state_dict()
    state["cncp_learning_rate"] = self.learning_rate
    state["cncp_iteration"] = self.iteration
    state["cncp_target_progress"] = self.target_progress
    if self.anchor_actor is not None:
      state["cncp_anchor_state_dict"] = self.anchor_actor.state_dict()
    return state

  def load(self, loaded_dict, load_cfg=None, strict=True):
    resume = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("optimizer", False):
      # 환경 수가 다르면 PID 대응이 깨지므로 strict shape 검사로 중단한다.
      self.pid.load_state_dict(loaded_dict["cncp_pid"], strict=True)
      self.learning_rate = loaded_dict["cncp_learning_rate"]
      self.iteration = loaded_dict.get("cncp_iteration", 0)
      self.target_progress = loaded_dict.get("cncp_target_progress", max(0, self.iteration - (
        self.curriculum.noise_start_iteration if self.curriculum is not None else 0)))
      if "cncp_anchor_state_dict" in loaded_dict:
        self.set_anchor(loaded_dict["cncp_anchor_state_dict"])
      self.episode_cost.reset()  # 재개 시 환경은 새 에피소드에서 시작한다.
    return resume
