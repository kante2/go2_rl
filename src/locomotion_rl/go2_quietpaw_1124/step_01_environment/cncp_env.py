"""CNCP 학습 환경: mjlab step에 substep 착지 측정, 식 (1) 비용, reset 전 관측 전달을 더한다.

step 순서는 mjlab 1.2.0의 ManagerBasedRlEnv를 따른다. 원본과 달리 substep 시작의
forward()로 충돌 직전 발 속도를 읽는다.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv

from .base_env import QuietPawEnvCfg
from ..step_02_learning.mdp.costs import LandingMeasurements, noise_cost
from ..step_02_learning.mdp.curriculums import gait_reward_scale
from ..step_02_learning.mdp.noise_condition import NoiseCondition


class QuietPawEnv(ManagerBasedRlEnv):
  cfg: QuietPawEnvCfg

  def load_managers(self):
    # ObservationManager는 생성 중 관측 함수를 호출한다. 조건을 먼저 준비한다.
    self.cfg.noise_cost.__post_init__()
    for name in ("actor", "critic"):
      group = self.cfg.observations[name]
      if list(group.terms)[-1] != "noise_condition" or not group.concatenate_terms:
        raise ValueError("CNCP observations require noise_condition (ε) as the last scalar.")
      for term in group.terms.values():
        history = group.history_length if group.history_length is not None else term.history_length
        if history not in (0, 1) or term.delay_max_lag > 0:
          raise ValueError("CNCP currently requires observations without stacked history or delay.")
    gait = self.cfg.gait_curriculum
    if gait.width_iterations <= 0 or gait.steps_per_iteration <= 0:
      raise ValueError("Gait curriculum width and rollout length must be positive.")
    if not 0 <= gait.minimum_scale <= 1:
      raise ValueError("Gait minimum_scale must be in [0, 1].")
    self.noise_condition = NoiseCondition(self.cfg.noise_condition, self.num_envs, self.device)
    self.landing = LandingMeasurements(self.num_envs, 4, self.device)
    super().load_managers()

    # 접촉 센서의 발 순서는 엔티티의 geom 순서(mjlab 1.2)다. 속도 site를 같은 순서로 맞춘다.
    robot = self.scene["robot"]
    sensor = self.scene["feet_ground_contact"]
    _, geoms = robot.find_geoms(sensor.cfg.primary.pattern)
    feet = [name.removesuffix("_foot_collision") for name in geoms]
    if len(feet) != 4 or set(feet) != {"FL", "FR", "RL", "RR"}:
      raise ValueError(f"Expected one contact channel per Go2 foot, got {geoms}")
    self.foot_names = feet
    self.foot_site_ids, _ = robot.find_sites(feet, preserve_order=True)
    names = self.reward_manager.active_terms
    self._termination_reward_idx = names.index("is_terminated") if "is_terminated" in names else None

  def _step_noise_cost(self):
    # 식 (1) 정규화 비용 ∈ [0, 1]. 실패 종료 처리는 cost critic bootstrap(cncp_ppo.py)에서 한다.
    return noise_cost(self.landing.peak_forces, self.landing.impact_speed_squared, self.cfg.noise_cost)

  def _shape_reward(self, reward):
    if not self.cfg.only_positive_rewards:
      return reward
    termination = 0.0
    if self._termination_reward_idx is not None:
      # RewardManager._step_reward는 dt를 곱하기 전 값이다.
      termination = self.reward_manager._step_reward[:, self._termination_reward_idx] * self.step_dt
    return (reward - termination).clamp_min(0.0) + termination

  def _reset_idx(self, env_ids=None):
    super()._reset_idx(env_ids)
    self.landing.reset(env_ids)

  def step(self, action):
    self.action_manager.process_action(action.to(self.device))
    self.landing.begin_step()
    robot = self.scene["robot"]
    sensor = self.scene["feet_ground_contact"]

    for _ in range(self.cfg.decimation):
      # 충돌 처리 후 속도로 착지 속도를 대신하지 않도록 미리 복사한다.
      self.sim.forward()
      pre_velocity = robot.data.site_lin_vel_w[:, self.foot_site_ids, 2].clone()
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.landing.sample(sensor.data.force, pre_velocity, self.cfg.noise_cost.contact_force_threshold)

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self._shape_reward(self.reward_manager.compute(dt=self.step_dt))
    self.metrics_manager.compute()

    # 비용은 reset 전 물리 상태에서 계산한다. reward와 분리하며 dt를 곱하지 않는다.
    costs = self._step_noise_cost()
    command = self.command_manager.get_command("twist")
    tracking_error = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2] - command[:, :2], dim=-1)
    terminal_obs = None
    if self.reset_time_outs.any() and not self.cfg.is_finite_horizon:
      # timeout bootstrap은 reset 후가 아니라 종료 직전 상태의 critic 관측을 쓴다.
      self.sim.forward()
      self.sim.sense()
      terminal_obs = {"critic": self.observation_manager.compute_group("critic", update_history=True).clone()}

    reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
      self._reset_idx(reset_ids)
      self.scene.write_data_to_sim()
    self.sim.forward()
    self.command_manager.compute(dt=self.step_dt)
    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    self.extras["noise_cost"] = costs
    self.extras["true_terminations"] = self.reset_terminated.clone()
    self.extras["quietpaw_metrics"] = {
      "tracking_error": tracking_error,
      "peak_force": self.landing.peak_forces.mean(dim=-1).clone(),
      "impact_speed_squared": self.landing.impact_speed_squared.mean(dim=-1).clone(),
    }
    self.extras["cost_limit"] = self.noise_condition.cost_limit.clone()
    self.extras["terminal_observations"] = terminal_obs
    # reset이 log 딕셔너리를 새로 만들 수 있으므로 진단값은 마지막에 기록한다.
    self.extras.setdefault("log", {}).update({
      "CNCP/noise_cost": costs.mean(),
      "CNCP/epsilon": self.noise_condition.cost_limit.mean(),
      "CNCP/peak_force": self.landing.peak_forces.mean(),
      "CNCP/impact_speed_squared": self.landing.impact_speed_squared.mean(),
      "CNCP/gait_scale": gait_reward_scale(self),
    })
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
