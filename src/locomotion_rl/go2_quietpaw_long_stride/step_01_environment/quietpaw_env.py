"""mjlab 환경에 substep 착지 측정과 reset 전 비용·관측 전달을 추가한다.

step 순서는 mjlab 1.2.0의 ManagerBasedRlEnv를 따른다. 원본과 달리 substep
시작의 forward()로 충돌 직전 발 속도를 읽는다. 정확한 측정을 위한 추가 계산이다.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv

from .base_env import QuietPawEnvCfg
from ..step_02_learning.mdp.costs import LandingMeasurements, noise_cost
from ..step_02_learning.mdp.curriculums import gait_reward_scale
from ..step_02_learning.mdp.noise_condition import NoiseCondition
from ..step_02_learning.stride import apply_stride_settings
from ..step_02_learning.mdp.stride_metrics import StrideMeasurements


class QuietPawEnv(ManagerBasedRlEnv):
  cfg: QuietPawEnvCfg

  def __init__(self, cfg, *args, **kwargs):
    # CLI와 체크포인트 값을 manager가 설정을 읽기 전에 반영한다.
    apply_stride_settings(cfg)
    super().__init__(cfg, *args, **kwargs)

  def load_managers(self):
    # ObservationManager는 생성 중 관측 함수를 호출한다. 먼저 조건을 준비한다.
    self.cfg.noise_cost.__post_init__()
    for name in ("actor", "critic"):
      group = self.cfg.observations[name]
      if list(group.terms)[-1] != "noise_condition" or not group.concatenate_terms:
        raise ValueError("QuietPaw observations require noise_condition as the last scalar.")
      for term in group.terms.values():
        history = group.history_length if group.history_length is not None else term.history_length
        if history not in (0, 1) or term.delay_max_lag > 0:
          raise ValueError("QuietPaw currently requires observations without stacked history or delay.")
    gait = self.cfg.gait_curriculum
    if gait.width_iterations <= 0 or gait.steps_per_iteration <= 0:
      raise ValueError("Gait curriculum width and rollout length must be positive.")
    if not 0 <= gait.minimum_scale <= 1:
      raise ValueError("Gait minimum_scale must be in [0, 1].")
    self.noise_condition = NoiseCondition(self.cfg.noise_condition, self.num_envs, self.device)
    self.foot_site_ids, _ = self.scene["robot"].find_sites(
      ["FR", "FL", "RR", "RL"], preserve_order=True
    )
    self.landing = LandingMeasurements(self.num_envs, 4, self.device)
    self.stride_measurements = StrideMeasurements(self.num_envs, 4, self.device)
    super().load_managers()

  def _reset_idx(self, env_ids=None):
    super()._reset_idx(env_ids)
    self.landing.reset(env_ids)
    self.stride_measurements.reset(env_ids)

  def step(self, action):
    self.action_manager.process_action(action.to(self.device))
    self.landing.begin_step()
    self.stride_measurements.begin_step()
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
      self.landing.sample(
        sensor.data.force, pre_velocity, self.cfg.noise_cost.contact_force_threshold
      )
      self.stride_measurements.sample(
        robot.data.site_pos_w[:, self.foot_site_ids], self.landing.touchdown,
        self.command_manager.get_command("twist"), self.physics_dt,
      )

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # 비용은 reset 전 물리 상태에서 계산한다. reward와 분리하며 dt를 곱하지 않는다.
    costs = noise_cost(
      self.landing.peak_forces, self.landing.impact_speed_squared, self.cfg.noise_cost
    )
    costs = torch.where(
      self.reset_terminated,
      costs.clamp_min(self.cfg.noise_cost.failure_cost),
      costs,
    )
    # reset 후 속도는 새 에피소드의 값이다. 평가용 추종 오차도 여기서 보존한다.
    command = self.command_manager.get_command("twist")
    tracking_error = torch.linalg.vector_norm(
      robot.data.root_link_lin_vel_b[:, :2] - command[:, :2], dim=-1
    )
    terminal_obs = None
    if self.reset_time_outs.any() and not self.cfg.is_finite_horizon:
      # timeout의 bootstrap은 reset 후 상태가 아니라 종료 직전 상태를 사용한다.
      self.sim.forward()
      self.sim.sense()
      # compute()의 기본값은 이전 스텝의 관측 cache를 반환한다. Critic 그룹을
      # 직접 다시 계산해야 실제 terminal state를 얻는다. history는 최대 1이다.
      terminal_obs = {
        "critic": self.observation_manager.compute_group("critic", update_history=True).clone()
      }

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
    self.extras["stride_metrics"] = {
      "distance_sum": self.stride_measurements.distance_sum.clone(),
      "period_sum": self.stride_measurements.period_sum.clone(),
      "count": self.stride_measurements.count.clone(),
    }
    self.extras["terminal_observations"] = terminal_obs
    # reset이 log 딕셔너리를 새로 만들 수 있으므로 진단값은 마지막에 기록한다.
    self.extras.setdefault("log", {}).update({
      "QuietPaw/noise_cost": costs.mean(),
      "QuietPaw/cost_limit": self.noise_condition.cost_limit.mean(),
      "QuietPaw/peak_force": self.landing.peak_forces.mean(),
      "QuietPaw/impact_speed_squared": self.landing.impact_speed_squared.mean(),
      "QuietPaw/gait_scale": gait_reward_scale(self),
      **self.stride_measurements.logs(),
    })
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
