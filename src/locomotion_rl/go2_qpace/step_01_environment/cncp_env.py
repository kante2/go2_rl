"""QPace 학습 환경: mjlab step에 소음 예측 모델(DiffImpact)을 연결한다.

정책 한 스텝 = 물리 substep 4번 (각 5 ms). substep마다 착지 추적기에 상태를 넘기고,
스텝 끝에 이번 스텝에 확정된 착지들의 에너지를 더해 로봇 전체 착지 에너지 E_t 를 얻는다.

  c_t = E_t / (Δt · P_ref)       P_ref = 10^(reference_db/10)       (cost critic 학습 신호, ≥ 0)
  제약 (cncp_ppo.py):  10·log10((1−γ)·Σγᵗ c_t) + reference_db  ≤  목표 dB

step 순서는 mjlab 1.2.0의 ManagerBasedRlEnv를 따른다.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from .base_env import QuietPawEnvCfg
from ..step_02_learning.mdp.costs import LandingMeasurements
from ..step_02_learning.mdp.curriculums import gait_reward_scale
from ..step_02_learning.mdp.noise_condition import NoiseCondition
from ..step_02_learning.noise_model import DiffImpactNoiseModel, DiffImpactStepNoise, unitree_joint_order

UNITREE_FEET = ("FR", "FL", "RR", "RL")  # 소음 모델의 발·관절 순서


class QuietPawEnv(ManagerBasedRlEnv):
  cfg: QuietPawEnvCfg

  def load_managers(self):
    # ObservationManager는 생성 중 관측 함수를 호출한다. 조건을 먼저 준비한다.
    self.cfg.noise_cost.__post_init__()
    for name in ("actor", "critic"):
      group = self.cfg.observations[name]
      if list(group.terms)[-1] != "noise_condition" or not group.concatenate_terms:
        raise ValueError("QPace observations require noise_condition (target dB) as the last scalar.")
      for term in group.terms.values():
        history = group.history_length if group.history_length is not None else term.history_length
        if history not in (0, 1) or term.delay_max_lag > 0:
          raise ValueError("QPace currently requires observations without stacked history or delay.")
    gait = self.cfg.gait_curriculum
    if gait.width_iterations <= 0 or gait.steps_per_iteration <= 0:
      raise ValueError("Gait curriculum width and rollout length must be positive.")
    if not 0 <= gait.minimum_scale <= 1:
      raise ValueError("Gait minimum_scale must be in [0, 1].")
    if abs(self.physics_dt - 0.005) > 1e-9:
      raise ValueError("The noise model assumes a 5 ms physics step.")
    self.noise_condition = NoiseCondition(self.cfg.noise_condition, self.num_envs, self.device)
    self.landing = LandingMeasurements(self.num_envs, 4, self.device)
    super().load_managers()

    robot = self.scene["robot"]
    sensor = self.scene["feet_ground_contact"]
    # 접촉 센서의 발 순서는 엔티티의 geom 순서(mjlab 1.2)다. 진단용 속도 site를 같은 순서로 맞춘다.
    _, geoms = robot.find_geoms(sensor.cfg.primary.pattern)
    feet = [name.removesuffix("_foot_collision") for name in geoms]
    if len(feet) != 4 or set(feet) != set(UNITREE_FEET):
      raise ValueError(f"Expected one contact channel per Go2 foot, got {geoms}")
    self.foot_names = feet
    self.foot_site_ids, _ = robot.find_sites(feet, preserve_order=True)
    # 소음 모델 입력은 Unitree 순서 (FR, FL, RR, RL). 센서·site·관절·액추에이터 순서를 각각 맞춘다.
    self.sensor_to_unitree = [feet.index(f) for f in UNITREE_FEET]
    self.unitree_site_ids, _ = robot.find_sites(list(UNITREE_FEET), preserve_order=True)
    self.unitree_joint_ids = unitree_joint_order(list(robot.joint_names))
    self.unitree_actuator_ids = unitree_joint_order(list(robot.actuator_names))  # 액추에이터는 관절 종류별 순서
    cost_cfg = self.cfg.noise_cost
    self.step_noise = DiffImpactStepNoise(
      self.num_envs, self.device, model=DiffImpactNoiseModel(device=self.device),
      foot_floor_db=cost_cfg.foot_floor_db, count_rejected=cost_cfg.count_rejected_contacts,
      air=tuple(cost_cfg.touchdown_air_n), on_delta=cost_cfg.touchdown_on_delta,
      off_delta=cost_cfg.touchdown_off_delta, rise_delta=cost_cfg.touchdown_rise_delta,
      min_swing=cost_cfg.touchdown_min_swing_s, min_stance=cost_cfg.touchdown_min_stance_s,
    )
    names = self.reward_manager.active_terms
    self._termination_reward_idx = names.index("is_terminated") if "is_terminated" in names else None
    self._noise_out = None

  # ------------------------------------------------------------------ 소음 비용
  @property
  def failure_cost(self) -> float:
    """실패 뒤 매 스텝 이어진다고 보는 비용 = (max_db + margin) 파워의 기준 대비 비율."""
    cond, cost = self.cfg.noise_condition, self.cfg.noise_cost
    return 10.0 ** ((cond.max_db + cost.failure_margin_db - cond.reference_db) / 10.0)

  def _sample_noise_model(self, sensor):
    d = self.scene["robot"].data
    s = self.unitree_site_ids
    quat = d.root_link_quat_w
    rel_pos_w = d.site_pos_w[:, s] - d.root_link_pos_w[:, None]
    foot_pos_b = quat_apply_inverse(quat[:, None].expand(-1, 4, -1), rel_pos_w)
    force = torch.linalg.vector_norm(sensor.data.force[:, self.sensor_to_unitree], dim=-1)
    self.step_noise.sample(
      d.site_lin_vel_w[:, s] - d.root_link_lin_vel_w[:, None],  # 발 속도 − 몸통 속도 (월드)
      foot_pos_b,
      force,
      d.joint_pos[:, self.unitree_joint_ids],
      d.actuator_force[:, self.unitree_actuator_ids],
      quat,
      self.command_manager.get_command("twist"),
    )

  def _step_noise_cost(self):
    self._noise_out = self.step_noise.end_step()
    reference_power = self.cfg.noise_condition.reference_power
    return self._noise_out["energy"] / (self.step_dt * reference_power)

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
    if hasattr(self, "step_noise"):  # 첫 reset은 load_managers 중에 일어날 수 있다.
      self.step_noise.reset(env_ids if env_ids is not None else slice(None))

  # ------------------------------------------------------------------ step
  def step(self, action):
    self.action_manager.process_action(action.to(self.device))
    self.landing.begin_step()
    self.step_noise.begin_step()
    robot = self.scene["robot"]
    sensor = self.scene["feet_ground_contact"]

    for _ in range(self.cfg.decimation):
      # 충돌 처리 후 속도로 착지 속도를 대신하지 않도록 미리 복사한다 (진단값).
      self.sim.forward()
      pre_velocity = robot.data.site_lin_vel_w[:, self.foot_site_ids, 2].clone()
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.landing.sample(sensor.data.force, pre_velocity, self.cfg.noise_cost.contact_force_threshold)
      self._sample_noise_model(sensor)

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self._shape_reward(self.reward_manager.compute(dt=self.step_dt))
    self.metrics_manager.compute()

    # 비용은 reset 전에 계산한다. reward와 분리하며 dt를 곱하지 않는다.
    costs = self._step_noise_cost()
    noise = self._noise_out
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
      "noise_energy": noise["energy"].clone(),  # 로봇 전체 착지 에너지 [FS²] (이번 스텝)
      "touchdown": noise["touchdown"].clone(),
      "peak_force": self.landing.peak_forces.mean(dim=-1).clone(),
    }
    self.extras["cost_limit"] = self.noise_condition.cost_limit.clone()
    self.extras["target_db"] = self.noise_condition.target_db.clone()
    self.extras["terminal_observations"] = terminal_obs
    # 배치 평균 로봇 소음 파워 [dB re FS²/s]: 모든 환경의 평균 에너지 ÷ 스텝 시간.
    mean_power = noise["energy"].mean() / self.step_dt
    log = {
      "QPace/robot_power_db": 10.0 * torch.log10(mean_power.clamp_min(1e-30)),
      "QPace/target_db": self.noise_condition.target_db.mean(),
      "QPace/touchdowns_per_step": noise["log"]["touchdowns"] / self.num_envs,
      "QPace/touchdowns_per_s": noise["log"]["touchdowns"] / (self.num_envs * self.step_dt),
      "QPace/touchdown_step_db": noise["log"]["step_db"],
      "QPace/ensemble_db_std": noise["log"]["ensemble_db_std"],
      "QPace/floored_frac": noise["log"]["floored_frac"],  # 발별 하한에 걸린 접촉 비율 (계속 커지면 모델 속이기 의심)
      "QPace/contact_speed": noise["log"]["contact_speed"],  # 모든 접촉(착지 + 탈락) 평균 접촉 속도 [m/s]
      "QPace/contacts_per_s": (noise["log"]["touchdowns"] + noise["log"]["rejected"]) / (self.num_envs * self.step_dt),
      "QPace/rejected_frac": noise["log"]["rejected_frac"],  # 접촉 중 착지 규칙에서 빠진 비율 (커지면 두드리기)
      "QPace/rejected_energy_frac": noise["log"]["rejected_energy_frac"],  # 소음 에너지 중 탈락 접촉 몫
      "QPace/peak_force": self.landing.peak_forces.mean(),
      "QPace/gait_scale": gait_reward_scale(self),
    }
    for feature in ("foot_vz_pre", "body_vz_pre", "foot_vx_pre", "foot_vy_pre", "cmd_vx"):
      log[f"QPace/ood_frac/{feature}"] = noise["log"][f"ood_frac/{feature}"]
    # reset이 log 딕셔너리를 새로 만들 수 있으므로 진단값은 마지막에 기록한다.
    self.extras.setdefault("log", {}).update(log)
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
