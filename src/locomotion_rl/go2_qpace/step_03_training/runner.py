"""기존 mjlab 학습 루프를 사용하며 QuietPaw 상태와 ONNX 메타데이터를 보존한다."""

import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata


class QuietPawOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    # PID의 환경별 상태를 여러 프로세스로 분산 저장하는 기능은 아직 지원하지 않는다.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("CNCP supports one training process/GPU; use --gpu-ids '[0]'.")
    env.unwrapped.cfg.gait_curriculum.steps_per_iteration = train_cfg["num_steps_per_env"]
    super().__init__(env, deepcopy(train_cfg), log_dir, device)
    self.alg.failure_cost = env.unwrapped.failure_cost
    self.alg.curriculum = env.unwrapped.cfg.qpace_curriculum
    self.alg.on_iteration = self._update_target_range
    self._update_target_range(self.alg.iteration, advance=False)

  def _update_target_range(self, iteration: int, advance: bool = True) -> float:
    """목표 dB 범위 커리큘럼: 다음 rollout 의 환경별 목표를 정한다. 현재 가장 엄격한 목표 [dB] 반환.

    게이트: noise_start 이후에는 방금 끝난 rollout 의 평균 속도 추종 오차가 한계 이하일 때만 ramp 를 한 칸
    진행한다 (alg.target_progress). 보행이 나빠지면 목표가 그 자리에서 멈춰 회복할 시간을 준다.
    """
    env = self.env.unwrapped
    cond = env.cfg.noise_condition
    cur = env.cfg.qpace_curriculum
    tracking = env.pop_tracking_error() if advance else float("nan")
    if advance and iteration > cur.noise_start_iteration and cur.target_gate_open(tracking):
      self.alg.target_progress += 1
    self.target_tracking_error = tracking
    effective = (iteration if iteration < cur.noise_start_iteration
                 else cur.noise_start_iteration + self.alg.target_progress)
    value = cur.target_min_db(effective, cond.min_db, cond.max_db)
    env.noise_condition.set_min_db(value)
    return env.noise_condition.current_min_db

  def warm_start(self):
    """QPaceCurriculumCfg.init_checkpoint의 baseline 보행으로 actor·critic을 초기화한다 (새 학습일 때만)."""
    from .warm_start import warm_start_from_baseline
    path = self.env.unwrapped.cfg.qpace_curriculum.init_checkpoint
    if not path:
      return None
    info = warm_start_from_baseline(self.alg.actor, self.alg.critic, path)
    self.alg.set_anchor()  # C 방식: 방금 불러온 baseline actor 의 고정 복사본 (anchor_coef = 0 이면 없음)
    print(f"[INFO] QPace warm start from {info['path']} (iter {info['iteration']}, "
          f"actor {info['actor_inputs']}+1, critic {info['critic_inputs']}+1 inputs)")
    return info

  def _environment_signature(self):
    cfg = self.env.unwrapped.cfg
    return {
      "noise_cost": asdict(cfg.noise_cost),
      "noise_condition": asdict(cfg.noise_condition),
      "gait_curriculum": asdict(cfg.gait_curriculum),
      "only_positive_rewards": cfg.only_positive_rewards,
      "qpace_curriculum": asdict(cfg.qpace_curriculum),
      "num_envs": self.env.num_envs,
      "step_dt": self.env.unwrapped.step_dt,
    }

  def save(self, path, infos=None):
    family = self.env.unwrapped.cfg.checkpoint_family
    infos = {
      **(infos or {}),
      family: self._environment_signature(),
      f"{family}_agent": deepcopy(self.cfg),  # 평가 시 CLI로 변경한 네트워크 크기도 복원.
    }
    super().save(path, infos)
    policy_dir = str(Path(path).parent)
    self.export_policy_to_onnx(policy_dir)
    metadata = get_base_metadata(self.env.unwrapped, family)
    cfg = self.env.unwrapped.cfg.noise_condition
    metadata.update({
      f"{family}_condition": "last observation: (target_db - min_db)/(max_db - min_db); lower means quieter",
      f"{family}_target_db_min": cfg.min_db,
      f"{family}_target_db_max": cfg.max_db,
      f"{family}_reference_db": cfg.reference_db,
      f"{family}_cost_units": self.env.unwrapped.cfg.cost_units,
      "policy_dt": self.env.unwrapped.step_dt,
    })
    attach_metadata_to_onnx(str(Path(policy_dir) / "policy.onnx"), metadata)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    loaded = torch.load(path, map_location=map_location or self.device, weights_only=False)
    family = self.env.unwrapped.cfg.checkpoint_family
    infos = loaded.get("infos") or {}
    signature = infos.get(family)
    # Older actor-only QuietPaw exports may predate configuration metadata.
    if signature is None and (any(key.endswith("_agent") for key in infos)):
      raise ValueError(f"Expected a {family} checkpoint.")
    if load_cfg is None or load_cfg.get("optimizer", False):
      if signature != self._environment_signature():
        raise ValueError(
          "QuietPaw resume requires the same environment count, cost/condition settings, "
          "control period and gait curriculum. Use actor-only loading for evaluation."
        )
    result = super().load(path, load_cfg, strict, map_location or self.device)
    if load_cfg is None or load_cfg.get("iteration", False):
      # 기본 Runner가 저장하는 iter는 방금 끝난 반복의 0-based index이다.
      self.current_learning_iteration += 1
    self._update_target_range(self.alg.iteration, advance=False)  # 재개 시 저장된 진행 기준 목표 범위
    return result
