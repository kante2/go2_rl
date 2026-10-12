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
      raise ValueError("QuietPaw supports one training process/GPU; use --gpu-ids '[0]'.")
    env.unwrapped.cfg.gait_curriculum.steps_per_iteration = train_cfg["num_steps_per_env"]
    super().__init__(env, deepcopy(train_cfg), log_dir, device)

  def _environment_signature(self):
    cfg = self.env.unwrapped.cfg
    return {
      "noise_cost": asdict(cfg.noise_cost),
      "noise_condition": asdict(cfg.noise_condition),
      "gait_curriculum": asdict(cfg.gait_curriculum),
      "num_envs": self.env.num_envs,
      "step_dt": self.env.unwrapped.step_dt,
    }

  def save(self, path, infos=None):
    infos = {
      **(infos or {}),
      "quietpaw": self._environment_signature(),
      "quietpaw_agent": deepcopy(self.cfg),  # 평가 시 CLI로 변경한 네트워크 크기도 복원.
    }
    super().save(path, infos)
    policy_dir = str(Path(path).parent)
    self.export_policy_to_onnx(policy_dir)
    metadata = get_base_metadata(self.env.unwrapped, "quietpaw")
    cfg = self.env.unwrapped.cfg.noise_condition
    metadata.update({
      "quietpaw_condition": "last observation: cost_limit; lower means quieter",
      "quietpaw_cost_limit_min": cfg.min_cost_limit,
      "quietpaw_cost_limit_max": cfg.max_cost_limit,
      "quietpaw_cost_units": "mean per-policy-step normalized proxy; not dB",
      "policy_dt": self.env.unwrapped.step_dt,
    })
    attach_metadata_to_onnx(str(Path(policy_dir) / "policy.onnx"), metadata)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    loaded = torch.load(path, map_location=map_location or self.device, weights_only=False)
    if load_cfg is None or load_cfg.get("optimizer", False):
      signature = (loaded.get("infos") or {}).get("quietpaw")
      if signature != self._environment_signature():
        raise ValueError(
          "QuietPaw resume requires the same environment count, cost/condition settings, "
          "control period and gait curriculum. Use actor-only loading for evaluation."
        )
    result = super().load(path, load_cfg, strict, map_location or self.device)
    if load_cfg is None or load_cfg.get("iteration", False):
      # 기본 Runner가 저장하는 iter는 방금 끝난 반복의 0-based index이다.
      self.current_learning_iteration += 1
    return result
