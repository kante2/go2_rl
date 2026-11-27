"""학습 때 사용한 모델 크기와 비용 척도를 평가에도 적용한다."""

from copy import deepcopy

import torch

from ..step_02_learning.mdp.noise_condition import NoiseConditionCfg


def load_evaluation_settings(env_cfg, path):
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  infos = checkpoint.get("infos") or {}
  family = env_cfg.checkpoint_family
  if f"{family}_agent" not in infos or family not in infos:
    raise ValueError(f"Expected a {family} checkpoint with saved model and cost settings.")
  saved = infos[family]
  cost = dict(saved["noise_cost"])
  cost.setdefault("uncertainty_k", 0.0)  # 2026-11-27 이전 체크포인트: 불확실성 반영 없이 학습 (평균 소음)
  env_cfg.noise_cost = type(env_cfg.noise_cost)(**cost)
  env_cfg.noise_condition = NoiseConditionCfg(**saved["noise_condition"])
  # Actor에 전달되는 ε의 범위를 학습과 같게 유지한다.
  return deepcopy(infos[f"{family}_agent"])
