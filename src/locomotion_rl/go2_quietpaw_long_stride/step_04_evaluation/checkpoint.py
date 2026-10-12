"""학습 때 사용한 모델 크기와 비용 척도를 평가에도 적용한다."""

from copy import deepcopy

import torch

from ..step_02_learning.costs import NoiseCostCfg
from ..step_02_learning.mdp.noise_condition import NoiseConditionCfg
from ..step_02_learning.stride import StrideCfg, apply_stride_settings


def load_evaluation_settings(env_cfg, path):
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  infos = checkpoint.get("infos") or {}
  if "quietpaw_agent" not in infos or "quietpaw" not in infos:
    raise ValueError("Expected a QuietPaw checkpoint with saved model and cost settings.")
  saved = infos["quietpaw"]
  if "stride" not in saved:
    raise ValueError("Expected a LongStride checkpoint with saved stride settings.")
  env_cfg.stride = StrideCfg(**saved["stride"])
  apply_stride_settings(env_cfg)
  env_cfg.noise_cost = NoiseCostCfg(**saved["noise_cost"])
  env_cfg.noise_condition = NoiseConditionCfg(**saved["noise_condition"])
  # Actor에 전달되는 ε의 범위를 학습과 같게 유지한다.
  return deepcopy(infos["quietpaw_agent"])
