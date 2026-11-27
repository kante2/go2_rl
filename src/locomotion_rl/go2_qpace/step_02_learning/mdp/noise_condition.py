"""목표 소음 dB 조건 (QuietPaw 식 (4)의 ε 자리).

학습: 환경 i마다 목표 L_i 를 [min_db, max_db]에 균등 배정하고 학습 내내 고정한다.
배치: target_db 하나를 지정한다 (방마다 다른 값). 실시간 입력은 없다.

정책·두 critic 입력:  (L − min_db)/(max_db − min_db) ∈ [0, 1]   (0 = 가장 조용, 1 = 가장 느슨)
학습 내부 한도:       ε = 10^((L − reference_db)/10)            (기준 파워 대비 허용 파워 비율)
dB 비교 L_robot ≤ L 은 에너지 비교 J_c ≤ ε 와 같다 (로그는 단조 함수).
"""

from dataclasses import dataclass

import torch


@dataclass
class NoiseConditionCfg:
  # 단위: 로봇 전체 착지 소음 파워 [dB re FS²/s, A-가중, 미보정 상대값].
  # 기본 범위는 측정값 기준이다 (2026-11-27, 소음 모델 go2_noise_aware 8d4270d, 탈락 접촉 포함):
  # baseline 보행(model_34997, actor 고정, 2048 env, 60 iter)의 로봇 소음 파워 −50.1 dB
  # (착지 초당 6.1회 + 탈락 접촉 5.0회, 탈락 접촉의 에너지 몫 6%).
  # baseline 대비 −10 ~ +5 dB를 학습 범위로, baseline −2.5 dB를 비용 정규화 기준으로 둔다.
  min_db: float = -60.0  # 가장 엄격한 목표 (baseline보다 약 10 dB 조용)
  max_db: float = -45.0  # 가장 느슨한 목표 (baseline보다 약 5 dB 시끄러움, 빠른 속도용)
  reference_db: float = -52.5  # 비용 정규화 기준. c_t = E_t / (Δt · 10^(reference_db/10))
  target_db: float | None = None  # None: 환경별 균등 배정 (학습), 숫자: 모든 환경 같은 목표 (play·배치)

  def __post_init__(self):
    if not self.min_db < self.max_db:
      raise ValueError("Require min_db < max_db.")
    if self.target_db is not None and not self.min_db <= self.target_db <= self.max_db:
      raise ValueError(f"target_db must be in [{self.min_db}, {self.max_db}] (trained range).")

  @property
  def reference_power(self) -> float:
    return 10.0 ** (self.reference_db / 10.0)


class NoiseCondition:
  def __init__(self, cfg: NoiseConditionCfg, num_envs: int, device: str):
    cfg.__post_init__()  # CLI 수정 이후에도 범위를 검사한다.
    self.cfg = cfg
    # 환경이 하나면 중간 목표. reset 때 재배정하지 않으므로 환경별 λ와 목표의 대응이 유지된다.
    self.target_db = (
      torch.linspace(cfg.min_db, cfg.max_db, num_envs, device=device)
      if num_envs > 1 else torch.full((1,), 0.5 * (cfg.min_db + cfg.max_db), device=device)
    )
    if cfg.target_db is not None:
      self.target_db.fill_(cfg.target_db)

  @property
  def normalized(self):
    return (self.target_db - self.cfg.min_db) / (self.cfg.max_db - self.cfg.min_db)

  @property
  def cost_limit(self):
    """학습 내부 한도 ε (기준 파워 대비 비율)."""
    return 10.0 ** ((self.target_db - self.cfg.reference_db) / 10.0)

  def set_target_db(self, value: float):
    if not self.cfg.min_db <= value <= self.cfg.max_db:
      raise ValueError(f"target_db must be in [{self.cfg.min_db}, {self.cfg.max_db}].")
    self.target_db.fill_(value)


def noise_condition(env):
  # Actor와 두 critic의 마지막 입력: 정규화한 목표 dB.
  return env.noise_condition.normalized.unsqueeze(-1)
