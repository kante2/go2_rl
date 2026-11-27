"""QPace 커리큘럼: 단계 학습(사전학습 보행 → critic → 소음 제약), 보행 리듬 보상 감소, 속도 명령·지형 단계."""

from dataclasses import dataclass

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from . import mdp


@dataclass
class GaitCurriculumCfg:
  # 논문 IV-A: 보행 관련 보상을 2k 지점에서 sigmoid로 감소 (2단계 커리큘럼). 폭은 구현 선택.
  center_iteration: float = 2000.0
  width_iterations: float = 250.0
  minimum_scale: float = 0.0
  steps_per_iteration: int = 24  # Runner가 실제 rollout 길이로 맞춘다.
  enabled: bool = True


@dataclass
class QPaceCurriculumCfg:
  """QPace 단계 학습: 사전학습 보행에서 시작 → critic 맞추기 → 소음 제약을 서서히 켬.

  iteration은 이번 학습의 PPO 업데이트 횟수다 (체크포인트 재개 시 이어서 센다).
    [0, critic_warmup)                       actor 고정, 두 critic만 학습. λ = 0
    [critic_warmup, noise_start)             actor·critic 학습, λ = 0 (소음 제약 없이 보행 적응)
    [noise_start, noise_start + ramp)        λ 갱신 시작, λ 상한을 0 → lambda_max로 선형 증가
    이후                                     λ 상한 = lambda_max
  처음부터 학습(init_checkpoint="")하면 noise_start를 보행이 자리 잡는 시점(예: 2000)으로 늦춘다.
  """

  # 사전학습 보행 체크포인트 (baseline 47차원 actor/74차원 critic). 저장소 루트 기준 경로. ""이면 처음부터.
  # 기본값은 실로봇에 배포된 baseline 정책이다.
  init_checkpoint: str = "src/go2_autonomous_driving/model_pt/model_10000.pt"
  critic_warmup_iterations: int = 100
  noise_start_iteration: int = 300
  lambda_ramp_iterations: int = 500
  lambda_max: float = 20.0  # 소음 우선 정도. 낮추면 속도 추종 우선 (PID max_multiplier보다 클 수 없다)

  def __post_init__(self):
    if min(self.critic_warmup_iterations, self.noise_start_iteration, self.lambda_ramp_iterations) < 0:
      raise ValueError("Curriculum iterations must be nonnegative.")
    if self.noise_start_iteration < self.critic_warmup_iterations:
      raise ValueError("noise_start_iteration must not precede critic warmup end.")
    if self.lambda_max < 0:
      raise ValueError("lambda_max must be nonnegative.")

  def actor_frozen(self, iteration: int) -> bool:
    return iteration < self.critic_warmup_iterations

  def noise_active(self, iteration: int) -> bool:
    return iteration >= self.noise_start_iteration

  def lambda_cap(self, iteration: int) -> float:
    if not self.noise_active(iteration):
      return 0.0
    if self.lambda_ramp_iterations == 0:
      return self.lambda_max
    progress = (iteration - self.noise_start_iteration + 1) / self.lambda_ramp_iterations
    return self.lambda_max * min(1.0, progress)


# 속도 명령 단계. 소음 모델은 명령 vx 0.7 m/s까지의 실측 보행으로 학습됐으므로 그 범위에서 먼저 학습한다.
# step은 모든 환경이 함께 진행한 정책 스텝 수 (iteration × 24). 속도 단위 x/y [m/s], z [rad/s].
VELOCITY_STAGES = [
  {"step": 0, "lin_vel_x": (-0.5, 0.7), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
  {"step": 4000 * 24, "lin_vel_x": (-0.7, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.5, 1.5)},
]


def make_curriculum() -> dict[str, CurriculumTermCfg]:
  # command_name은 commands.py의 키와 같아야 한다. 여기서는 twist 속도 명령을 조정한다.
  return {
    "terrain_levels": CurriculumTermCfg(
      # 이동 거리와 명령 속도를 보고 지형 난이도를 조절한다. Flat에서는 이 항목을 제거한다.
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": VELOCITY_STAGES,
      },
    ),
  }
