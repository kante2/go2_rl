"""QPace 커리큘럼: 단계 학습(사전학습 보행 → critic → 소음 제약), 보행 리듬 보상 감소, 속도 명령·지형 단계."""

from dataclasses import dataclass

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from . import mdp


@dataclass
class GaitCurriculumCfg:
  # 논문 IV-A: 보행 관련 보상을 2k 지점에서 sigmoid로 감소 (2단계 커리큘럼, 처음부터 학습할 때).
  # QPace 는 완성된 보행(model_34997)에서 시작하므로 리듬 보상을 끝까지 유지한다 (minimum_scale = 1).
  # 2026-11-27 학습: 0 으로 줄이자 리듬 보상이 사라진 2500 iter 부터 trot 이 풀리고 발 떨기·끌기로 붕괴.
  center_iteration: float = 2000.0
  width_iterations: float = 250.0
  minimum_scale: float = 1.0
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
  # 기본값 model_34997: 실로봇 기본 보행이자 소음 모델 학습 녹음(real_sim*)에 쓴 정책 (kante_.md, final_cmd.txt).
  # 소음 모델이 본 보행에서 시작하므로 초기 예측이 가장 믿을 만하다. 이전 기본값은 model_10000.
  init_checkpoint: str = "src/go2_autonomous_driving/model_pt/model_34997.pt"
  critic_warmup_iterations: int = 100
  noise_start_iteration: int = 300
  # 2026-11-27 확인 학습: ramp 500·상한 20 에서는 목표 조건을 배우기 전에 모든 환경이 "거의 안 걷기"로 무너졌다.
  lambda_ramp_iterations: int = 2000
  lambda_max: float = 5.0  # 소음 우선 정도. 낮추면 속도 추종 우선 (PID max_multiplier보다 클 수 없다)
  # 목표 dB 범위 커리큘럼: 가장 엄격한 목표를 baseline 근처(target_start_min_db)에서 시작해
  # noise_start 부터 target_ramp_iterations 동안 NoiseConditionCfg.min_db 까지 선형으로 내린다.
  # 환경별 목표 = 현재 하한 + u_i·(max_db − 현재 하한), u_i ∈ [0, 1] 고정. 정책 입력 정규화는 전체 범위 기준이다.
  # None 이면 처음부터 전체 범위.
  target_start_min_db: float | None = -52.0  # baseline(model_34997) −50.1 dB 보다 약 2 dB 조용
  target_ramp_iterations: int = 3000
  # 목표 커리큘럼 게이트: rollout 평균 xy 속도 추종 오차 [m/s] (QPace/tracking_error_mps) 가 이 값보다 크면
  # 그 iteration 은 목표를 더 내리지 않는다 (ramp 를 멈추고 지금 목표에서 보행을 회복할 시간을 준다).
  # baseline(model_34997) 0.166 m/s. 2026-11-27 학습이 무너지기 시작한 2300 iter 는 약 0.19 m/s
  # (Metrics/twist/error_vel_xy 0.50 ÷ 2.65; 그 지표는 m/s 가 아니다). None 이면 게이트 없이 iteration 수대로 내린다.
  target_gate_tracking_error: float | None = 0.20

  def __post_init__(self):
    if min(self.critic_warmup_iterations, self.noise_start_iteration, self.lambda_ramp_iterations,
           self.target_ramp_iterations) < 0:
      raise ValueError("Curriculum iterations must be nonnegative.")
    if self.noise_start_iteration < self.critic_warmup_iterations:
      raise ValueError("noise_start_iteration must not precede critic warmup end.")
    if self.lambda_max < 0:
      raise ValueError("lambda_max must be nonnegative.")
    if self.target_gate_tracking_error is not None and not self.target_gate_tracking_error > 0:
      raise ValueError("target_gate_tracking_error must be positive or None.")

  def target_gate_open(self, tracking_error: float) -> bool:
    """이번 rollout 의 추종 오차로 목표 ramp 를 한 칸 진행해도 되는지. 값이 없으면(nan) 진행한다."""
    gate = self.target_gate_tracking_error
    return gate is None or not tracking_error > gate

  def target_min_db(self, iteration: int, min_db: float, max_db: float) -> float:
    """이번 iteration 의 가장 엄격한 목표 [dB]. 게이트를 쓰면 iteration 대신 '게이트를 통과한 진행 iteration'
    (noise_start + 통과 횟수)을 넣는다 (runner.py)."""
    if self.target_start_min_db is None:
      return min_db
    start = min(max(self.target_start_min_db, min_db), max_db)
    if iteration < self.noise_start_iteration or self.target_ramp_iterations == 0:
      return start if iteration < self.noise_start_iteration else min_db
    progress = min(1.0, (iteration - self.noise_start_iteration) / self.target_ramp_iterations)
    return start + (min_db - start) * progress

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
# 넓히는 시점은 목표 ramp(300 + 3000 iter)가 끝나고 충분히 지난 뒤로 둔다 (소음 압박과 겹치지 않게).
# step은 모든 환경이 함께 진행한 정책 스텝 수 (iteration × 24). 속도 단위 x/y [m/s], z [rad/s].
VELOCITY_STAGES = [
  {"step": 0, "lin_vel_x": (-0.5, 0.7), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
  {"step": 5000 * 24, "lin_vel_x": (-0.7, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.5, 1.5)},
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
