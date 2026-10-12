"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from ..step_01_environment.quietpaw_env import QuietPawEnv
from .checkpoint import load_evaluation_settings
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  quietness: float = 0.5  # 0: 느슨한 소음 제약, 1: 가장 조용한 조건.
  agent: Literal["zero", "random", "trained"] = "trained"  # 영 행동/랜덤 행동/학습 모델 중 선택.
  checkpoint_file: str | None = None  # 로컬 .pt 경로. 학습 모델 모드에서 W&B 경로보다 우선한다.
  wandb_run_path: str | None = None  # 로컬 파일 미지정 시 내려받을 W&B 실행 경로.
  num_envs: int | None = None  # 평가할 병렬 환경 수. None이면 태스크 설정값(기본 1)을 사용한다.
  device: str | None = None  # cuda:0 또는 cpu 등. None이면 사용 가능한 CUDA를 우선한다.
  video: bool = False  # 학습된 모델의 평가 영상을 저장할지 여부.
  video_length: int = 200  # 기록할 환경 스텝 수; 기본 50 Hz에서 약 4초 분량.
  video_height: int | None = None  # 영상 높이 [pixel], None이면 viewer 기본 설정 유지.
  video_width: int | None = None  # 영상 너비 [pixel].
  camera: int | str | None = None  # 현재 run_play()에는 이 값을 적용하는 코드가 없다.
  viewer: Literal["auto", "native", "viser"] = "auto"  # 데스크톱 창/native 또는 웹/viser.
  no_terminations: bool = False  # True면 넘어짐을 포함한 모든 종료 조건을 비워 동작 관찰을 계속한다.
  """Disable all termination conditions when inspecting policy behavior."""


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)  # 관측 노이즈/외란 등을 끈 평가용 환경 설정.
  if not 0 <= cfg.quietness <= 1:
    raise ValueError("quietness must be in [0, 1].")
  agent_cfg = load_rl_cfg(task_id)  # 체크포인트와 같은 신경망 구조를 생성하기 위한 설정.

  DUMMY_MODE = cfg.agent in {"zero", "random"}  # 체크포인트 없이 환경만 확인하는 모드.
  TRAINED_MODE = not DUMMY_MODE  # 저장된 모델의 추론을 사용하는 모드이며, 학습 재개가 아니다.

  # Disable terminations if requested.
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  log_dir: Path | None = None  # 체크포인트 부모 폴더. 평가 영상도 이 폴더 아래에 저장한다.
  resume_path: Path | None = None  # 변수명은 resume지만 이 파일에서는 추론 모델을 불러올 경로이다.
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(  # 로컬 캐시를 재사용하거나 W&B에서 내려받는다.
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  runner_cfg = asdict(agent_cfg)
  if TRAINED_MODE:
    runner_cfg = load_evaluation_settings(env_cfg, resume_path)
  env_cfg.noise_condition.quietness = cfg.quietness

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None  # 녹화 시 이미지 배열 출력을 활성화.
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = QuietPawEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",  # 평가 영상 저장 디렉터리.
      step_trigger=lambda step: step == 0,  # 시작 시 한 번 녹화를 시작한다.
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg["clip_actions"])
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # 전체 환경의 행동 텐서 모양 [B, 12].
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs  # 관측에 관계없이 영 행동. 기본 offset 사용 시 목표각은 기본 관절 자세가 된다.
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          # 각 행동을 [-1, 1)에서 균등 샘플링한다. 학습된 보행 정책이 아니다.
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, runner_cfg, device=device)
    runner.load(
      # 추론에는 Actor만 복원한다. strict=True는 누락되거나 맞지 않는 가중치 구성을 오류로 처리한다.
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)  # 관측을 받아 행동을 내는 추론 함수.

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    # 디스플레이 환경 변수가 있으면 native 창, 없으면 브라우저에서 보는 viser를 선택한다.
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    # viewer.run() 안에서 관측 → 정책 → env.step을 반복한다. 여기서는 PPO 업데이트가 없다.
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()  # 뷰어 종료 후 시뮬레이션 리소스 해제.


def main():
  # Parse first argument to choose the task.
  # Import local tasks to populate the registry.
  import src.locomotion_rl  # noqa: F401

  all_tasks = [task for task in list_tasks() if task.startswith("Unitree-Go2-QuietPaw-LongStride-")]
  # chosen_task는 CLI 첫 태스크 이름, remaining_args는 --checkpoint-file/--viewer 등의 옵션이다.
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # 나머지 인자는 아래 PlayConfig 필드만 받는다. train.py의 --env.* / --agent.*와 구분한다.
  agent_cfg = load_rl_cfg(chosen_task)  # 여기서는 조회 후 버린다. 실제 runner 설정은 run_play()에서 조회한다.

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
