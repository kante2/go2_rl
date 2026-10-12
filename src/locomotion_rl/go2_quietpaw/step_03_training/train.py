"""QuietPaw 학습 진입점. baseline과 같은 CLI·registry·로그 구조를 사용한다."""

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import mjlab
import tyro
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from ..step_01_environment.base_env import QuietPawEnvCfg
from ..step_01_environment.quietpaw_env import QuietPawEnv


@dataclass(frozen=True)
class TrainConfig:
  env: QuietPawEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  video: bool = False
  video_length: int = 200  # 50 Hz에서 4초.
  video_interval: int = 2000
  enable_nan_guard: bool = False
  gpu_ids: list[int] | None = field(default_factory=lambda: [0])  # 단일 GPU 또는 CPU(None).

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  # launch_training에서 선택한 물리 GPU를 현재 프로세스의 cuda:0으로 사용한다.
  device = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else "cuda:0"
  configure_torch_backends()
  cfg.env.seed = cfg.agent.seed
  cfg.env.sim.nan_guard.enabled = cfg.enable_nan_guard
  cfg.env.gait_curriculum.steps_per_iteration = cfg.agent.num_steps_per_env
  print(f"[INFO] QuietPaw training: device={device}, logs={log_dir}")

  # PID 상태는 환경별 조건과 대응한다. resume에서는 같은 환경 수·비용 설정을 검사한다.
  resume_path = None
  if cfg.agent.resume:
    resume_path = get_checkpoint_path(log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint)

  env = QuietPawEnv(cfg.env, device=device, render_mode="rgb_array" if cfg.video else None)
  try:
    if cfg.video:
      env = VideoRecorder(
        env, video_folder=log_dir / "videos" / "train",
        step_trigger=lambda step: step % cfg.video_interval == 0,
        video_length=cfg.video_length, disable_logger=True,
      )
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
      raise ValueError(f"No QuietPaw runner registered for {task_id}.")
    runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
      runner.load(str(resume_path))
    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

    # Runner: rollout 수집 -> Reward/Cost GAE -> PID -> PPO 업데이트 -> 저장.
    # max_iterations는 이번 실행에서 추가로 학습할 반복 횟수이다.
    runner.learn(cfg.agent.max_iterations, init_at_random_ep_len=True)
  finally:
    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  if num_gpus > 1:
    raise ValueError("QuietPaw currently supports a single GPU; select --gpu-ids '[0]'.")
  os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_gpus is None else ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    run_name += f"_{args.agent.run_name}"
  log_dir = Path("logs") / "rsl_rl" / args.agent.experiment_name / run_name
  run_train(task_id, args, log_dir)


def main():
  import src.locomotion_rl  # noqa: F401; registry 등록.

  tasks = [task for task in list_tasks() if task.startswith("Unitree-Go2-QuietPaw-")]
  task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tasks), add_help=False,
    return_unknown_args=True, config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig, args=remaining_args, default=TrainConfig.from_task(task),
    prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS,
  )
  launch_training(task, args)


if __name__ == "__main__":
  main()
