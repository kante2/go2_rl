"""Script to train RL agent with RSL-RL."""
'''
register_mjlab_task

'''

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

import mjlab
# ManagerBasedRlEnv가 실제 RL simulation environment
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
# 1. MjlabOnPolicyRunner - RL학습을 총괄
# 2. RslRlVecEnvWrapper - MjLab 환경을 RSL-RL이 이해할 수 있는 형태로 바꿈
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
# 3. 이 함수들을 호출하면 앞서 등록한 _REGISTRY에서 정보를 꺼내준다.
from mjlab.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls
)
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


# TrainConfig 은, 전체 training옵션을 모아놓은 config class
@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg # 실제 RL simulation environment
  agent: RslRlBaseRunnerCfg # MjLab 환경을 RSL-RL이 이해할 수 있는 형태로 바꿈
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  # registry.py에서 등록된 환경을 가져오는 부분
  # task_id : unitree-go2-flat, unitree-go2-rough가 존재 --
  # from_task함수에서, 이름으로 설정을 조회
  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id) # env_cfg
    agent_cfg = load_rl_cfg(task_id) # rl_cfg
    return TrainConfig(env=env_cfg, agent=agent_cfg)

# 실제 학습 한번을 실행하는 함수,
def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  # CPU/ GPU을 결정
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    # multi gpu에서 각 gpu가 완전히 같은 random sequence를 만들면 안되므로, rank를 더해주어 각 gpu마다 다른 seed를 갖도록 한다.
    '''
    예)
    GPU0 → seed 42
    GPU1 → seed 43
    GPU2 → seed 44
      '''
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  # 실제 환경 생성 -> go2 simulation환경이 생성되는 지점
  # num_envs수 만큼의 개수를 GPU에서 돌린다
  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  # resume옵션이 켜져있으면, 과거 checkpoint를 찾는다 -> 다음, runner.load()에서 checkpoint를 불러와서 학습을 이어간다.
  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    # 학습 도중 특정 interval마다 영상을 저장
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  # RslRlVecEnvWrapper - adapter
  # MjLab과 RLS-RL이 요구하는 환경 interface가 조금 다르기 때문
  '''
            MjLab
  ManagerBasedRlEnv
            │
            ▼
  RslRlVecEnvWrapper
            │
            ▼
          RSL-RL
  '''
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  # cfg을 python dictionary 형태로 바꾼다. (yaml로 저장하기 위해)
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  # task 별 custom runner가 있으면 가져온다,
  #  없으면 기본 runner (MjlabOnPolicyRunner)를 사용한다.
  #  여기서, 환경과 RL알고리즘이 실제로 만나는 곳
  runner_cls = load_runner_cls(task_id) # runner_cls
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  '''
          Go2 Environment
              │
              ▼
        PPO Runner
              │
      ┌───────┴────────┐
      │                │
 Actor Network    Critic Network
      │
      ▼
   Action
      │
      ▼
     Go2
      │
      ▼
    Reward
  '''

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path)) # resume옵션이 켜져있으면 과거 checkpoint를 찾고, 기존 policy를 불러와서 학습을 이어간다.

  # config 저장 ---------------
  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  # 실제 ppo학습이 진행 --------------
  # runner.learn -> 내부에서는 반복해서, 1,rollout, 2,update, 3,log를 수행한다.
  '''
  for iteration in range(max_iterations):

    # 1. rollout
    obs = env.get_observation()

    for step in range(num_steps):

        action = actor(obs)

        obs, reward, done = env.step(action)

        storage.add(
            obs,
            action,
            reward,
            done
        )

    # 2. advantage 계산
    compute_returns()

    # 3. PPO update
    for epoch in range(num_epochs):
        loss = compute_ppo_loss()
        loss.backward()
        optimizer.step()

    # 4. checkpoint / logging
    save_model()
  '''
  # init_at_random_ep_len=True : 초기 episode proress을 랜덤하게 만들어 분산시킨다,
  #  즉, parallel environment간 동기화를 깨뜨리는 역할을 한다. (모든 parallel environment가 같은 시점에서 시작하면, 학습이 잘 안될 수 있음)
  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )
  # 학습이 끝나면, 환경을 닫는다.
  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # 로그 폴더 생성 ----------------
  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # GPU 선택 ----------------
  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  # Mujuco가 rendering할 GPU를 지정
  # 특히 headless training에서도 rendering context가 필요할 수 있어서 EGL을 사용
  '''
  MuJoCo Rendering
       ↓
      EGL
       ↓
   NVIDIA GPU --> 이 구조가 화면 없이도, 학습이 되는 이유
  '''
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import local tasks to populate the registry.
  import src.locomotion_rl  # noqa: F401

  all_tasks = [task for task in list_tasks() if task.startswith("Unitree-Go2-")]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
