"""논문 표 I 형식 평가: ε × 목표 전진 속도마다 비용 위반량, 속도 추종 오차, 실패 수, Pareto 여부.

python -m src.locomotion_rl.go2_quietpaw_1124.step_04_evaluation.evaluate \
  --checkpoint-file logs/rsl_rl/go2_quietpaw_1124/RUN/model_9999.pt
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from tensordict import TensorDict

from .checkpoint import load_evaluation_settings
from ..step_03_training.cncp_ppo import EpisodeCostReturn


@dataclass
class EvaluationConfig:
  checkpoint_file: Path
  task: Literal["Unitree-Go2-QuietPaw-1124-Flat", "Unitree-Go2-QuietPaw-1124-Rough"] = "Unitree-Go2-QuietPaw-1124-Flat"
  epsilons: tuple[float, ...] = (0., .1, .2, .3, .4, .6, .8, 1.)  # 논문 평가 ε
  velocities: tuple[float, ...] = (.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25)  # 논문 0.5~2.25 m/s
  num_envs: int = 32
  steps: int = 1000
  seed: int = 42
  device: str = "cuda:0"
  output: Path = Path("logs/go2_quietpaw_1124_evaluation.csv")


def mark_pareto(rows):
  """같은 목표 속도 안에서 비용과 추종 오차가 모두 더 좋은 점이 없으면 Pareto 해."""
  for row in rows:
    row["pareto"] = not any(
      other["target_velocity"] == row["target_velocity"]
      and other["normalized_cost"] <= row["normalized_cost"]
      and other["tracking_error"] <= row["tracking_error"]
      and (other["normalized_cost"] < row["normalized_cost"] or other["tracking_error"] < row["tracking_error"])
      for other in rows
    )


@torch.inference_mode()
def evaluate(cfg: EvaluationConfig):
  if min(cfg.steps, cfg.num_envs) < 1 or not cfg.velocities or not cfg.epsilons:
    raise ValueError("Provide positive steps and environments and a nonempty sweep.")
  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_evaluation_settings(env_cfg, cfg.checkpoint_file)
  bounds = env_cfg.noise_condition
  if any(not bounds.min_cost_limit <= eps <= bounds.max_cost_limit for eps in cfg.epsilons):
    raise ValueError("Evaluation ε must lie in the training interval.")
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  command = env_cfg.commands["twist"]
  command.heading_command = False
  command.rel_standing_envs = 0.0
  command.ranges.heading = None
  command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0., 0.)
  raw = env_cfg.build_env(device=cfg.device)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg["clip_actions"])
  rows = []
  try:
    runner = load_runner_cls(cfg.task)(env, agent_cfg, device=cfg.device)
    runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True, "critic": True}, map_location=cfg.device)
    policy = runner.get_inference_policy(cfg.device)
    alg = runner.alg
    for speed in cfg.velocities:
      raw.command_manager.get_term("twist").cfg.ranges.lin_vel_x = (speed, speed)
      for eps in cfg.epsilons:
        quietness = (bounds.max_cost_limit - eps) / (bounds.max_cost_limit - bounds.min_cost_limit)
        raw.noise_condition.set_quietness(quietness)
        raw.reset(seed=cfg.seed)
        obs = env.get_observations()
        costs = torch.empty((cfg.steps, cfg.num_envs), device=cfg.device)
        dones = torch.empty_like(costs, dtype=torch.bool)
        raw_sum = torch.zeros(cfg.num_envs, device=cfg.device)
        tracking_sum = torch.zeros_like(raw_sum)
        failures = torch.zeros((), dtype=torch.long, device=cfg.device)
        for t in range(cfg.steps):
          obs, _, done, extras = env.step(policy(obs))
          raw_cost = extras["noise_cost"]
          failed = extras["true_terminations"].bool()
          raw_sum += raw_cost
          costs[t] = raw_cost + alg.gamma * alg.failure_value * failed
          dones[t] = done.bool()
          failures += failed.sum()
          tracking_sum += extras["quietpaw_metrics"]["tracking_error"]
          if extras["terminal_observations"] is not None:
            timeout = extras["time_outs"].bool() & ~failed
            terminal = TensorDict(extras["terminal_observations"], batch_size=[cfg.num_envs])
            costs[t] += alg.gamma * alg.critic(terminal)[:, 1] * timeout
        estimator = EpisodeCostReturn(cfg.num_envs, cfg.device, alg.gamma)
        normalized = (1 - alg.gamma) * estimator.estimate(costs, dones, alg.critic(obs)[:, 1])
        mean_cost = raw_sum / cfg.steps
        rows.append(dict(
          task=cfg.task, seed=cfg.seed, num_envs=cfg.num_envs, steps=cfg.steps,
          target_velocity=speed, epsilon=eps,
          mean_step_cost=mean_cost.mean().item(),
          normalized_cost=normalized.mean().item(),
          cost_violation=(normalized - eps).clamp_min(0).mean().item(),  # 표 I: E[max(0, c − ε)]
          tracking_error=(tracking_sum.mean() / cfg.steps).item(),
          failures=failures.item(),
        ))
        print(f"v={speed:.2f} ε={eps:.1f} cost={rows[-1]['normalized_cost']:.3f} "
              f"violation={rows[-1]['cost_violation']:.3f} tracking={rows[-1]['tracking_error']:.3f} "
              f"failures={rows[-1]['failures']}")
  finally:
    env.close()
  mark_pareto(rows)
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  with cfg.output.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  return rows


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluationConfig, config=mjlab.TYRO_FLAGS))
