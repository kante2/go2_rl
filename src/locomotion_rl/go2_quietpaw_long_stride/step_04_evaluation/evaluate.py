"""같은 속도·초기화 조건에서 quietness를 바꿔 CSV로 비교한다.

python -m src.locomotion_rl.go2_quietpaw_long_stride.step_04_evaluation.evaluate \
  --checkpoint-file logs/rsl_rl/go2_quietpaw_long_stride/RUN/model_1000.pt
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import mjlab
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from .checkpoint import load_evaluation_settings
from ..step_01_environment.quietpaw_env import QuietPawEnv
from ..step_03_training.runner import QuietPawOnPolicyRunner


@dataclass
class EvaluationConfig:
  checkpoint_file: Path
  task: Literal["Unitree-Go2-QuietPaw-LongStride-Flat", "Unitree-Go2-QuietPaw-LongStride-Rough"] = "Unitree-Go2-QuietPaw-LongStride-Flat"
  quietness: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
  velocities: tuple[float, ...] = (1.0, 1.5)
  num_envs: int = 32
  steps: int = 1000
  seed: int = 42
  device: str = "cuda:0"
  output: Path = Path("logs/quietpaw_long_stride_evaluation.csv")


def mark_pareto(rows):
  """동일 목표 속도 내에서 비용과 추종 오차가 동시에 더 좋은 점이 있는지 확인한다."""
  for row in rows:
    row["pareto"] = not any(
      other["target_velocity"] == row["target_velocity"]
      and other["noise_cost"] <= row["noise_cost"]
      and other["tracking_error"] <= row["tracking_error"]
      and (other["noise_cost"] < row["noise_cost"] or other["tracking_error"] < row["tracking_error"])
      for other in rows
    )


def evaluate(cfg: EvaluationConfig):
  if cfg.steps < 1 or cfg.num_envs < 1 or not cfg.quietness or not cfg.velocities:
    raise ValueError("Evaluation requires positive steps/envs and nonempty condition/speed lists.")
  if any(not 0 <= q <= 1 for q in cfg.quietness):
    raise ValueError("All quietness values must be in [0, 1].")
  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_evaluation_settings(env_cfg, cfg.checkpoint_file)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.noise_condition.quietness = cfg.quietness[0]
  command_cfg = env_cfg.commands["twist"]
  command_cfg.heading_command = False
  command_cfg.ranges.heading = None
  command_cfg.rel_standing_envs = 0.0
  command_cfg.ranges.lin_vel_y = (0.0, 0.0)
  command_cfg.ranges.ang_vel_z = (0.0, 0.0)
  raw_env = QuietPawEnv(env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg["clip_actions"])
  rows = []
  try:
    runner = QuietPawOnPolicyRunner(env, agent_cfg, device=cfg.device)
    runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, map_location=cfg.device)
    policy = runner.get_inference_policy(cfg.device)
    for velocity in cfg.velocities:
      raw_env.command_manager.get_term("twist").cfg.ranges.lin_vel_x = (velocity, velocity)
      for quietness in cfg.quietness:
        raw_env.noise_condition.set_quietness(quietness)
        # 조건 사이에 동일 seed로 재초기화한다. 넘어지면 종료·reset하며 실패 횟수도 기록한다.
        # mjlab의 센서 history도 inference 중 생성되므로 reset을 같은 모드에서 수행한다.
        with torch.inference_mode():
          raw_env.reset(seed=cfg.seed)
          obs = env.get_observations()
        costs = torch.zeros(cfg.num_envs, device=cfg.device)
        totals = {key: 0.0 for key in ("tracking_error", "peak_force", "impact_speed_squared")}
        failures = 0
        stride_totals = {key: 0.0 for key in ("distance_sum", "period_sum", "count")}
        with torch.inference_mode():
          for _ in range(cfg.steps):
            obs, _, _, extras = env.step(policy(obs))
            costs += extras["noise_cost"]
            for key in totals:
              totals[key] += extras["quietpaw_metrics"][key].mean().item()
            failures += int(raw_env.reset_terminated.sum().item())
            for key in stride_totals:
              stride_totals[key] += extras["stride_metrics"][key].sum().item()
        costs /= cfg.steps
        limits = raw_env.noise_condition.cost_limit
        row = {
          "target_velocity": velocity, "quietness": quietness,
          "cost_limit": limits.mean().item(), "noise_cost": costs.mean().item(),
          "cost_violation": (costs - limits).clamp_min(0).mean().item(),
          **{key: value / cfg.steps for key, value in totals.items()},
          "failures": failures, "num_envs": cfg.num_envs, "steps": cfg.steps,
          "seed": cfg.seed,
          "stride_samples": int(stride_totals["count"]),
          "stride_length_m": (stride_totals["distance_sum"] / stride_totals["count"]
                              if stride_totals["count"] else None),
          "stride_period_s": (stride_totals["period_sum"] / stride_totals["count"]
                              if stride_totals["count"] else None),
        }
        rows.append(row)
        print(row)
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
