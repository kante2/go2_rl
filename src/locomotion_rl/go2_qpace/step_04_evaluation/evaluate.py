"""목표 dB × 목표 전진 속도 평가: 실제 로봇 소음(dB), 목표 초과량, 속도 추종 오차, 실패 수, Pareto 여부.

python -m src.locomotion_rl.go2_qpace.step_04_evaluation.evaluate \
  --checkpoint-file logs/rsl_rl/go2_qpace/RUN/model_9999.pt
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_runner_cls

from .checkpoint import load_evaluation_settings


@dataclass
class EvaluationConfig:
  checkpoint_file: Path
  task: Literal["Unitree-Go2-QPace-Flat", "Unitree-Go2-QPace-Rough"] = "Unitree-Go2-QPace-Flat"
  target_dbs: tuple[float, ...] | None = None  # None: 학습 범위를 8등분
  velocities: tuple[float, ...] = (.3, .5, .7, 1., 1.5, 2.)
  num_envs: int = 32
  steps: int = 1000
  warmup_steps: int = 100  # 가속 구간 제외
  seed: int = 42
  device: str = "cuda:0"
  output: Path = Path("logs/go2_qpace_evaluation.csv")


def mark_pareto(rows):
  """같은 목표 속도 안에서 소음과 추종 오차가 모두 더 좋은 점이 없으면 Pareto 해."""
  for row in rows:
    row["pareto"] = not any(
      other["target_velocity"] == row["target_velocity"]
      and other["robot_db"] <= row["robot_db"]
      and other["tracking_error"] <= row["tracking_error"]
      and (other["robot_db"] < row["robot_db"] or other["tracking_error"] < row["tracking_error"])
      for other in rows
    )


@torch.inference_mode()
def evaluate(cfg: EvaluationConfig):
  if min(cfg.steps, cfg.num_envs) < 1 or cfg.warmup_steps >= cfg.steps or not cfg.velocities:
    raise ValueError("Provide positive steps/envs, warmup < steps and a nonempty speed sweep.")
  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_evaluation_settings(env_cfg, cfg.checkpoint_file)
  bounds = env_cfg.noise_condition
  targets = cfg.target_dbs or tuple(
    round(bounds.min_db + i * (bounds.max_db - bounds.min_db) / 7, 2) for i in range(8))
  if any(not bounds.min_db <= t <= bounds.max_db for t in targets):
    raise ValueError(f"target_dbs must lie in the trained range [{bounds.min_db}, {bounds.max_db}].")
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
    runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, map_location=cfg.device)
    policy = runner.get_inference_policy(cfg.device)
    dt = raw.step_dt
    for speed in cfg.velocities:
      raw.command_manager.get_term("twist").cfg.ranges.lin_vel_x = (speed, speed)
      for target in targets:
        raw.noise_condition.set_target_db(target)
        raw.reset(seed=cfg.seed)
        obs = env.get_observations()
        energy = torch.zeros(cfg.num_envs, device=cfg.device)
        touchdowns = torch.zeros_like(energy)
        tracking = torch.zeros_like(energy)
        failures = torch.zeros((), dtype=torch.long, device=cfg.device)
        for t in range(cfg.steps):
          obs, _, _, extras = env.step(policy(obs))
          failures += extras["true_terminations"].sum()
          if t >= cfg.warmup_steps:
            metrics = extras["quietpaw_metrics"]
            energy += metrics["noise_energy"]
            touchdowns += metrics["touchdown"].float()
            tracking += metrics["tracking_error"]
        duration = (cfg.steps - cfg.warmup_steps) * dt
        robot_db = 10 * torch.log10((energy / duration).clamp_min(1e-30))  # 환경별 Leq [dB re FS²/s]
        rows.append(dict(
          task=cfg.task, seed=cfg.seed, num_envs=cfg.num_envs, steps=cfg.steps,
          target_velocity=speed, target_db=target,
          robot_db=10 * math.log10(max((energy.sum() / (duration * cfg.num_envs)).item(), 1e-30)),
          robot_db_env_mean=robot_db.mean().item(),
          excess_db=(robot_db - target).mean().item(),
          violation_db=(robot_db - target).clamp_min(0).mean().item(),
          satisfied_frac=(robot_db <= target).float().mean().item(),
          touchdown_steps_per_s=(touchdowns.mean() / duration).item(),
          tracking_error=(tracking.mean() / (cfg.steps - cfg.warmup_steps)).item(),
          failures=failures.item(),
        ))
        r = rows[-1]
        print(f"v={speed:.2f} target={target:.1f} dB → robot={r['robot_db']:.1f} dB "
              f"(violation {r['violation_db']:.2f}) tracking={r['tracking_error']:.3f} failures={r['failures']}")
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
