"""Evaluate one trained seed across the noise limits and forward speeds of Table I."""

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

from ..go2_quietpaw.step_04_evaluation.checkpoint import load_evaluation_settings
from ..go2_quietpaw.step_04_evaluation.evaluate import mark_pareto
from .algorithm import EpisodeCostReturn


@dataclass
class EvaluationConfig:
  checkpoint_file: Path
  task: Literal["Unitree-Go2-QuietWalk-1124-Flat", "Unitree-Go2-QuietWalk-1124-Rough"] = "Unitree-Go2-QuietWalk-1124-Flat"
  limits: tuple[float, ...] = (0., .1, .2, .3, .4, .6, .8, 1.)
  velocities: tuple[float, ...] = (.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25)
  num_envs: int = 32
  steps: int = 1000
  seed: int = 42
  device: str = "cuda:0"
  output: Path = Path("logs/go2_quietwalk_1124_evaluation.csv")


@torch.inference_mode()
def evaluate(cfg: EvaluationConfig):
  if min(cfg.steps, cfg.num_envs) < 1 or not cfg.velocities or not cfg.limits:
    raise ValueError("Provide positive steps and environments and a nonempty sweep.")
  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_evaluation_settings(env_cfg, cfg.checkpoint_file)
  bounds = env_cfg.noise_condition
  if any(not bounds.min_cost_limit <= limit <= bounds.max_cost_limit for limit in cfg.limits):
    raise ValueError("Evaluation limits must lie in the training interval.")
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
      for limit in cfg.limits:
        quietness = (bounds.max_cost_limit - limit) / (bounds.max_cost_limit - bounds.min_cost_limit)
        raw.noise_condition.set_quietness(quietness)
        raw.reset(seed=cfg.seed)
        obs = env.get_observations()
        costs = torch.empty((cfg.steps, cfg.num_envs), device=cfg.device)
        dones = torch.empty_like(costs, dtype=torch.bool)
        raw_cost_sum = torch.zeros(cfg.num_envs, device=cfg.device)
        tracking_sum = torch.zeros_like(raw_cost_sum)
        failures = torch.zeros((), dtype=torch.long, device=cfg.device)
        for t in range(cfg.steps):
          obs, _, done, extras = env.step(policy(obs))
          raw_cost = extras["noise_cost"]
          raw_cost_sum += raw_cost
          costs[t] = raw_cost
          dones[t] = done.bool()
          failures += extras["true_terminations"].sum()
          tracking_sum += extras["quietpaw_metrics"]["tracking_error"]
          if extras["terminal_observations"] is not None:
            timeout = extras["time_outs"].bool() & ~extras["true_terminations"]
            terminal = TensorDict(extras["terminal_observations"], batch_size=[cfg.num_envs])
            costs[t] += alg.gamma * alg.critic(terminal)[:, 1] * timeout
        mean_cost = raw_cost_sum / cfg.steps
        estimator = EpisodeCostReturn(cfg.num_envs, cfg.device, alg.gamma)
        cost_return = estimator.estimate(costs, dones, alg.critic(obs)[:, 1])
        rows.append(dict(
          task=cfg.task, seed=cfg.seed, num_envs=cfg.num_envs, steps=cfg.steps,
          target_velocity=speed, cost_limit=limit, noise_cost=mean_cost.mean().item(),
          normalized_cost_violation=(mean_cost - limit).clamp_min(0).mean().item(),
          discounted_cost_return=cost_return.mean().item(),
          discounted_cost_violation=(cost_return - limit).clamp_min(0).mean().item(),
          tracking_error=(tracking_sum.mean() / cfg.steps).item(), failures=failures.item(),
        ))
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
