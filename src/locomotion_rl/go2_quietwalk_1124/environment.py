"""Go2 physics, constraint conditioning and the normalized Eq. (1) proxy."""

import math
from dataclasses import dataclass, field, fields, replace
from typing import ClassVar, Literal

import torch

from ..go2_quietpaw.step_01_environment.base_env import QuietPawEnvCfg
from ..go2_quietpaw.step_01_environment.go2_env import unitree_go2_flat_env_cfg, unitree_go2_rough_env_cfg
from ..go2_quietpaw.step_01_environment.quietpaw_env import QuietPawEnv
from ..go2_quietpaw.step_02_learning.mdp.noise_condition import NoiseConditionCfg


@dataclass
class ImpactCostCfg:
  force_threshold: float = 50.0  # Fmax in N, unspecified in the paper
  force_scale: float = 50.0
  velocity_squared_scale: float = 0.5
  contact_force_threshold: float = 1.0
  orientation: Literal["increasing", "printed"] = "increasing"

  def __post_init__(self):
    if not all(math.isfinite(x) for x in (
      self.force_threshold, self.force_scale, self.velocity_squared_scale, self.contact_force_threshold
    )):
      raise ValueError("Impact cost parameters must be finite.")
    if min(self.force_scale, self.velocity_squared_scale) <= 0 or min(self.force_threshold, self.contact_force_threshold) < 0:
      raise ValueError("Impact scales must be positive and thresholds nonnegative.")
    if self.orientation not in ("increasing", "printed"):
      raise ValueError("Unknown Eq. (1) orientation.")


def impact_cost(force, speed_squared, cfg):
  # Divide Eq. (1) by its value at rest. Equal lambda weights cancel.
  # Scaling both coefficients by exp(-Fmax/sigma_F) avoids overflow.
  velocity_weight = math.exp(-cfg.force_threshold / cfg.force_scale)
  force_exponent = -force / cfg.force_scale
  speed_exponent = -speed_squared / cfg.velocity_squared_scale
  if cfg.orientation == "printed":
    terms = torch.exp(force_exponent) + velocity_weight * torch.exp(speed_exponent)
  else:
    terms = -torch.expm1(force_exponent) - velocity_weight * torch.expm1(speed_exponent)
  return (terms.mean(dim=-1) / (1 + velocity_weight)).clamp(0, 1)


@dataclass
class QuietWalkEnvCfg(QuietPawEnvCfg):
  checkpoint_family: ClassVar[str] = "go2_quietwalk_1124"
  cost_units: ClassVar[str] = "expected discounted sum of normalized impact costs from episode start"
  noise_cost: ImpactCostCfg = field(default_factory=ImpactCostCfg)

  def build_env(self, **kwargs):
    return QuietWalkEnv(self, **kwargs)


class QuietWalkEnv(QuietPawEnv):
  def load_managers(self):
    super().load_managers()
    # mjlab 1.2's ContactSensor resolves primary geoms in entity order.
    robot = self.scene["robot"]
    sensor = self.scene["feet_ground_contact"]
    _, geoms = robot.find_geoms(sensor.cfg.primary.pattern)
    feet = [name.removesuffix("_foot_collision") for name in geoms]
    if len(feet) != 4 or set(feet) != {"FL", "FR", "RL", "RR"}:
      raise ValueError(f"Expected one contact channel per Go2 foot, got {geoms}")
    self.foot_site_ids, _ = robot.find_sites(feet, preserve_order=True)

  def _step_noise_cost(self):
    return impact_cost(self.landing.peak_forces, self.landing.impact_speed_squared, self.cfg.noise_cost)


def environment_cfg(terrain: Literal["flat", "rough"], play: bool = False):
  factory = {"flat": unitree_go2_flat_env_cfg, "rough": unitree_go2_rough_env_cfg}[terrain]
  base = factory(play=play)
  values = {item.name: getattr(base, item.name) for item in fields(base)}
  values.update(noise_cost=ImpactCostCfg(), noise_condition=NoiseConditionCfg(
    min_cost_limit=0.0, max_cost_limit=1.0, quietness=0.5 if play else None,
  ))
  cfg = QuietWalkEnvCfg(**values)
  cfg.scene.num_envs = 1 if play else 4000
  # Go2 configurations share the original articulation. Copy before changing gains.
  robot = cfg.scene.entities["robot"]
  actuators = tuple(replace(a, stiffness=25.0, damping=0.6) for a in robot.articulation.actuators)
  robot.articulation = replace(robot.articulation, actuators=actuators)
  return cfg
