import os
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

import torch
from mjlab.tasks.registry import load_env_cfg
from tensordict import TensorDict

from src.locomotion_rl.go2_quietwalk_1124.algorithm import EpisodeCostReturn, QuietWalkPPO, runner_cfg
from src.locomotion_rl.go2_quietwalk_1124.environment import ImpactCostCfg, impact_cost


class CncpTests(unittest.TestCase):
  def test_eq1_and_normalization_direction(self):
    cfg = ImpactCostCfg()
    f = torch.tensor([[0., 20., 50., 100.]])
    v2 = torch.tensor([[0., .05, .25, 1.]])
    raw = (.5 * torch.exp((50. - f) / 50.) + .5 * torch.exp(-v2 / .5)).sum(-1)
    maximum = 4 * (.5 * torch.exp(torch.tensor(1.)) + .5)
    printed = impact_cost(f, v2, replace(cfg, orientation="printed"))
    torch.testing.assert_close(printed, raw / maximum)
    torch.testing.assert_close(impact_cost(f, v2, cfg), 1. - printed)
    torch.testing.assert_close(impact_cost(f*0, v2*0, cfg), torch.zeros(1))

  def test_discounted_return_spans_rollouts_and_stops_at_resets(self):
    ret = EpisodeCostReturn(2, "cpu", .5)
    first = ret.estimate(torch.tensor([[1., 2.]]), torch.tensor([[False, True]]), torch.tensor([4., 99.]))
    torch.testing.assert_close(first, torch.tensor([3., 2.]))
    second = ret.estimate(torch.tensor([[2., 3.], [4., 1.]]), torch.tensor([[True, False], [False, True]]), torch.tensor([6., 99.]))
    torch.testing.assert_close(second, torch.tensor([(2.+7.)/2, 3.5]))

  def test_advantage_uses_multiplier_before_pid_update(self):
    cfg = asdict(runner_cfg())
    cfg["num_steps_per_env"] = 1
    cfg["actor"]["hidden_dims"] = (8,)
    cfg["critic"]["hidden_dims"] = (8,)
    obs = TensorDict({name: torch.zeros(2, 4) for name in ("actor", "critic")}, batch_size=[2])
    alg = QuietWalkPPO.construct_algorithm(obs, SimpleNamespace(num_actions=2, num_envs=2), cfg, "cpu")
    self.assertIsInstance(alg, QuietWalkPPO)
    alg.normalize_advantage_per_mini_batch = True  # inspect the unstandardized combination
    with torch.no_grad():
      for p in alg.critic.parameters():
        p.zero_()
      alg.act(obs)
      alg.process_env_step(obs, torch.tensor([2., 3.]), torch.ones(2), {
        "noise_cost": torch.ones(2), "cost_limit": torch.zeros(2),
      })
      alg.compute_returns(obs)
    torch.testing.assert_close(alg.storage.advantages[0, :, 0], torch.tensor([2., 3.]))
    self.assertTrue((alg.pid.multipliers > 0).all())

  def test_registered_defaults_are_isolated(self):
    cfg = load_env_cfg("Unitree-Go2-QuietWalk-1124-Flat")
    old = load_env_cfg("Unitree-Go2-QuietPaw-Flat")
    self.assertEqual(cfg.scene.num_envs, 4000)
    self.assertEqual((cfg.noise_condition.min_cost_limit, cfg.noise_condition.max_cost_limit), (0., 1.))
    self.assertEqual(cfg.sim.mujoco.timestep * cfg.decimation, .02)
    def gains(env):
      return [(a.stiffness, a.damping) for a in env.scene.entities["robot"].articulation.actuators]
    self.assertEqual(gains(cfg), [(25., .6)] * 3)
    self.assertEqual(gains(old), [(20., 1.), (20., 1.), (40., 2.)])

  @unittest.skipUnless(os.environ.get("QUIETWALK_GPU_TESTS") == "1", "requires GPU simulation")
  def test_velocity_sites_match_compiled_force_channels(self):
    cfg = load_env_cfg("Unitree-Go2-QuietWalk-1124-Flat")
    cfg.scene.num_envs = 4
    env = cfg.build_env(device="cuda:0")
    try:
      sensor = env.scene["feet_ground_contact"]
      contact_feet = [s.primary_name.removesuffix("_foot_collision") for s in sensor._slots if s.field_name == "force"]
      velocity_feet = [env.scene["robot"].site_names[i] for i in env.foot_site_ids]
      self.assertEqual(len(contact_feet), 4)
      self.assertEqual(contact_feet, velocity_feet)
      torch.testing.assert_close(env.noise_condition.cost_limit, torch.linspace(1., 0., 4, device=env.device))
    finally:
      env.close()
