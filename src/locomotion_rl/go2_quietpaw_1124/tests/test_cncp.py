"""CNCP 구성요소 검증. 마지막 두 테스트는 GPU 시뮬레이션이 필요하다 (QUIETPAW_1124_GPU_TESTS=1)."""

import ast
import math
import os
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from tensordict import TensorDict

from src.locomotion_rl.go2_quietpaw_1124.step_02_learning.costs import NoiseCostCfg
from src.locomotion_rl.go2_quietpaw_1124.step_02_learning.mdp.costs import noise_cost
from src.locomotion_rl.go2_quietpaw_1124.step_03_training.cncp_ppo import EpisodeCostReturn, QuietPawPPO

PACKAGE = Path(__file__).resolve().parents[1]
TASK = "Unitree-Go2-QuietPaw-1124-Flat"


def _algorithm(num_envs=2):
  cfg = asdict(load_rl_cfg(TASK))
  cfg["num_steps_per_env"] = 1
  cfg["actor"]["hidden_dims"] = (8,)
  cfg["critic"]["hidden_dims"] = (8,)
  obs = TensorDict({name: torch.zeros(num_envs, 4) for name in ("actor", "critic")}, batch_size=[num_envs])
  alg = QuietPawPPO.construct_algorithm(obs, SimpleNamespace(num_actions=2, num_envs=num_envs), cfg, "cpu")
  alg.normalize_advantage_per_mini_batch = True  # 표준화 전 결합값을 본다.
  with torch.no_grad():
    for p in alg.critic.parameters():
      p.zero_()
  return alg, obs


class CncpTests(unittest.TestCase):
  def test_package_does_not_import_other_locomotion_packages(self):
    root = "src.locomotion_rl.go2_quietpaw_1124"
    for path in PACKAGE.rglob("*.py"):
      parts = (root,) + path.relative_to(PACKAGE).with_suffix("").parts
      for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
          if node.level:  # 상대 import를 절대 이름으로 바꾼다.
            base = ".".join(".".join(parts).split(".")[:-node.level])
            module = f"{base}.{node.module}" if node.module else base
          else:
            module = node.module or ""
          names = [module]
        elif isinstance(node, ast.Import):
          names = [alias.name for alias in node.names]
        else:
          continue
        for name in names:
          if name.startswith("src.locomotion_rl.") or name == "src.locomotion_rl":
            self.assertTrue(name.startswith(root), f"{path}: imports {name}")

  def test_eq1_normalization_and_direction(self):
    cfg = NoiseCostCfg()
    f = torch.tensor([[0., 20., 50., 100.]])
    v2 = torch.tensor([[0., .05, .25, 1.]])
    printed_sum = (.5 * torch.exp((50. - f) / 50.) + .5 * torch.exp(-v2 / .5)).sum(-1)
    s0 = 4 * (.5 * math.e + .5)
    printed = noise_cost(f, v2, replace(cfg, orientation="printed"))
    torch.testing.assert_close(printed, printed_sum / s0)
    torch.testing.assert_close(noise_cost(f, v2, cfg), 1. - printed)
    torch.testing.assert_close(noise_cost(f * 0, v2 * 0, cfg), torch.zeros(1))
    self.assertLess(noise_cost(f, v2, cfg).item(), noise_cost(f * 2, v2 * 2, cfg).item())

  def test_episode_cost_return_spans_rollouts(self):
    ret = EpisodeCostReturn(2, "cpu", .5)
    first = ret.estimate(torch.tensor([[1., 2.]]), torch.tensor([[False, True]]), torch.tensor([4., 99.]))
    torch.testing.assert_close(first, torch.tensor([3., 2.]))
    second = ret.estimate(torch.tensor([[2., 3.], [4., 1.]]), torch.tensor([[True, False], [False, True]]),
                          torch.tensor([6., 99.]))
    torch.testing.assert_close(second, torch.tensor([(2. + 7.) / 2, 3.5]))

  def test_failure_bootstraps_maximum_cost_and_advantage_precedes_pid(self):
    alg, obs = _algorithm()
    alg.failure_cost = 1.0
    with torch.no_grad():
      alg.act(obs)
      alg.process_env_step(obs, torch.tensor([2., 3.]), torch.ones(2), {
        "noise_cost": torch.tensor([.5, .5]), "cost_limit": torch.tensor([.2, .9]),
        "true_terminations": torch.tensor([True, False]), "time_outs": torch.tensor([False, True]),
        "terminal_observations": {"critic": torch.zeros(2, 4)},
      })
      expected_cost = torch.tensor([.5 + alg.gamma / (1 - alg.gamma), .5])
      torch.testing.assert_close(alg.storage.costs[0, :, 0], expected_cost)
      alg.compute_returns(obs)
    # λ는 처음 0이므로 결합 advantage는 reward advantage와 같다 (식 8, Algorithm 1 4→5행).
    torch.testing.assert_close(alg.storage.advantages[0, :, 0], torch.tensor([2., 3.]))
    # 실패한 환경(정규화 비용 ≈ 1 > ε = 0.2)만 λ가 커진다.
    self.assertGreater(alg.pid.multipliers[0].item(), 0.0)
    self.assertEqual(alg.pid.multipliers[1].item(), 0.0)

  def test_registered_defaults(self):
    cfg = load_env_cfg(TASK)
    play = load_env_cfg(TASK, play=True)
    agent = load_rl_cfg(TASK)
    self.assertEqual((cfg.scene.num_envs, play.scene.num_envs), (4000, 1))
    self.assertEqual((cfg.noise_condition.min_cost_limit, cfg.noise_condition.max_cost_limit), (0., 1.))
    self.assertAlmostEqual(cfg.sim.mujoco.timestep * cfg.decimation, .02)
    self.assertTrue(cfg.only_positive_rewards)
    gains = [(a.stiffness, a.damping) for a in cfg.scene.entities["robot"].articulation.actuators]
    self.assertEqual(gains, [(20., 1.), (20., 1.), (40., 2.)])
    self.assertFalse(agent.algorithm.use_clipped_value_loss)
    self.assertEqual((agent.max_iterations, agent.experiment_name), (10000, "go2_quietpaw_1124"))
    self.assertIn("go2_quietpaw_1124", agent.algorithm.class_name)
    self.assertIn("go2_quietpaw_1124", agent.critic.class_name)

  @unittest.skipUnless(os.environ.get("QUIETPAW_1124_GPU_TESTS") == "1", "requires GPU simulation")
  def test_feet_match_force_channels_and_cost_range(self):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 64
    env = cfg.build_env(device="cuda:0")
    try:
      sensor = env.scene["feet_ground_contact"]
      contact_feet = [s.primary_name.removesuffix("_foot_collision") for s in sensor._slots if s.field_name == "force"]
      velocity_feet = [env.scene["robot"].site_names[i] for i in env.foot_site_ids]
      self.assertEqual(contact_feet, velocity_feet)
      env.reset()
      action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
      for _ in range(50):
        _, _, _, _, extras = env.step(action)
        self.assertTrue(((extras["noise_cost"] >= 0) & (extras["noise_cost"] <= 1)).all())
    finally:
      env.close()

  @unittest.skipUnless(os.environ.get("QUIETPAW_1124_GPU_TESTS") == "1", "requires GPU simulation")
  def test_positive_reward_clip_keeps_termination_penalty(self):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 4
    env = cfg.build_env(device="cuda:0")
    try:
      env.reset()
      action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
      _, reward, terminated, _, _ = env.step(action)
      idx = env._termination_reward_idx
      term = env.reward_manager._step_reward[:, idx] * env.step_dt
      self.assertTrue(((reward - term) >= -1e-6).all())
    finally:
      env.close()


if __name__ == "__main__":
  unittest.main()
