"""QPace 구성요소 검증. GPU 테스트는 QPACE_GPU_TESTS=1 이 필요하다."""

import ast
import math
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from tensordict import TensorDict

from src.locomotion_rl.go2_qpace.step_02_learning.mdp.noise_condition import NoiseCondition, NoiseConditionCfg
from src.locomotion_rl.go2_qpace.step_02_learning.noise_model import DiffImpactNoiseModel
from src.locomotion_rl.go2_qpace.step_02_learning.curriculum import QPaceCurriculumCfg
from src.locomotion_rl.go2_qpace.step_03_training.cncp_ppo import EpisodeCostReturn, QuietPawPPO
from src.locomotion_rl.go2_qpace.step_03_training.warm_start import warm_start_from_baseline

PACKAGE = Path(__file__).resolve().parents[1]
TASK = "Unitree-Go2-QPace-Flat"
BASELINE = Path(__file__).resolve().parents[3] / "go2_autonomous_driving/model_pt/model_10000.pt"


def _algorithm(num_envs=2):
  cfg = asdict(load_rl_cfg(TASK))
  cfg["num_steps_per_env"] = 1
  cfg["actor"]["hidden_dims"] = (8,)
  cfg["critic"]["hidden_dims"] = (8,)
  cfg["algorithm"]["num_mini_batches"] = 1
  cfg["algorithm"]["num_learning_epochs"] = 1
  obs = TensorDict({name: torch.zeros(num_envs, 4) for name in ("actor", "critic")}, batch_size=[num_envs])
  alg = QuietPawPPO.construct_algorithm(obs, SimpleNamespace(num_actions=2, num_envs=num_envs), cfg, "cpu")
  alg.normalize_advantage_per_mini_batch = True  # 표준화 전 결합값을 본다.
  with torch.no_grad():
    for p in alg.critic.parameters():
      p.zero_()
  return alg, obs


class QPaceTests(unittest.TestCase):
  def test_package_does_not_import_other_locomotion_packages(self):
    root = "src.locomotion_rl.go2_qpace"
    for path in PACKAGE.rglob("*.py"):
      parts = (root,) + path.relative_to(PACKAGE).with_suffix("").parts
      for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
          if node.level:
            base = ".".join(".".join(parts).split(".")[:-node.level])
            names = [f"{base}.{node.module}" if node.module else base]
          else:
            names = [node.module or ""]
        elif isinstance(node, ast.Import):
          names = [alias.name for alias in node.names]
        else:
          continue
        for name in names:
          if name.startswith("src.locomotion_rl.") or name == "src.locomotion_rl":
            self.assertTrue(name.startswith(root), f"{path}: imports {name}")

  def test_target_db_condition(self):
    cfg = NoiseConditionCfg(min_db=-64., max_db=-49., reference_db=-56.)
    cond = NoiseCondition(cfg, 4, "cpu")
    torch.testing.assert_close(cond.target_db, torch.tensor([-64., -59., -54., -49.]))
    torch.testing.assert_close(cond.normalized, torch.tensor([0., 1 / 3, 2 / 3, 1.]))
    # dB 비교와 에너지 비교가 같다: ε = 10^((L − ref)/10)
    torch.testing.assert_close(10 * torch.log10(cond.cost_limit) + cfg.reference_db, cond.target_db)
    cond.set_target_db(-60.)
    self.assertTrue(torch.all(cond.target_db == -60.))
    with self.assertRaises(ValueError):
      cond.set_target_db(-70.)
    with self.assertRaises(ValueError):
      NoiseConditionCfg(target_db=-40.).__post_init__()

  def test_body_vz_extrapolation_is_capped(self):
    free = DiffImpactNoiseModel()
    capped = DiffImpactNoiseModel(extrapolation_limits={"body_vz_pre": 0.05})
    k = free.features.index("body_vz_pre")
    hi = free.hi[0, k].item()
    x = free.mu[0].clone().repeat(2, 1)
    x[0, k] = hi + 0.05
    x[1, k] = hi + 0.30
    legs, offset = torch.tensor([3, 3]), torch.zeros(2)
    db_capped = capped.foot_energy(x, legs, offset)["db"]
    db_free = free.foot_energy(x, legs, offset)["db"]
    torch.testing.assert_close(db_capped[0], db_capped[1])  # 상한 너머는 같은 값
    self.assertLess(db_free[1].item(), db_capped[1].item() - 1.0)  # 원본은 더 조용하다고 외삽

  def test_episode_cost_return_spans_rollouts(self):
    ret = EpisodeCostReturn(2, "cpu", .5)
    first = ret.estimate(torch.tensor([[1., 2.]]), torch.tensor([[False, True]]), torch.tensor([4., 99.]))
    torch.testing.assert_close(first, torch.tensor([3., 2.]))

  def test_failure_bootstrap_and_db_pid(self):
    alg, obs = _algorithm()
    alg.failure_cost = 10 ** 0.3  # +3 dB
    limits = torch.tensor([0.1, 1.0])  # 기준 대비 −10 dB, 0 dB
    with torch.no_grad():
      alg.act(obs)
      alg.process_env_step(obs, torch.tensor([2., 3.]), torch.ones(2), {
        "noise_cost": torch.tensor([.5, .5]), "cost_limit": limits,
        "true_terminations": torch.tensor([True, False]), "time_outs": torch.tensor([False, True]),
        "terminal_observations": {"critic": torch.zeros(2, 4)},
      })
      torch.testing.assert_close(alg.storage.costs[0, :, 0],
                                 torch.tensor([.5 + alg.gamma * alg.failure_value, .5]))
      alg.compute_returns(obs)
    # λ는 처음 0 → 결합 advantage = reward advantage (식 8, advantage 후 λ 갱신)
    torch.testing.assert_close(alg.storage.advantages[0, :, 0], torch.tensor([2., 3.]))
    m = alg.constraint_metrics
    self.assertGreater(alg.pid.multipliers[0].item(), 0.0)  # 넘어짐 → 목표 초과
    self.assertEqual(alg.pid.multipliers[1].item(), 0.0)  # 조용한 환경 → 만족
    self.assertEqual(m["noise_satisfied_frac"], 0.5)

  def test_negative_cost_rejected(self):
    alg, obs = _algorithm()
    with torch.no_grad():
      alg.act(obs)
      with self.assertRaises(ValueError):
        alg.process_env_step(obs, torch.zeros(2), torch.zeros(2), {
          "noise_cost": torch.tensor([-1., 0.]), "cost_limit": torch.ones(2)})

  def test_registered_defaults(self):
    cfg = load_env_cfg(TASK)
    play = load_env_cfg(TASK, play=True)
    agent = load_rl_cfg(TASK)
    self.assertEqual((cfg.scene.num_envs, play.scene.num_envs), (4000, 1))
    c = cfg.noise_condition
    self.assertTrue(c.min_db < c.reference_db < c.max_db)
    self.assertIsNone(c.target_db)
    self.assertIsNotNone(play.noise_condition.target_db)
    self.assertAlmostEqual(cfg.sim.mujoco.timestep * cfg.decimation, .02)
    self.assertTrue(cfg.only_positive_rewards)
    gains = [(a.stiffness, a.damping) for a in cfg.scene.entities["robot"].articulation.actuators]
    self.assertEqual(gains, [(20., 1.), (20., 1.), (40., 2.)])
    self.assertFalse(agent.algorithm.use_clipped_value_loss)
    self.assertEqual((agent.max_iterations, agent.experiment_name), (10000, "go2_qpace"))

  def test_curriculum_schedule(self):
    c = QPaceCurriculumCfg(critic_warmup_iterations=10, noise_start_iteration=20,
                           lambda_ramp_iterations=10, lambda_max=20.)
    self.assertTrue(c.actor_frozen(9) and not c.actor_frozen(10))
    self.assertEqual([c.lambda_cap(i) for i in (0, 19, 20, 24, 29, 100)], [0., 0., 2., 10., 20., 20.])
    with self.assertRaises(ValueError):
      QPaceCurriculumCfg(critic_warmup_iterations=50, noise_start_iteration=10)

  def test_frozen_actor_and_lambda_cap(self):
    alg, obs = _algorithm()
    alg.curriculum = QPaceCurriculumCfg(critic_warmup_iterations=1, noise_start_iteration=1,
                                        lambda_ramp_iterations=1, lambda_max=0.5)
    alg.failure_cost = 10.
    before = {k: v.clone() for k, v in alg.actor.state_dict().items()}
    lr = alg.learning_rate

    def rollout():
      with torch.no_grad():
        alg.act(obs)
        alg.process_env_step(obs, torch.tensor([1., 3.]), torch.ones(2), {
          "noise_cost": torch.tensor([.5, .5]), "cost_limit": torch.tensor([.01, .01]),
          "true_terminations": torch.tensor([True, True])})
        alg.compute_returns(obs)
      return alg.update()

    rollout()  # iteration 0: actor 고정, λ = 0
    for k, v in alg.actor.state_dict().items():
      if "normalizer" not in k:
        torch.testing.assert_close(v, before[k])
    self.assertEqual(alg.learning_rate, lr)
    self.assertTrue(torch.all(alg.pid.multipliers == 0))
    rollout()  # iteration 1: λ 켜짐, 상한 0.5
    self.assertTrue(torch.all(alg.pid.multipliers <= 0.5) and torch.all(alg.pid.multipliers > 0))
    changed = any(not torch.equal(v, before[k]) for k, v in alg.actor.state_dict().items()
                  if "normalizer" not in k)
    self.assertTrue(changed)

  @unittest.skipUnless(BASELINE.is_file(), "baseline checkpoint not present")
  def test_warm_start_reproduces_baseline_actor(self):
    cfg = asdict(load_rl_cfg(TASK))
    n = 47
    obs = TensorDict({"actor": torch.zeros(3, n + 1), "critic": torch.zeros(3, 75)}, batch_size=[3])
    alg = QuietPawPPO.construct_algorithm(obs, SimpleNamespace(num_actions=12, num_envs=3), cfg, "cpu")
    info = warm_start_from_baseline(alg.actor, alg.critic, BASELINE)
    self.assertEqual((info["actor_inputs"], info["critic_inputs"]), (47, 74))
    src = torch.load(BASELINE, map_location="cpu", weights_only=False)["actor_state_dict"]
    x = torch.randn(3, n)
    ref = (x - src["obs_normalizer._mean"]) / (src["obs_normalizer._std"] + 1e-2)
    for i in (0, 2, 4):
      ref = torch.nn.functional.elu(ref @ src[f"mlp.{i}.weight"].T + src[f"mlp.{i}.bias"])
    ref = ref @ src["mlp.6.weight"].T + src["mlp.6.bias"]
    alg.actor.eval()
    for cond in (0., 0.5, 1.):  # 조건 열 가중치가 0이므로 목표 dB와 무관하게 baseline과 같다.
      obs_q = TensorDict({"actor": torch.cat([x, torch.full((3, 1), cond)], -1),
                          "critic": torch.zeros(3, 75)}, batch_size=[3])
      out = alg.actor(obs_q)
      torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

  @unittest.skipUnless(os.environ.get("QPACE_GPU_TESTS") == "1", "requires GPU simulation")
  def test_noise_model_wiring_in_simulation(self):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 16
    env = cfg.build_env(device="cuda:0")
    try:
      robot = env.scene["robot"]
      unitree = ["FR", "FL", "RR", "RL"]
      self.assertEqual([env.foot_names[i] for i in env.sensor_to_unitree], unitree)
      self.assertEqual([robot.site_names[i] for i in env.unitree_site_ids], unitree)
      joints = [robot.joint_names[i] for i in env.unitree_joint_ids]
      actuators = [robot.actuator_names[i] for i in env.unitree_actuator_ids]
      self.assertEqual(joints, actuators)
      self.assertEqual(joints[:3], ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"])
      env.reset()
      action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
      total = torch.zeros(env.num_envs, device=env.device)
      for _ in range(50):
        _, _, _, _, extras = env.step(action)
        cost = extras["noise_cost"]
        self.assertTrue(torch.isfinite(cost).all() and (cost >= 0).all())
        total += extras["quietpaw_metrics"]["noise_energy"]
      self.assertTrue((total > 0).any(), "no touchdown energy detected")
      ratio = env.extras["cost_limit"]
      torch.testing.assert_close(10 * torch.log10(ratio) + cfg.noise_condition.reference_db,
                                 env.extras["target_db"])
      self.assertGreater(env.failure_cost, 10 ** ((cfg.noise_condition.max_db - cfg.noise_condition.reference_db) / 10))
    finally:
      env.close()

  @unittest.skipUnless(os.environ.get("QPACE_GPU_TESTS") == "1", "requires GPU simulation")
  def test_positive_reward_clip_keeps_termination_penalty(self):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 4
    env = cfg.build_env(device="cuda:0")
    try:
      env.reset()
      action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
      _, reward, _, _, _ = env.step(action)
      term = env.reward_manager._step_reward[:, env._termination_reward_idx] * env.step_dt
      self.assertTrue(((reward - term) >= -1e-6).all())
    finally:
      env.close()


if __name__ == "__main__":
  unittest.main()
