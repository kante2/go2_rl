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
from src.locomotion_rl.go2_qpace.step_02_learning.noise_model import (
  SPL_OFFSET_DB,
  DiffImpactNoiseModel,
  DiffImpactStepNoise,
)
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

  def test_gentle_rl_landing_is_floored(self):
    # 물리 법칙 모델은 v → 0 에서 RL 다리를 약 −106 dB 로 본다. 발별 하한(기본 db_ref p1)에서 멈춰야 한다.
    model = DiffImpactNoiseModel()
    kf, kb = model.features.index("foot_vz_pre"), model.features.index("body_vz_pre")
    x = model.mu[0].clone().repeat(2, 1)
    x[:, kb] = 0.0
    x[0, kf], x[1, kf] = -0.02, 0.0  # 거의 멈춘 채 내려놓기
    raw = model.foot_energy(x, torch.tensor([3, 3]))["db"]
    floor = model.db_ref["p1"]
    self.assertTrue(torch.all(raw < floor - 5.0), raw)  # 원본은 하한보다 훨씬 조용하다고 본다
    noise = DiffImpactStepNoise(1, "cpu", model=model)
    self.assertAlmostEqual(noise.foot_floor_db, floor)
    noise.step_ready[0, 3] = True
    noise.step_feats[0, 3] = x[1]
    out = noise.end_step()
    self.assertAlmostEqual(out["foot_db"][0, 3].item(), floor, places=4)
    self.assertAlmostEqual(out["log"]["floored_frac"].item(), 1.0)
    self.assertAlmostEqual(out["db_spl"][0].item(), floor + SPL_OFFSET_DB, places=4)
    off = DiffImpactStepNoise(1, "cpu", model=model, foot_floor_db=float("-inf"))
    off.step_ready[0, 3] = True
    off.step_feats[0, 3] = x[1]
    self.assertAlmostEqual(off.end_step()["foot_db"][0, 3].item(), raw[1].item(), places=3)

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
    # C 방식 재학습 구조 (2026-11-27 붕괴 대책)
    self.assertEqual(cfg.gait_curriculum.minimum_scale, 1.0)  # 보행 리듬 보상 유지
    self.assertGreater(agent.algorithm.anchor_coef, 0)  # baseline 앵커
    self.assertIsNotNone(agent.algorithm.max_std)
    self.assertGreater(cfg.noise_cost.uncertainty_k, 0)
    self.assertIsNotNone(cfg.qpace_curriculum.target_gate_tracking_error)

  def test_target_gate_holds_ramp(self):
    from src.locomotion_rl.go2_qpace.step_03_training.runner import QuietPawOnPolicyRunner
    cur = QPaceCurriculumCfg(critic_warmup_iterations=1, noise_start_iteration=2, target_start_min_db=-50.,
                             target_ramp_iterations=4, target_gate_tracking_error=0.5)
    self.assertTrue(cur.target_gate_open(0.4) and cur.target_gate_open(float("nan")))
    self.assertFalse(cur.target_gate_open(0.6))
    cond_cfg = NoiseConditionCfg(min_db=-58., max_db=-45., reference_db=-52.5)
    errors = iter([0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.4])
    env = SimpleNamespace(cfg=SimpleNamespace(noise_condition=cond_cfg, qpace_curriculum=cur),
                          noise_condition=NoiseCondition(cond_cfg, 2, "cpu"),
                          pop_tracking_error=lambda: next(errors))
    fake = SimpleNamespace(env=SimpleNamespace(unwrapped=env), alg=SimpleNamespace(target_progress=0))
    got = [QuietPawOnPolicyRunner._update_target_range(fake, i) for i in range(1, 8)]
    # iter 1, 2: ramp 전. 3, 4: 오차 0.9 → 멈춤. 이후 다시 진행.
    self.assertEqual(got, [-50., -50., -50., -50., -52., -54., -56.])

  def test_anchor_pulls_toward_baseline_and_std_cap(self):
    alg, obs = _algorithm()
    alg.anchor_coef, alg.max_std = 10.0, 0.5
    alg.set_anchor()
    with torch.no_grad():
      alg.actor.mlp[-1].bias.add_(1.0)  # 평균 행동을 baseline 에서 행동 차원마다 1 만큼 옮긴다
      alg.actor.distribution.std_param.fill_(2.0)

    def rollout():
      with torch.no_grad():
        alg.act(obs)
        alg.process_env_step(obs, torch.zeros(2), torch.zeros(2), {
          "noise_cost": torch.zeros(2), "cost_limit": torch.ones(2)})
        alg.compute_returns(obs)
      return alg.update()["anchor_mse"]

    mse = [rollout() for _ in range(5)]
    self.assertAlmostEqual(mse[0], 2.0, places=4)  # Σ_dim 1²
    self.assertTrue(all(b < a for a, b in zip(mse, mse[1:])), mse)  # 보상·비용 신호가 없으면 baseline 으로 돌아간다
    self.assertTrue(torch.all(alg.actor.distribution.std_param <= 0.5))
    state = alg.save()
    self.assertIn("cncp_anchor_state_dict", state)

  def test_curriculum_schedule(self):
    c = QPaceCurriculumCfg(critic_warmup_iterations=10, noise_start_iteration=20,
                           lambda_ramp_iterations=10, lambda_max=20.)
    self.assertTrue(c.actor_frozen(9) and not c.actor_frozen(10))
    self.assertEqual([c.lambda_cap(i) for i in (0, 19, 20, 24, 29, 100)], [0., 0., 2., 10., 20., 20.])
    with self.assertRaises(ValueError):
      QPaceCurriculumCfg(critic_warmup_iterations=50, noise_start_iteration=10)

  def test_target_range_curriculum(self):
    c = QPaceCurriculumCfg(critic_warmup_iterations=10, noise_start_iteration=20, target_start_min_db=-52.,
                           target_ramp_iterations=10)
    lo, hi = -60., -45.
    self.assertEqual([c.target_min_db(i, lo, hi) for i in (0, 19, 20, 25, 30, 100)],
                     [-52., -52., -52., -56., -60., -60.])
    self.assertEqual(QPaceCurriculumCfg(target_start_min_db=None).target_min_db(0, lo, hi), lo)
    cond = NoiseCondition(NoiseConditionCfg(min_db=lo, max_db=hi, reference_db=-52.5), 4, "cpu")
    cond.set_min_db(-51.)
    torch.testing.assert_close(cond.target_db, torch.tensor([-51., -49., -47., -45.]))
    # 정책 입력 정규화는 전체 범위 기준 (목표 dB 의 의미가 커리큘럼 중에도 같다)
    torch.testing.assert_close(cond.normalized, (cond.target_db - lo) / (hi - lo))
    cond.set_target_db(-50.)  # 고정 목표(평가·배치)는 커리큘럼이 바꾸지 않는다
    cond.set_min_db(-60.)
    self.assertTrue(torch.all(cond.target_db == -50.))

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
