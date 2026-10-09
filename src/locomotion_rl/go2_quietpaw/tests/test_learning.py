"""소음 제약 학습의 수학·저장 경계를 검증한다. 물리 시뮬레이터 실행은 필요 없다."""

import unittest
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from src.locomotion_rl.go2_quietpaw.step_02_learning.costs import NoiseCostCfg
from src.locomotion_rl.go2_quietpaw.step_02_learning.mdp.costs import LandingMeasurements, noise_cost
from src.locomotion_rl.go2_quietpaw.step_02_learning.mdp.noise_condition import NoiseCondition, NoiseConditionCfg
from src.locomotion_rl.go2_quietpaw.step_03_training.critics import DecomposedCritic
from src.locomotion_rl.go2_quietpaw.step_03_training.pid_lagrangian import PidLagrangian, PidLagrangianCfg
from src.locomotion_rl.go2_quietpaw.step_03_training.ppo import unitree_go2_ppo_runner_cfg
from src.locomotion_rl.go2_quietpaw.step_03_training.ppo_lagrangian import QuietPawPPO
from src.locomotion_rl.go2_quietpaw.step_03_training.storage import generalized_advantage


def make_algorithm(num_envs=3, steps=3):
  obs = TensorDict({"actor": torch.randn(num_envs, 48), "critic": torch.randn(num_envs, 75)}, [num_envs])
  limits = torch.linspace(0.35, 0.02, num_envs)
  obs["actor"][:, -1] = limits
  obs["critic"][:, -1] = limits
  cfg = asdict(unitree_go2_ppo_runner_cfg())
  cfg["num_steps_per_env"] = steps
  cfg["actor"]["hidden_dims"] = (16,)
  cfg["critic"].update(hidden_dims=(16,), feature_dim=4, condition_hidden_dims=(8,))
  cfg["algorithm"].update(num_mini_batches=2, num_learning_epochs=1)
  alg = QuietPawPPO.construct_algorithm(obs, SimpleNamespace(num_envs=num_envs, num_actions=12), cfg, "cpu")
  return alg, obs


class NoiseTests(unittest.TestCase):
  def test_cost_is_bounded_and_increases_with_impact(self):
    cfg = NoiseCostCfg()
    forces = torch.tensor([0., 50., 100., 10000.])[:, None].expand(-1, 4)
    force_cost = noise_cost(forces, torch.zeros_like(forces), cfg)
    velocity_cost = noise_cost(torch.zeros_like(forces), torch.tensor([0., .1, .5, 100.])[:, None].expand(-1, 4), cfg)
    self.assertTrue(torch.all(force_cost.diff() >= 0))
    self.assertTrue(torch.all(velocity_cost.diff() > 0))
    combined = noise_cost(forces, forces, cfg)
    self.assertTrue(torch.all((combined >= 0) & (combined <= 1)))

  def test_landing_uses_preimpact_velocity_once_and_keeps_peak(self):
    measurements = LandingMeasurements(1, 1, "cpu")
    force = torch.tensor([[[0., 0., 80.]]])
    measurements.begin_step()
    measurements.sample(force, torch.tensor([[-2.]]), 1.)
    measurements.sample(force / 2, torch.tensor([[-4.]]), 1.)
    self.assertEqual(measurements.impact_speed_squared.item(), 4.)
    self.assertEqual(measurements.peak_forces.item(), 80.)
    self.assertEqual(measurements.landing_count.item(), 1.)
    measurements.begin_step()
    measurements.sample(force, torch.tensor([[-4.]]), 1.)
    self.assertEqual(measurements.impact_speed_squared.item(), 0.)
    measurements.reset(torch.tensor([0]))
    measurements.sample(force, torch.tensor([[-3.]]), 1.)
    self.assertEqual(measurements.impact_speed_squared.item(), 9.)

  def test_condition_direction(self):
    condition = NoiseCondition(NoiseConditionCfg(), 3, "cpu")
    torch.testing.assert_close(condition.cost_limit, torch.tensor([.35, .185, .02]))
    condition.set_quietness(1.)
    torch.testing.assert_close(condition.cost_limit, torch.full((3,), .02))
    with self.assertRaises(ValueError):
      condition.set_quietness(1.1)

  def test_pid_updates_independently_and_restores_history(self):
    cfg = PidLagrangianCfg(kp=1., ki=.2, kd=.1, smoothing=0.)
    pid = PidLagrangian(2, cfg, "cpu")
    result = pid.update(torch.tensor([.8, .1]), torch.tensor([.2, .5]))
    self.assertGreater(result[0], 0)
    self.assertEqual(result[1], 0)
    restored = PidLagrangian(2, cfg, "cpu")
    restored.load_state_dict(deepcopy(pid.state_dict()))
    torch.testing.assert_close(
      pid.update(torch.tensor([.1, .9]), torch.tensor([.2, .5])),
      restored.update(torch.tensor([.1, .9]), torch.tensor([.2, .5])),
    )


class LearningTests(unittest.TestCase):
  def test_gae_does_not_cross_episode_boundaries(self):
    signals = torch.tensor([[[1., .1]], [[2., .2]]])
    values = torch.zeros_like(signals)
    done = torch.tensor([[[1]], [[0]]])
    returns, _ = generalized_advantage(signals, values, done, torch.tensor([[10., 1.]]), .9, 1.)
    torch.testing.assert_close(returns[0], signals[0])
    torch.testing.assert_close(returns[1], torch.tensor([[11., 1.1]]))

  def test_critic_condition_has_its_own_path(self):
    critic = DecomposedCritic(5, hidden_dims=(8,), feature_dim=4, condition_hidden_dims=(8,))
    state = torch.randn(2, 5)
    condition = torch.tensor([[.1], [.3]], requires_grad=True)
    seen = []
    hook = critic.state_features.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    value = critic(state, condition)
    critic(state, condition + .2)
    hook.remove()
    torch.testing.assert_close(seen[0], seen[1])
    value.sum().backward()
    self.assertGreater(condition.grad.abs().sum().item(), 0)

  def test_timeout_bootstraps_terminal_observation_and_not_a_fall(self):
    alg, obs = make_algorithm(3, 1)
    terminal = obs.clone()
    terminal["critic"][:, 0] += 100.
    with torch.no_grad():
      alg.act(obs)
      terminal_values = alg.critic(terminal)
      alg.process_env_step(obs, torch.ones(3), torch.tensor([1, 1, 0]), {
        "noise_cost": torch.full((3,), .2), "cost_limit": obs["actor"][:, -1],
        "time_outs": torch.tensor([True, True, False]),
        "true_terminations": torch.tensor([False, True, False]),
        "terminal_observations": terminal,
      })
    self.assertAlmostEqual(alg.storage.rewards[0, 0, 0].item(), 1 + alg.gamma * terminal_values[0, 0].item(), places=5)
    self.assertAlmostEqual(alg.storage.costs[0, 0, 0].item(), .2 + alg.gamma * terminal_values[0, 1].item(), places=5)
    self.assertEqual(alg.storage.rewards[0, 1, 0].item(), 1.)
    torch.testing.assert_close(alg.storage.raw_costs[0, :, 0], torch.full((3,), .2))

  def test_full_update_and_checkpoint_round_trip(self):
    alg, obs = make_algorithm()
    before = [p.detach().clone() for p in alg.actor.parameters()]
    with torch.no_grad():
      for _ in range(3):
        alg.act(obs)
        alg.process_env_step(obs, torch.rand(3), torch.zeros(3), {
          "noise_cost": torch.tensor([.1, .3, .8]), "cost_limit": obs["actor"][:, -1],
        })
      alg.compute_returns(obs)
    # 9개 sample / 2개 minibatch에서도 마지막 sample이 보존되는지 확인한다.
    batches = list(alg.storage.mini_batch_generator(2, 1))
    self.assertEqual(sum(b.actions.shape[0] for b in batches), 9)
    losses = alg.update()
    self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, alg.actor.parameters())))
    state = deepcopy(alg.save())
    restored, _ = make_algorithm()
    restored.load(state, None, True)
    torch.testing.assert_close(alg.actor(obs), restored.actor(obs))
    torch.testing.assert_close(alg.critic(obs), restored.critic(obs))
    torch.testing.assert_close(alg.pid.multipliers, restored.pid.multipliers)
    self.assertEqual(alg.learning_rate, restored.learning_rate)
    # 평가에서는 환경 수가 달라도 Actor만 불러올 수 있다.
    inference, _ = make_algorithm(1, 1)
    inference.load(state, {"actor": True}, True)


if __name__ == "__main__":
  unittest.main()
