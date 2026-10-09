"""실제 GPU 환경·timeout·저장·ONNX·play 연결을 확인하는 선택적 통합 테스트.

QUIETPAW_GPU_TESTS=1 python -m unittest discover -s src/locomotion_rl/go2_quietpaw/tests -v
"""

import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from src.locomotion_rl.go2_quietpaw.step_01_environment.quietpaw_env import QuietPawEnv
from src.locomotion_rl.go2_quietpaw.step_03_training.ppo import unitree_go2_ppo_runner_cfg
from src.locomotion_rl.go2_quietpaw.step_03_training.runner import QuietPawOnPolicyRunner
from src.locomotion_rl.go2_quietpaw.step_04_evaluation import play


@unittest.skipUnless(os.environ.get("QUIETPAW_GPU_TESTS") == "1", "Set QUIETPAW_GPU_TESTS=1 to run physics integration.")
class GpuPipelineTests(unittest.TestCase):
  def test_timeout_update_resume_export_and_play(self):
    cfg = load_env_cfg("Unitree-Go2-QuietPaw-Flat")
    cfg.scene.num_envs = 4
    cfg.episode_length_s = .04  # 두 정책 스텝마다 timeout을 발생시킨다.
    cfg.seed = 42
    agent = unitree_go2_ppo_runner_cfg()
    agent.actor.hidden_dims = (32, 16)
    agent.critic.hidden_dims = (32, 16)
    agent.num_steps_per_env = 4
    agent.algorithm.num_mini_batches = 2
    agent.algorithm.num_learning_epochs = 1
    env = RslRlVecEnvWrapper(QuietPawEnv(cfg, "cuda:0"))
    try:
      with tempfile.TemporaryDirectory(prefix="quietpaw_test_") as directory:
        runner = QuietPawOnPolicyRunner(env, asdict(agent), device="cuda:0")
        captured = {}
        reset = env.unwrapped._reset_idx

        def record_terminal(env_ids=None):
          captured["critic"] = env.unwrapped.observation_manager.compute_group("critic", update_history=True).clone()
          reset(env_ids)

        env.unwrapped._reset_idx = record_terminal
        obs = env.get_observations()
        self.assertEqual(obs["actor"].shape, (4, 48))
        self.assertEqual(obs["critic"].shape, (4, 75))
        timeout_count = 0
        with torch.inference_mode():
          for _ in range(agent.num_steps_per_env):
            old_critic = obs["critic"].clone()
            actions = runner.alg.act(obs)
            obs, reward, done, extras = env.step(actions)
            if extras["time_outs"].any():
              timeout_count += 1
              terminal = extras["terminal_observations"]["critic"]
              torch.testing.assert_close(terminal, captured["critic"])
              self.assertFalse(torch.equal(terminal, old_critic))
            runner.alg.process_env_step(obs, reward, done, extras)
          runner.alg.compute_returns(obs)
        self.assertGreater(timeout_count, 0)
        losses = runner.alg.update()
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        checkpoint = Path(directory) / "model_0.pt"
        runner.save(str(checkpoint))
        policy = runner.get_inference_policy("cuda:0")
        with torch.inference_mode():
          expected = policy(obs)[:1].cpu().numpy()

        # PyTorch와 독립된 ONNX reference 실행기로 실제 내보낸 그래프를 비교한다.
        onnx_path = str(Path(directory) / "policy.onnx")
        onnx.checker.check_model(onnx_path)
        output = ReferenceEvaluator(onnx_path).run(None, {"obs": obs["actor"][:1].cpu().numpy()})[0]
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

        restored = QuietPawOnPolicyRunner(env, asdict(agent), device="cuda:0")
        restored.load(str(checkpoint))
        self.assertEqual(restored.current_learning_iteration, 1)
        torch.testing.assert_close(restored.alg.pid.multipliers, runner.alg.pid.multipliers)
        with torch.inference_mode():
          torch.testing.assert_close(restored.alg.actor(obs), policy(obs))

        # 창 대신 짧은 루프를 넣어 play.py의 실제 모델 복원·추론 경로를 검증한다.
        class HeadlessViewer:
          def __init__(self, play_env, play_policy):
            self.env, self.policy = play_env, play_policy

          def run(self):
            with torch.inference_mode():
              observations = self.env.get_observations()
              for _ in range(4):
                observations, _, _, _ = self.env.step(self.policy(observations))

        with patch.object(play, "NativeMujocoViewer", HeadlessViewer):
          play.run_play("Unitree-Go2-QuietPaw-Flat", play.PlayConfig(
            checkpoint_file=str(checkpoint), num_envs=2, device="cuda:0", viewer="native", quietness=.8,
          ))
    finally:
      env.close()
