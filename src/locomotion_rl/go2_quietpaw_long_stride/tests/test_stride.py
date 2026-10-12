"""CLI 설정 전파, checkpoint 복원, 착지 측정의 에피소드 경계를 확인한다."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
import tyro
import mjlab
from mjlab.tasks.registry import load_env_cfg

from src.locomotion_rl.go2_quietpaw_long_stride.step_02_learning.stride import apply_stride_settings
from src.locomotion_rl.go2_quietpaw_long_stride.step_02_learning.mdp.stride_metrics import StrideMeasurements
from src.locomotion_rl.go2_quietpaw_long_stride.step_03_training.train import TrainConfig
from src.locomotion_rl.go2_quietpaw_long_stride.step_04_evaluation.checkpoint import load_evaluation_settings


class StrideTests(unittest.TestCase):
  def test_cli_overrides_both_phases_and_pose_without_changing_original(self):
    cfg = tyro.cli(TrainConfig, default=TrainConfig.from_task("Unitree-Go2-QuietPaw-LongStride-Flat"),
                   args=["--env.stride.gait-period", "0.83",
                         "--env.stride.thigh-pose-std", "0.51",
                         "--env.stride.calf-pose-std", "0.67"], config=mjlab.TYRO_FLAGS)
    apply_stride_settings(cfg.env)
    self.assertEqual(cfg.env.rewards["foot_gait"].params["period"], .83)
    for name in ("actor", "critic"):
      self.assertEqual(cfg.env.observations[name].terms["phase"].params["period"], .83)
    for regime in ("std_walking", "std_running"):
      values = cfg.env.rewards["pose"].params[regime]
      self.assertEqual(values[r".*(FR|FL|RR|RL)_thigh_joint.*"], .51)
      self.assertEqual(values[r".*(FR|FL|RR|RL)_calf_joint.*"], .67)
      self.assertEqual(values[r".*(FR|FL|RR|RL)_hip_joint.*"], .15)
    original = load_env_cfg("Unitree-Go2-QuietPaw-Flat")
    self.assertEqual(original.rewards["foot_gait"].params["period"], .6)
    self.assertEqual(original.gait_curriculum.minimum_scale, 0.)
    self.assertEqual(cfg.env.gait_curriculum.minimum_scale, .2)

  def test_invalid_cli_values_fail_before_environment_creation(self):
    for field in ("gait_period", "thigh_pose_std", "calf_pose_std"):
      for value in (0., -1., float("nan"), float("inf")):
        cfg = load_env_cfg("Unitree-Go2-QuietPaw-LongStride-Flat")
        setattr(cfg.stride, field, value)
        with self.assertRaises(ValueError):
          apply_stride_settings(cfg)

  def test_checkpoint_restores_nondefault_period_in_flat_and_rough(self):
    cfg = load_env_cfg("Unitree-Go2-QuietPaw-LongStride-Flat")
    cfg.stride.gait_period = .83
    saved = {key: asdict(getattr(cfg, key)) for key in ("stride", "noise_cost", "noise_condition")}
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.pt"
      torch.save({"infos": {"quietpaw": saved, "quietpaw_agent": {"example": 1}}}, path)
      for terrain in ("Flat", "Rough"):
        play_cfg = load_env_cfg(f"Unitree-Go2-QuietPaw-LongStride-{terrain}", play=True)
        self.assertEqual(load_evaluation_settings(play_cfg, path), {"example": 1})
        self.assertEqual(play_cfg.stride.gait_period, .83)
        self.assertEqual(play_cfg.observations["actor"].terms["phase"].params["period"], .83)
      del saved["stride"]
      torch.save({"infos": {"quietpaw": saved, "quietpaw_agent": {}}}, path)
      with self.assertRaisesRegex(ValueError, "LongStride"):
        load_evaluation_settings(cfg, path)

  def test_measurements_exclude_first_contact_reset_and_standing(self):
    tracker = StrideMeasurements(2, 1, "cpu")
    positions = torch.zeros(2, 1, 3)
    command = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    landing = torch.ones(2, 1, dtype=torch.bool)
    tracker.sample(positions, landing, command, .1)
    self.assertEqual(tracker.count.sum().item(), 0.)
    tracker.begin_step()
    for _ in range(6):
      tracker.sample(positions, ~landing, command, .1)
    positions[:, :, 0] = .7
    tracker.sample(positions, landing, command, .1)
    torch.testing.assert_close(tracker.distance_sum, torch.tensor([.7, 0.]))
    torch.testing.assert_close(tracker.period_sum, torch.tensor([.7, 0.]))
    self.assertEqual(tracker.count.sum().item(), 1.)
    tracker.reset(torch.tensor([0]))
    self.assertEqual(tracker.count.sum().item(), 1.)  # reset 전 집계를 보존한다.
    tracker.begin_step()
    positions[:, :, 0] = 10.
    tracker.sample(positions, landing, command, .1)
    self.assertEqual(tracker.count.sum().item(), 0.)
    self.assertEqual(tracker.logs()["LongStride/measured_feet"].item(), 0)


if __name__ == "__main__":
  unittest.main()
