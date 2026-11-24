"""DiffImpact 앙상블 소음 모델(RL 비용)의 방향·클리핑·비용 경계를 검증한다. 시뮬레이터 불필요.

원본 DiffImpact 합성과의 수치 일치는 로컬 검증 스크립트(src/diffimpact, 저장소 밖)로 확인했다.
"""

import unittest

import torch

from src.locomotion_rl.go2_noise_aware.noise_model.diffimpact import (
  DiffImpactNoiseModel,
  touchdown_cost,
)
from src.locomotion_rl.go2_noise_aware.noise_model.touchdown_features import (
  DiffImpactNoiseCost,
  TouchdownFeatureTracker,
  leg_kinematics,
  torque_grf_z,
)


class NoiseModelTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.nm = DiffImpactNoiseModel()
    cls.mid = ((cls.nm.lo + cls.nm.hi) / 2).mean(0)  # 학습 범위 중앙의 착지
    cls.legs = torch.arange(4)

  def energy(self, change=None, offset=None):
    x = self.mid.repeat(4, 1)
    for name, value in (change or {}).items():
      x[:, self.nm.feature_index(name)] = value
    return self.nm.foot_energy(x, self.legs, offset)

  def test_shapes_and_units(self):
    out = self.energy()
    self.assertEqual(out["energy"].shape, (4,))
    self.assertTrue(torch.all(out["energy"] > 0))
    self.assertTrue(torch.all(out["db"] < 0))  # dB re FS²
    self.assertTrue(torch.all(out["ood"] == 0))

  def test_slower_descent_is_quieter(self):
    k = self.nm.feature_index("foot_vz_pre")
    base = self.energy()["db"]
    slow = self.energy({"foot_vz_pre": self.mid[k] * 0.7})["db"]  # vz < 0: 하강 30% 느리게
    self.assertTrue(torch.all(slow <= base + 1e-4), (base, slow))

  def test_slower_body_drop_is_not_louder(self):
    k = self.nm.feature_index("body_vz_pre")
    base = self.energy()["db"]
    slow = self.energy({"body_vz_pre": self.mid[k] * 0.7})["db"]
    self.assertTrue(torch.all(slow <= base + 1e-4), (base, slow))

  def test_clipping_freezes_prediction_and_adds_ood(self):
    k = self.nm.feature_index("foot_vz_pre")
    edge = self.nm.lo[:, k].max()
    at_edge = self.energy({"foot_vz_pre": edge})
    beyond = self.energy({"foot_vz_pre": edge - 1.0})  # 범위보다 1 m/s 더 빠른 하강
    self.assertTrue(torch.allclose(at_edge["db"], beyond["db"], atol=1e-3))
    self.assertTrue(torch.all(beyond["ood"] > at_edge["ood"]))

  def test_posture_outside_range_has_no_penalty(self):
    k = self.nm.feature_index("foot_z")
    out = self.energy({"foot_z": self.nm.lo[:, k].min() - 0.05})
    self.assertTrue(torch.all(out["ood"] == 0))
    self.assertGreater(out["ood_frac"][k].item(), 0)  # 감시용 비율에는 남음

  def test_offset_inside_window(self):
    first = self.energy(offset=torch.zeros(4))["energy"]
    later = self.energy(offset=torch.full((4,), 0.01))["energy"]
    self.assertTrue(torch.all(torch.isfinite(later)))
    self.assertFalse(torch.allclose(first, later))

  def test_cost_floor_and_no_ceiling(self):
    out = {"db": torch.tensor([-90.0, -60.0, -30.0]), "ood": torch.zeros(3)}
    cost = touchdown_cost(out, floor_db=-70.0, ref_db=-60.0, ood_weight=0.1)
    self.assertEqual(cost[0].item(), 0.0)
    self.assertAlmostEqual(cost[1].item(), 1.0, places=5)
    self.assertGreater(cost[2].item(), 1.0)
    with self.assertRaises(ValueError):
      touchdown_cost(out, floor_db=-60.0, ref_db=-60.0, ood_weight=0.1)


class TrackerTest(unittest.TestCase):
  """MuJoCo 입력 → 특징. 학습 데이터와의 수치 일치는 로컬 검증 스크립트로 확인했다."""

  def test_leg_kinematics_matches_urdf(self):
    p, jac = leg_kinematics(torch.zeros(1, 12))
    self.assertTrue(torch.allclose(p[0, 0], torch.tensor([0.1934, -0.142, -0.426]), atol=1e-4))
    self.assertTrue(torch.allclose(p[0, 3], torch.tensor([-0.1934, 0.142, -0.426]), atol=1e-4))
    q = torch.randn(1, 12) * 0.3  # 야코비안 = 수치 미분
    eps = 1e-3
    for j in range(3):
      dq = torch.zeros(1, 12); dq[0, j] = eps
      num = (leg_kinematics(q + dq)[0] - leg_kinematics(q - dq)[0])[0, 0] / (2 * eps)
      self.assertTrue(torch.allclose(num, leg_kinematics(q)[1][0, 0, :, j], atol=1e-3))

  def test_torque_grf_supports_weight(self):
    # 정지 자세에서 발에 위로 50 N 을 주는 토크 τ = −Jᵀ F 면 추정 Fz = 50 N
    q = torch.tensor([[0.0, 0.8, -1.6] * 4])
    _, jac = leg_kinematics(q)
    f = torch.zeros(1, 4, 3); f[..., 2] = 50.0
    tau = -(jac.transpose(-1, -2) @ f[..., None])[..., 0].reshape(1, 12)
    fz = torque_grf_z(q, tau, torch.tensor([[1.0, 0, 0, 0]]))
    self.assertTrue(torch.allclose(fz, torch.full((1, 4), 50.0), atol=1e-3))

  def run_pair(self, tracker, second_delay_steps=2):
    """FR 착지 후 second_delay_steps substep 뒤 RL 착지. 확정된 (substep, 다리) 목록."""
    B = 1
    q = torch.tensor([[0.0, 0.8, -1.6] * 4])
    pos, _ = leg_kinematics(q)
    vel = torch.zeros(B, 4, 3); vel[..., 2] = -0.6
    contact = torch.zeros(B, 4); contact[0, [1, 2]] = 100.0  # FL, RR 입각
    out = []
    for i in range(12):
      if i >= 2:
        contact[0, 0] = 100.0
      if i >= 2 + second_delay_steps:
        contact[0, 3] = 100.0
      ready = tracker.sample(vel, pos, contact.clone(), q, torch.zeros(B, 12), torch.tensor([[1.0, 0, 0, 0]]),
                             torch.tensor([[0.5, 0.0, 0.0]]))
      out += [(i, leg) for leg in ready[0].nonzero().flatten().tolist()]
    return out

  def test_tracker_timing_offset_and_second(self):
    names = DiffImpactNoiseModel().features
    tr = TouchdownFeatureTracker(1, names, "cpu")
    events = self.run_pair(tr)
    # 처음부터 닿아 있던 FL·RR 도 reset 직후 첫 접촉으로 착지 처리 (QuietPaw 와 같은 규칙).
    # 착지 4 substep (20 ms) 뒤 확정.
    self.assertEqual(events, [(4, 1), (4, 2), (6, 0), (8, 3)])
    self.assertAlmostEqual(tr.offset[0, 0].item(), 0.0, places=6)
    self.assertAlmostEqual(tr.offset[0, 3].item(), 0.010, places=6)
    k = names.index("is_second")
    self.assertEqual(tr.feats[0, 0, k].item(), 0.0)
    self.assertEqual(tr.feats[0, 3, k].item(), 1.0)
    self.assertAlmostEqual(tr.feats[0, 0, names.index("foot_vz_pre")].item(), -0.6, places=5)
    self.assertAlmostEqual(tr.feats[0, 0, names.index("body_vz_pre")].item(), 0.6, places=5)  # FL·RR 입각 발 −(−0.6)

  def test_cost_wrapper(self):
    cost = DiffImpactNoiseCost(1, "cpu")
    cost.begin_step()
    self.run_pair(cost.tracker)  # 직접 tracker 로는 비용이 쌓이지 않음
    c, _ = cost.end_step()
    self.assertEqual(c.item(), 0.0)
    cost = DiffImpactNoiseCost(1, "cpu")
    q = torch.tensor([[0.0, 0.8, -1.6] * 4]); pos, _ = leg_kinematics(q)
    vel = torch.zeros(1, 4, 3); vel[..., 2] = -0.6
    contact = torch.zeros(1, 4); contact[0, [1, 2]] = 100.0
    cost.begin_step()
    for i in range(8):
      if i >= 2:
        contact[0, 0] = 100.0
      cost.sample(vel, pos, contact.clone(), q, torch.zeros(1, 12), torch.tensor([[1.0, 0, 0, 0]]),
                  torch.tensor([[0.5, 0.0, 0.0]]))
    c, log = cost.end_step()
    self.assertGreater(c.item(), 0.0)
    self.assertEqual(log["touchdowns"].item(), 3.0)  # 처음 닿아 있던 FL·RR + FR


if __name__ == "__main__":
  unittest.main()
