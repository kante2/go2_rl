"""DiffImpact 물리 법칙 소음 모델(충격량 ∝ v, Hertz τ ∝ v^-1/5)과 착지 추적·스텝 소음(에너지 합 → dB)을 검증한다. 시뮬레이터 불필요.

원본 DiffImpact 합성과의 수치 일치는 로컬 검증 스크립트(src/diffimpact, 저장소 밖)로 확인했다.
"""

import unittest

import torch

from src.locomotion_rl.go2_noise_aware.noise_model.diffimpact import DiffImpactNoiseModel
from src.locomotion_rl.go2_noise_aware.noise_model.touchdown_features import (
  DiffImpactStepNoise,
  TouchdownFeatureTracker,
  leg_kinematics,
  torque_grf_z,
)


class NoiseModelTest(unittest.TestCase):
  """충돌 물리 법칙 표: A = sqrt((m (v/v_ref)^α)² + A0²), τ = τ0 (v/v_ref)^(−β), v = −(발 vz + 몸통 vz)."""

  @classmethod
  def setUpClass(cls):
    cls.nm = DiffImpactNoiseModel()
    cls.mid = ((cls.nm.lo + cls.nm.hi) / 2).mean(0)  # 학습 범위 중앙의 착지
    cls.legs = torch.arange(4)
    cls.kf, cls.kb = cls.nm.feature_index("foot_vz_pre"), cls.nm.feature_index("body_vz_pre")

  def energy(self, change=None, offset=None):
    x = self.mid.repeat(4, 1)
    for name, value in (change or {}).items():
      x[:, self.nm.feature_index(name)] = value
    return self.nm.foot_energy(x, self.legs, offset)

  def db(self, foot_vz, body_vz=0.0):
    return self.energy({"foot_vz_pre": foot_vz, "body_vz_pre": body_vz})

  def test_shapes_and_units(self):
    out = self.energy()
    self.assertEqual(out["energy"].shape, (4,))
    self.assertTrue(torch.all(out["energy"] > 0))
    self.assertTrue(torch.all(out["db"] < 0))  # dB re FS²
    self.assertTrue(torch.all(out["ood"] == 0))

  def test_theory_exponents(self):
    self.assertEqual(self.nm.amp_mode, "phys")
    self.assertTrue(torch.allclose(self.nm.alpha, torch.ones_like(self.nm.alpha)))  # 충격량 ∝ v
    self.assertTrue(torch.allclose(self.nm.beta, torch.full_like(self.nm.beta, 0.2)))  # Hertz τ ∝ v^(−1/5)

  def test_slower_descent_is_quieter(self):
    base = self.energy()["db"]
    slow = self.energy({"foot_vz_pre": self.mid[self.kf] * 0.7})["db"]  # vz < 0: 하강 30% 느리게
    self.assertTrue(torch.all(slow < base), (base, slow))

  def test_faster_contact_is_louder_beyond_range(self):
    # 학습 범위(가장 빠른 하강 ≈ −1 m/s)를 넘어서도 계속 커짐 (클리핑 없음)
    dbs = torch.stack([self.db(v)["db"] for v in (-0.2, -0.6, -1.0, -1.5, -2.0)])
    self.assertTrue(torch.all(dbs[1:] > dbs[:-1]), dbs)
    self.assertTrue(torch.all(self.db(-2.0)["ood"] > 0))  # 학습 범위 밖은 진단으로 셈

  def test_contact_speed_is_foot_plus_body(self):
    a = self.db(-0.6, 0.0)
    b = self.db(-0.4, -0.2)  # 같은 v = 0.6
    self.assertTrue(torch.allclose(a["db"], b["db"], atol=1e-4))
    self.assertTrue(torch.allclose(a["contact_speed"], torch.full((4,), 0.6), atol=1e-6))

  def test_contact_time_follows_hertz(self):
    t1, t2 = self.db(-0.4)["tau_ms"], self.db(-0.8)["tau_ms"]  # 속도 2배 → τ × 2^(−0.2) (클램프 안쪽)
    self.assertTrue(torch.allclose(t2 / t1, torch.full((4,), 2 ** -0.2), rtol=1e-3), (t1, t2))

  def test_gentle_contact_floor(self):
    # v → 0 이어도 하중 소리 A0 가 남아 유한한 값에서 멈춤
    d0 = self.db(0.0)["db"]
    self.assertTrue(torch.all(torch.isfinite(d0)))
    self.assertTrue(torch.all(d0 < self.db(-0.6)["db"] - 1.0))

  def test_other_features_do_not_change_prediction(self):
    k = self.nm.feature_index("foot_z")
    out = self.energy({"foot_z": self.nm.lo[:, k].min() - 0.05})
    self.assertTrue(torch.allclose(out["db"], self.energy()["db"]))
    self.assertTrue(torch.all(out["ood"] == 0))  # 자세는 ood 합에서 제외
    self.assertGreater(out["ood_frac"][k].item(), 0)  # 감시용 비율에는 남음

  def test_spl_offset(self):
    out = self.energy()
    self.assertTrue(torch.allclose(out["db_spl"], out["db"] + 118.0))
    self.assertTrue(torch.all((out["db_spl"] > 30) & (out["db_spl"] < 90)))  # 마이크 위치 착지 소리, 상식 범위

  def test_offset_inside_window(self):
    first = self.energy(offset=torch.zeros(4))["energy"]
    later = self.energy(offset=torch.full((4,), 0.01))["energy"]
    self.assertTrue(torch.all(torch.isfinite(later)))
    self.assertFalse(torch.allclose(first, later))


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

  def test_step_noise_sums_energy(self):
    noise = DiffImpactStepNoise(1, "cpu")
    q = torch.tensor([[0.0, 0.8, -1.6] * 4]); pos, _ = leg_kinematics(q)
    vel = torch.zeros(1, 4, 3); vel[..., 2] = -0.6
    contact = torch.zeros(1, 4); contact[0, [1, 2]] = 100.0  # FL, RR 입각 → reset 직후 착지로 처리
    steps = []
    for k in range(3):  # 정책 스텝 3 개 × substep 4
      noise.begin_step()
      for i in range(4):
        n = 4 * k + i
        if n >= 2:
          contact[0, 0] = 100.0  # FR 착지
        noise.sample(vel, pos, contact.clone(), q, torch.zeros(1, 12), torch.tensor([[1.0, 0, 0, 0]]),
                     torch.tensor([[0.5, 0.0, 0.0]]))
      steps.append(noise.end_step())
    first, second, third = steps
    # 스텝 0: 확정 없음 (substep 4 에 FL·RR 확정 = 스텝 1 의 첫 substep)
    self.assertFalse(first["touchdown"].item())
    self.assertEqual(first["energy"].item(), 0.0)
    self.assertEqual(first["db"].item(), float("-inf"))
    # 스텝 1: FL·RR (substep 4), FR (substep 6) 확정 → 에너지 선형 합의 dB
    self.assertTrue(second["touchdown"].item())
    self.assertEqual(second["log"]["touchdowns"].item(), 3.0)
    fe = second["foot_energy"][0]
    self.assertEqual(int((fe > 0).sum()), 3)
    self.assertAlmostEqual(second["energy"].item(), fe.sum().item(), places=10)
    self.assertAlmostEqual(second["db"].item(), 10 * torch.log10(fe.sum()).item(), places=4)
    self.assertGreater(second["db"].item(), second["foot_db"][0].max().item())  # 합 > 가장 큰 발
    self.assertEqual(second["foot_db"][0, 3].item(), float("-inf"))  # RL 은 착지 없음
    self.assertFalse(third["touchdown"].item())
    nt = DiffImpactStepNoise(1, "cpu", model=noise.model, no_touchdown_db=-100.0)
    nt.begin_step()
    self.assertEqual(nt.end_step()["db"].item(), -100.0)


if __name__ == "__main__":
  unittest.main()
