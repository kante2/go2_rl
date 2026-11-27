"""DiffImpact 앙상블 소음 모델의 방향·클리핑·조용한 쪽 외삽·스텝 소음(에너지 합 → dB)을 검증한다. 시뮬레이터 불필요.

원본 DiffImpact 합성과의 수치 일치는 로컬 검증 스크립트(src/diffimpact, 저장소 밖)로 확인했다.
"""

import unittest

import numpy as np
import torch

from src.locomotion_rl.go2_qpace.step_02_learning.noise_model.diffimpact import DiffImpactNoiseModel
from src.locomotion_rl.go2_qpace.step_02_learning.noise_model.touchdown_features import (
  DiffImpactStepNoise,
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

  def test_quiet_side_extrapolates(self):
    # 발 vz: 학습 범위의 조용한 끝(hi)보다 더 느리게 내려도 예측이 계속 작아짐 (클리핑 안 함)
    for name in ("foot_vz_pre", "body_vz_pre"):
      k = self.nm.feature_index(name)
      self.assertEqual(self.nm.quiet_direction[name], +1)
      edge = self.nm.hi[:, k].min()
      at_edge = self.energy({name: edge})
      beyond = self.energy({name: edge + 0.1})
      self.assertTrue(torch.all(beyond["db"] < at_edge["db"] - 1e-3), (name, at_edge["db"], beyond["db"]))
      self.assertTrue(torch.all(beyond["ood"] > 0))  # 학습 범위 밖으로는 계속 셈 (진단)
    off = DiffImpactNoiseModel(extrapolate_quiet=())  # 끄면 양쪽 클리핑
    k = off.feature_index("foot_vz_pre")
    x = self.mid.repeat(4, 1)
    x[:, k] = off.hi[:, k].max(); a = off.foot_energy(x, self.legs)["db"]
    x[:, k] = off.hi[:, k].max() + 0.1; b = off.foot_energy(x, self.legs)["db"]
    self.assertTrue(torch.allclose(a, b, atol=1e-4))

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

  # ---- [go2_qpace] 착지 판정: src/diffimpact/extract_touchdowns.py detect_leg 와 같은 결과인지 ----
  @staticmethod
  def reference_events(f, air, on=10., off=4., rise=2., min_swing=.06, min_stance=.04, dt=.005):
    """extract_touchdowns.detect_leg 를 그대로 옮긴 기준 구현. 착지 시각 j 의 샘플 인덱스 목록."""
    t = np.arange(len(f)) * dt
    above = f > air + on
    out, last_end = [], -np.inf
    for c in np.flatnonzero(above[1:] & ~above[:-1]) + 1:
      if t[c] <= last_end:
        continue
      k = c - 1
      while k >= 0 and f[k] <= air + on:
        k -= 1
      swing = np.arange(k + 1, c)
      low = swing[f[swing] < air + off] if len(swing) else swing
      if len(low) == 0 or t[low[-1]] - t[low[0]] < min_swing - 1e-9:
        continue
      j = c
      while j > low[-1] and f[j - 1] > air + rise:
        j -= 1
      e = c
      while e < len(f) and f[e] >= air + off:
        e += 1
      if t[min(e, len(t) - 1)] - t[j] < min_stance - 1e-9:
        continue
      out.append(int(j))
      last_end = t[min(e, len(t) - 1)]
    return out

  def run_force(self, forces, leg=0):
    """한 다리에 힘 시퀀스를 넣고 확정된 착지의 (확정 substep, 착지 시각 substep) 목록."""
    names = DiffImpactNoiseModel().features
    tr = TouchdownFeatureTracker(1, names, "cpu")
    q = torch.tensor([[0.0, 0.8, -1.6] * 4]); pos, _ = leg_kinematics(q)
    vel = torch.zeros(1, 4, 3); vel[..., 2] = -0.6
    out = []
    for i, fi in enumerate(forces):
      contact = torch.zeros(1, 4); contact[0, leg] = float(fi)
      ready = tr.sample(vel, pos, contact, q, torch.zeros(1, 12), torch.tensor([[1.0, 0, 0, 0]]),
                        torch.tensor([[0.5, 0.0, 0.0]]))
      if ready[0, leg]:
        out.append((i, int(round(tr.last_td[0, leg].item() / 0.005)) - 1))
    return out

  def test_touchdown_rule_matches_training_detection(self):
    air = 8.0  # FR 시뮬 공중값
    rng = np.random.default_rng(0)
    seq = [0.] * 20
    # 정상 보행, 짧은 유각기(<60 ms), 짧은 입각기(<40 ms), 떨림, 느린 상승, rise~off 사이 머무름
    for swing, stance in [(16, 40), (8, 40), (20, 6), (14, 30), (12, 9), (30, 50)]:
      seq += [0.] * swing + [60.] * stance
    seq += [0., 30.] * 10                       # 떨림: 매 substep 0/30
    seq += [0.] * 15 + [11., 12., 13., 15., 19., 25., 60.] + [60.] * 20   # 느린 상승 (rise 시작 시각 확인)
    seq += [0.] * 15 + [11., 11., 13., 60.] + [60.] * 20                  # off 아래·rise 위 구간 뒤 교차
    seq += list(rng.uniform(0, 70, 60))         # 무작위
    seq += [0.] * 15 + [60.] * 20
    f = np.array(seq)
    expected = self.reference_events(f, air)
    got = self.run_force(f, leg=0)
    self.assertEqual([j for _, j in got], expected)
    # 확정은 착지 시각 j 뒤 35 ms(7 substep) 또는 그 뒤 on 교차 시점
    self.assertTrue(all(i - j >= 7 for i, j in got))
    self.assertGreater(len(expected), 4)

  def test_initial_contact_is_not_a_touchdown(self):
    self.assertEqual(self.run_force([60.] * 40), [])  # reset 직후 이미 닿아 있음 → 유각기 없음

  def test_step_noise_sums_energy(self):
    noise = DiffImpactStepNoise(1, "cpu")
    q = torch.tensor([[0.0, 0.8, -1.6] * 4]); pos, _ = leg_kinematics(q)
    vel = torch.zeros(1, 4, 3); vel[..., 2] = -0.6
    steps = []
    for k in range(10):  # 정책 스텝 10 개 × substep 4
      noise.begin_step()
      for i in range(4):
        n = 4 * k + i
        contact = torch.zeros(1, 4)
        if n >= 20:  # 100 ms 공중 뒤 FR·RL 동시 착지 (substep 20), 35 ms 뒤 substep 27 = 스텝 6 에서 확정
          contact[0, [0, 3]] = 100.0
        noise.sample(vel, pos, contact, q, torch.zeros(1, 12), torch.tensor([[1.0, 0, 0, 0]]),
                     torch.tensor([[0.5, 0.0, 0.0]]))
      steps.append(noise.end_step())
    for k, out in enumerate(steps):
      if k != 6:
        self.assertFalse(out["touchdown"].item(), k)
        self.assertEqual(out["energy"].item(), 0.0)
    out = steps[6]
    self.assertEqual(out["log"]["touchdowns"].item(), 2.0)
    fe = out["foot_energy"][0]
    self.assertEqual((fe > 0).tolist(), [True, False, False, True])
    self.assertAlmostEqual(out["energy"].item(), fe.sum().item(), places=10)
    self.assertAlmostEqual(out["db"].item(), 10 * torch.log10(fe.sum()).item(), places=4)
    self.assertGreater(out["db"].item(), out["foot_db"][0].max().item())  # 합 > 가장 큰 발
    nt = DiffImpactStepNoise(1, "cpu", model=noise.model, no_touchdown_db=-100.0)
    nt.begin_step()
    self.assertEqual(nt.end_step()["db"].item(), -100.0)


if __name__ == "__main__":
  unittest.main()
