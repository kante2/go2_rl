"""DiffImpact 물리 법칙 소음 모델(충격량 ∝ v, Hertz τ ∝ v^-1/5)과 착지 추적·스텝 소음(에너지 합 → dB)을 검증한다.
시뮬레이터 불필요. NoiseModelTest 는 go2_noise_aware 8d4270d 테스트와 같고, TrackerTest 의 착지 판정은 go2_qpace 규칙이다.

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
    self.assertEqual(self.run_rejected([60.] * 40), 0)  # 탈락 접촉으로도 세지 않음

  def run_rejected(self, forces, leg=0):
    """한 다리 힘 시퀀스의 탈락 접촉 수."""
    names = DiffImpactNoiseModel().features
    tr = TouchdownFeatureTracker(1, names, "cpu")
    q = torch.tensor([[0.0, 0.8, -1.6] * 4]); pos, _ = leg_kinematics(q)
    vel = torch.zeros(1, 4, 3); vel[..., 2] = -0.6
    n = 0
    for fi in forces:
      contact = torch.zeros(1, 4); contact[0, leg] = float(fi)
      tr.sample(vel, pos, contact, q, torch.zeros(1, 12), torch.tensor([[1.0, 0, 0, 0]]),
                torch.tensor([[0.5, 0.0, 0.0]]))
      n += int(tr.rejected[0, leg])
      if tr.rejected[0, leg]:
        self.assertAlmostEqual(tr.rejected_feats[0, leg, names.index("foot_vz_pre")].item(), -0.6, places=5)
    return n

  def test_short_contacts_are_rejected_events(self):
    # 정상 착지 1 + 짧은 유각기(20 ms) 3 + 짧은 입각기(15 ms) 2 → 착지 1, 탈락 5
    seq = [0.] * 20 + [60.] * 20                 # 정상
    seq += ([0.] * 4 + [60.] * 10) * 3           # 유각기 20 ms < 60 ms
    seq += [0.] * 20 + [60.] * 3 + [0.] * 20 + [60.] * 3 + [0.] * 20   # 입각기 15 ms < 40 ms
    self.assertEqual(len(self.run_force(seq)), 1)
    self.assertEqual(self.run_rejected(seq), 5)

  def test_step_noise_charges_rejected_contacts(self):
    q = torch.tensor([[0.0, 0.8, -1.6] * 4]); pos, _ = leg_kinematics(q)
    vel = torch.zeros(1, 4, 3); vel[..., 2] = -0.6
    seq = [0.] * 20 + [60.] * 20 + ([0.] * 2 + [60.] * 2) * 10  # 정상 착지 뒤 20 ms 주기로 두드리기

    def total(count_rejected):
      noise = DiffImpactStepNoise(1, "cpu", count_rejected=count_rejected)
      e, rej = 0.0, 0.0
      for k in range(len(seq) // 4):
        noise.begin_step()
        for i in range(4):
          contact = torch.zeros(1, 4); contact[0, 0] = seq[4 * k + i]
          noise.sample(vel, pos, contact, q, torch.zeros(1, 12), torch.tensor([[1.0, 0, 0, 0]]),
                       torch.tensor([[0.5, 0.0, 0.0]]))
        out = noise.end_step()
        e += out["energy"].item(); rej += out["log"]["rejected"].item()
      return e, rej, noise

    e_on, rej_on, noise = total(True)
    e_off, rej_off, _ = total(False)
    self.assertEqual(rej_off, 0)
    self.assertGreaterEqual(rej_on, 9)  # 두드리기 10 번 (마지막은 시퀀스 끝과 겹칠 수 있음)
    # 탈락 접촉 하나 = 같은 접촉 속도(0.6 m/s) 착지 한 번 (세트 첫 발 기준) 에너지
    x = noise.tracker.rejected_feats[0, 0][None]
    one = noise.model.foot_energy(x, torch.tensor([0]))["energy"].item()
    self.assertAlmostEqual(e_on - e_off, rej_on * one, delta=1e-6 * e_on + 1e-12)

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
