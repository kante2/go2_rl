"""MuJoCo 물리 substep 상태 → DiffImpact 입력 특징 (실로봇 학습 데이터와 같은 정의).

학습 데이터(src/diffimpact/extract_touchdowns.py, make_pairs.py, train_diffimpact.load_pairs)의 정의를
MuJoCo 에서 그대로 재현한다. 물리 dt 5 ms (200 Hz) 를 가정한다 — 실로봇 데이터도 200 Hz 로 줄여 학습했다.

| 특징 | 학습 데이터 (실로봇) | MuJoCo (이 모듈 입력) |
|---|---|---|
| foot_v{x,y,z}_pre | R(J·dq + ω×p), 착지 전 10 ms 평균 | site_lin_vel_w − root_link_lin_vel_w (월드) |
| foot_{x,y,z} | 몸체 좌표 발 위치 (URDF foot = calf 끝 0.213 m) | R_baseᵀ(site_pos_w − root_pos_w), 같은 점 |
| cmd_{vx,vy,wz} | 착지 순간 정책 명령 | command manager 의 twist |
| body_vz_pre | 반대 대각 입각 발 운동학 추정: −평균(입각 발의 몸 기준 vz) | 같은 추정 (기본). 정답 사용은 body_vz_source="true" |
| log_grf_impulse20 | ln Σ(Fz − Fz_pre) · 1 ms, 착지 후 0~20 ms, Fz = [R(−J⁻ᵀτ)]_z | 같은 식: 관절 각 q·토크 τ·자세 쿼터니언으로 계산 |

학습과 같은 추정식을 쓰는 이유: MuJoCo 정답과 추정값이 다르다 (녹화된 MuJoCo 세션으로 확인:
충격량은 접촉 센서 정답이 토크 추정보다 약 2.4 배, 몸통 vz 는 중앙값 0.02 m/s 차이). 모델은 추정값으로 학습됐다.
관절·발 순서는 Unitree 순서 (FR, FL, RR, RL) × (hip, thigh, calf). mjlab 관절 순서와 다르면 넘기기 전에 재배열한다.
| is_second | 대각 짝이 먼저 닿음 (세트 = 첫 착지 후 80 ms) | 같은 규칙 |
| Δ (offset) | 세트 첫 착지 기준 이 발의 착지 시각 | 같은 규칙 |

착지 전 10 ms 평균: 학습은 200 Hz 값을 1 kHz 로 보간해 -1~-10 ms 10점 평균 → 200 Hz 샘플
(착지 substep, -5 ms, -10 ms) 에 가중치 (0.2, 0.5, 0.3). 충격량도 같은 보간의 0~20 ms 21점 합 →
(0, 5, 10, 15, 20 ms) 에 (3, 5, 5, 5, 3) ms. 따라서 착지 특징은 착지 후 4 substep (20 ms) 에 확정된다.
"""

import torch

LEGS = ("FR", "FL", "RR", "RL")
DIAGONAL = (3, 2, 1, 0)  # FR↔RL, FL↔RR
PHYSICS_DT = 0.005
PRE_WEIGHTS = (0.3, 0.5, 0.2)  # (-10 ms, -5 ms, 0 ms)
IMPULSE_WEIGHTS_S = (3e-3, 5e-3, 5e-3, 5e-3, 3e-3)  # (0, 5, 10, 15, 20 ms)
GROUP_WINDOW = 0.08  # 대각 쌍으로 묶는 시간 [s] (학습 데이터와 같음)
MIN_IMPULSE = 1e-2  # ln 전 하한 [N·s] (학습 데이터와 같음)
DIAG_LEGS = ((0, 3), (1, 2))  # 대각 쌍 0 = FR+RL, 1 = FL+RR
LEG_DIAG = (0, 1, 1, 0)
# Go2 다리 (URDF go2_description, MuJoCo go2.xml 과 같음): hip(x축) → thigh(y축) → calf(y축) → foot
HIP_OFFSET = ((0.1934, -0.0465, 0.0), (0.1934, 0.0465, 0.0), (-0.1934, -0.0465, 0.0), (-0.1934, 0.0465, 0.0))
THIGH_Y = (-0.0955, 0.0955, -0.0955, 0.0955)
LINK = 0.213  # thigh, calf 길이 [m]


def unitree_joint_order(joint_names: list[str]) -> list[int]:
  """시뮬 관절 이름 목록 → Unitree 순서 (FR, FL, RR, RL) × (hip, thigh, calf) 로 고르는 인덱스 12 개.

  MuJoCo go2.xml 의 관절 순서는 FL, FR, RL, RR 이라 그대로 넘기면 다리가 뒤바뀐다.
  예: q_unitree = robot.data.joint_pos[:, unitree_joint_order(robot.joint_names)]
  """
  want = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]
  missing = [w for w in want if w not in joint_names]
  if missing:
    raise ValueError(f"Joint names not found: {missing}")
  return [list(joint_names).index(w) for w in want]


_CONST = {}


def _consts(device):
  if device not in _CONST:
    h = torch.tensor(HIP_OFFSET, device=device)  # (4, 3)
    ty = torch.tensor(THIGH_Y, device=device)  # (4,)
    _CONST[device] = (h, ty)
  return _CONST[device]


def leg_kinematics(q: torch.Tensor):
  """q (B, 12) Unitree 순서 → 몸체 좌표 발 위치 (B, 4, 3), 위치 야코비안 (B, 4, 3, 3) (열 = hip, thigh, calf).

  hip 은 x축, thigh·calf 는 hip 회전 뒤 y축으로 돈다. 닫힌 식으로 계산 (작은 GPU 연산 수를 줄이려고).
  """
  B = q.shape[0]
  q = q.view(B, 4, 3).float()
  h, ty = _consts(q.device)
  q1, q2, q23 = q[..., 0], q[..., 1], q[..., 1] + q[..., 2]
  c1, s1 = torch.cos(q1), torch.sin(q1)
  # hip 프레임(회전 전)에서 thigh·calf 평면 위치: x = −L(sin q2 + sin q23), z = −L(cos q2 + cos q23)
  px = -LINK * (torch.sin(q2) + torch.sin(q23))
  pz = -LINK * (torch.cos(q2) + torch.cos(q23))
  y = ty.expand_as(px)
  # Rx(q1) 적용: (x, y, z) → (x, c1 y − s1 z, s1 y + c1 z)
  p = torch.stack([px, c1 * y - s1 * pz, s1 * y + c1 * pz], -1) + h
  # 야코비안 열: hip = e_x × (p − h), thigh = Rx e_y × (p − o2), calf = Rx e_y × (p − o3)
  r = p - h
  j1 = torch.stack([torch.zeros_like(px), -r[..., 2], r[..., 1]], -1)
  # thigh·calf 는 hip 프레임에서 y축 회전: d/dq2 (x, z) = (−L(cos q2 + cos q23), L(sin q2 + sin q23)), calf 는 q23 항만
  dx2, dz2 = -LINK * (torch.cos(q2) + torch.cos(q23)), LINK * (torch.sin(q2) + torch.sin(q23))
  dx3, dz3 = -LINK * torch.cos(q23), LINK * torch.sin(q23)
  rot = lambda dx, dz: torch.stack([dx, -s1 * dz, c1 * dz], -1)  # noqa: E731  (y 성분 0 을 Rx 로 회전)
  jac = torch.stack([j1, rot(dx2, dz2), rot(dx3, dz3)], -1)
  return p, jac


def quat_rotate(q_wxyz: torch.Tensor, v: torch.Tensor):
  """몸체 → 월드 회전. q (B, 4) wxyz, v (B, ..., 3)."""
  w, xyz = q_wxyz[:, :1], q_wxyz[:, 1:]
  shape = (q_wxyz.shape[0],) + (1,) * (v.dim() - 2) + (3,)
  w, xyz = w.view(shape[:-1] + (1,)), xyz.view(shape)
  t = 2 * torch.cross(xyz.expand_as(v), v, dim=-1)
  return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def torque_grf_z(q, tau, quat_wxyz):
  """관절 토크 기반 접촉력 수직 성분 (B, 4) [N]: F = −J⁻ᵀτ (몸체 좌표) → 월드 z. 학습 데이터와 같은 식.

  Jᵀ F = −τ 를 닫힌 식으로 푼다: 열 j1·j2·j3 에 대해 F = −(τ1 j2×j3 + τ2 j3×j1 + τ3 j1×j2) / (j1·(j2×j3)).
  """
  _, jac = leg_kinematics(q)
  j1, j2, j3 = jac[..., 0], jac[..., 1], jac[..., 2]
  c23, c31, c12 = torch.linalg.cross(j2, j3), torch.linalg.cross(j3, j1), torch.linalg.cross(j1, j2)
  det = (j1 * c23).sum(-1, keepdim=True)
  t = tau.view(-1, 4, 3).float()
  f_b = -(t[..., 0:1] * c23 + t[..., 1:2] * c31 + t[..., 2:3] * c12) / det
  w, x, y, z = quat_wxyz.float().unbind(-1)
  rz = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)  # 회전행렬 3행
  return (f_b * rz[:, None]).sum(-1)


class TouchdownFeatureTracker:
  """환경 B 개 × 발 4 개의 착지를 substep 마다 추적해, 확정된 착지의 특징을 내놓는다.

  사용: 매 물리 substep 후 sample(...) 호출 → 반환된 ready 마스크 (B, 4) 의 착지가 이번 substep 에 확정.
  확정된 착지의 특징은 features (B, 4, F), 세트 안 시각은 offset (B, 4) [s].
  """

  def __init__(self, num_envs: int, feature_names: list[str], device: str,
               contact_threshold: float = 1.0, physics_dt: float = PHYSICS_DT,
               body_vz_source: str = "kinematic"):
    if body_vz_source not in ("kinematic", "true"):
      raise ValueError("body_vz_source must be 'kinematic' or 'true'.")
    self.body_vz_source = body_vz_source
    if abs(physics_dt - PHYSICS_DT) > 1e-9:
      raise ValueError("TouchdownFeatureTracker assumes a 5 ms physics step (training data rate).")
    self.names = list(feature_names)
    self.idx = {n: i for i, n in enumerate(self.names)}
    unknown = set(self.names) - {
      "foot_vx_pre", "foot_vy_pre", "foot_vz_pre", "foot_x", "foot_y", "foot_z",
      "cmd_vx", "cmd_vy", "cmd_wz", "body_vz_pre", "log_grf_impulse20", "is_second",
    }
    if unknown:
      raise ValueError(f"Unsupported features: {sorted(unknown)}")
    self.threshold = contact_threshold
    self.device = device
    B, F = num_envs, len(self.names)
    z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
    self.time = z(B)  # 환경별 경과 시간 [s]
    self.in_contact = torch.zeros(B, 4, dtype=torch.bool, device=device)
    self.hist_vel = z(3, B, 4, 3)  # 최근 3 substep 의 몸 기준 발 속도 (가장 최근이 [-1])
    self.hist_body_vz = z(3, B, 2)  # 대각 쌍별 몸통 vz 추정 (입각 발 없으면 직전 값 유지)
    self.hist_fz = z(3, B, 4)
    self.pending = torch.full((B, 4), -1, dtype=torch.long, device=device)  # 착지 후 지난 substep, -1 = 없음
    self.feats = z(B, 4, F)
    self.fz_pre = z(B, 4)
    self.impulse = z(B, 4)
    self.offset = z(B, 4)
    self.last_td = torch.full((B, 4), -1e9, device=device)  # 발별 마지막 착지 시각
    self.set_start = torch.full((B, 4), -1e9, device=device)  # 발별 소속 세트의 첫 착지 시각
    self.body_hold = z(B, 2)
    self._pw = torch.tensor(PRE_WEIGHTS, device=device)
    self._iw = torch.tensor(IMPULSE_WEIGHTS_S, device=device)
    self._other = torch.tensor([1 - d for d in LEG_DIAG], device=device)

  def reset(self, env_ids):
    self.in_contact[env_ids] = False
    self.pending[env_ids] = -1
    self.hist_vel[:, env_ids] = 0
    self.hist_body_vz[:, env_ids] = 0
    self.body_hold[env_ids] = 0
    self.hist_fz[:, env_ids] = 0
    self.last_td[env_ids] = -1e9
    self.set_start[env_ids] = -1e9
    self.time[env_ids] = 0

  @torch.no_grad()
  def sample(self, foot_vel_rel_w, foot_pos_b, contact_force, joint_pos, joint_torque, base_quat_w, command,
             base_vz_w=None):
    """물리 substep 1 회 후 상태 (발·관절은 Unitree 순서 FR, FL, RR, RL).

    foot_vel_rel_w (B, 4, 3): site_lin_vel_w − root_link_lin_vel_w (월드)         [m/s]
    foot_pos_b     (B, 4, 3): 몸체 좌표 발 site 위치 (root_link 기준)             [m]
    contact_force  (B, 4):    발 접촉력 크기 (착지·입각 판정용)                    [N]
    joint_pos      (B, 12):   관절 각 (hip, thigh, calf) × 4                      [rad]
    joint_torque   (B, 12):   관절 토크 (액추에이터 출력)                          [N·m]
    base_quat_w    (B, 4):    몸통 자세 쿼터니언 wxyz
    command        (B, 3):    (vx, vy, wz)
    base_vz_w      (B,):      root_link_lin_vel_w z — body_vz_source="true" 일 때만 사용
    반환: ready (B, 4) bool — 이번 substep 에 특징이 확정된 착지.
    """
    self.time += PHYSICS_DT
    contact = contact_force > self.threshold
    foot_fz = torque_grf_z(joint_pos, joint_torque, base_quat_w)
    if self.body_vz_source == "true":
      body = base_vz_w.float()[:, None].expand(-1, 2)
    else:  # 학습과 같은 추정: 대각 쌍별 −평균(입각 발의 몸 기준 vz)
      vz = foot_vel_rel_w[..., 2].float()
      cols = []
      for k, (a, b) in enumerate(DIAG_LEGS):
        st = contact[:, [a, b]].float()
        n = st.sum(-1)
        est = -(vz[:, [a, b]] * st).sum(-1) / n.clamp_min(1)
        cols.append(torch.where(n > 0, est, self.body_hold[:, k]))
      body = torch.stack(cols, -1)
      self.body_hold = body
    self.hist_vel = torch.cat([self.hist_vel[1:], foot_vel_rel_w[None].float()])
    self.hist_body_vz = torch.cat([self.hist_body_vz[1:], body[None]])
    self.hist_fz = torch.cat([self.hist_fz[1:], foot_fz[None]])
    touchdown = contact & ~self.in_contact
    self.in_contact = contact

    # 진행 중인 착지: 충격량 누적 (착지 substep 이 k=0, 20 ms 뒤 k=4 에서 확정)
    active = self.pending >= 0
    k = self.pending.clamp(0, 4)
    w = self._iw[k]
    self.impulse += torch.where(active, w * (foot_fz - self.fz_pre), torch.zeros_like(foot_fz))
    ready = active & (self.pending == 4)
    if "log_grf_impulse20" in self.idx:  # GPU→CPU 동기화를 피하려고 조건 분기 없이 항상 계산
      k_imp = self.idx["log_grf_impulse20"]
      self.feats[..., k_imp] = torch.where(ready, torch.log(self.impulse.clamp_min(MIN_IMPULSE)), self.feats[..., k_imp])
    self.pending = torch.where(ready, torch.full_like(self.pending, -1),
                               torch.where(active, self.pending + 1, self.pending))

    self._start(touchdown, foot_pos_b, command)  # 착지가 없으면 torch.where 로 아무것도 바뀌지 않음
    return ready

  def _start(self, td, foot_pos_b, command):
    pw = self._pw
    vel_pre = (pw[:, None, None, None] * self.hist_vel).sum(0)  # (B, 4, 3)
    body_diag = (pw[:, None, None] * self.hist_body_vz).sum(0)  # (B, 2)
    body_pre = body_diag[:, self._other]  # (B, 4): 착지하는 발의 반대 대각 쌍 추정
    # 착지 전 접촉력 기준: 학습과 같은 착지 전 -1~-10 ms 보간 평균 (속도와 같은 가중치)
    fz_pre = (pw[:, None, None] * self.hist_fz).sum(0)
    t = self.time[:, None].expand(-1, 4)
    partner_t = self.last_td[:, DIAGONAL]
    partner_set = self.set_start[:, DIAGONAL]
    # 짝 발이 이 착지보다 80 ms 안쪽 앞에 세트를 시작했으면 같은 세트의 두 번째 발
    joins = (t - partner_set < GROUP_WINDOW - 1e-9) & (partner_t <= t)
    start = torch.where(joins, partner_set, t)
    second = joins & (partner_t < t - 1e-9)

    vals = {
      "foot_vx_pre": vel_pre[..., 0], "foot_vy_pre": vel_pre[..., 1], "foot_vz_pre": vel_pre[..., 2],
      "foot_x": foot_pos_b[..., 0], "foot_y": foot_pos_b[..., 1], "foot_z": foot_pos_b[..., 2],
      "cmd_vx": command[:, None, 0].expand(-1, 4), "cmd_vy": command[:, None, 1].expand(-1, 4),
      "cmd_wz": command[:, None, 2].expand(-1, 4),
      "body_vz_pre": body_pre, "is_second": second.float(),
    }
    for name, v in vals.items():
      if name in self.idx:
        f = self.feats[..., self.idx[name]]
        self.feats[..., self.idx[name]] = torch.where(td, v.float(), f)
    self.fz_pre = torch.where(td, fz_pre, self.fz_pre)
    self.impulse = torch.where(td, torch.zeros_like(self.impulse), self.impulse)
    # 착지 substep 의 충격량 기여 (k=0)
    self.impulse += torch.where(td, IMPULSE_WEIGHTS_S[0] * (self.hist_fz[-1] - fz_pre), torch.zeros_like(fz_pre))
    self.pending = torch.where(td, torch.ones_like(self.pending), self.pending)
    self.offset = torch.where(td, t - start, self.offset)
    self.last_td = torch.where(td, t, self.last_td)
    self.set_start = torch.where(td, start, self.set_start)


class DiffImpactNoiseCost:
  """착지 추적 + 앙상블 소음 모델 + 비용. 정책 한 스텝 = 물리 substep 여러 번.

  env 에서: begin_step() → substep 마다 sample(...) → end_step() 이 스텝 비용 (B,) 과 진단값을 돌려준다.
  스텝 비용 = 그 스텝에 확정된 착지들의 비용 합 (착지가 없으면 0).
  substep 에서는 특징만 모으고, 모델 추론은 end_step 에서 B×4 전체에 한 번 한다 (GPU→CPU 동기화 없음).
  한 발은 정책 스텝(20 ms) 안에 착지가 많아야 한 번 확정된다 (확정까지 20 ms).
  """

  def __init__(self, num_envs: int, device: str, cfg=None, model=None, **tracker_kwargs):
    from .diffimpact import DiffImpactCostCfg, DiffImpactNoiseModel
    self.cfg = cfg or DiffImpactCostCfg()
    self.model = model or DiffImpactNoiseModel(self.cfg.table_path, device)
    self.floor_db = self.cfg.floor_db if self.cfg.floor_db is not None else self.model.db_ref["p1"]
    self.ref_db = self.cfg.ref_db if self.cfg.ref_db is not None else self.model.db_ref["p50"]
    self.tracker = TouchdownFeatureTracker(num_envs, self.model.features, device, **tracker_kwargs)
    self.num_envs, self.device = num_envs, device
    self.step_feats = torch.zeros_like(self.tracker.feats)
    self.step_offset = torch.zeros(num_envs, 4, device=device)
    self.step_ready = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
    self.legs = torch.arange(4, device=device).repeat(num_envs)

  def begin_step(self):
    self.step_ready.zero_()

  def reset(self, env_ids):
    self.tracker.reset(env_ids)

  @torch.no_grad()
  def sample(self, *args, **kwargs):
    """TouchdownFeatureTracker.sample 과 같은 인자."""
    ready = self.tracker.sample(*args, **kwargs)
    self.step_feats = torch.where(ready[..., None], self.tracker.feats, self.step_feats)
    self.step_offset = torch.where(ready, self.tracker.offset, self.step_offset)
    self.step_ready |= ready

  @torch.no_grad()
  def end_step(self):
    from .diffimpact import touchdown_cost
    B, F = self.num_envs, len(self.model.features)
    out = self.model.foot_energy(self.step_feats.view(B * 4, F), self.legs, self.step_offset.view(B * 4),
                                 mask=self.step_ready.view(B * 4))
    mask = self.step_ready.float()
    cost = touchdown_cost(out, self.floor_db, self.ref_db, self.cfg.ood_weight).view(B, 4)
    step_cost = (cost * mask).sum(-1)
    n = mask.sum()
    denom = n.clamp_min(1)
    log = {
      "touchdowns": n,
      "touchdown_db": (out["db"].view(B, 4) * mask).sum() / denom,
      "ensemble_db_std": (out["db_std"].view(B, 4) * mask).sum() / denom,
    }
    log.update({f"ood_frac/{f}": v for f, v in zip(self.model.features, out["ood_frac"])})
    return step_cost, log
