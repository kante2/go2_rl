from __future__ import annotations

# 로컬 목표 명령 구현. 현재 학습의 commands.py는 외부 mjlab의 UniformVelocityCommandCfg를 쓴다.
# 따라서 이 파일의 샘플링/GUI 로직만 수정하면 현재 학습 명령에는 반영되지 않는다.
# 공통 표기: B=환경 수, _b=몸체 좌표계, _w=월드 좌표계, env_ids=갱신할 환경 인덱스 목록.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

'''
목표 속도 명령 생성/ 관리
-> 어느 방향으로 얼마나 빠르게 움직여야 하는지 판단
'''

class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg  # 아래 데이터 클래스의 샘플링 범위, 비율, 시각화 설정.

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)  # [B, 3]: vx, vy [m/s], wz [rad/s].
    self.heading_target = torch.zeros(self.num_envs, device=self.device)  # [B], 월드 기준 목표 방위각 [rad].
    self.heading_error = torch.zeros(self.num_envs, device=self.device)  # [B], 최단 방향 오차 [-pi, pi].
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)  # [B] bool, 정지 명령을 적용할 환경.

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)  # 수평 속도 추종 오차 누적.
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)  # 회전 속도 추종 오차 누적.

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]  # 목표 유지 시간 상한 [s].
    max_command_step = max_command_time / self._env.step_dt  # 해당 시간 동안의 정책 스텝 수로 누적값 정규화.
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # 부모 CommandTerm이 타이머 만료/리셋 시 호출한다. uniform_는 축마다 새 균등 난수를 뽑는다.
    r = torch.empty(len(env_ids), device=self.device)  # 갱신 대상 환경 수만큼의 임시 난수 버퍼.
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    # 아주 작은 명령을 0으로 만드는 로컬 처리. heading 모드의 wz는 아래 갱신에서 다시 계산된다.
    self.vel_command_b[env_ids, :] *= (torch.norm(self.vel_command_b[env_ids, :], dim=1) > 0.1).unsqueeze(1)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs  # 목표 방향 제어 선택 마스크.
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # 확률이 0보다 크면 선택된 환경의 실제 몸체 속도도 목표 속도로 초기화하는 선택 기능이다.
    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)  # 몸체 선속도를 월드 선속도로 회전 변환.
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(  # 위치3 + quaternion4 + 선속도3 + 각속도3 = 13개 상태값.
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      # wrap_to_pi로 최단 회전 오차를 만들고, 비례 제어한 wz를 설정 범위 안으로 자른다.
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0  # 정지는 앞서 만든 전후/좌우/회전 명령 모두에 우선한다.

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [  # GUI 초기 범위는 각 설정의 양의 상한으로 대칭 범위를 만든다.
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)  # 켜야 슬라이더가 명령을 덮어쓴다.

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,  # GUI에서 상한을 조절하는 눈금 간격.
          min=0.1,
          max=10.0,  # 상한 조절 UI의 최대값이며, 기본 학습 명령 상한을 뜻하지 않는다.
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,  # vx/vy는 m/s, wz는 rad/s 단위로 조절한다.
          initial_value=0.0,  # UI 시작 시 목표 속도 0.
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          # 기본 인자로 각 반복의 핸들을 묶어, 자기 축 슬라이더 범위만 갱신한다.
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)  # dt [s]만큼 타이머를 진행하고 샘플링/추종 갱신을 수행한다.
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()  # 뷰어에서 선택한 한 환경에만 수동 속도를 적용한다.
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    # 시각화 API용으로 GPU 텐서를 CPU NumPy 배열로 변환한다. 정책 관측 변환과는 별개이다.
    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale  # 화살표의 표시 크기 배율.
    z_offset = self.cfg.viz.z_offset  # 몸체 위에 표시할 화살표 시작점 오프셋.

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:  # 원점에 있는 미초기화 상태를 거르기 위한 작은 허용값 [m].
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec  # 회전행렬 [3,3]과 몸체 위치로 로컬 점을 월드 점으로 변환.

      # Command linear velocity arrow (blue).
      # 아래 color는 RGBA(0~1), width는 화살표 표시 두께. 이 숫자는 물리/학습에 영향을 주지 않는다.
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str  # scene.entities에서 제어할 로봇 이름.
  heading_command: bool = False  # True면 랜덤 wz 대신 목표 방위각 오차를 이용한다.
  heading_control_stiffness: float = 1.0  # 방향 오차 [rad]를 wz [rad/s]로 바꾸는 비례 이득 [1/s].
  rel_standing_envs: float = 0.0  # 샘플링할 때 정지를 선택할 확률 [0~1].
  rel_heading_envs: float = 1.0  # heading 모드일 때 방위각 추종을 사용할 환경 비율 [0~1].
  init_velocity_prob: float = 0.0  # 실제 몸체 속도도 명령에 맞춰 초기화할 확률. 0이면 비활성.

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]  # (최솟값, 최댓값), 전후 선속도 [m/s].
    lin_vel_y: tuple[float, float]  # 좌우 선속도 [m/s].
    ang_vel_z: tuple[float, float]  # yaw 회전 속도 [rad/s].
    heading: tuple[float, float] | None = None  # 월드 방위각 [rad]; heading 모드에서는 필수.

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2  # 화살표 시작점 높이 오프셋; 실제 그리기에서 scale도 곱한다.
    scale: float = 0.5  # 속도/오프셋을 화면 화살표 길이로 바꾸는 배율.

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
