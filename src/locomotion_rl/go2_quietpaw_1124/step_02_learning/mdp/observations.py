from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
'''
observation : 관측값 계산
->
go2입장에서, 지금 내 발 + 보행 주기가 어떤 상태인지 파악

1.foot height           : 각 발의 월드 좌표계 높이
2.foot air time         : 각 발이 현재 공중에 떠있는 시간
3.foot contact          : 각 발이 현재 지면에 닿아있는지 여부
4.foot contact forces   : 각 발의 접촉력의 X,Y,Z성분을 변환한 값
5.phase                 : 보행 주기의 현재 단계


sensor_data.force
-> 해당 데이터 구조
forces = env.scene["feet_ground_contact"].data.force

Fx = forces[:, :, 0]  # 각 발의 x축 힘
Fy = forces[:, :, 1]  # 각 발의 y축 힘
Fz = forces[:, :, 2]  # 각 발의 z축 힘


'''
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")  # 별도 선택 조건이 없을 때 참조할 로봇.
# 공통 표기: B=병렬 환경 수, N=선택한 발 수(Go2는 4). _w는 월드, _b는 몸체 좌표계.


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  # site_ids는 go2_env.py에서 정한 발 site 순서이다. 2번 축은 월드 z 높이 [m].
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time  # [B, N], 현재 공중 체류 시간 [s].
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()  # 접촉 검출 수를 정책 입력용 0.0/1.0으로 바꾼다.


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  # 원래 힘은 [N] 단위. 로그 압축으로 큰 충격의 입력 크기를 줄이면서 방향 부호는 보존한다.
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    # episode_length_buf=환경별 정책 스텝 수, step_dt=0.02 s, period=보행 주기 [s].
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)  # [B, 2]: sin/cos로 주기 경계 불연속을 줄인다.
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    # 명령 [vx, vy, wz]의 L2 크기로 정지를 판단한다. 다른 보상의 선속도+회전속도 기준과는 다르다.
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
