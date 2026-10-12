from __future__ import annotations

# 로컬 종료 판정 구현. 현재 go2_env.py의 illegal_contact 설정은 외부 mjlab 함수를 사용한다.
# 반환값 True는 해당 환경 에피소드의 종료/리셋을 뜻하며 전체 학습 종료를 뜻하지 않는다.

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

'''
termination : 종료 여부 판정
-> 에이전트가 특정 조건을 충족하면 학습을 종료
'''

def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,  # 접촉력 크기 임계값 [N]; 이 값보다 큰 충돌을 검출한다.
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]  # 검사할 형상들이 연결된 접촉 센서.
  data = sensor.data  # B=환경 수, N=감시 형상 수, H=이력 길이.
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  # 힘 이력이 없는 센서에서는 힘의 크기 대신 접촉 존재 여부로 판정한다.
  return torch.any(data.found, dim=-1)
