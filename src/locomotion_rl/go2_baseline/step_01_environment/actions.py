"""Joint position targets applied by the Go2 policy."""

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg


def make_actions() -> dict[str, ActionTermCfg]:
  # joint_pos는 행동 관리자에서 사용하는 이름이며, 정책의 12개 출력이 관절 목표각으로 변환된다.
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",  # scene.entities의 로봇 이름.
      actuator_names=(".*",),  # 정규식으로 모든 관절 액추에이터를 선택한다.
      scale=0.25,  # 행동 1.0당 목표각 0.25 rad 변화. 모터 토크 배율이 아니다.
      use_default_offset=True,  # 목표각 = 기본 관절 자세 + scale × 정책 출력.
    )
  }
