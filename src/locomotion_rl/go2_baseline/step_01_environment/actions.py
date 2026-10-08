"""Joint position targets applied by the Go2 policy."""

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg


def make_actions() -> dict[str, ActionTermCfg]:
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }
