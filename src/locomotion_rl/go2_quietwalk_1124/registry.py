from mjlab.tasks.registry import register_mjlab_task

from ..go2_quietpaw.step_03_training.runner import QuietPawOnPolicyRunner
from .algorithm import runner_cfg
from .environment import environment_cfg


for terrain in ("flat", "rough"):
  register_mjlab_task(
    task_id=f"Unitree-Go2-QuietWalk-1124-{terrain.title()}",
    env_cfg=environment_cfg(terrain),
    play_env_cfg=environment_cfg(terrain, play=True),
    rl_cfg=runner_cfg(),
    runner_cls=QuietPawOnPolicyRunner,
  )
