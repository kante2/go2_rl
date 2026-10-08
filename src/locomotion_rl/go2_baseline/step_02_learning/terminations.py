"""Episode timeout and fall conditions."""

import math

from mjlab.managers.termination_manager import TerminationTermCfg

from . import mdp


def make_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }
