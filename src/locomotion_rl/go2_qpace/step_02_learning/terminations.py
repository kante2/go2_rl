"""Episode timeout and fall conditions."""

import math

from mjlab.managers.termination_manager import TerminationTermCfg

from . import mdp


def make_terminations() -> dict[str, TerminationTermCfg]:
  # 종료는 해당 환경의 에피소드를 리셋한다는 뜻이다. PPO 학습 프로세스 전체를 끝내지 않는다.
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),  # 실패와 구분하는 시간 제한 종료.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},  # 수직 자세에서 기운 각도 70도를 rad로 변환.
    ),
  }
