"""Diagnostic metrics recorded during training."""

from mjlab.managers.metrics_manager import MetricsTermCfg

from . import mdp


def make_metrics() -> dict[str, MetricsTermCfg]:
  # 보상 가중치가 없는 진단용 기록이다. 값을 기록하는 것만으로 학습 보상이 바뀌지는 않는다.
  return {
    "mean_action_acc": MetricsTermCfg(
      # 평균 |a_t - 2*a_(t-1) + a_(t-2)|: 행동의 2차 차분. dt²로 나눈 물리 가속도는 아니다.
      func=mdp.mean_action_acc,
    ),
  }
