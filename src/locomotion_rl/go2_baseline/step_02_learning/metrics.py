"""Diagnostic metrics recorded during training."""

from mjlab.managers.metrics_manager import MetricsTermCfg

from . import mdp


def make_metrics() -> dict[str, MetricsTermCfg]:
  return {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }
