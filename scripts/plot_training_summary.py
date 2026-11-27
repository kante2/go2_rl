"""Two English summary figures showing whether an rsl_rl run trained well.

  fig1_task_performance : return, survival, velocity tracking, early terminations
  fig2_optimization     : critic loss, policy noise, learning rate, surrogate loss

Usage:
  python scripts/plot_training_summary.py logs/rsl_rl/go2_velocity/2026-10-25_12-52-32_go2_flat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_training_curves import (
  GRID, SERIES, STAGE_BG, SURFACE, TEXT, TEXT_MUTED, WINDOW,
  ema, fmt, load_params, stage_means, stages, style_axis, to_true_error,
)

plt.rcParams["font.family"] = "DejaVu Sans"


def load_scalars(run_dir: Path) -> dict:
  ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  out = {}
  for tag in ea.Tags()["scalars"]:
    if not tag.endswith("/time"):
      events = ea.Scalars(tag)
      out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
  return out


def early_termination_rate(scalars: dict):
  """Share of episode ends caused by falling or illegal contact instead of the 20 s time limit [%]."""
  steps, timeout = scalars["Episode_Termination/time_out"]
  early = sum(scalars[t][1] for t in ("Episode_Termination/fell_over", "Episode_Termination/illegal_contact")
              if t in scalars)
  return steps, 100.0 * early / np.maximum(early + timeout, 1e-6)


def panel(ax, steps, values, title, note, stage_list, alpha, log=False, raw=True):
  style_axis(ax)
  for i, (start, end, _) in enumerate(stage_list):
    if i % 2 == 1:
      ax.axvspan(start, end, color=STAGE_BG, zorder=0)
    ax.text((start + end) / 2, 1.0, f"Stage {i + 1}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color=TEXT_MUTED)
  if raw:
    ax.plot(steps, values, color=SERIES[0], alpha=0.18, linewidth=0.8)
  smooth = ema(values, alpha)
  ax.plot(steps, smooth, color=SERIES[0], linewidth=2.0)
  if log:
    ax.set_yscale("log")
  if len(stage_list) > 1:
    means = stage_means(steps, values, stage_list)
    text = "  →  ".join(f"S{i + 1} end: {fmt(m)}" for i, m in enumerate(means))
    ax.text(0.99, 0.03, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=TEXT, bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID))
  ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=TEXT, pad=22)
  ax.text(0, 1.02, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=TEXT_MUTED)
  ax.set_xlabel("training iteration", color=TEXT_MUTED, fontsize=9)
  return smooth


def mark_convergence(ax, steps, smooth, target, label):
  """Mark the first iteration where the smoothed curve reaches the target."""
  hit = np.flatnonzero(smooth >= target) if target >= smooth[0] else np.flatnonzero(smooth <= target)
  if hit.size == 0:
    return
  x, y = steps[hit[0]], smooth[hit[0]]
  ax.plot([x], [y], "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
  ax.annotate(f"{label} @ iter {x}", (x, y), xytext=(18, 22 if target < smooth[0] else -18),
              textcoords="offset points",
              fontsize=8.5, color=TEXT, arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, lw=0.8))


def figure(title, subtitle):
  fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
  fig.patch.set_facecolor(SURFACE)
  fig.suptitle(title, x=0.01, ha="left", fontsize=16, fontweight="bold", color=TEXT)
  fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=TEXT_MUTED)
  return fig, axes.flat


def save(fig, out_dir: Path, name: str):
  fig.tight_layout(rect=(0, 0, 1, 0.91))
  paths = []
  for ext in ("png", "pdf"):
    path = out_dir / f"{name}.{ext}"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    paths.append(path)
  plt.close(fig)
  return paths


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("--out", type=Path, default=None, help="default: plots/<run dir name>/summary")
  parser.add_argument("--smooth", type=float, default=0.98, help="EMA factor")
  args = parser.parse_args()

  out_dir = args.out or Path("plots") / args.run_dir.name / "summary"
  out_dir.mkdir(parents=True, exist_ok=True)
  scalars = load_scalars(args.run_dir)
  env, agent = load_params(args.run_dir)
  to_true_error(scalars, env)
  last_iter = int(max(s[0][-1] for s in scalars.values())) + 1
  stage_list = stages(env, agent, last_iter)
  legend = ("Thin line = raw per-iteration value, thick line = EMA.  "
            + "  |  ".join(label for *_, label in stage_list)
            + f".  Boxes = mean of the last {WINDOW} iterations of each stage.")
  a = args.smooth
  first_end = stage_list[0][1]

  # Figure 1: does the robot do the task?
  fig, ax = figure("Training outcome: stable walking learned, tracking converged in Stage 1", legend)
  steps, v = scalars["Train/mean_reward"]
  s = panel(next(ax), steps, v, "Mean episode return  (higher is better)",
            "Sum of rewards over one episode (max 20 s). Rises, then plateaus = converged.", stage_list, a)
  plateau = stage_means(steps, v, stage_list)[0]
  mark_convergence(fig.axes[0], steps[steps < first_end], s[steps < first_end], 0.9 * plateau, "90% of plateau")

  steps, v = scalars["Train/mean_episode_length"]
  axis = next(ax)
  panel(axis, steps, v, "Mean episode length  (higher is better)",
        "Dashed line = max 1000 steps: the robot survives the full 20 s episode.", stage_list, a)
  axis.axhline(1000, color=TEXT_MUTED, linestyle="--", linewidth=1)

  steps, v = early_termination_rate(scalars)
  panel(next(ax), steps, v, "Early termination rate [%]  (lower is better)",
        "Episodes ended by falling or body contact instead of the time limit.", stage_list, a)

  steps, v = scalars["Metrics/twist/error_vel_xy"]
  s = panel(next(ax), steps, v, "Linear velocity tracking error [m/s]  (lower is better)",
            "Mean |commanded − actual| base velocity. Stage 2 doubles the command range.", stage_list, a)
  plateau = stage_means(steps, v, stage_list)[0]
  mark_convergence(fig.axes[3], steps[steps < first_end], s[steps < first_end], 1.1 * plateau, "within 10% of best")
  paths = save(fig, out_dir, "fig1_task_performance")

  # Figure 2: was the optimization stable?
  fig, ax = figure("Training health: stable, non-divergent PPO optimization", legend)
  steps, v = scalars["Loss/value"]
  panel(next(ax), steps, v, "Value loss (critic error, log scale)  (lower is better)",
        "Critic learns to predict returns: drops ~2 orders of magnitude, no divergence.", stage_list, a, log=True)
  steps, v = scalars["Policy/mean_std"]
  panel(next(ax), steps, v, "Action noise std  (decreasing = policy becomes confident)",
        "Exploration noise shrinks smoothly and does not collapse to zero.", stage_list, a)
  steps, v = scalars["Loss/learning_rate"]
  panel(next(ax), steps, v, "Adaptive learning rate (log scale)  (stable is good)",
        "Auto-tuned to keep policy KL near 0.01. Flat band = well-sized updates.", stage_list, a, log=True)
  steps, v = scalars["Loss/surrogate"]
  panel(next(ax), steps, v, "PPO surrogate loss  (small and stable is good)",
        "Policy objective stays small and bounded with no spikes or NaNs.", stage_list, a)
  paths += save(fig, out_dir, "fig2_optimization")
  for p in paths:
    print(f"saved {p}")


if __name__ == "__main__":
  main()
