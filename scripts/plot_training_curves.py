"""rsl_rl tensorboard 로그에서 설명이 붙은 학습 곡선 PNG를 만든다 (그래프 문구는 영어).

사용 예:
  python scripts/plot_training_curves.py logs/rsl_rl/go2_velocity/2026-10-25_12-52-32_go2_flat
  python scripts/plot_training_curves.py <run_dir> --out plots/go2_flat --smooth 0.98

각 패널에는 지표의 뜻과 좋은 방향을 적고, command_vel 커리큘럼 단계를 배경으로 나눈다.
GAE/advantage/KL 태그는 baseline의 DiagnosticPPO로 학습한 run에만 있고, 없는 태그는 건너뛴다.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# dataviz 기본 팔레트 (라이트 모드). 원본은 얇게, EMA는 같은 색으로 진하게 그린다.
SERIES = ("#2a78d6", "#eb6834")  # 범주 1, 2번 슬롯.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
STAGE_BG = "#f0efec"

UP, DOWN, FLAT = "(higher is better)", "(lower is better)", "(stable is good)"

# 태그 -> (제목, 설명, 좋은 방향). 설명은 "무엇을 보는지"와 "어떻게 읽는지"를 한 번에 적는다.
INFO: dict[str, tuple[str, str, str | None]] = {
  "Train/mean_reward": ("Mean episode return", "Sum of rewards over one episode (max 20 s). The most basic sign of learning progress.", UP),
  "Train/mean_episode_length": ("Mean episode length [steps]", "1000 steps (20 s) = the robot survives the whole episode without falling.", UP),
  "Loss/value": ("Value loss (critic error)", "Squared error between the critic's predicted return and the actual return. Log scale.", DOWN),
  "Loss/surrogate": ("Surrogate loss (actor objective)", "PPO policy loss. Look for a small, bounded value without large spikes, not its exact level.", FLAT),
  "Loss/entropy": ("Policy entropy (exploration)", "How spread out the action distribution is. Falls while learning, rises again when the task gets harder.", None),
  "Loss/learning_rate": ("Learning rate (KL-adaptive)", "Lowered when the policy changes too much (KL > 0.02), raised when too little (KL < 0.005). Log scale.", FLAT),
  "Policy/mean_std": ("Action noise std", "Mean std of the Gaussian noise added to joint actions. Smaller = more confident policy.", None),
  "Loss/kl": ("KL divergence (policy change)", "Difference between the policy before and after an update. Near the 0.01 target = well-sized steps.", FLAT),
  "Loss/returns_mean": ("GAE return target (mean)", "Discounted return the critic is trained to predict: R = A + V.", UP),
  "Loss/returns_std": ("GAE return target (std)", "Spread of returns across environments. Large = very different difficulty per situation.", None),
  "Loss/values_mean": ("Value estimate (mean)", "Critic prediction V(s). Should follow the mean return.", None),
  "Loss/advantages_mean": ("Advantage (mean, before normalization)", "How much better than expected the actions were. Near 0 = unbiased critic.", FLAT),
  "Loss/advantages_std": ("Advantage (std, before normalization)", "Size of the good-vs-bad action signal. Near 0 = little left to learn.", None),
  "Loss/explained_variance": ("Explained variance", "Fraction of return variation the critic explains. Close to 1 is good, <= 0 means the critic fails.", UP),
  "Metrics/twist/error_vel_xy": ("Linear velocity tracking error [m/s]", "Mean difference between commanded and actual forward/lateral base velocity.", DOWN),
  "Metrics/twist/error_vel_yaw": ("Yaw rate tracking error [rad/s]", "Mean difference between commanded and actual turning rate.", DOWN),
  "Episode_Termination/time_out": ("Terminations: time limit", "Environments that reached the 20 s limit this iteration. Normal ending.", None),
  "Episode_Termination/fell_over": ("Terminations: fell over", "Environments that ended because the body tipped over.", DOWN),
  "Episode_Termination/illegal_contact": ("Terminations: illegal contact", "Environments that ended because a non-foot body part touched the ground.", DOWN),
  "Episode_Metrics/mean_action_acc": ("Action acceleration", "How abruptly joint commands change. Smaller = smoother gait.", DOWN),
  "Metrics/slip_velocity_mean": ("Foot slip velocity", "Mean sliding speed of feet that are in contact with the ground.", DOWN),
  "Metrics/landing_force_mean": ("Landing force", "Mean contact force at touchdown. Smaller = softer steps.", DOWN),
}

# 보상 항목 설명. Episode_Reward/* 값은 에피소드 평균 초당 보상(가중치 포함)이다.
REWARD_INFO = {
  "track_linear_velocity": "Track commanded linear velocity",
  "track_angular_velocity": "Track commanded yaw rate",
  "pose": "Keep the default posture",
  "foot_gait": "Follow the gait rhythm",
  "action_rate_l2": "Penalty: abrupt action changes",
  "foot_clearance": "Penalty: foot lift height error",
  "joint_acc_l2": "Penalty: joint acceleration",
  "body_ang_vel": "Penalty: body wobble",
  "foot_slip": "Penalty: foot slip",
  "soft_landing": "Penalty: hard landing",
  "body_orientation_l2": "Penalty: body tilt",
  "angular_momentum": "Penalty: angular momentum",
  "stand_still": "Penalty: moving on a stop command",
  "joint_pos_limits": "Penalty: near joint limits",
  "is_terminated": "Penalty: early termination",
}

PAGES: dict[str, tuple[str, list[str]]] = {
  "0_overview": ("Overview", [
    "Train/mean_reward",
    "Metrics/twist/error_vel_xy",
    "Policy/mean_std",
    "Train/mean_episode_length",
    "Episode_Termination/fell_over",
    "Loss/value",
  ]),
  "1_ppo_losses": ("PPO training signals (losses, exploration, learning rate)", [
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_std",
    "Loss/kl",
  ]),
  "2_return_advantage": ("Return, value and advantage (GAE)", [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/returns_mean",
    "Loss/returns_std",
    "Loss/values_mean",
    "Loss/advantages_mean",
    "Loss/advantages_std",
    "Loss/explained_variance",
  ]),
  "4_tracking_terminations": ("Velocity tracking, terminations and gait quality", [
    "Metrics/twist/error_vel_xy",
    "Metrics/twist/error_vel_yaw",
    "Episode_Termination/fell_over",
    "Episode_Termination/illegal_contact",
    "Episode_Metrics/mean_action_acc",
    "Metrics/slip_velocity_mean",
    "Metrics/landing_force_mean",
  ]),
}

LOG_Y = {"Loss/value", "Loss/learning_rate"}
WINDOW = 500  # 단계 끝 평균을 낼 iteration 수.

plt.rcParams["font.family"] = "DejaVu Sans"


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
  out = np.empty_like(values)
  acc = values[0]
  for i, v in enumerate(values):
    acc = alpha * acc + (1.0 - alpha) * v
    out[i] = acc
  return out


class _ParamsLoader(yaml.SafeLoader):
  """!!python/* 태그를 import 없이 읽는다 (tuple은 list, 함수/객체는 None)."""


def _construct_python_tag(loader, suffix, node):
  if suffix == "tuple" or isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  return None


_ParamsLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def load_params(run_dir: Path) -> tuple[dict, dict]:
  try:
    env = yaml.load((run_dir / "params/env.yaml").read_text(), Loader=_ParamsLoader)
    agent = yaml.load((run_dir / "params/agent.yaml").read_text(), Loader=_ParamsLoader)
    return env or {}, agent or {}
  except (OSError, yaml.YAMLError):
    return {}, {}


def stages(env: dict, agent: dict, last_iter: int) -> list[tuple[int, int, str]]:
  """command_vel 커리큘럼을 (시작 iter, 끝 iter, 라벨) 목록으로 바꾼다."""
  try:
    raw = env["curriculum"]["command_vel"]["params"]["velocity_stages"]
    steps_per_iter = agent["num_steps_per_env"]
  except (KeyError, TypeError):
    return [(0, last_iter, "")]
  out, vx, vy = [], None, None
  for i, s in enumerate(raw):
    vx, vy = s.get("lin_vel_x") or vx, s.get("lin_vel_y") or vy
    start = s["step"] // steps_per_iter
    end = raw[i + 1]["step"] // steps_per_iter if i + 1 < len(raw) else last_iter
    label = f"Stage {i + 1}: vx {vx[0]:g}~{vx[1]:g}, vy ±{vy[1]:g} m/s" if vx and vy else f"Stage {i + 1}"
    out.append((start, end, label))
  return out


def to_true_error(scalars: dict, env: dict) -> None:
  """error_vel_*는 오차를 '최대 명령 시간'으로 나눠 누적하므로 에피소드 평균 오차로 되돌린다."""
  try:
    max_cmd_time = env["commands"]["twist"]["resampling_time_range"][1]
    step_dt = env["decimation"] * env["sim"]["mujoco"]["timestep"]
  except (KeyError, TypeError):
    return
  if "Train/mean_episode_length" not in scalars:
    return
  len_steps, ep_len = scalars["Train/mean_episode_length"]
  for tag in ("Metrics/twist/error_vel_xy", "Metrics/twist/error_vel_yaw"):
    if tag in scalars:
      steps, v = scalars[tag]
      ep_time = np.interp(steps, len_steps, ep_len) * step_dt
      scalars[tag] = (steps, v * max_cmd_time / np.maximum(ep_time, step_dt))


def stage_means(steps, values, stage_list) -> list[float]:
  """각 단계 마지막 WINDOW iteration의 평균."""
  return [float(np.mean(values[(steps >= max(start, end - WINDOW)) & (steps < end)]))
          for start, end, _ in stage_list]


def fmt(v: float) -> str:
  return f"{v:.3g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"


def style_axis(ax) -> None:
  ax.set_facecolor(SURFACE)
  ax.tick_params(colors=TEXT_MUTED, labelsize=8)
  ax.grid(True, color=GRID, linewidth=0.6)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)


def draw_panel(ax, tag, title, desc, direction, scalars, stage_list, alpha) -> None:
  steps, values = scalars[tag]
  style_axis(ax)
  for i, (start, end, _) in enumerate(stage_list):
    if i % 2 == 1:
      ax.axvspan(start, end, color=STAGE_BG, zorder=0)
  ax.plot(steps, values, color=SERIES[0], alpha=0.18, linewidth=0.8)
  ax.plot(steps, ema(values, alpha), color=SERIES[0], linewidth=2.0)
  if tag in LOG_Y:
    ax.set_yscale("log")
  for i, (start, end, _) in enumerate(stage_list):
    ax.text((start + end) / 2, 1.0, f"S{i + 1}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color=TEXT_MUTED)
  if len(stage_list) > 1:
    means = stage_means(steps, values, stage_list)
    note = " → ".join(f"S{i + 1} {fmt(m)}" for i, m in enumerate(means))
    ax.text(0.99, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=TEXT, bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID))
  # 설명을 항상 두 줄로 맞춰 패널마다 제목 높이가 같게 한다.
  wrapped = "\n".join((textwrap.wrap(desc, 70) + ["", ""])[:2])
  head = f"{title}  {direction}" if direction else title
  ax.set_title(head, loc="left", fontsize=11, color=TEXT, fontweight="bold", pad=34)
  ax.text(0, 1.02, wrapped, transform=ax.transAxes, ha="left", va="bottom",
          fontsize=8.5, color=TEXT_MUTED, linespacing=1.3)
  ax.set_xlabel("training iteration", color=TEXT_MUTED, fontsize=9)


def new_figure(n_panels: int, page_title: str, subtitle: str):
  cols = 3
  rows = int(np.ceil(n_panels / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.1 * rows + 0.9), squeeze=False)
  fig.patch.set_facecolor(SURFACE)
  for ax in axes.flat[n_panels:]:
    ax.set_visible(False)
  page_header(fig, page_title, subtitle)
  return fig, axes


def page_header(fig, title: str, subtitle: str) -> None:
  # 그림 높이와 무관하게 위에서부터 같은 인치 간격으로 제목과 부제를 둔다.
  h = fig.get_figheight()
  fig.text(0.01, 1 - 0.15 / h, title, ha="left", va="top", fontsize=16, fontweight="bold", color=TEXT)
  fig.text(0.01, 1 - 0.5 / h, subtitle, ha="left", va="top", fontsize=10, color=TEXT_MUTED)


def legend_line(stage_list) -> str:
  parts = ["Thin line = raw value, thick line = EMA"]
  if len(stage_list) > 1:
    parts.append("grey background = even stages")
    parts.append(" / ".join(label for *_, label in stage_list))
    parts.append(f"box = mean of the last {WINDOW} iterations of each stage")
  return "   ·   ".join(parts)


def save(fig, out_dir: Path, name: str) -> Path:
  fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
  path = out_dir / f"{name}.png"
  fig.savefig(path, dpi=130, facecolor=SURFACE)
  plt.close(fig)
  return path


def draw_page(name, page_title, tags, scalars, stage_list, alpha, out_dir):
  tags = [t for t in tags if t in scalars]
  if not tags:
    return None
  fig, axes = new_figure(len(tags), page_title, legend_line(stage_list))
  for ax, tag in zip(axes.flat, tags):
    title, desc, direction = INFO.get(tag, (tag, "", None))
    draw_panel(ax, tag, title, desc, direction, scalars, stage_list, alpha)
  return save(fig, out_dir, name)


def draw_reward_terms(scalars, stage_list, alpha, out_dir):
  tags = sorted((t for t in scalars if t.startswith("Episode_Reward/")),
                key=lambda t: -abs(scalars[t][1][-WINDOW:].mean()))
  if not tags:
    return []
  paths = []
  # (a) 단계별 보상 구성 비교: 어떤 항목이 커지고 작아졌는지.
  if len(stage_list) > 1:
    names = [t.split("/", 1)[1] for t in tags]
    means = np.array([stage_means(*scalars[t], stage_list) for t in tags])  # [항목, 단계]
    fig, ax = plt.subplots(figsize=(13, 0.48 * len(tags) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    y = np.arange(len(tags))
    h = 0.38
    for i, (_, _, label) in enumerate(stage_list[:2]):
      ax.barh(y + (i - 0.5) * h, means[:, i], height=h - 0.04, color=SERIES[i], label=label)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.8)
    ax.set_yticks(y, [f"{REWARD_INFO.get(n, '')}  ({n})" for n in names], fontsize=9, color=TEXT)
    ax.invert_yaxis()
    xmax = np.abs(means).max() * 1.05
    ax.set_xlim(-xmax * 0.45, xmax * 1.25)
    for yi, row in zip(y, means):
      delta = row[1] - row[0]
      label = "no change" if abs(delta) < 5e-4 else f"change {delta:+.3f}"
      ax.text(xmax * 1.23, yi, label, ha="right", va="center", fontsize=8.5, color=TEXT_MUTED)
    ax.set_xlabel("Mean reward per second of episode (weighted; right = reward, left = penalty)",
                  color=TEXT_MUTED, fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    page_header(fig, "Reward composition by stage",
                f"Mean of the last {WINDOW} iterations of each stage. A shorter bar means less of that reward"
                " or a larger penalty.")
    paths.append(save(fig, out_dir, "3a_reward_breakdown"))
  # (b) 항목별 시계열.
  fig, axes = new_figure(len(tags), "Reward terms over training", legend_line(stage_list))
  for ax, tag in zip(axes.flat, tags):
    n = tag.split("/", 1)[1]
    values = scalars[tag][1]
    direction = UP if values[-WINDOW:].mean() > 0 else "(closer to 0 is better)"
    draw_panel(ax, tag, n, REWARD_INFO.get(n, ""), direction, scalars, stage_list, alpha)
  paths.append(save(fig, out_dir, "3b_reward_terms"))
  return paths


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("--out", type=Path, default=None, help="기본값: plots/<run 폴더 이름>")
  parser.add_argument("--smooth", type=float, default=0.98, help="EMA 계수 (0이면 평활화 없음)")
  args = parser.parse_args()

  out_dir = args.out or Path("plots") / args.run_dir.name
  out_dir.mkdir(parents=True, exist_ok=True)

  ea = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  scalars = {}
  for tag in ea.Tags()["scalars"]:
    if tag.endswith("/time"):
      continue  # 같은 값을 wall-clock 축으로 다시 기록한 태그.
    events = ea.Scalars(tag)
    scalars[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))

  env, agent = load_params(args.run_dir)
  to_true_error(scalars, env)
  last_iter = int(max(s[0][-1] for s in scalars.values())) + 1
  stage_list = stages(env, agent, last_iter)

  paths = [draw_page(name, title, tags, scalars, stage_list, args.smooth, out_dir)
           for name, (title, tags) in PAGES.items()]
  paths += draw_reward_terms(scalars, stage_list, args.smooth, out_dir)
  for path in sorted(p for p in paths if p):
    print(f"saved {path}")
  missing = sorted({t for _, tags in PAGES.values() for t in tags if t not in scalars})
  if missing:
    print("not in log (skipped):", ", ".join(missing))


if __name__ == "__main__":
  main()
