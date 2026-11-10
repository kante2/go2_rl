#!/usr/bin/env python3
"""Find steady walking from the low-frequency derivative of measured sp_vel.

Direct execution only displays a diagnostic graph.  It does not write data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common_time import TrialReader, create_common_time, discover_trials
import numpy as np


def _smooth(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples % 2 == 0:
        samples += 1
    finite = np.isfinite(values)
    filled = np.interp(np.arange(len(values)), np.flatnonzero(finite), values[finite])
    kernel = np.ones(samples, dtype=float) / samples
    padded = np.pad(filled, samples // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _fill_short_false(mask: np.ndarray, max_samples: int) -> np.ndarray:
    out = mask.copy()
    changes = np.flatnonzero(np.diff(np.r_[True, out, True]))
    for start, stop in zip(changes[::2], changes[1::2]):
        if stop - start <= max_samples and start > 0 and stop < len(out):
            out[start:stop] = True
    return out


def _remove_short_true(mask: np.ndarray, min_samples: int) -> np.ndarray:
    out = mask.copy()
    changes = np.flatnonzero(np.diff(np.r_[False, out, False]))
    for start, stop in zip(changes[::2], changes[1::2]):
        if stop - start < min_samples:
            out[start:stop] = False
    return out


def detect_steady_state(
    data: dict[str, np.ndarray],
    acceleration_threshold: float = 0.15,
    smoothing_s: float = 0.50,
    speed_floor: float = 0.03,
    fill_gap_s: float = 0.10,
    min_segment_s: float = 2.0,
) -> dict[str, np.ndarray]:
    """Return a steady mask based on measured velocity trend, not command error."""
    hz = float(data["target_hz"])
    time_abs = data["time_abs"]
    labels = data["event_label"]
    event_t = data["event_t"]
    starts = np.flatnonzero(labels == "move_start")
    stops = np.flatnonzero(labels == "move_stop")
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError("steady-state detection requires one move_start and one move_stop")
    move_start, move_stop = float(event_t[starts[0]]), float(event_t[stops[0]])

    vx = data["sp_vel"][:, 0]
    smooth_vx = _smooth(vx, round(smoothing_s * hz))
    acceleration = np.gradient(smooth_vx, time_abs)
    move_mask = (time_abs >= move_start) & (time_abs <= move_stop)
    raw = (
        move_mask
        & data["valid"]
        & (np.abs(acceleration) <= acceleration_threshold)
        & (np.abs(smooth_vx) >= speed_floor)
    )
    cleaned = _fill_short_false(raw, round(fill_gap_s * hz))
    cleaned &= move_mask & data["valid"]
    cleaned = _remove_short_true(cleaned, round(min_segment_s * hz))

    return {
        # Keep every sufficiently long interval. Long ROS dropouts remain gaps,
        # but do not discard otherwise valid steady walking later in the trial.
        "steady": cleaned,
        "steady_raw": raw,
        "move_mask": move_mask,
        "sp_vel_x_smooth": smooth_vx,
        "sp_acc_x": acceleration,
        "acceleration_threshold": np.array(acceleration_threshold),
    }


def plot_steady_state(
    trial: TrialReader,
    data: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    time = data["time"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time, data["sp_vel"][:, 0], alpha=0.35, lw=0.6, label="sp_vel x")
    axes[0].plot(time, state["sp_vel_x_smooth"], lw=1.2, label="0.5 s trend")
    axes[0].fill_between(time, 0, 1, where=state["steady"],
                         transform=axes[0].get_xaxis_transform(), color="green",
                         alpha=0.2, label="selected steady state")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].legend()
    limit = float(state["acceleration_threshold"])
    axes[1].plot(time, state["sp_acc_x"], lw=0.8)
    axes[1].axhline(limit, color="red", ls="--", lw=0.8)
    axes[1].axhline(-limit, color="red", ls="--", lw=0.8)
    axes[1].fill_between(time, 0, 1, where=state["steady"],
                         transform=axes[1].get_xaxis_transform(), color="green", alpha=0.2)
    axes[1].set_ylabel("Trend acceleration (m/s²)")
    axes[1].set_xlabel("Time from move_start (s)")
    duration = np.sum(state["steady"]) / float(data["target_hz"])
    fig.suptitle(f"{trial.name}: steady walking selection ({duration:.2f} s)")
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", nargs="?", type=Path)
    parser.add_argument("--acceleration-threshold", type=float, default=0.15)
    args = parser.parse_args()
    if args.trial is not None:
        path = args.trial
    else:
        from preprocessing import configured_trial_path
        path = configured_trial_path()
    trial = TrialReader(path)
    data = create_common_time(trial)
    state = detect_steady_state(data, args.acceleration_threshold)
    plot_steady_state(trial, data, state)
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
