#!/usr/bin/env python3
"""Place audio, LowState, and SportMode velocity on a common time axis.

Running this module directly is diagnostic-only: it displays a graph and does
not write processed data.  ``preprocessing.py`` imports the functions below to
build CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import site
import sys
from typing import Any

# This workstation may expose NumPy 2.x from the user site together with a
# system Matplotlib built for NumPy 1.x.  Match the existing plotting tools by
# preferring the mutually compatible system pair when this file is run alone.
_user_sites = site.getusersitepackages()
if isinstance(_user_sites, str):
    _user_sites = [_user_sites]
sys.path[:] = [p for p in sys.path if p not in _user_sites]

import numpy as np


DEFAULT_HZ = 500.0
DEFAULT_MAX_GAP = 0.020
DEFAULT_RMS_WINDOW = 0.020
KNEE_INDICES = np.array([2, 5, 8, 11])
MOTOR_NAMES = tuple(f"{foot}_{joint}" for foot in ("fr", "fl", "rr", "rl") for joint in ("hip", "thigh", "calf"))


class TrialReader:
    """Read either an extracted trial directory or one recorder NPZ file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir() and self.path.suffix != ".npz":
            raise ValueError(f"trial must be a directory or .npz file: {self.path}")
        self.name = self.path.name if self.path.is_dir() else self.path.stem

    def load(self, name: str) -> np.ndarray:
        if self.path.is_dir():
            return np.load(self.path / f"{name}.npy", allow_pickle=False)
        with np.load(self.path, allow_pickle=False) as archive:
            if name not in archive:
                raise KeyError(f"{self.path}: missing {name}")
            return archive[name]

    @property
    def keys(self) -> list[str]:
        if self.path.is_dir():
            return sorted(p.stem for p in self.path.glob("*.npy"))
        with np.load(self.path, allow_pickle=False) as archive:
            return list(archive.files)

    @property
    def meta(self) -> dict[str, Any]:
        value = self.load("meta")
        return json.loads(str(value.item() if value.ndim == 0 else value))


def discover_trials(data_dir: str | Path = "data") -> list[Path]:
    """Return each trial once, preferring extracted directories over NPZ."""
    root = Path(data_dir)
    found: dict[str, Path] = {}
    if not root.exists():
        return []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "audio.npy").exists():
            found[path.name] = path
        elif path.suffix == ".npz" and path.stem not in found:
            found[path.stem] = path
    return [found[name] for name in sorted(found)]


def recorded_commands(trial, target_t):
    """Hold recorded [vx, vy, yaw-rate] commands; legacy metadata is fallback."""
    target_t = np.asarray(target_t, dtype=float)
    if {'cmd_t', 'cmd_vel'}.issubset(trial.keys):
        times = np.asarray(trial.load('cmd_t'), dtype=float)
        values = np.asarray(trial.load('cmd_vel'), dtype=float)
        if (not len(times) or values.shape != (len(times), 3)
                or not np.isfinite(times).all() or not np.isfinite(values).all()
                or np.any(np.diff(times) < 0)):
            raise ValueError(f'{trial.name}: invalid recorded commands')
        # Commands persist until replaced; their age is not a sensor dropout.
        indices = np.searchsorted(times, target_t, side='right') - 1
        result = np.zeros((len(target_t), 3))
        valid = indices >= 0
        result[valid] = values[indices[valid]]
        return result
    meta = trial.meta
    return np.tile([float(meta.get(key, 0.)) for key in ('vx', 'vy', 'vyaw')], (len(target_t), 1))


def interpolate_continuous(
    source_t: np.ndarray,
    values: np.ndarray,
    target_t: np.ndarray,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate without extrapolating across long ROS gaps."""
    source_t = np.asarray(source_t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(source_t) < 2 or len(values) != len(source_t):
        raise ValueError("a source stream needs at least two matching samples")
    if np.any(np.diff(source_t) <= 0):
        keep = np.concatenate(([True], np.diff(source_t) > 0))
        source_t, values = source_t[keep], values[keep]

    right = np.searchsorted(source_t, target_t, side="left")
    left = right - 1
    safe_left = np.clip(left, 0, len(source_t) - 1)
    safe_right = np.clip(right, 0, len(source_t) - 1)
    valid = (
        (left >= 0)
        & (right < len(source_t))
        & ((source_t[safe_right] - source_t[safe_left]) <= max_gap)
    )
    output = np.full((len(target_t),) + values.shape[1:], np.nan)
    if np.any(valid):
        li, ri = left[valid], right[valid]
        denom = source_t[ri] - source_t[li]
        weight = (target_t[valid] - source_t[li]) / denom
        shape = (len(weight),) + (1,) * (values.ndim - 1)
        output[valid] = values[li] + weight.reshape(shape) * (values[ri] - values[li])
    return output, valid


def interpolate_previous(source_t, values, target_t, max_gap):
    """Causal sample-and-hold: never uses a ROS sample after the target time."""
    source_t = np.asarray(source_t, dtype=float)
    values = np.asarray(values)
    if len(source_t) < 2 or len(values) != len(source_t) or np.any(np.diff(source_t) <= 0):
        raise ValueError("previous interpolation requires matching strictly increasing timestamps")
    idx = np.searchsorted(source_t, target_t, side="right") - 1
    safe = np.clip(idx, 0, len(source_t)-1)
    valid = (idx >= 0) & (target_t <= source_t[-1]) & (target_t-source_t[safe] <= max_gap)
    out = np.full((len(target_t),)+values.shape[1:], np.nan)
    out[valid] = values[safe[valid]]
    valid &= np.all(np.isfinite(out.reshape(len(target_t), -1)), axis=1)
    return out, valid


def interpolate_angles(
    source_t: np.ndarray,
    values: np.ndarray,
    target_t: np.ndarray,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate radian angles without crossing the +/-pi branch cut."""
    values = np.asarray(values, dtype=np.float64)
    unwrapped = np.unwrap(values, axis=0)
    output, valid = interpolate_continuous(source_t, unwrapped, target_t, max_gap)
    output[valid] = (output[valid] + np.pi) % (2.0 * np.pi) - np.pi
    return output, valid


def interpolate_quaternion(
    source_t: np.ndarray,
    values: np.ndarray,
    target_t: np.ndarray,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate [w, x, y, z] quaternions along their shortest chord."""
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternion values must have shape (N, 4)")

    # q and -q encode the same attitude. Keep adjacent samples in the same
    # hemisphere so component interpolation cannot pass through zero.
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0.0:
            values[index] *= -1.0
    output, valid = interpolate_continuous(source_t, values, target_t, max_gap)
    norms = np.linalg.norm(output, axis=1)
    normalizable = valid & np.isfinite(norms) & (norms > 0.0)
    output[normalizable] /= norms[normalizable, None]
    valid &= normalizable
    output[~valid] = np.nan
    return output, valid


def _audio_windows(
    audio: np.ndarray,
    slope: float,
    t0: float,
    target_t: np.ndarray,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DC-removed RMS, absolute peak, and validity per centered window."""
    audio = np.asarray(audio, dtype=np.float64)
    half = window_s / 2.0
    left = np.ceil(((target_t - half) - t0) / slope - 1).astype(np.int64)
    right = np.ceil(((target_t + half) - t0) / slope - 1).astype(np.int64)
    valid = (left >= 0) & (right <= len(audio)) & (right > left)
    left = np.clip(left, 0, len(audio))
    right = np.clip(right, 0, len(audio))

    # Prefix sums make overlapping 20 ms windows inexpensive.
    prefix = np.concatenate(([0.0], np.cumsum(audio, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(audio * audio, dtype=np.float64)))
    count = right - left
    mean = np.full(len(target_t), np.nan)
    rms = np.full(len(target_t), np.nan)
    good = valid & (count > 0)
    mean[good] = (prefix[right[good]] - prefix[left[good]]) / count[good]
    mean_sq = np.full(len(target_t), np.nan)
    mean_sq[good] = (prefix_sq[right[good]] - prefix_sq[left[good]]) / count[good]
    rms[good] = np.sqrt(np.maximum(mean_sq[good] - mean[good] ** 2, 0.0))

    peak = np.full(len(target_t), np.nan)
    for i in np.flatnonzero(good):
        centered = audio[left[i]:right[i]] - mean[i]
        peak[i] = np.max(np.abs(centered))
    return rms, peak, valid


def create_common_time(
    trial: TrialReader,
    hz: float = DEFAULT_HZ,
    max_gap: float = DEFAULT_MAX_GAP,
    rms_window: float = DEFAULT_RMS_WINDOW,
    robot_interpolation: str = "previous",
) -> dict[str, np.ndarray]:
    """Resample the requested robot and audio features."""
    if hz <= 0 or max_gap <= 0 or rms_window <= 0:
        raise ValueError("hz, max_gap, and rms_window must be positive")

    ls_t = trial.load("ls_t").astype(np.float64)
    sp_t = trial.load("sp_t").astype(np.float64)
    audio = trial.load("audio")
    slope = float(trial.load("audio_slope"))
    t0 = float(trial.load("audio_t0"))
    audio_start = t0 + slope
    audio_stop = t0 + slope * len(audio)
    start = max(float(ls_t[0]), float(sp_t[0]), audio_start)
    stop = min(float(ls_t[-1]), float(sp_t[-1]), audio_stop)
    first = int(np.ceil(start * hz))
    last = int(np.floor(stop * hz))
    time_abs = np.arange(first, last + 1, dtype=np.float64) / hz
    if not len(time_abs):
        raise ValueError(f"{trial.path}: streams have no common interval")

    labels = trial.load("ev_label").astype(str)
    event_t = trial.load("ev_t").astype(np.float64)
    move = np.flatnonzero(labels == "move_start")
    reference = float(event_t[move[0]]) if len(move) else float(time_abs[0])
    result: dict[str, np.ndarray] = {
        "time_abs": time_abs,
        "time": time_abs - reference,
        "reference_time": np.array(reference),
        "target_hz": np.array(hz),
        "max_gap_s": np.array(max_gap),
        "event_t": event_t,
        "event_label": labels,
        "event_api": trial.load("ev_api"),
    }

    low_fields = [key for key in trial.keys if key.startswith("ls_") and key != "ls_t"]
    result["lowstate_keys"] = np.asarray(low_fields)
    for key in low_fields:
        field = key.removeprefix("ls_")
        values = trial.load(key)
        if field in {"q", "dq", "tau"}:
            if values.ndim != 2 or values.shape[1] != 12:
                raise ValueError(f"{trial.path}: ls_{field} does not contain 12 motors")
        if values.dtype.kind not in "biuf" or values.ndim not in (1, 2) or len(values) != len(ls_t):
            raise ValueError(f"{trial.path}: unsupported recorded LowState field {key}, shape={values.shape}")
        if robot_interpolation == "previous" or field == "tick":
            interpolate = interpolate_previous
        elif robot_interpolation != "linear":
            raise ValueError("robot_interpolation must be previous or linear")
        elif field == "rpy":
            interpolate = interpolate_angles
        elif field == "quaternion":
            interpolate = interpolate_quaternion
        else:
            interpolate = interpolate_continuous
        result[key], result[f"{key}_valid"] = interpolate(
            ls_t, values, time_abs, max_gap
        )
        # Compatibility aliases for existing diagnostics; full motor arrays
        # remain the primary data and are also exported to CSV.
        if field in {"q", "dq", "tau"}:
            result[f"knee_{field}"] = result[key][:, KNEE_INDICES]
            result[f"knee_{field}_valid"] = result[f"{key}_valid"]

    for key in ("sp_vel", "sp_foot_vel"):
        interpolate = interpolate_previous if robot_interpolation == "previous" else interpolate_continuous
        result[key], result[f"{key}_valid"] = interpolate(
            sp_t, trial.load(key), time_abs, max_gap
        )

    audio_rms, _, audio_rms_valid = _audio_windows(
        audio, slope, t0, time_abs, rms_window
    )
    _, audio_peak, audio_peak_valid = _audio_windows(
        audio, slope, t0, time_abs, 1.0 / hz
    )
    result["audio_rms_20ms"] = audio_rms
    result["audio_dbfs_20ms"] = 20.0 * np.log10(np.maximum(audio_rms, 1e-12) / 32768.0)
    result["audio_peak_2ms"] = audio_peak
    result["audio_valid"] = audio_rms_valid & audio_peak_valid

    valid = result["audio_valid"].copy()
    result["robot_keys"] = np.asarray(low_fields + ["sp_vel", "sp_foot_vel"])
    for key in result["robot_keys"]:
        valid &= result[f"{key}_valid"]
    result["valid"] = valid
    return result


def plot_common_time(trial: TrialReader, data: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    time = data["time"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(time, data["audio_dbfs_20ms"], lw=0.8)
    axes[0].set_ylabel("Audio (dBFS, 20 ms)")
    axes[1].plot(time, data["sp_vel"][:, 0], lw=0.8, label="sp_vel x")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend()
    axes[2].plot(time, np.nansum(data["ls_ff"], axis=1), lw=0.7, label="sum foot force")
    bad = ~data["valid"]
    axes[2].fill_between(time, 0, 1, where=bad, transform=axes[2].get_xaxis_transform(),
                         color="red", alpha=0.18, label="invalid")
    axes[2].set_ylabel("Foot force sum")
    axes[2].set_xlabel("Time from move_start (s)")
    axes[2].legend()
    fig.suptitle(f"{trial.name}: common {float(data['target_hz']):g} Hz time axis")
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", nargs="?", type=Path)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP)
    args = parser.parse_args()
    if args.trial is not None:
        path = args.trial
    else:
        from preprocessing import configured_trial_path
        path = configured_trial_path()
    trial = TrialReader(path)
    data = create_common_time(trial, args.hz, args.max_gap)
    plot_common_time(trial, data)
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
