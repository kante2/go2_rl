"""Shared data contract for offline Go2 identification (SDK order: FR FL RR RL)."""
from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL = DATA_ROOT / "real/NORMAL_20261019_155919.npz"
DEFAULT_MODEL = ROOT / "src/assets/robots/unitree_go2/xmls/go2.xml"
LEGS = ("FR", "FL", "RR", "RL")
JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
STATE_KEYS = ("q", "dq", "gyro", "quaternion", "ff", "tau", "acc")
COMMAND_KEYS = ("q", "dq", "kp", "kd", "tau")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    # Optimizer checkpoints remain readable if a run is interrupted mid-write.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name,
                                     suffix=".tmp", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        required = ["ls_t", "ls_tick", "lc_t"]
        required += ["ls_" + k for k in STATE_KEYS] + ["lc_" + k for k in COMMAND_KEYS]
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Missing state-action fields: {sorted(missing)}")
        raw = {key: archive[key].copy() for key in required}
        raw["meta"] = json.loads(str(archive["meta"])) if "meta" in archive else {}
    for prefix, keys in (("ls", STATE_KEYS), ("lc", COMMAND_KEYS)):
        t = raw[prefix + "_t"]
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"{prefix}_t must be finite, strictly increasing receive times")
        for key in keys:
            a = raw[prefix + "_" + key]
            width = 12 if prefix == "lc" or key in ("q", "dq", "tau") else (4 if key in ("ff", "quaternion") else 3)
            if a.shape != (len(t), width) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {prefix}_{key}: expected finite ({len(t)}, {width})")
    tick = raw["ls_tick"]
    if tick.shape != raw["ls_t"].shape or not np.isfinite(tick).all() or np.any(np.diff(tick) < 0):
        raise ValueError("Invalid/reset/wrapped ls_tick; split the recording at clock resets first")
    if np.any(np.linalg.norm(raw["ls_quaternion"], axis=1) < 1e-6):
        raise ValueError("Invalid zero quaternion")
    if np.any(raw["lc_kp"] < 0) or np.any(raw["lc_kd"] < 0):
        raise ValueError("Negative PD gain")
    return raw


def state_clock(raw):
    """Retain last duplicate tick; map tick milliseconds to receive-clock offset.

    This removes receive jitter, not transport latency. Clock drift is reported
    by inspect_dataset; no motion/torque fit is used to warp timestamps.
    """
    tick = raw["ls_tick"].astype(float)
    keep = np.r_[np.diff(tick) != 0, True]
    t = (tick[keep] - tick[0]) * .001
    offset = float(np.median(raw["ls_t"][keep] - t))
    return t + offset, keep, offset


def command_at(times, query, delay=0.):
    indices = np.searchsorted(times, np.asarray(query) - delay, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("No causal command history for this clip/delay")
    return indices


def rotation(quaternion):
    q = np.asarray(quaternion)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def interpolate_quaternion(t, q, query):
    return Slerp(t, rotation(q))(query).as_quat()[..., [3, 0, 1, 2]]


def hysteresis(values, on, off, dt, min_duration=.006):
    """Confirm transitions after dwell; timestamp retained at confirmation (causal)."""
    values = np.asarray(values)
    on, off = np.asarray(on), np.asarray(off)
    if on.shape != (4,) or off.shape != (4,) or not np.isfinite(on).all() or not np.isfinite(off).all() or np.any(on <= off):
        raise ValueError("Contact thresholds need four on > off values")
    result = np.zeros(values.shape, dtype=bool)
    state = values[0] >= on
    count = np.zeros(4, dtype=int)
    dwell = max(1, int(np.ceil(min_duration / dt)))
    result[0] = state
    for i in range(1, len(values)):
        transition = np.where(state, values[i] <= off, values[i] >= on)
        count = np.where(transition, count + 1, 0)
        switch = count >= dwell
        state = np.where(switch, ~state, state)
        count[switch] = 0
        result[i] = state
    return result


def load_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"]))
    if meta.get("schema") != "go2-real2sim-v1":
        raise ValueError("Unsupported prepared dataset")
    return data, meta


def select_clips(data, split, max_clips=0):
    selected = np.flatnonzero(data["clip_split"] == split)
    if max_clips > 0 and len(selected) > max_clips:
        selected = selected[np.linspace(0, len(selected) - 1, max_clips).astype(int)]
    if not len(selected):
        raise ValueError(f"No clips in split {split!r}")
    return selected
