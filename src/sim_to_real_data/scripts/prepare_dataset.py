"""Build contiguous uniform state clips; preserve raw causal command history."""
import argparse
import json
from pathlib import Path
import numpy as np
from real2sim_common import (COMMAND_KEYS, DEFAULT_REAL, JOINTS, STATE_KEYS, digest,
                            hysteresis, interpolate_quaternion, load_raw, state_clock, write_json)


def prepare(path, output, dt=.002, horizon=.2, stride=.2, max_gap=.01,
            max_delay=.04, contact_config=None, split_block=1.5):
    if not all(np.isfinite(v) and v > 0 for v in (dt, horizon, stride, max_gap)) or not 0 <= max_delay < 1:
        raise ValueError("Invalid sampling/clip/gap/delay parameters")
    steps = int(round(horizon / dt))
    if steps < 2 or not np.isclose(steps * dt, horizon):
        raise ValueError("horizon must be an integer multiple of dt and at least 2 steps")
    raw = load_raw(path)
    t, keep, offset = state_clock(raw)
    start = max(t[0], raw["lc_t"][0] + max_delay)
    end = min(t[-1], raw["lc_t"][-1])
    if end - start < 3 * horizon:
        raise ValueError("Insufficient shared state/command duration")
    if not np.isfinite(split_block) or split_block <= horizon + max_delay:
        raise ValueError("split-block must be longer than horizon + max-delay")
    # Repeating whole time blocks include walking and standing in held-out data.
    # Never random-split overlapping windows; only TRAIN determines thresholds.
    edges = np.r_[np.arange(start, end, split_block), end]
    pattern = ("train", "train", "train", "validation", "test")
    blocks = [(float(a), float(b), pattern[i % 5]) for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]
    train = np.zeros(len(t), dtype=bool)
    for a, b, split in blocks:
        if split == "train":
            train |= (t >= a) & (t < b)
    if contact_config:
        contact = json.loads(Path(contact_config).read_text())
        contact["status"] = "user_supplied_unverified"
    else:
        lo, hi = np.percentile(raw["ls_ff"][keep][train], [5, 95], axis=0)
        span = np.maximum(hi - lo, 1.)
        contact = {"on": (lo + .30 * span).tolist(), "off": (lo + .15 * span).tolist(),
                   "status": "provisional_train_quantiles"}
    contact["min_duration_s"] = float(contact.get("min_duration_s", .006))
    if not np.isfinite(contact["min_duration_s"]) or contact["min_duration_s"] < 0:
        raise ValueError("Invalid contact dwell")
    # State and command gaps both invalidate clips; interpolation cannot bridge them.
    bad = [(t[i], t[i+1]) for i in np.flatnonzero(np.diff(t) > max_gap)]
    ct = raw["lc_t"]
    bad += [(ct[i], ct[i+1]) for i in np.flatnonzero(np.diff(ct) > max_gap)]
    state = {k: raw["ls_" + k][keep] for k in STATE_KEYS}
    samples = {k: [] for k in STATE_KEYS}
    samples.update(t=[], contact=[])
    clips, splits = [], []
    for a, b, split in blocks:
        # A delay may query earlier commands; keep that history in the same block.
        for clip_start in np.arange(a + max_delay, b - horizon, stride):
            clip_end = clip_start + horizon
            if any(left < clip_end and right > clip_start - max_delay for left, right in bad):
                continue
            query = clip_start + np.arange(steps + 1) * dt
            index = len(clips) * (steps + 1)
            clips.append([index, index + steps + 1])
            splits.append(split)
            samples["t"].append(query)
            # Raw force is held, not interpolated; preserves causal threshold crossings.
            ff = state["ff"][np.searchsorted(t, query, side="right") - 1]
            samples["contact"].append(hysteresis(ff, contact["on"], contact["off"], dt, contact["min_duration_s"]))
            for k in STATE_KEYS:
                if k == "quaternion":
                    value = interpolate_quaternion(t, state[k], query)
                elif k == "ff":
                    value = ff
                else:
                    value = np.stack([np.interp(query, t, col) for col in state[k].T], axis=1)
                samples[k].append(value)
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("No usable clips in one or more splits; shorten horizon or inspect gaps")
    metadata = {"schema": "go2-real2sim-v1", "source": str(path.resolve()), "source_sha256": digest(path),
                "dt": dt, "horizon_s": horizon, "stride_s": stride, "max_gap_s": max_gap,
                "max_delay_s": max_delay, "joint_order": list(JOINTS),
                "clock": "robot tick milliseconds + median receive offset; unknown transport latency",
                "tick_receive_offset_s": offset, "split_blocks_clock_s": blocks,
                "split_block_s": split_block,
                "split_type": "single_trial_temporal_blocks", "contact": contact,
                "base_initialization": "foot geometry/contact heuristic; stance no-slip velocity estimate",
                "clips": {s: splits.count(s) for s in ("train", "validation", "test")},
                "independent_base_observation": False, "force_calibrated": False}
    result = {k: np.concatenate(v) for k, v in samples.items()}
    result.update(clip_bounds=np.array(clips, dtype=int), clip_split=np.array(splits), command_t=ct)
    result.update({"command_" + k: raw["lc_" + k] for k in COMMAND_KEYS})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result, metadata=json.dumps(metadata, allow_nan=False))
    write_json(output.with_suffix(".json"), metadata)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="?", type=Path, default=DEFAULT_REAL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dt", type=float, default=.002)
    p.add_argument("--horizon", type=float, default=.2)
    p.add_argument("--stride", type=float, default=.2)
    p.add_argument("--max-gap", type=float, default=.01)
    p.add_argument("--max-delay", type=float, default=.04)
    p.add_argument("--contact-config", type=Path)
    p.add_argument("--split-block", type=float, default=1.5)
    args = p.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.dt, args.horizon, args.stride,
                             args.max_gap, args.max_delay, args.contact_config, args.split_block), indent=2))
