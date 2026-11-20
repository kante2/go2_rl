"""Inspect raw state/action NPZ without importing ROS, mjlab or MuJoCo."""
import argparse
import json
from pathlib import Path
import numpy as np
from real2sim_common import DEFAULT_REAL, digest, load_raw, state_clock, write_json


def inspect(path):
    raw = load_raw(path)
    t, keep, _ = state_clock(raw)
    report = {"source": str(path.resolve()), "sha256": digest(path), "streams": {}}
    for prefix in ("ls", "lc"):
        times = raw[prefix + "_t"]
        delta = np.diff(times)
        report["streams"][prefix] = {
            "samples": len(times), "duration_s": float(np.ptp(times)),
            "receive_dt_percentiles_s": np.percentile(delta, [0, 1, 50, 99, 100]).tolist(),
            "receive_gaps_over_10ms": int(np.sum(delta > .01)),
        }
    residual = raw["ls_t"][keep] - t
    slope = np.polyfit(t - t[0], raw["ls_t"][keep] - raw["ls_t"][keep][0], 1)[0]
    report.update({
        "duplicate_state_ticks": int(np.sum(~keep)),
        "tick_to_receive_clock_drift_ppm": float((slope - 1) * 1e6),
        "tick_mapping_residual_percentiles_s": np.percentile(residual, [0, 1, 50, 99, 100]).tolist(),
        "target_change_hz": float(np.any(np.diff(raw["lc_q"], axis=0) != 0, axis=1).sum() / np.ptp(raw["lc_t"])),
        "kp_min_max": [raw["lc_kp"].min(axis=0).tolist(), raw["lc_kp"].max(axis=0).tolist()],
        "kd_min_max": [raw["lc_kd"].min(axis=0).tolist(), raw["lc_kd"].max(axis=0).tolist()],
        "joint_position_std_rad": raw["ls_q"].std(axis=0).tolist(),
        "foot_raw_percentiles": np.percentile(raw["ls_ff"], [0, 5, 50, 95, 100], axis=0).tolist(),
        "foot_fraction_above_10": (raw["ls_ff"] > 10).mean(axis=0).tolist(),
        "sport_samples": int(raw["meta"].get("counts", {}).get("sp", 0)),
        "limitations": ["Receive timestamps do not measure hardware actuation latency.",
                         "Foot raw counts are not calibrated newtons.",
                         "Single-trial temporal splits do not establish motion generalization."]})
    relative_time = raw["ls_t"] - raw["ls_t"][0]
    report["motion_windows"] = []
    for start in np.arange(0., np.ptp(relative_time), 2.):
        active = (relative_time >= start) & (relative_time < start + 2.)
        report["motion_windows"].append({"start_s": float(start),
            "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(raw["ls_dq"][active] ** 2)))})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.input)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
