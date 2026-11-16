"""Evaluate saved rollouts, including provisional contact events and missed events."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from real2sim_common import digest, hysteresis, load_dataset, write_json
from mujoco_replay import errors, prediction_cost


def match_events(real, simulated, tolerance=.08):
    """One-to-one event matching; missing/extra events are never hidden."""
    real, simulated = np.asarray(real), np.asarray(simulated)
    matched = []
    if len(real) and len(simulated):
        distances = abs(real[:, None] - simulated[None, :])
        # Invalid pairs cost more than any combination of valid distances.
        penalized = np.where(distances <= tolerance, distances, 1e6)
        rows, cols = linear_sum_assignment(penalized)
        matched = [float(distances[r, c]) for r, c in zip(rows, cols) if distances[r, c] <= tolerance]
    return {"matched": len(matched), "missed": len(real) - len(matched),
            "extra": len(simulated) - len(matched), "absolute_errors_s": matched}


def evaluate(dataset_path, predictions_path, output, compare=None):
    data, meta = load_dataset(dataset_path)
    with np.load(predictions_path, allow_pickle=False) as z:
        prediction = {k: z[k].copy() for k in z.files if k != "metadata"}
        info = json.loads(str(z["metadata"]))
    if info["dataset_sha256"] != digest(dataset_path):
        raise ValueError("Predictions belong to a different dataset")
    if info.get("schema") != "go2-real2sim-predictions-v1":
        raise ValueError("Invalid predictions schema")
    all_errors = {k: [] for k in ("q", "dq", "gyro", "orientation")}
    contact_agreement, event_records, per_clip = [], [], []
    motion_errors = {label: {k: [] for k in all_errors} for label in ("moving", "near_stationary")}
    for i, clip in enumerate(prediction["clip_ids"]):
        start, end = data["clip_bounds"][clip]
        pred = {k: prediction[k][i] for k in ("q", "dq", "gyro", "quaternion")}
        err = errors(pred, data, clip)
        for key, value in err.items():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite prediction")
            all_errors[key].append(value)
        measured_dq_rms = float(np.sqrt(np.mean(data["dq"][start+1:end] ** 2)))
        category = "moving" if measured_dq_rms > .3 else "near_stationary"
        for key, value in err.items():
            motion_errors[category][key].append(value)
        per_clip.append({"clip_id": int(clip), "start_clock_s": float(data["t"][start]),
                         "measured_dq_rms_rad_s": measured_dq_rms, "motion_category": category,
                         "prediction_cost": prediction_cost(err)})
        force = prediction["force_world"][i, :, :, 2]
        sim_contact = hysteresis(force, np.full(4, 1.), np.full(4, .5),
                                 meta["dt"], meta["contact"]["min_duration_s"])
        real_contact = data["contact"][start:end]
        contact_agreement.append((sim_contact[1:] == real_contact[1:]).mean(axis=0))
        t = data["t"][start:end]
        for foot in range(4):
            # Clip-start contact is censored, not a touchdown. Same rule for both.
            re = t[1:][np.diff(real_contact[:, foot].astype(int)) == 1]
            se = t[1:][np.diff(sim_contact[:, foot].astype(int)) == 1]
            event_records.append(match_events(re, se))
    merged = {k: np.concatenate(v) for k, v in all_errors.items()}
    rmse = {k: float(np.sqrt(np.mean(v ** 2))) for k, v in merged.items()}
    # Orientation is angular distance (rotvec norm), not component RMSE.
    rmse["orientation_angle_rad"] = float(np.sqrt(np.mean(np.sum(merged["orientation"] ** 2, axis=1))))
    event_errors = [x for record in event_records for x in record["absolute_errors_s"]]
    report = {"dataset_sha256": digest(dataset_path), "predictions_sha256": digest(predictions_path),
              "predictions_file": str(predictions_path.resolve()),
              "split": info["split"], "clip_ids": prediction["clip_ids"].tolist(),
              "split_type": meta["split_type"], "parameters": info["parameters"], "model": info["model"],
              "rmse": rmse, "joint_q_rmse_rad": np.sqrt(np.mean(merged["q"] ** 2, axis=0)).tolist(),
              "joint_dq_rmse_rad_s": np.sqrt(np.mean(merged["dq"] ** 2, axis=0)).tolist(),
              "prediction_cost": prediction_cost(merged), "per_clip": per_clip,
              "contact": {"status": meta["contact"]["status"], "force_rmse_N": None,
                          "stance_agreement_per_foot": np.mean(contact_agreement, axis=0).tolist(),
                          "matched_touchdowns": sum(r["matched"] for r in event_records),
                          "missed_touchdowns": sum(r["missed"] for r in event_records),
                          "extra_touchdowns": sum(r["extra"] for r in event_records),
                          "touchdown_mae_s": float(np.mean(event_errors)) if event_errors else None,
                          "matching_tolerance_s": .08,
                          "censoring": "No touchdown at clip start; no matching across clip boundaries."},
              "limitations": ["Contact metrics are provisional until sensor calibration and delay validation.",
                               "Initial state excluded from RMSE; subsequent states are free rollout.",
                               "No independent base velocity/position ground truth."]}
    report["motion_groups"] = {}
    for category, grouped in motion_errors.items():
        if not grouped["q"]:
            continue
        joined = {k: np.concatenate(v) for k, v in grouped.items()}
        report["motion_groups"][category] = {
            "clips": len(grouped["q"]), "joint_velocity_classification_threshold_rad_s": .3,
            "prediction_cost": prediction_cost(joined),
            "q_rmse_rad": float(np.sqrt(np.mean(joined["q"] ** 2))),
            "dq_rmse_rad_s": float(np.sqrt(np.mean(joined["dq"] ** 2)))}
    if compare:
        baseline = json.loads(compare.read_text())
        if baseline["dataset_sha256"] != report["dataset_sha256"] or baseline["clip_ids"] != report["clip_ids"]:
            raise ValueError("Comparison must use identical dataset and clips")
        if baseline["model"] != report["model"]:
            raise ValueError("Comparison must use identical physics and initialization settings")
        report["improvement_percent"] = {
            k: 100 * (1 - value / baseline["rmse"][k]) if baseline["rmse"][k] > 0 else None
            for k, value in rmse.items()}
    write_json(output, report)
    lines = ["# Go2 real2sim evaluation", "", f"Split: `{report['split']}` ({len(per_clip)} clips).",
             "Single-trial temporal split; independent trial generalization is unverified.", "",
             "| Metric | RMSE |", "|---|---:|", f"| Joint position [rad] | {rmse['q']:.6f} |",
             f"| Joint velocity [rad/s] | {rmse['dq']:.6f} |", f"| Angular velocity component [rad/s] | {rmse['gyro']:.6f} |",
             f"| Orientation angular distance [rad] | {rmse['orientation_angle_rad']:.6f} |", "",
             f"Normalized prediction cost: {report['prediction_cost']:.6f}", "",
             "Contact results are provisional; raw foot counts are not newtons.",
             f"Touchdowns: {report['contact']['matched_touchdowns']} matched, "
             f"{report['contact']['missed_touchdowns']} missed, {report['contact']['extra_touchdowns']} extra.", "",
             "Base height and velocity are initialized from geometry and estimated stance."]
    lines += ["", "| Motion group | Clips | q RMSE [rad] | dq RMSE [rad/s] |", "|---|---:|---:|---:|"]
    for label, group in report["motion_groups"].items():
        lines.append(f"| {label} | {group['clips']} | {group['q_rmse_rad']:.6f} | {group['dq_rmse_rad_s']:.6f} |")
    if "improvement_percent" in report:
        lines += ["", "Improvement against baseline (negative = worse):", ""]
        lines += [f"- {k}: {v:.2f}%" if v is not None else f"- {k}: undefined" for k, v in report["improvement_percent"].items()]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare", type=Path, help="Baseline evaluation JSON (same clips/settings)")
    args = p.parse_args()
    result = evaluate(args.dataset, args.predictions, args.output, args.compare)
    print(json.dumps({k: result[k] for k in ("split", "rmse", "prediction_cost", "contact")}, indent=2))
