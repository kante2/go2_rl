"""Export joint trajectory and prediction horizon error plots for matched clips."""
import argparse
import json
from pathlib import Path
import numpy as np
from real2sim_common import JOINTS, digest, load_dataset
from mujoco_replay import errors


def plot(dataset_path, baseline_path, identified_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, dataset_meta = load_dataset(dataset_path)
    loaded = []
    for path in (baseline_path, identified_path):
        with np.load(path, allow_pickle=False) as z:
            record = {k: z[k].copy() for k in z.files if k != "metadata"}
            meta = json.loads(str(z["metadata"]))
        if meta["dataset_sha256"] != digest(dataset_path):
            raise ValueError("Dataset mismatch")
        loaded.append((record, meta))
    baseline, bm = loaded[0]
    identified, im = loaded[1]
    if not np.array_equal(baseline["clip_ids"], identified["clip_ids"]) or bm["model"] != im["model"]:
        raise ValueError("Plot comparisons require identical clips and model settings")
    # Show the held-out clip with the largest observed motion, independent of fit quality.
    rms = []
    for clip in baseline["clip_ids"]:
        a, b = data["clip_bounds"][clip]
        rms.append(np.mean(data["dq"][a:b] ** 2))
    i = int(np.argmax(rms))
    clip = baseline["clip_ids"][i]
    start, end = data["clip_bounds"][clip]
    time = data["t"][start:end] - data["t"][start]
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(time, data["q"][start:end, j], color="black", label="Real")
        ax.plot(time, baseline["q"][i, :, j], "--", label="Baseline")
        ax.plot(time, identified["q"][i, :, j], label="Identified")
        ax.set_title(JOINTS[j].removesuffix("_joint"))
        ax.set_ylabel("Joint angle [rad]")
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time from clip start [s]")
    axes[0, 0].legend()
    fig.suptitle(f"Held-out {bm['split']} clip {clip}; highest observed joint-speed RMS\n"
                 "Initial state copied once; subsequent states freely simulated")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for record, label in ((baseline, "Baseline"), (identified, "Identified")):
        grouped = {k: [] for k in ("q", "dq", "gyro")}
        for index, clip in enumerate(record["clip_ids"]):
            err = errors({k: record[k][index] for k in ("q", "dq", "gyro", "quaternion")}, data, clip)
            for key in grouped:
                grouped[key].append(err[key])
        for ax, key, unit in zip(axes, grouped, ("rad", "rad/s", "rad/s")):
            values = np.stack(grouped[key])
            curve = np.sqrt(np.mean(values ** 2, axis=(0, 2)))
            ax.plot(time[1:], curve, label=label)
            ax.set(title=key, xlabel="Prediction horizon [s]", ylabel=f"RMSE [{unit}]")
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(f"All {len(baseline['clip_ids'])} held-out clips; single-trial temporal blocks")
    fig.tight_layout()
    horizon_path = output.with_name(output.stem + "_horizon" + output.suffix)
    fig.savefig(horizon_path, dpi=150)
    plt.close(fig)
    return output, horizon_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("identified", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(plot(args.dataset, args.baseline, args.identified, args.output))
