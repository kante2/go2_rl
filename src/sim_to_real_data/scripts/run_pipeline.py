"""Run inspection, preparation, baseline, identification and held-out evaluation."""
import argparse
from pathlib import Path
from evaluate_real2sim import evaluate
from identify_dynamics import identify
from inspect_dataset import inspect
from plot_real2sim import plot
from prepare_dataset import prepare
from real2sim_common import DATA_ROOT, DEFAULT_REAL, write_json
from replay_mujoco import replay


def check_long_horizon(input_path, output, parameters, plots=True):
    """Diagnose horizon sensitivity without refitting or selecting parameters."""
    dataset = output / "dataset.npz"
    prepare(input_path, dataset, horizon=1., stride=1.)
    replay(dataset, output / "baseline.npz")
    before = evaluate(dataset, output / "baseline.npz", output / "baseline_metrics.json")
    replay(dataset, output / "identified.npz", parameters_path=parameters)
    after = evaluate(dataset, output / "identified.npz", output / "identified_metrics.json",
                     compare=output / "baseline_metrics.json")
    if plots:
        plot(dataset, output / "baseline.npz", output / "identified.npz", output / "comparison.png")
    return {"horizon_s": 1., "clips": len(after["clip_ids"]), "refitted": False,
            "baseline_rmse": before["rmse"], "identified_rmse": after["rmse"],
            "improvement_percent": after["improvement_percent"]}


def run(input_path, output, iterations=20, max_clips=12, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    print("Inspecting raw state-action recording", flush=True)
    write_json(output / "inspection.json", inspect(input_path))
    dataset = output / "dataset.npz"
    meta = prepare(input_path, dataset)
    print(f"Prepared clips: {meta['clips']}", flush=True)
    replay(dataset, output / "baseline.npz")
    before = evaluate(dataset, output / "baseline.npz", output / "baseline_metrics.json")
    parameters = output / "identified_parameters.json"
    identify(dataset, parameters, iterations=iterations, max_clips=max_clips)
    replay(dataset, output / "identified.npz", parameters_path=parameters)
    after = evaluate(dataset, output / "identified.npz", output / "identified_metrics.json",
                     compare=output / "baseline_metrics.json")
    if plots:
        plot(dataset, output / "baseline.npz", output / "identified.npz", output / "comparison.png")
    print("Checking 1s prediction horizon without refitting", flush=True)
    longer = check_long_horizon(input_path, output / "horizon_1s", parameters, plots)
    summary = {"clips": meta["clips"], "test_baseline_rmse": before["rmse"],
               "test_identified_rmse": after["rmse"], "test_improvement_percent": after["improvement_percent"],
               "test_motion_groups_baseline": before["motion_groups"],
               "test_motion_groups_identified": after["motion_groups"],
               "long_horizon_diagnostic": longer,
               "limitations": after["limitations"]}
    write_json(output / "summary.json", summary)
    print(f"Results: {output / 'identified_metrics.md'}", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="?", type=Path, default=DEFAULT_REAL)
    p.add_argument("--output-dir", type=Path, default=DATA_ROOT / "scripts/outputs")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--max-clips", type=int, default=12, help="Training/validation clip cap; test always uses all clips")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    run(args.input, args.output_dir, args.iterations, args.max_clips, not args.no_plots)
