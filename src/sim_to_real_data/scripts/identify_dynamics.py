"""Fit a small actuator parameter set with CMA-ES on TRAIN clips only."""
import argparse
from pathlib import Path
import time
import numpy as np

from real2sim_common import digest, load_dataset, select_clips, write_json
from mujoco_replay import DEFAULT_PARAMETERS, Replay, errors, prediction_cost


def identify(dataset_path, output, iterations=20, max_clips=12, seed=7,
             timestep=.001, fit_mass=False, regularization=.05):
    import cma
    if iterations < 1 or seed < 1 or regularization < 0:
        raise ValueError("iterations/seed must be positive; regularization nonnegative")
    data, meta = load_dataset(dataset_path)
    engine = Replay(timestep=timestep)
    train = select_clips(data, "train", max_clips)
    validation = select_clips(data, "validation", max_clips)
    specs = [("gain_" + j, .5, 1.5) for j in ("hip", "thigh", "calf")]
    specs += [("damping_" + j, .4, 1.6) for j in ("hip", "thigh", "calf")]
    if meta["max_delay_s"] > 0:
        specs += [("delay_s", 0., min(.04, meta["max_delay_s"]))]
    if fit_mass:
        specs += [("base_mass_scale", .8, 1.3)]
    nominal = np.array([(DEFAULT_PARAMETERS[k] - lo) / (hi - lo) for k, lo, hi in specs])

    def unpack(vector):
        return DEFAULT_PARAMETERS | {k: float(lo + x * (hi - lo)) for x, (k, lo, hi) in zip(vector, specs)}

    def score(p, clips):
        values = []
        try:
            for clip in clips:
                pred, _ = engine.rollout(data, clip, p, include_forces=False)
                values.append(prediction_cost(errors(pred, data, clip)))
        except FloatingPointError:
            return 1e12
        result = float(np.mean(values))
        return result if np.isfinite(result) else 1e12

    baseline_train = score(DEFAULT_PARAMETERS, train)
    baseline_validation = score(DEFAULT_PARAMETERS, validation)
    best_cost, best_vector = baseline_train, nominal.copy()
    history = []
    started = time.monotonic()
    initial = np.clip(nominal, .1, .9)
    optimizer = cma.CMAEvolutionStrategy(initial.tolist(), .15,
        {"bounds": [0., 1.], "seed": seed, "maxiter": iterations,
         "popsize": 10, "verbose": -9, "verb_log": 0, "verb_disp": 0})

    def checkpoint():
        write_json(output, {"schema": "go2-real2sim-parameters-v1", "parameters": unpack(best_vector),
                    "dataset_sha256": digest(dataset_path), "dataset": str(dataset_path.resolve()),
                    "optimizer": "CMA-ES", "seed": seed, "regularization": regularization,
                    "parameter_bounds": {k: [lo, hi] for k, lo, hi in specs},
                    "train_clip_ids": train.tolist(), "validation_clip_ids": validation.tolist(),
                    "selection": "minimum regularized TRAIN loss, including nominal baseline",
                    "baseline_train_prediction_cost": baseline_train,
                    "baseline_validation_prediction_cost": baseline_validation,
                    "best_regularized_train_cost": best_cost, "history": history,
                    "elapsed_s": time.monotonic() - started, "model": engine.metadata(),
                    "limitations": ["Single-trial split; no independent trial validation.",
                                     "Base initialization and contact thresholds are heuristic.",
                                     "Effective PD scales can absorb unmodeled dynamics; not hardware calibration."]})

    while not optimizer.stop():
        candidates = optimizer.ask()
        costs = []
        for vector in candidates:
            cost = score(unpack(vector), train) + regularization * float(np.mean((vector - nominal) ** 2))
            costs.append(cost)
            if cost < best_cost:
                best_cost, best_vector = cost, np.asarray(vector).copy()
        optimizer.tell(candidates, costs)
        history.append({"iteration": optimizer.countiter, "best_regularized_train_cost": best_cost})
        checkpoint()
        print(f"iteration {optimizer.countiter}/{iterations}: train objective {best_cost:.6f}", flush=True)
    checkpoint()
    import json
    report = json.loads(output.read_text())
    report["identified_train_prediction_cost"] = score(unpack(best_vector), train)
    report["identified_validation_prediction_cost"] = score(unpack(best_vector), validation)
    report["elapsed_s"] = time.monotonic() - started
    write_json(output, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--max-clips", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--timestep", type=float, default=.001)
    p.add_argument("--fit-mass", action="store_true")
    p.add_argument("--regularization", type=float, default=.05)
    args = p.parse_args()
    report = identify(args.dataset, args.output, args.iterations, args.max_clips, args.seed,
                      args.timestep, args.fit_mass, args.regularization)
    print({k: report[k] for k in ("parameters", "baseline_train_prediction_cost", "identified_train_prediction_cost",
                                 "baseline_validation_prediction_cost", "identified_validation_prediction_cost")})
