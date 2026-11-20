"""Replay recorded LowCmd in free MuJoCo clips and save aligned predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from real2sim_common import DEFAULT_MODEL, digest, load_dataset, select_clips, write_json
from mujoco_replay import Replay, parameters_from_file


def replay(dataset_path, output, split="test", parameters_path=None, timestep=.001,
           max_clips=0, height_offset=0., model=DEFAULT_MODEL):
    data, meta = load_dataset(dataset_path)
    parameters = parameters_from_file(parameters_path)
    if parameters["delay_s"] > meta["max_delay_s"]:
        raise ValueError("Delay exceeds prepared command-history guard")
    engine = Replay(model, timestep, height_offset)
    selected = select_clips(data, split, max_clips)
    predictions, initializations = [], []
    for clip in selected:
        prediction, initialization = engine.rollout(data, clip, parameters)
        predictions.append(prediction)
        initializations.append(initialization)
    metadata = {"schema": "go2-real2sim-predictions-v1", "dataset": str(dataset_path.resolve()),
                "dataset_sha256": digest(dataset_path), "split": split,
                "parameters": parameters, "model": engine.metadata(),
                "initialization": initializations, "contact_status": meta["contact"]["status"]}
    if parameters_path:
        metadata["parameter_file_sha256"] = digest(parameters_path)
        metadata["parameter_file"] = str(parameters_path.resolve())
    arrays = {key: np.stack([p[key] for p in predictions]) for key in predictions[0]}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, clip_ids=selected, metadata=json.dumps(metadata))
    write_json(output.with_suffix(".json"), metadata)
    output.with_suffix(".model.xml").write_text(engine.xml)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation", "test"), default="test")
    p.add_argument("--parameters", type=Path)
    p.add_argument("--timestep", type=float, default=.001)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--height-offset", type=float, default=0.)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = p.parse_args()
    meta = replay(args.dataset, args.output, args.split, args.parameters, args.timestep,
                  args.max_clips, args.height_offset, args.model)
    print(json.dumps({"output": str(args.output), "clips": len(meta["initialization"]),
                      "model": meta["model"]}, indent=2))
