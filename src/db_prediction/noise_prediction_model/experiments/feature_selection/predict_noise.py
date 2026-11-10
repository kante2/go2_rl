#!/usr/bin/env python3
"""Apply the saved whole-walking predictor without requiring audio labels."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import build_windows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[2]/"models/feature_selection/noise_predictor.joblib")
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args(); artifact=joblib.load(a.model); cfg=artifact["config"]
    x,_,meta,names,_=build_windows(pd.read_csv(a.input),artifact["columns"],cfg["lags_ms"],cfg["sample_ms"],cfg["horizon_ms"],
                                  with_targets=False,include_standing=cfg["include_standing"],contact_config=cfg.get("contact_feature_config"))
    if names != artifact["feature_names"]: raise ValueError("feature schema mismatch")
    meta["predicted_dbfs"]=artifact["model"].predict(x[:,artifact["indices"]])
    a.output.parent.mkdir(parents=True,exist_ok=True); meta.to_csv(a.output,index=False)
    print(f"Saved {len(meta)} predictions -> {a.output}")


if __name__ == "__main__":
    with threadpool_limits(limits=1): main()
