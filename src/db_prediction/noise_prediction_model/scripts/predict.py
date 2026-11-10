#!/usr/bin/env python3
"""Predict future noise from one preprocessed CSV; audio columns are optional."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from noise_prediction_model.scripts.features import OUT, FEATURE_SCHEMA, physical_arrays, causal_windows
from noise_prediction_model.scripts.sparse_gp import SparseGaussianProcessRegressor
from noise_prediction_model.scripts.torch_models import load_bundle

MODEL_KEYS = dict(ridge='01_ridge_regression', hgb='02_hgb', sparse_gpr='03_sparse_gpr')


def predict_frame(frame, bundle, trial='inference'):
    if bundle['task'] != 'causal_forecast' or not bundle['causal'] or bundle['feature_schema'] != FEATURE_SCHEMA:
        raise ValueError('Expected a compatible causal LowState model')
    arrays, segments = physical_arrays(frame)
    x, _, meta, names, _ = causal_windows(frame, arrays, segments, trial, 'inference', with_targets=False)
    columns = np.asarray(bundle['columns'], dtype=int)
    if [names[i] for i in columns] != bundle['feature_names']:
        raise ValueError('Feature names differ from the frozen model')
    if not len(x):
        raise ValueError('No usable causal windows')
    model = bundle['model']
    with threadpool_limits(limits=1):
        if isinstance(model, SparseGaussianProcessRegressor):
            mean, latent, observation = model.predict_distribution_numpy(x[:, columns])
            meta['predicted_noise_dbfs'] = mean
            meta['latent_std_db'], meta['observation_std_db'] = latent, observation
            meta['lower_95_dbfs'], meta['upper_95_dbfs'] = mean-1.96*observation, mean+1.96*observation
        else:
            meta['predicted_noise_dbfs'] = model.predict(x[:, columns])
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=MODEL_KEYS, default='sparse_gpr')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error('Output exists; use --overwrite to replace it')
    path = args.model_path or OUT / 'models' / MODEL_KEYS[args.model] / 'model.pt'
    bundle = load_bundle(path)
    table = predict_frame(pd.read_csv(args.input, float_precision='round_trip'), bundle, args.input.stem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format='%.17g')
    print(f'{len(table)} causal predictions -> {args.output}')


if __name__ == '__main__':
    main()
