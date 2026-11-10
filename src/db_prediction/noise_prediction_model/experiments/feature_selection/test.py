#!/usr/bin/env python3
"""Write held-out model predictions. Run model_evaluation.py for metrics and figures."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import build_windows, validate_preprocessing_metadata
from noise_prediction_model.experiments.feature_selection.model_evaluation import ROOT, MODEL_DIR, EVALUATION_DIR


def validate_test_split(config, requested):
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('Test trials must be nonempty and unique')
    if set(config['training_trials']) & set(requested):
        raise ValueError('Requested test trials were used for feature selection/training')
    if not set(requested).issubset(config['excluded_test_trials']):
        raise ValueError('Requested trials were not reserved by feature extraction')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=MODEL_DIR/'noise_models.joblib')
    parser.add_argument('--test-input',type=Path,default=ROOT/'preprocessed/1007/test/all_trials.csv')
    parser.add_argument('--test-trials',nargs='+',help='Defaults to trials reserved in the fitted bundle')
    parser.add_argument('--output',type=Path,default=EVALUATION_DIR/'test')
    args=parser.parse_args()
    if not args.model.exists():
        parser.error('Run train.py first to create models/noise_models.joblib')
    bundle=joblib.load(args.model); cfg=bundle['config']
    requested=args.test_trials or cfg['excluded_test_trials']
    validate_test_split(cfg,requested)
    metadata=json.loads((args.test_input.parent/'metadata.json').read_text())
    training_metadata=cfg['preprocessing_metadata']
    validate_preprocessing_metadata(metadata)
    validate_preprocessing_metadata(training_metadata)
    for field in ('audio_delay_s','audio_rms_window_s'):
        if abs(metadata.get(field,0)-training_metadata.get(field,0))>1e-6:
            raise ValueError(f'Train/test preprocessing differs: {field}')
    df=pd.read_csv(args.test_input)
    df=df[df.trial.isin(requested)].copy()
    if set(df.trial)!=set(requested):
        raise ValueError('Missing requested trials in test input')
    reference=next(iter(bundle['models'].values()))
    x,y,meta,names,coverage=build_windows(df,reference['columns'],cfg['lags_ms'],cfg['sample_ms'],cfg['horizon_ms'],include_standing=cfg['include_standing'], contact_config=cfg.get('contact_feature_config'))
    if set(meta.trial)!=set(requested):
        raise ValueError('No valid evaluation windows for a requested test trial')
    predictions=meta.copy(); predictions['actual_dbfs']=y
    for name,artifact in bundle['models'].items():
        validate_test_split(artifact['config'],requested)
        if names!=artifact['feature_names'] or reference['columns']!=artifact['columns']:
            raise ValueError('Feature schema mismatch across models')
        pred=np.asarray(artifact['model'].predict(x[:,artifact['indices']]),dtype=np.float64); predictions[name]=pred
    run=dict(train_trials=cfg['training_trials'],test_trials=requested,n_train_windows=cfg['n_windows'],n_test_windows=len(y),
             trained_model_bundle=str(args.model.resolve()),original_model_used_for_predictions=True,
             test_input=str(args.test_input.resolve()),preprocessing_metadata=dict(train=training_metadata,test=metadata),
             lags_ms=cfg['lags_ms'],horizon_ms=cfg['horizon_ms'],
             feature_mode=cfg.get('feature_mode','raw'),contact_feature_config=cfg.get('contact_feature_config'),
             selected_lowstate_features=cfg['selected_lowstate_features'],feature_definition_sha256=cfg['feature_definition_sha256'],
             loud_threshold_train_dbfs=cfg['loud_threshold_train_dbfs'],retrained_in_test=False,
             model_names=list(bundle['models']),selected_model=cfg['selected_model'])
    args.output.mkdir(parents=True,exist_ok=True)
    coverage.to_csv(args.output/'test_coverage.csv',index=False)
    (args.output/'run.json').write_text(json.dumps(run,indent=2))
    predictions.to_csv(args.output/'test_predictions.csv',index=False)
    print(f'Saved {len(y)} test windows / {len(bundle["models"])} model predictions -> {args.output}',flush=True)
    if cfg.get('experiment')=='temporal_comparison':
        print(f'Run model_evaluation.py --kind temporal --input {args.output.parent} for metrics, figures and report.',flush=True)
    else:
        print(f'Run model_evaluation.py --kind holdout --input {args.output} for metrics, figures and report.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
