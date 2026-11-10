#!/usr/bin/env python3
"""Train noise models from a saved feature definition, reserving whole test trials.

Top-K correlation selection is repeated inside outer and inner training splits.
Only train.py fits models; test.py evaluates the saved fitted bundle.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import (FAMILIES, build_windows, load_feature_definition,
                                      select_from_windows, validate_preprocessing_metadata, model_feature_ranking, select_ranked_features)
from noise_prediction_model.experiments.feature_selection.model_evaluation import (FEATURE_DIR, MODEL_DIR, EVALUATION_DIR, metrics, trial_rmse,
                              importance, prediction_metrics, summarize_validation)


def weights(groups):
    counts = pd.Series(groups).value_counts()
    w = np.asarray([1/counts[g] for g in groups])
    return w/w.mean()


def estimator(kind, param):
    if kind == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=float(param)))
    if kind == 'hist':
        return HistGradientBoostingRegressor(max_leaf_nodes=int(param), max_iter=120, learning_rate=.08,
            l2_regularization=10, min_samples_leaf=40, early_stopping=False, random_state=42)
    return DummyRegressor(strategy='mean')


def fit(model, x, y, groups):
    w = weights(groups)
    if hasattr(model, 'named_steps'):
        model.fit(x, y, standardscaler__sample_weight=w, ridge__sample_weight=w)
    else:
        model.fit(x, y, sample_weight=w)
    return model


def tune(kind, x, y, groups, inner, indices=None, inner_indices=None):
    """Inner validation uses features selected using only its training trials."""
    indices = list(range(x.shape[1])) if indices is None else indices
    inner_indices = indices if inner_indices is None else inner_indices
    params = [1., 10., 100., 1000.] if kind == 'ridge' else [7, 15]
    train, valid = inner
    scores = []
    for param in params:
        m = fit(estimator(kind, param), x[train][:,inner_indices], y[train], groups[train])
        scores.append((trial_rmse(y[valid], m.predict(x[valid][:,inner_indices]), groups[valid]), param))
    score, best = min(scores)
    return fit(estimator(kind, best), x[:,indices], y, groups), best, score


def model_specs(names, selected):
    low = [i for i,n in enumerate(names) if n.split('__lag')[0] in selected]
    speed = [i for i,n in enumerate(names) if n.startswith('sp_vel_')]
    combined = [i for i in range(len(names)) if i in set(low+speed)]
    return {'mean_baseline': ('dummy', speed), 'speed_ridge': ('ridge', speed), 'speed_hist': ('hist', speed),
            'lowstate_ridge': ('ridge', low), 'lowstate_hist': ('hist', low),
            'speed_lowstate_ridge': ('ridge', combined), 'speed_lowstate_hist': ('hist', combined)}


def select_for_split(x, y, meta, names, cfg):
    if not cfg['selection_uses_noise_ranking']:
        return cfg['selected_lowstate_features']
    if cfg.get('selection_backend')=='extra-trees':
        settings=cfg.get('ranking_model_settings') or {}
        ranking=model_feature_ranking(x,y,meta.trial.to_numpy(),names,settings.get('n_estimators',96),settings.get('random_state',42))
        return select_ranked_features(ranking.rename(columns={'model_importance':'mean_abs_rho'}),cfg['candidate_lowstate_features'],cfg['top_k'])
    return select_from_windows(x, y, meta, names, cfg['candidate_lowstate_features'], cfg['top_k'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=FEATURE_DIR/'selected_features.json')
    p.add_argument('--input', type=Path, help='Defaults to the CSV recorded in selected_features.json')
    p.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    p.add_argument('--output', type=Path, default=EVALUATION_DIR/'cross_validation')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--block-ms', type=float, default=500)
    p.add_argument('--importance-repeats', type=int, default=2)
    args = p.parse_args()
    if args.folds < 3 or args.block_ms <= 0 or args.importance_repeats <= 0:
        p.error('At least 3 folds and positive block/repeats required')
    if not args.features.exists():
        p.error('Run noise_feature_extraction.py first to create selected_features.json')
    cfg = load_feature_definition(args.features)
    source = args.input or Path(cfg['input'])
    if source.resolve() != Path(cfg['input']).resolve() or (source.stat().st_size, source.stat().st_mtime_ns) != (cfg['input_size'], cfg['input_mtime_ns']):
        raise ValueError('Input changed after feature analysis; rerun noise_feature_extraction.py')
    metadata = json.loads((source.parent/'metadata.json').read_text())
    validate_preprocessing_metadata(metadata)
    if metadata != cfg['preprocessing_metadata']:
        raise ValueError('Preprocessing metadata changed; rerun feature extraction')
    df = pd.read_csv(source)
    df = df[~df.trial.isin(cfg['test_trials'])].copy()
    candidates = cfg['candidate_lowstate_features']+cfg['speed_features']
    x,y,meta,names,coverage = build_windows(df, candidates, cfg['lags_ms'], cfg['sample_ms'], cfg['horizon_ms'], include_standing=cfg['include_standing'], contact_config=cfg.get('contact_feature_config'))
    del df
    groups = meta.trial.to_numpy()
    if set(groups) != set(cfg['training_trials']) or set(groups) & set(cfg['test_trials']):
        raise ValueError('Training trials differ from feature-definition split')
    if len(set(groups)) < max(args.folds, 5):
        raise ValueError('Need at least 5 training trials and at least as many as folds')
    args.output.mkdir(parents=True, exist_ok=True); args.model_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output/'coverage.csv', index=False)
    final_specs = model_specs(names, cfg['selected_lowstate_features'])
    predictions = meta.copy(); predictions['actual_dbfs'] = y
    for name in final_specs:
        predictions[name] = np.nan  # Float64 CSV storage preserves float32 model outputs exactly.
    trial_rows, tuning, importance_rows, coef_rows, selection_rows = [], [], [], [], []
    cv = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    print(f'{len(y)} windows / {len(set(groups))} training trials / {len(cfg["selected_lowstate_features"])} selected LowState', flush=True)
    for fold, (train, valid) in enumerate(cv.split(x,y,groups), 1):
        inner = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed+fold).split(x[train],y[train],groups[train]))
        selected = select_for_split(x[train],y[train],meta.iloc[train].reset_index(drop=True),names,cfg)
        itrain = train[inner[0]]
        inner_selected = select_for_split(x[itrain],y[itrain],meta.iloc[itrain].reset_index(drop=True),names,cfg)
        specs, inner_specs = model_specs(names, selected), model_specs(names, inner_selected)
        for stage, features, used in [('outer_train',selected,train),('inner_train',inner_selected,itrain)]:
            selection_rows.extend(dict(fold=fold, stage=stage, feature=f, selection_trials=';'.join(sorted(set(groups[used]))), excluded_outer_trials=';'.join(sorted(set(groups[valid])))) for f in features)
        validmeta = meta.iloc[valid].reset_index(drop=True)
        for name,(kind,idx) in specs.items():
            if kind == 'dummy':
                model = fit(estimator(kind,None),x[train][:,idx],y[train],groups[train]); param=None; score=np.nan
            else:
                model,param,score=tune(kind,x[train],y[train],groups[train],inner,idx,inner_specs[name][1])
            pred=model.predict(x[valid][:,idx]); predictions.loc[valid,name]=pred
            trial_rows.extend(prediction_metrics(y[valid],pred,validmeta,name,fold=fold))
            tuning.append(dict(fold=fold,model=name,param=param,inner_mean_trial_rmse_db=score,
                               train_trials=';'.join(sorted(set(groups[train]))),test_trials=';'.join(sorted(set(groups[valid])))))
            local_names=[names[i] for i in idx]
            if name=='speed_lowstate_ridge':
                coef_rows.extend(dict(fold=fold,feature=n,coefficient_per_train_sd=float(c)) for n,c in zip(local_names,model.named_steps['ridge'].coef_))
            if name in ('speed_lowstate_ridge','speed_lowstate_hist'):
                imp_specs=[('family',family,[j for j,n in enumerate(local_names) if n.startswith(prefix)]) for family,prefix in FAMILIES.items()]
                if kind=='hist':
                    imp_specs += [('raw_feature',raw,[j for j,n in enumerate(local_names) if n.split('__lag')[0]==raw]) for raw in selected]
                importance_rows.extend(importance(model,x[valid][:,idx],y[valid],validmeta,local_names,imp_specs,args.importance_repeats,
                    max(1,round(args.block_ms/cfg['sample_ms'])),args.seed+fold,fold,name))
            print(f'fold {fold}/{args.folds} {name}: RMSE {metrics(y[valid],pred)["rmse_db"]:.3f} dB',flush=True)
    selection_table=pd.DataFrame(selection_rows)
    selection_table.to_csv(args.features.parent/'cross_validation_feature_selection.csv',index=False)
    frequency=selection_table[selection_table.stage=='outer_train'].groupby('feature',as_index=False).agg(n_selected_folds=('fold','nunique'))
    frequency['fold_fraction']=frequency.n_selected_folds/args.folds
    frequency.to_csv(args.features.parent/'selection_frequency.csv',index=False)
    pd.DataFrame(tuning).to_csv(args.output/'fold_tuning.csv',index=False)
    predictions.to_csv(args.output/'out_of_fold_predictions.csv',index=False)
    pd.DataFrame(coef_rows).to_csv(args.output/'ridge_coefficients.csv',index=False)
    summary,ranks=summarize_validation(pd.DataFrame(trial_rows),predictions,pd.DataFrame(importance_rows),args.output)
    selected_model=summary[summary.model!='mean_baseline'].iloc[0].model
    inner=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed).split(x,y,groups))
    inner_selected=select_for_split(x[inner[0]],y[inner[0]],meta.iloc[inner[0]].reset_index(drop=True),names,cfg)
    inner_specs=model_specs(names,inner_selected)
    final_columns=cfg['selected_lowstate_features']+cfg['speed_features']
    final_names=cfg['feature_names']
    # Each model uses its own indices into one shared selected-feature schema.
    local_specs=model_specs(final_names,cfg['selected_lowstate_features'])
    config=dict(cfg, n_trials=len(set(groups)),n_windows=len(y),n_raw_lowstate=len(cfg['selected_lowstate_features']),
        n_predictors=len(final_names),seed=args.seed,selected_model=selected_model,training_trials=sorted(set(groups)),
        preprocessed_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        excluded_test_trials=cfg['test_trials'],feature_definition=str(args.features.resolve()),
        feature_definition_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
        model_names=list(final_specs),output=str(args.output.resolve()),interpretation='predictive contribution, not causal effect',
        loud_threshold_train_dbfs=float(np.quantile(y,.9)))
    artifacts={}; final_tuning=[]
    for name,(kind,idx) in final_specs.items():
        if kind=='dummy':
            model=fit(estimator(kind,None),x[:,idx],y,groups); param=None; score=np.nan
        else:
            model,param,score=tune(kind,x,y,groups,inner,idx,inner_specs[name][1])
        final_tuning.append(dict(model=name,param=param,inner_mean_trial_rmse_db=score))
        artifacts[name]=dict(model=model,indices=local_specs[name][1],columns=final_columns,feature_names=final_names,
                             config=dict(config,selected_model=name,selected_param=param))
    pd.DataFrame(final_tuning).to_csv(args.model_dir/'training_tuning.csv',index=False)
    joblib.dump(dict(models=artifacts,config=config),args.model_dir/'noise_models.joblib')
    joblib.dump(artifacts[selected_model],args.model_dir/'noise_predictor.joblib')
    (args.model_dir/'run.json').write_text(json.dumps(config,indent=2))
    (args.output/'run.json').write_text(json.dumps(config,indent=2))
    ranks.to_csv(args.features.parent/'model_feature_importance.csv',index=False)
    print(f'Saved fitted models -> {args.model_dir}; cross-validation predictions -> {args.output}',flush=True)
    print('Run model_evaluation.py for evaluation figures and report.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
