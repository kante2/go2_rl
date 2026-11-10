#!/usr/bin/env python3
"""Independently audit source integrity, split isolation and saved metrics."""
from pathlib import Path
import json
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from noise_prediction_model.scripts.features import OUT, SOURCE, digest, load_dataset, write_json
from noise_prediction_model.scripts.train_evaluate import MODELS, metrics
from noise_prediction_model.scripts.torch_models import load_bundle


def verify_analysis(output, data, folds):
    folder = output/'results/analysis'
    if not (folder/'evaluation_predictions.csv.gz').exists():
        return None
    table = pd.read_csv(folder/'evaluation_predictions.csv.gz',float_precision='round_trip')
    meta = data['meta']; train = meta.split.eq('train').to_numpy()
    reference = meta.assign(y=data['y']).set_index('sample_id')
    for (name,split),part in table.groupby(['model','split']):
        expected_ids = meta.loc[meta.split.eq('train' if split=='train_oof' else 'test'),'sample_id']
        assert part.sample_id.tolist()==expected_ids.tolist()
        original=reference.loc[part.sample_id]
        np.testing.assert_allclose(part.y_true_dbfs,original.y,rtol=0,atol=1e-12)
        if name not in {'train_mean','command_speed_only'}:
            continue
        expected=np.empty(len(part))
        if split=='test':
            assignments=[(np.arange(len(part)),meta[train].trial.unique())]
        else:
            assignments=[]
            for _,fold in folds.groupby('fold'):
                heldout=set(fold.loc[fold.role.eq('heldout'),'trial'])
                fit=fold.loc[fold.role.eq('training'),'trial'].to_numpy()
                assignments.append((np.flatnonzero(part.trial.isin(heldout)),fit))
        for ids,fit_trials in assignments:
            fit=meta.trial.isin(fit_trials).to_numpy() & train
            mean=float(data['y'][fit].mean())
            if name=='train_mean':expected[ids]=mean
            else:
                for i in ids:
                    speed=part.command_vx.iloc[i]
                    same=fit & np.isclose(meta.command_vx,speed,atol=.0005)
                    expected[i]=data['y'][same].mean() if same.any() else mean
        np.testing.assert_allclose(part.y_pred_dbfs,expected,rtol=0,atol=1e-10)
    groups=pd.read_csv(folder/'metrics_by_speed_sync_and_loudness.csv',float_precision='round_trip')
    cutoff=float(np.quantile(data['y'][train],.9))
    for row in groups.itertuples():
        part=table[table.model.eq(row.model)&table.split.eq(row.split)]
        if row.dimension=='speed':part=part[np.isclose(part.command_vx,float(row.group))]
        elif row.dimension=='sync_group':part=part[part.sync_group.eq(row.group)]
        else:part=part[part.y_true_dbfs>=cutoff]
        assert len(part)==row.n_samples and part.trial.nunique()==row.n_trials
        error=part.y_pred_dbfs.to_numpy()-part.y_true_dbfs.to_numpy()
        np.testing.assert_allclose(row.rmse_db,np.sqrt(np.mean(error**2)),atol=1e-10)
        np.testing.assert_allclose(row.mae_db,np.mean(abs(error)),atol=1e-10)
        np.testing.assert_allclose(row.bias_db,np.mean(error),atol=1e-10)
    return dict(baseline_fit_isolation=True,baseline_predictions_recomputed=True,
                subgroup_metric_rows_recomputed=len(groups),all_trial_predictions_checked=True)


def verify(output=OUT, source=None):
    output = Path(output).resolve()
    manifest = json.loads((output / 'datasets/source_manifest.json').read_text())
    source = Path(source or manifest['root']).resolve()
    assert Path(manifest['root']) == source
    for relative, expected in manifest['sha256'].items():
        assert digest(source / relative) == expected, relative
    source_split = json.loads((source / 'split.json').read_text())
    data = load_dataset(output)
    meta = data['meta']
    assert set(meta.split) == {'train', 'test'}
    assert np.isfinite(data['x']).all() and np.isfinite(data['y']).all()
    assert data['x'].shape == (len(meta), 64)
    assert not any('audio' in name or 'command' in name for name in data['names'])
    for split in ('train', 'test'):
        assert set(meta[meta.split.eq(split)].trial) == set(source_split[f'{split}_trials'])
    assert not set(source_split['train_trials']) & set(source_split['test_trials'])
    assert ((meta.target_support_start-meta.anchor_time) > 0).all()
    folds = pd.read_csv(output / 'datasets/cv_recordings.csv')
    for _, fold in folds.groupby('fold'):
        tr, va = set(fold[fold.role.eq('training')].trial), set(fold[fold.role.eq('heldout')].trial)
        assert not tr & va
        assert tr | va == set(source_split['train_trials'])
    assert folds[folds.role.eq('heldout')].trial.value_counts().eq(1).all()
    frozen = json.loads((output / 'models/frozen_manifest.json').read_text())
    expected_models = {f'models/{key}/model.pt' for key, _ in MODELS}
    assert set(frozen['models']) == expected_models
    assert {str(path.relative_to(output)) for path in (output / 'models').rglob('*.pt')} == expected_models
    assert frozen['test_evaluation_started'] is False  # snapshot was written before evaluation
    assert frozen['data_manifest_sha256'] == digest(output / 'datasets/source_manifest.json')
    for relative, expected in frozen['models'].items():
        assert digest(output / relative) == expected, relative
    comparison = pd.read_csv(output / 'results/model_comparison.csv')
    assert set(comparison.model) == {key for key, _ in MODELS}
    assert set(comparison.split) == {'train', 'train_oof', 'test'}
    reports = []
    for key, _ in MODELS:
        folder = output / 'results' / key
        bundle = load_bundle(output / 'models' / key / 'model.pt')
        assert bundle['backend'] == 'pytorch'
        assert isinstance(bundle['model'], torch.nn.Module)
        if key == '02_hgb':
            assert bundle['training_backend'] == 'sklearn'
            assert bundle['model'].threshold.dtype == bundle['model'].value.dtype == torch.float64
            assert bundle['model'].feature.dtype == torch.int64
            parity = json.loads((folder / 'conversion_parity.json').read_text())
            assert {entry['split'] for entry in parity['splits']} == {'train', 'test'}
            assert all(entry['max_abs_error_db'] < 1e-10 for entry in parity['splits'])
        else:
            assert all(t.dtype == torch.float32 for t in bundle['model'].state_dict().values())
        assert set(bundle['fit_trials']) == set(source_split['train_trials'])
        assert bundle['selection']['used_test_to_select'] is False
        assert bundle['feature_names'] == data['names'].tolist()
        prediction = pd.read_csv(folder / 'predictions.csv.gz', float_precision='round_trip')
        scores = pd.read_csv(folder / 'metrics.csv')
        for split in ('train', 'train_oof', 'test'):
            p = prediction[prediction.split.eq(split)]
            original = meta[meta.split.eq('train' if split == 'train_oof' else split)]
            assert p.sample_id.tolist() == original.sample_id.tolist()
            assert p.sample_id.is_unique
            np.testing.assert_allclose(p.y_true_dbfs, data['y'][original.index], rtol=0, atol=1e-12)
            actual = metrics(p.y_true_dbfs.to_numpy(), p.y_pred_dbfs.to_numpy())
            for column, value in actual.items():
                np.testing.assert_allclose(scores[scores.split.eq(split)][column].iloc[0], value, atol=1e-10)
                np.testing.assert_allclose(comparison[comparison.model.eq(key) & comparison.split.eq(split)][column].iloc[0], value, atol=1e-10)
            if key == '03_sparse_gpr':
                assert np.isfinite(p[['latent_std_db', 'observation_std_db']]).all().all()
                assert (p.observation_std_db > 0).all()
                assert (p.observation_std_db >= p.latent_std_db).all()
                uncertainty = pd.read_csv(folder / 'uncertainty_metrics.csv')
                coverage = ((p.y_true_dbfs >= p.lower_95_dbfs) & (p.y_true_dbfs <= p.upper_95_dbfs)).mean()
                np.testing.assert_allclose(coverage, uncertainty[uncertainty.split.eq(split)].coverage_95.iloc[0], atol=1e-12)
            if split != 'train_oof':
                ids = original.index.to_numpy() if key == '02_hgb' else original.index.to_numpy()[:20]
                with threadpool_limits(limits=1):
                    predicted = bundle['model'].predict(data['x'][ids])
                if key == '02_hgb':
                    np.testing.assert_allclose(predicted, p.y_pred_dbfs, rtol=1e-12, atol=1e-10)
                else:
                    # Float32 GEMM can round differently across batch sizes.
                    np.testing.assert_allclose(predicted, p.y_pred_dbfs.iloc[:20], rtol=1e-5, atol=.002)
        for split, ids in [('train', source_split['train_trials']), ('test', source_split['test_trials'])]:
            for trial in ids:
                assert (folder / 'figures' / split / f'{trial}_predictions.png').is_file()
        reports.append(dict(model=key, backend='pytorch', metrics_recomputed=True, model_roundtrip=True, train_test_isolated=True))
    actual_folders = {p.name for p in (output / 'results').iterdir() if p.is_dir()}
    required_folders = {key for key, _ in MODELS}
    assert required_folders <= actual_folders
    assert actual_folders <= required_folders | {'analysis', 'feature_selection'}
    for key in required_folders | {'causal_forecast'}:
        assert not list((output / 'models' / key).rglob('*.joblib'))
    assert not (output / 'models/pytorch').exists() and not (output / 'results/pytorch').exists()
    run = json.loads((output / 'results/run.json').read_text())
    assert run['status'] == 'complete' and run['all_models_frozen_before_test'] and run['backend'] == 'pytorch'
    report = dict(status='passed', backend='pytorch', checkpoint_format='metadata + state_dict', n_source_files=len(manifest['sha256']),
                  train_samples=int(meta.split.eq('train').sum()), test_samples=int(meta.split.eq('test').sum()),
                  n_features=64, models=reports)
    report['analysis_verification'] = verify_analysis(output, data, folds)
    write_json(output / 'results/verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify(args.output, args.source)
