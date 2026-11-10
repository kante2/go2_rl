#!/usr/bin/env python3
"""Train/evaluate Ridge -> HGB -> sparse GPR with PyTorch inference on an explicit dataset."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from noise_prediction_model.scripts.features import OUT, SOURCE, SEED, FEATURE_SCHEMA, prepare, digest, write_json
from noise_prediction_model.scripts.sparse_gp import SparseGaussianProcessRegressor
from noise_prediction_model.scripts.torch_models import TorchRidge, TorchHGB, save_bundle, load_bundle
from noise_prediction_model.scripts.plots import (
    prediction_figures, training_curve, comparison_figure, uncertainty_figure,
)

MODELS = [('01_ridge_regression', 'Ridge Regression'), ('02_hgb', 'HGB'),
          ('03_sparse_gpr', 'Sparse Gaussian Process Regression')]


def metrics(y, prediction):
    mse = float(mean_squared_error(y, prediction))
    return dict(rmse_db=float(np.sqrt(mse)), mae_db=float(np.mean(abs(prediction-y))),
                r2=float(r2_score(y, prediction)) if len(y)>1 and np.var(y)>1e-12 else None,
                bias_db=float(np.mean(prediction-y)))


def by_trial(y, prediction, meta):
    return pd.DataFrame([dict(trial=trial, n_samples=len(ids), **metrics(y[ids], prediction[ids]))
                         for trial, ids in meta.groupby('trial', sort=False).indices.items()])


def candidates(key, gp_epochs):
    if key == '01_ridge_regression':
        return [dict(alpha=alpha) for alpha in (1., 100., 1000.)]
    if key == '02_hgb':
        return [dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=25,
                     l2_regularization=1., learning_rate=.08),
                dict(max_iter=200, max_leaf_nodes=31, min_samples_leaf=30,
                     l2_regularization=3., learning_rate=.06)]
    return [dict(n_inducing=m, epochs=gp_epochs) for m in (128, 256)]


def make_model(key, parameters):
    if key == '01_ridge_regression':
        return TorchRidge(**parameters)
    if key == '02_hgb':
        return TorchHGB(**parameters, early_stopping=False, random_state=SEED)
    return SparseGaussianProcessRegressor(**parameters, random_state=SEED)


def select_and_fit(key, title, x, y, meta, folds, names, gp_epochs, output=OUT):
    """This function receives TRAIN ONLY; test arrays cannot affect selection."""
    folder = output / 'results' / key
    folder.mkdir(parents=True, exist_ok=True)
    scores, details, predictions = [], [], {}
    for index, parameters in enumerate(candidates(key, gp_epochs)):
        candidate = f'candidate_{index+1}'
        oof = np.full(len(y), np.nan)
        latent, observation = np.full(len(y), np.nan), np.full(len(y), np.nan)
        started = time.perf_counter()
        for fold, (tr, va) in enumerate(folds):
            if set(meta.trial.iloc[tr]) & set(meta.trial.iloc[va]):
                raise AssertionError('Group leakage')
            print(f'{title} | {candidate} | fold {fold+1}/{len(folds)} | {parameters}', flush=True)
            model = make_model(key, parameters).fit(x[tr], y[tr])
            if key == '03_sparse_gpr':
                oof[va], latent[va], observation[va] = model.predict_distribution_numpy(x[va])
            else:
                oof[va] = model.predict(x[va])
            conversion = dict(conversion_max_abs_error_db=model.check_reference(x[va])) if key == '02_hgb' else {}
            details.append(dict(**conversion, candidate=candidate, fold=fold, n_train=len(tr), n_heldout=len(va),
                                **metrics(y[va], oof[va])))
        if not np.isfinite(oof).all():
            raise AssertionError('Incomplete OOF predictions')
        trial_table = by_trial(y, oof, meta)
        score = dict(candidate=candidate, parameters=json.dumps(parameters, sort_keys=True),
                     cv_mean_trial_rmse_db=float(trial_table.rmse_db.mean()),
                     seconds=time.perf_counter()-started, **metrics(y, oof))
        scores.append(score)
        predictions[candidate] = (oof, latent, observation)
        pd.DataFrame(scores).to_csv(folder / 'cv_candidates.csv', index=False)
        pd.DataFrame(details).to_csv(folder / 'cv_folds.csv', index=False)
        print(f'{title} | {candidate} | CV recording-mean RMSE={score["cv_mean_trial_rmse_db"]:.4f} dB', flush=True)
    best = min(scores, key=lambda r: (r['cv_mean_trial_rmse_db'], r['candidate']))
    parameters = json.loads(best['parameters'])
    print(f'{title} | FINAL FIT on {len(y)} train samples | {parameters}', flush=True)
    started = time.perf_counter()
    model = make_model(key, parameters).fit(x, y)
    selection = dict(model=key, title=title, training_backend='sklearn' if key == '02_hgb' else 'pytorch', inference_backend='pytorch', candidate=best['candidate'], parameters=parameters,
                     cv_mean_trial_rmse_db=best['cv_mean_trial_rmse_db'],
                     n_train_samples=len(y), train_trials=sorted(meta.trial.unique()),
                     n_features=x.shape[1], fit_seconds=time.perf_counter()-started,
                     used_test_to_select=False, selection_metric='train-only grouped CV equal-recording mean RMSE')
    model_folder = output / 'models' / key
    model_folder.mkdir(parents=True, exist_ok=True)
    bundle = dict(schema_version=4, backend='pytorch', task='causal_forecast', causal=True,
                  model=model, model_family=key, feature_schema=FEATURE_SCHEMA,
                  training_backend=selection['training_backend'], inference_dtype='float64' if key == '02_hgb' else 'float32',
                  columns=list(range(x.shape[1])), feature_names=names.tolist(),
                  fit_trials=selection['train_trials'], target='20ms RMS centered at t+20ms, in dBFS',
                  required_ros_topics=['/lowstate'], selection=selection)
    save_bundle(model_folder / 'model.pt', model, {k: v for k, v in bundle.items() if k != 'model'})
    bundle = load_bundle(model_folder / 'model.pt')
    if key == '02_hgb':
        bundle['_reference_estimator'] = model.reference_estimator_
    write_json(model_folder / 'selection.json', selection)
    write_json(folder / 'selection.json', selection)
    figures = folder / 'figures'
    if key == '01_ridge_regression':
        coefficients = model.coefficient.detach().cpu().numpy()
        pd.DataFrame(dict(feature=names, coefficient=coefficients)).to_csv(model_folder / 'coefficients.csv', index=False)
        write_json(model_folder / 'linear_equation.json', dict(intercept=float(model.intercept),
                   formula='predicted_noise_dBFS = intercept + sum(coefficient_j * raw_feature_j)'))
    else:
        history = pd.DataFrame(model.history_)
        history.to_csv(folder / 'training_history.csv', index=False)
        if key == '02_hgb':
            training_curve(history, figures, 'Train RMSE (dB)', 'iteration', 'train_rmse_db')
            write_json(model_folder / 'tree_structure.json', dict(n_trees=model.feature.shape[0],
                       max_depth=model.max_depth, max_nodes=model.feature.shape[1],
                       threshold_dtype='float64', value_dtype='float64', training_backend='sklearn',
                       inference_backend='pytorch', training_conversion_max_abs_error_db=model.train_conversion_max_abs_error_db_))
        else:
            training_curve(history, figures, 'Negative ELBO per sample (standardized target)', 'epoch', 'negative_elbo_per_sample')
            pd.DataFrame(dict(feature=names, lengthscale_standardized=model.lengthscale.cpu().numpy())).to_csv(model_folder / 'kernel_lengthscales.csv', index=False)
            write_json(model_folder / 'gp_parameters.json', dict(kernel='Matern 3/2, ARD',
                       n_inducing=len(model.inducing), n_training_samples=model.n_training_samples_,
                       observation_noise_std_db=float(model.y_scale*model.noise_variance.sqrt()),
                       likelihood='homoscedastic Gaussian', posterior='optimal Gaussian variational posterior conditioned on all train samples',
                       backend='pytorch', posterior_solve_dtype='float64', inference_dtype='float32',
                       uncertainty='latent_std excludes observation noise; observation_std includes observation noise'))
    return bundle, predictions[best['candidate']]


def uncertainty_metrics(y, mean, latent, observation):
    from scipy.stats import norm
    error = y-mean
    row = dict(gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*observation**2)+error**2/observation**2))),
               mean_latent_std_db=float(latent.mean()), mean_observation_std_db=float(observation.mean()),
               mean_interval_width_95_db=float((2*1.96*observation).mean()))
    for level in (.5, .8, .9, .95, .99):
        row[f'coverage_{int(level*100)}'] = float(np.mean(abs(error) <= norm.ppf((1+level)/2)*observation))
    return row


def runtime(model, x):
    """Measure resident tensor calls and a separate NumPy adapter overhead."""
    result = dict(backend='pytorch', device='cpu', input_dtype='float32',
                  model_dtype='float64 + int64/bool' if isinstance(model, TorchHGB) else 'float32',
                  threads=torch.get_num_threads())
    def measure(operation):
        for _ in range(10):
            operation()
        times = []
        for _ in range(100):
            started = time.perf_counter()
            operation()
            times.append((time.perf_counter()-started)*1000)
        return dict(median_ms=float(np.median(times)), p95_ms=float(np.quantile(times, .95)),
                    p99_ms=float(np.quantile(times, .99)), max_ms=float(max(times)))
    with torch.inference_mode():
        for batch in (1, 64, 1024):
            xx = torch.as_tensor(x[:batch], dtype=torch.float32)
            entry = dict(mean_only=measure(lambda: model(xx)))
            entry['samples_per_second'] = len(xx)*1000/entry['mean_only']['median_ms']
            if isinstance(model, SparseGaussianProcessRegressor):
                entry['mean_and_std'] = measure(lambda: model.predict_distribution(xx))
            result[str(batch)] = entry
        result['numpy_adapter_single'] = measure(lambda: model.predict(x[:1]))
    result['scope'] = 'Host CPU, resident tensor calls, one thread, milliseconds per batch; excludes tensor conversion/device transfer, kinematics, feature extraction, MuJoCo physics and policy training. NumPy adapter is measured separately.'
    return result


def evaluate(key, title, bundle, oof, data, output=OUT):
    folder = output / 'results' / key
    model = bundle['model']
    tables, rows, trial_rows, uncertainty_rows, conversion_rows = [], [], [], [], []
    for split in ('train', 'train_oof', 'test'):
        mask = data['meta'].split.eq('train' if split == 'train_oof' else split).to_numpy()
        x, y = data['x'][mask], data['y'][mask]
        meta = data['meta'][mask].reset_index(drop=True).copy()
        meta['split'] = split
        if split == 'train_oof':
            prediction, latent, observation = oof
        elif key == '03_sparse_gpr':
            prediction, latent, observation = model.predict_distribution_numpy(x)
        else:
            prediction, latent, observation = model.predict(x), None, None
        if key == '02_hgb' and split != 'train_oof':
            expected = bundle['_reference_estimator'].predict(x)
            np.testing.assert_allclose(prediction, expected, atol=1e-10, rtol=1e-12)
            conversion_rows.append(dict(split=split, n_samples=len(x), max_abs_error_db=float(abs(prediction-expected).max())))
        trial_table = by_trial(y, prediction, meta)
        trial_rows.append(trial_table.assign(split=split))
        rows.append(dict(model=key, title=title, split=split, n_samples=len(y), n_trials=meta.trial.nunique(),
                         mean_trial_rmse_db=float(trial_table.rmse_db.mean()), **metrics(y, prediction)))
        meta['y_true_dbfs'], meta['y_pred_dbfs'] = y, prediction
        meta['error_db'] = prediction-y
        if key == '03_sparse_gpr':
            meta['latent_std_db'], meta['observation_std_db'] = latent, observation
            meta['lower_95_dbfs'], meta['upper_95_dbfs'] = prediction-1.96*observation, prediction+1.96*observation
            uncertainty_rows.append(dict(split=split, **uncertainty_metrics(y, prediction, latent, observation)))
        tables.append(meta)
    if conversion_rows:
        write_json(folder / 'conversion_parity.json', dict(reference_backend='sklearn', inference_backend='pytorch', splits=conversion_rows))
    table, score_table = pd.concat(tables, ignore_index=True), pd.DataFrame(rows)
    table.to_csv(folder / 'predictions.csv.gz', index=False, float_format='%.17g', compression=dict(method='gzip', mtime=0))
    score_table.to_csv(folder / 'metrics.csv', index=False)
    write_json(folder / 'metrics.json', rows)
    pd.concat(trial_rows, ignore_index=True).to_csv(folder / 'metrics_by_trial.csv', index=False)
    if uncertainty_rows:
        pd.DataFrame(uncertainty_rows).to_csv(folder / 'uncertainty_metrics.csv', index=False)
        uncertainty_figure(table, folder / 'figures')
    write_json(folder / 'inference_runtime.json', runtime(model, data['x'][data['meta'].split.eq('train').to_numpy()]))
    prediction_figures(table, folder / 'figures', title)
    return score_table


def write_report(comparison, output):
    protocol = json.loads((output / 'protocol.json').read_text())
    lines = ['# Causal forecast: PyTorch 세 모델 비교', '',
             f"입력: `{protocol['source']}`. Train {len(protocol['train_trials'])}개, test {len(protocol['test_trials'])}개 녹화.",
             protocol.get('synchronization_policy', ''),
             '모든 모델은 동일한 64차원 LowState 입력과 녹화 단위 5-fold 분할을 사용한다.',
             '`train`은 최종 모델의 학습 적합도이고, `train_oof`는 해당 녹화를 제외하고 학습한 모델의 예측이다.',
             '설정 선택은 train 내부의 녹화별 RMSE 평균으로 수행하고 최종 모델은 train 전체로 학습한다.',
             '아래 RMSE, MAE, R²는 전체 표본을 합쳐 계산한다. 녹화별 RMSE 평균도 CSV에 별도 저장한다.', '',
             '| 모델 | 분할 | RMSE (dB) | MAE (dB) | R² |', '|---|---|---:|---:|---:|']
    for r in comparison.itertuples():
        lines.append(f'| {r.title} | {r.split} | {r.rmse_db:.4f} | {r.mae_db:.4f} | {r.r2:.4f} |')
    lines += ['', '## 해석과 한계', '',
              '- 세 모델의 추론은 PyTorch 기반이며 model.pt는 metadata + state_dict 형식이다. Ridge와 GPR은 PyTorch로 학습하고, HGB는 sklearn으로 학습한 트리를 PyTorch로 변환한다. HGB 임계값과 리프 값은 float64로 유지한다.',
              '- Sparse GPR은 ARD Matérn-3/2 커널과 일정한 Gaussian 관측 잡음을 가정한다. 함수 불확실성과 관측 잡음 포함 불확실성을 구분해 저장한다.',
              '- GP 예측 구간은 모형 가정에 의존한다. 포함률은 이 데이터에서의 측정치이며 새로운 보행·바닥·센서 조건의 보장이 아니다.',
              '- 함수 표준편차가 작아도 큰 오차가 발생할 수 있다. 불확실성을 보상 벌점이나 데이터 범위 이탈 지표로 사용할 타당성은 아직 검증하지 않았다.',
              '- Test는 설정 선택에서 제외했다. 동기화 검토 시퀀스도 포함하므로 예측 오차에는 정렬 불확실성이 섞일 수 있다. 동일 수집 환경의 반복 녹화에 대한 평가이며 다른 바닥·로봇·마이크로의 일반화는 별도 검증이 필요하다.',
              '- 속도는 CPU 단일 스레드의 PyTorch 텐서 호출이다. 로봇 I/O·특징 추출·장치 전송·MuJoCo를 제외하며 GPU 성능은 측정하지 않았다.',
              '- 정답은 보정된 dB SPL이 아닌 dBFS다. 접촉력은 센서 단위, 속도는 몸체 병진을 제외한 발 상대 수직 속도다.',
              '- Test 성능으로 세 모델 중 하나를 자동 선정하지 않았다. 실시간 기본 모델은 사전에 지정한 HGB다.', '',
              '## GPR 불확실성 검증', '',
              '아래는 Gaussian 관측 예측 구간의 포함률과 평균 폭이다. 함수 불확실성만의 구간과 구분해야 한다.', '',
              '| 분할 | 95% 구간 실제 포함률 | 평균 구간 폭 (dB) | Gaussian NLL |',
              '|---|---:|---:|---:|']
    uncertainty = pd.read_csv(output / 'results/03_sparse_gpr/uncertainty_metrics.csv')
    for r in uncertainty.itertuples():
        lines.append(f'| {r.split} | {100*r.coverage_95:.2f}% | {r.mean_interval_width_95_db:.3f} | {r.gaussian_nll:.4f} |')
    lines += ['', '## GP 방법론 출처', '',
              '- https://proceedings.mlr.press/v5/titsias09a.html',
              '- https://arxiv.org/abs/1309.6835', '']
    lines += ['', '## PyTorch 텐서 추론 시간', '', '| 모델 | 배치 | 평균만 중앙값 (ms) |', '|---|---:|---:|']
    for key, title in MODELS:
        timing = json.loads((output / 'results' / key / 'inference_runtime.json').read_text())
        for batch in ('1', '64', '1024'):
            lines.append(f'| {title} | {batch} | {timing[batch]["mean_only"]["median_ms"]:.4f} |')
    (output / 'results/model_comparison.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--gp-epochs', type=int, default=30)
    parser.add_argument('--models', nargs='+', choices=['ridge', 'hgb', 'sparse_gpr'],
                        default=['ridge', 'hgb', 'sparse_gpr'], help='Train selected families; audit and reuse the other saved models/results')
    args = parser.parse_args()
    if args.gp_epochs < 1:
        parser.error('--gp-epochs must be positive')
    output = args.output.resolve()
    if output == args.source.resolve():
        parser.error('output must differ from the source dataset')
    torch.set_num_threads(1)
    started = time.perf_counter()
    previous_frozen = json.loads((output / 'models/frozen_manifest.json').read_text()) if (output / 'models/frozen_manifest.json').exists() else None
    write_json(output / 'results/run.json', dict(status='running', backend='pytorch'))
    with threadpool_limits(limits=1):
        data, protocol = prepare(source=args.source, output=output)
        train = data['meta'].split.eq('train').to_numpy()
        x, y = data['x'][train], data['y'][train]
        meta = data['meta'][train].reset_index(drop=True)
        folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y, meta.trial))
        fold_rows = []
        for fold, (tr, va) in enumerate(folds):
            for role, ids in [('training', tr), ('heldout', va)]:
                fold_rows.extend(dict(fold=fold, role=role, trial=trial) for trial in sorted(meta.trial.iloc[ids].unique()))
        (output / 'results').mkdir(exist_ok=True)
        pd.DataFrame(fold_rows).to_csv(output / 'datasets/cv_recordings.csv', index=False)
        fitted, retained_scores, trained, reused = [], [], [], []
        requested = dict(ridge='01_ridge_regression', hgb='02_hgb', sparse_gpr='03_sparse_gpr')
        selected = {requested[name] for name in args.models}
        for key, title in MODELS:
            if key in selected:
                bundle, oof = select_and_fit(key, title, x, y, meta, folds, data['names'], args.gp_epochs, output=output)
                fitted.append((key, title, bundle, oof))
                trained.append(key)
            else:
                relative = f'models/{key}/model.pt'
                if previous_frozen is None or digest(output/relative) != previous_frozen['models'].get(relative):
                    raise ValueError(f'Cannot reuse an unverified checkpoint: {relative}')
                if previous_frozen['data_manifest_sha256'] != digest(output/'datasets/source_manifest.json'):
                    raise ValueError('Cannot reuse models across changed source data')
                retained = load_bundle(output/relative)
                if retained['feature_names'] != data['names'].tolist() or set(retained['fit_trials']) != set(meta.trial):
                    raise ValueError('Reused model schema or train recordings differ')
                table = pd.read_csv(output/'results'/key/'predictions.csv.gz', float_precision='round_trip')
                oof_table = table[table.split.eq('train_oof')]
                if oof_table.sample_id.tolist() != meta.sample_id.tolist():
                    raise ValueError('Reused OOF rows differ from the current dataset')
                retained_scores.append(pd.read_csv(output/'results'/key/'metrics.csv'))
                reused.append(key)
        # The streaming default is explicit, not selected by test performance.
        compatibility = output / 'models/causal_forecast'
        compatibility.mkdir(exist_ok=True)
        write_json(compatibility / 'selection.json', dict(task='causal_forecast', estimator='hgb',
                   model_path='../02_hgb/model.pt', backend='pytorch', used_test_to_select=False,
                   role='explicit streaming default; not selected against other models on test'))
        hashes = {str(p.relative_to(output)): digest(p) for p in sorted((output / 'models').rglob('*.pt'))}
        manifest = dict(task='causal_forecast', models=hashes, test_evaluation_started=False,
                        data_manifest_sha256=digest(output / 'datasets/source_manifest.json'))
        write_json(output / 'models/frozen_manifest.json', manifest)
        print('All three models frozen. Starting evaluation.', flush=True)
        scores = retained_scores + [evaluate(*row, data, output=output) for row in fitted]
        comparison = pd.concat(scores, ignore_index=True)
        comparison = pd.concat([comparison[comparison.model.eq(key)] for key, _ in MODELS], ignore_index=True)
        comparison.to_csv(output / 'results/model_comparison.csv', index=False)
        write_json(output / 'results/model_comparison.json', comparison.to_dict('records'))
        comparison_figure(comparison, output / 'results/model_comparison.png')
        write_report(comparison, output)
        if any(digest(output / path) != value for path, value in hashes.items()):
            raise AssertionError('Frozen model changed during evaluation')
        write_json(output / 'environment.json', dict(python=platform.python_version(), numpy=np.__version__,
                   pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                   torch=torch.__version__, backend='pytorch', training_backends={'ridge': 'pytorch', 'hgb': 'sklearn', 'sparse_gpr': 'pytorch'},
                   inference_dtype={'ridge': 'float32', 'hgb': 'float64 thresholds/values; integer nodes', 'sparse_gpr': 'float32'}, device='CPU', threads=1))
        write_json(output / 'results/run.json', dict(status='complete', task='causal_forecast', seed=SEED,
                   backend='pytorch', training_backends={'ridge': 'pytorch', 'hgb': 'sklearn', 'sparse_gpr': 'pytorch'},
                   models_trained=trained, models_reused=reused, gp_epochs=args.gp_epochs, seconds=time.perf_counter()-started,
                   train_recordings=protocol['train_trials'], test_recordings=protocol['test_trials'],
                   source_manifest_sha256=digest(output / 'datasets/source_manifest.json'),
                   train_samples=len(y), test_samples=int((~train).sum()), all_models_frozen_before_test=True))
        print(comparison.to_string(index=False), flush=True)
        print(f'Complete: {time.perf_counter()-started:.1f} seconds', flush=True)


if __name__ == '__main__':
    main()
