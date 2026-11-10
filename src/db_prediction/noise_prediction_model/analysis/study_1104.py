#!/usr/bin/env python3
"""Train-only exploration and post-freeze evaluation for the all-trial 1104 study."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from noise_prediction_model.scripts.features import prepare, load_dataset, write_json, SEED
from noise_prediction_model.scripts.train_evaluate import metrics, by_trial, MODELS
from noise_prediction_model.scripts.plots import plt, save

DEFAULT_OUTPUT = ROOT/'noise_prediction_model/runs/1104'
SOURCE = ROOT/'preprocessed/1104'


def associations(x, y):
    rank_x = np.apply_along_axis(rankdata, 0, x)
    rank_y = rankdata(y)
    rank_x -= rank_x.mean(axis=0)
    rank_y -= rank_y.mean()
    denominator = np.sqrt((rank_x**2).sum(axis=0)*(rank_y**2).sum())
    return np.divide(rank_x.T@rank_y, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 0)


def analyze(source, output):
    data, protocol = prepare(source, output)
    folder = output/'results/analysis'; folder.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(output/'datasets/coverage.csv')
    coverage.to_csv(folder/'coverage.csv', index=False)
    mask = data['meta'].split.eq('train').to_numpy()
    x, y, meta = data['x'][mask], data['y'][mask], data['meta'][mask].reset_index(drop=True)
    # Test target values are never used in exploratory plots or thresholds.
    trial_rows = []
    for trial, ids in meta.groupby('trial', sort=False).indices.items():
        values = y[ids]
        trial_rows.append(dict(trial=trial, command_vx=float(meta.command_vx.iloc[ids].median()),
                               sync_group=meta.sync_group.iloc[ids[0]], n_samples=len(ids),
                               mean_dbfs=float(values.mean()), std_db=float(values.std()),
                               p10_dbfs=float(np.quantile(values,.1)), p50_dbfs=float(np.median(values)),
                               p90_dbfs=float(np.quantile(values,.9))))
    trial_table = pd.DataFrame(trial_rows)
    trial_table.to_csv(folder/'train_noise_by_trial.csv', index=False)
    speeds = sorted(meta.command_vx.round(3).unique())
    speed_rows = []
    for speed in speeds:
        ids = np.flatnonzero(np.isclose(meta.command_vx, speed))
        values = y[ids]
        speed_rows.append(dict(command_vx=speed, recordings=meta.trial.iloc[ids].nunique(),
                               samples=len(ids), mean_dbfs=values.mean(), std_db=values.std(),
                               median_dbfs=np.median(values), p90_dbfs=np.quantile(values,.9)))
    pd.DataFrame(speed_rows).to_csv(folder/'train_noise_by_speed.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([y[np.isclose(meta.command_vx, speed)] for speed in speeds],
               tick_labels=[f'{v:g}' for v in speeds], showfliers=False)
    ax.set(xlabel='Commanded forward speed (m/s)', ylabel='Future 20 ms RMS noise (dBFS)',
           title='Training recordings only: noise by speed')
    ax.grid(axis='y', alpha=.2); save(fig, folder/'noise_by_speed.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    for speed in speeds:
        part = trial_table[np.isclose(trial_table.command_vx, speed)]
        ax.scatter(part.trial, part.mean_dbfs, label=f'{speed:g} m/s', s=45)
    ax.tick_params(axis='x', rotation=70); ax.legend(ncol=5, fontsize=8)
    ax.set(ylabel='Mean future noise (dBFS)', title='Training recordings: repeat-to-repeat variation')
    ax.grid(axis='y', alpha=.2); save(fig, folder/'noise_by_recording.png')
    correlations = []
    for trial, ids in meta.groupby('trial', sort=False).indices.items():
        correlations.append(associations(x[ids], y[ids]))
    correlation_by_trial = pd.DataFrame(correlations, index=meta.trial.drop_duplicates(), columns=data['names'])
    correlation_by_trial.to_csv(folder/'feature_spearman_by_trial.csv', index_label='trial')
    ranked = pd.DataFrame(dict(feature=data['names'], pooled_spearman=associations(x,y),
                               median_within_trial_spearman=correlation_by_trial.median().to_numpy(),
                               median_absolute_within_trial=correlation_by_trial.abs().median().to_numpy()))
    ranked = ranked.sort_values('median_absolute_within_trial', ascending=False)
    ranked.to_csv(folder/'feature_associations.csv', index=False)
    top = ranked.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10,6))
    positions = np.arange(len(top))
    ax.barh(positions-.17, top.pooled_spearman, height=.32, label='Pooled train')
    ax.barh(positions+.17, top.median_within_trial_spearman, height=.32, label='Median within recording')
    ax.set_yticks(positions, top.feature); ax.axvline(0,color='black',lw=.8)
    ax.set(xlabel='Spearman correlation with future noise', title='Training data only: association is not causation')
    ax.legend(); save(fig, folder/'feature_associations.png')
    # Offline contact analysis may contain post-contact quantities; never model inputs.
    event_path = source/'train/contact_event_features.csv'
    if event_path.exists():
        events = pd.read_csv(event_path)
        events = events[events.feature_valid & events.trial.isin(protocol['train_trials'])].copy()
        features = ['pre_downward_speed_mps','force_post_peak_sensor_units',
                    'force_post_max_rise_sensor_units_per_s','force_post_integral_sensor_unit_s']
        event_corr = []
        for feature in features:
            for trial, group in events.groupby('trial'):
                sub = group[[feature,'noise_post_rms_dbfs']].dropna()
                if len(sub)>3:
                    event_corr.append(dict(trial=trial, feature=feature, events=len(sub),
                                           spearman=sub[feature].corr(sub.noise_post_rms_dbfs,method='spearman')))
        pd.DataFrame(event_corr).to_csv(folder/'contact_associations_by_trial.csv', index=False)
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.hexbin(events.pre_downward_speed_mps, events.noise_post_rms_dbfs, gridsize=35,mincnt=1,cmap='viridis')
        ax.set(xlabel='Pre-contact downward foot speed (m/s)', ylabel='Post-contact 40 ms RMS noise (dBFS)',
               title='Train-only descriptive contact analysis; shared microphone')
        save(fig,folder/'contact_speed_noise.png')
    loud_threshold = float(np.quantile(y,.9))
    write_json(folder/'analysis_protocol.json',dict(source=str(source), training_trials=protocol['train_trials'],
               test_targets_used_for_exploration=False, all_review_trials_included=True,
               loud_threshold_train_p90_dbfs=loud_threshold, selected_features='fixed existing 64; no ranking-based selection',
               association_policy='train only; pooled and within-recording; descriptive post-contact features excluded from prediction inputs'))
    lines = ['# 1104 데이터 분석', '',
             f"전체 train {len(protocol['train_trials'])}개 / test {len(protocol['test_trials'])}개. 동기화 검토 대상 포함.",
             '분석 그래프와 특징 관계는 train만 사용했다. Test는 분할·수량만 기록하며 정답 분포를 분석하지 않았다.',
             '예측: 과거 0/10/20/40ms LowState 특징 64개 → t+20ms 중심 20ms RMS(dBFS).',
             '접촉 후 힘/소음 분석은 기술 분석용이며 예측 입력에 사용하지 않는다.', '',
             '|속도(m/s)|train 녹화|표본|평균(dBFS)|표준편차(dB)|', '|---|---:|---:|---:|---:|']
    for row in speed_rows:
        lines.append(f"|{row['command_vx']:g}|{row['recordings']}|{row['samples']}|{row['mean_dbfs']:.2f}|{row['std_db']:.2f}|")
    lines += ['', '## 해석', '',
              '속도 차이로 생긴 pooled 상관과 같은 녹화 안에서의 상관을 구분한다. 표본 수가 많아도 독립적인 수집 단위는 24개 train 녹화다.',
              '이번 모델은 고정 64개 특징으로 비교하므로 이 분석의 순위를 보고 특징을 선택하지 않는다.',
              '정렬 검토 시퀀스의 오차에는 모델 오차와 시간 정렬 불확실성이 함께 포함될 수 있다.',
              '학습/검증은 녹화 단위 5-fold이며 test는 모델 고정 후 평가한다.']
    (folder/'data_report.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(speed_rows).to_string(index=False), flush=True)


def speed_mean_reference(fit_speed, fit_y, query_speed):
    """Train-only lookup of the mean noise at each commanded speed; unseen -> train mean."""
    means = pd.Series(fit_y).groupby(np.round(np.asarray(fit_speed).reshape(-1), 3)).mean()
    return np.array([means.get(round(float(value), 3), float(np.mean(fit_y)))
                     for value in np.asarray(query_speed).reshape(-1)])


def complete_evaluation(output):
    data = load_dataset(output); meta = data['meta']
    folder = output/'results/analysis'; folder.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((output/'models/frozen_manifest.json').read_text())
    assert frozen['test_evaluation_started'] is False
    train = meta.split.eq('train').to_numpy(); xmeta=meta[train].reset_index(drop=True)
    train_y = data['y'][train]; speeds=meta.command_vx.to_numpy(float).reshape(-1,1)
    folds=list(GroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(speeds[train],train_y,xmeta.trial))
    baseline_tables=[]; baseline_scores=[]; subgroup_rows=[]
    threshold=json.loads((folder/'analysis_protocol.json').read_text())['loud_threshold_train_p90_dbfs']
    for name in ('train_mean','command_speed_only'):
        oof=np.full(len(train_y),np.nan)
        for tr,va in folds:
            if name=='train_mean':oof[va]=train_y[tr].mean()
            else:oof[va]=speed_mean_reference(speeds[train][tr],train_y[tr],speeds[train][va])
        for split in ('train_oof','test'):
            ids=train if split=='train_oof' else ~train
            pred=oof if split=='train_oof' else speed_mean_reference(speeds[train],train_y,speeds[ids]) if name!='train_mean' else np.full(ids.sum(),train_y.mean())
            table=meta[ids].copy();table['split']=split;table['y_true_dbfs']=data['y'][ids];table['y_pred_dbfs']=pred;table['model']=name
            baseline_tables.append(table)
            baseline_scores.append(dict(model=name,split=split,n_trials=table.trial.nunique(),n_samples=len(table),
                                        mean_trial_rmse_db=float(by_trial(table.y_true_dbfs.to_numpy(),pred,table.reset_index(drop=True)).rmse_db.mean()),
                                        **metrics(table.y_true_dbfs.to_numpy(),pred)))
    pd.DataFrame(baseline_scores).to_csv(folder/'baseline_metrics.csv',index=False)
    speed_levels=pd.Series(train_y).groupby(speeds[train].reshape(-1).round(3)).mean()
    write_json(folder/'baseline_parameters.json',dict(train_mean_dbfs=float(train_y.mean()),
               command_speed_means_dbfs={str(k):float(v) for k,v in speed_levels.items()},
               unseen_speed_policy='use global train mean',selection='none; OOF means use each fold training recordings only'))
    model_tables=[]
    for key,_ in MODELS:
        table=pd.read_csv(output/'results'/key/'predictions.csv.gz',float_precision='round_trip')
        table=table[table.split.ne('train')].copy()
        missing=[name for name in ('command_vx','sync_group') if name not in table]
        if missing:
            table=table.merge(meta[['sample_id',*missing]],on='sample_id',validate='many_to_one')
        table['model']=key;model_tables.append(table)
    all_predictions=pd.concat(baseline_tables+model_tables,ignore_index=True)
    all_predictions['speed']=all_predictions.command_vx.round(3)
    for (model,split),part in all_predictions.groupby(['model','split']):
        for dimension in ('sync_group','speed'):
            for group,rows in part.groupby(dimension):
                subgroup_rows.append(dict(model=model,split=split,dimension=dimension,group=str(group),
                                           n_trials=rows.trial.nunique(),n_samples=len(rows),
                                           **metrics(rows.y_true_dbfs.to_numpy(),rows.y_pred_dbfs.to_numpy())))
        loud=part[part.y_true_dbfs>=threshold]
        subgroup_rows.append(dict(model=model,split=split,dimension='loud_noise',group='train_p90_or_above',
                                   n_trials=loud.trial.nunique(),n_samples=len(loud),
                                   **metrics(loud.y_true_dbfs.to_numpy(),loud.y_pred_dbfs.to_numpy())))
    pd.DataFrame(subgroup_rows).to_csv(folder/'metrics_by_speed_sync_and_loudness.csv',index=False)
    all_predictions.to_csv(folder/'evaluation_predictions.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    summary=pd.concat([pd.DataFrame(baseline_scores),pd.read_csv(output/'results/model_comparison.csv')],ignore_index=True)
    summary.to_csv(folder/'models_and_baselines.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4.5))
    names=['train_mean','command_speed_only',*[key for key,_ in MODELS]]
    xx=np.arange(len(names))
    for offset,split in [(-.18,'train_oof'),(.18,'test')]:
        values=summary[summary.split.eq(split)].set_index('model').loc[names]
        ax.bar(xx+offset,values.mean_trial_rmse_db,width=.35,label=split)
    ax.set_xticks(xx,['Train mean','Speed only','Ridge','HGB','Sparse GPR'])
    ax.set(ylabel='Mean recording RMSE (dB)',title='All 29 recordings included; held-out recordings in each split')
    ax.legend();ax.grid(axis='y',alpha=.2);save(fig,folder/'models_and_baselines.png')
    lines=['# 1104 전체 시퀀스 모델 평가','','동기화 검토 대상 포함: train 24개, test 5개. 기존 분리 유지.',
           '모델 설정은 train 녹화 단위 CV로 선택했다. 아래 test 성능으로 기본 모델을 재선정하지 않았다.',
           '평균값·명령 속도별 평균값 기준 모델도 같은 녹화 fold로 평가했다. 속도별 평균은 해당 fold train에서만 구하며, 없는 속도는 전체 train 평균으로 대체한다.', '',
           '|모델|분할|녹화 평균 RMSE(dB)|전체 RMSE(dB)|MAE(dB)|R²|','|---|---|---:|---:|---:|---:|']
    for row in summary[summary.split.ne('train')].itertuples():
        lines.append(f'|{row.model}|{row.split}|{row.mean_trial_rmse_db:.3f}|{row.rmse_db:.3f}|{row.mae_db:.3f}|{row.r2:.3f}|')
    lines+=['','세부 속도·동기화 상태·큰 소음의 성능은 metrics_by_speed_sync_and_loudness.csv에 기록했다.',
            f'큰 소음의 기준은 train 미래 정답의 90 분위수({threshold:.2f} dBFS)다.',
            '동기화 상태별 오차는 속도·표본 구성도 다르므로 동기화 영향의 인과 추정이 아니다.',
            '평균 소음을 맞추는 것과 개별 착지 소음을 맞추는 성능을 기준 모델 대비 개선량으로 구분한다.']
    eligible=summary[summary.model.isin([key for key,_ in MODELS]) & summary.split.eq('train_oof')]
    selected=eligible.sort_values('mean_trial_rmse_db').iloc[0].model
    test_row=summary[summary.model.eq(selected)&summary.split.eq('test')].iloc[0]
    reference=summary[summary.model.eq('command_speed_only')&summary.split.eq('test')].iloc[0]
    loud_row=pd.DataFrame(subgroup_rows).loc[lambda f:f.model.eq(selected)&f.split.eq('test')&f.dimension.eq('loud_noise')].iloc[0]
    interpretation=dict(preferred_by_train_cv=selected, test_rmse_db=float(test_row.rmse_db),
                        test_mae_db=float(test_row.mae_db), test_r2=float(test_row.r2),
                        speed_baseline_rmse_db=float(reference.rmse_db),
                        rmse_reduction_vs_speed_baseline_percent=float((1-test_row.rmse_db/reference.rmse_db)*100),
                        test_loud_noise_bias_db=float(loud_row.bias_db),
                        selection_uses_test=False)
    write_json(folder/'evaluation_summary.json',interpretation)
    lines += ['', '## 결과 해석', '',
              f"Train 녹화 평균 CV RMSE가 가장 낮은 모델은 {selected}다. 이 모델의 test RMSE는 {test_row.rmse_db:.3f} dB, MAE는 {test_row.mae_db:.3f} dB, R²는 {test_row.r2:.3f}다.",
              f"속도별 평균값 기준의 test RMSE {reference.rmse_db:.3f} dB 대비 RMSE가 {(1-test_row.rmse_db/reference.rmse_db)*100:.1f}% 감소했다. 평균 속도에 따른 차이 이상의 예측 정보를 사용한 결과다.",
              f"큰 소음 구간의 test 평균 예측−실측은 {loud_row.bias_db:.3f} dB로, 피크 소음의 과소 예측은 추가 개선 대상이다.",
              '동기화 검토 시퀀스 포함 결과이며, 더 나은 수치가 시간 정렬 정확도를 입증하지는 않는다.']
    (folder/'evaluation_report.md').write_text('\n'.join(lines)+'\n')
    print(summary[summary.split.ne('train')].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--stage',choices=['analyze','evaluate'],default='analyze')
    a=p.parse_args()
    if a.stage=='analyze':analyze(a.source.resolve(),a.output.resolve())
    else:complete_evaluation(a.output.resolve())
