#!/usr/bin/env python3
"""Noise model evaluation, feature analysis, figures and reports.

Run on saved predictions to rebuild evaluation without retraining. Permutation
importance is computed on held-out fold models during training; the standalone
command aggregates its saved rows rather than using the final fitted model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/low_noise_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from noise_prediction_model.paths import (ROOT, MODEL_ROOT, FEATURE_SELECTION_RESULTS as EVALUATION_DIR,
                                          FEATURE_DIR, FEATURE_SELECTION_MODELS as MODEL_DIR)

def trial_rmse(y, pred, groups):
    return float(np.mean([np.sqrt(np.mean((y[groups == g]-pred[groups == g])**2)) for g in np.unique(groups)]))

def metrics(y, p):
    return dict(rmse_db=float(np.sqrt(np.mean((y-p)**2))), mae_db=float(np.mean(np.abs(y-p))),
                r2=float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else np.nan)

def block_permutation(meta, block_samples, rng):
    order = np.arange(len(meta))
    for _, positions in meta.groupby(["trial", "segment"], sort=False).groups.items():
        pos = np.asarray(list(positions)); blocks = [pos[i:i+block_samples] for i in range(0, len(pos), block_samples)]
        if len(blocks) < 2: continue
        perm = rng.permutation(len(blocks))
        if np.array_equal(perm, np.arange(len(blocks))): perm = np.roll(perm, 1)
        order[pos] = np.concatenate([blocks[i] for i in perm])
    return order

def importance(model, x, y, meta, names, specs, repeats, block_samples, seed, fold, model_name):
    rows = []; groups = meta.trial.to_numpy(); base = model.predict(x)
    baseline = {g: np.sqrt(np.mean((y[groups == g]-base[groups == g])**2)) for g in np.unique(groups)}
    rng = np.random.default_rng(seed)
    orders = [block_permutation(meta, block_samples, rng) for _ in range(repeats)]
    work = x.copy()
    for level, name, cols in specs:
        if not cols: continue
        for repeat, order in enumerate(orders):
            work[:, cols] = x[order][:, cols]
            pred = model.predict(work)
            for g in np.unique(groups):
                mask = groups == g
                rows.append(dict(fold=fold, model=model_name, level=level, feature=name, repeat=repeat, trial=g,
                                 rmse_increase_db=float(np.sqrt(np.mean((y[mask]-pred[mask])**2))-baseline[g])))
            work[:, cols] = x[:, cols]
    return rows


def plots(predictions, ranks, output, selected_model="speed_lowstate_hist", noise_label="RMS (dBFS)"):
    output.mkdir(parents=True, exist_ok=True)
    for trial, g in predictions.groupby("trial", sort=False):
        fig, ax = plt.subplots(figsize=(14, 4))
        for i, (_, s) in enumerate(g.groupby("segment", sort=False)):
            ax.plot(s.time_from_move_start, s.actual_dbfs, c="black", lw=.6, label="Measured noise" if i == 0 else None)
            ax.plot(s.time_from_move_start, s[selected_model], c="tab:orange", lw=.8, alpha=.8,
                    label=model_label(selected_model) if i == 0 else None)
        ax.set(xlabel="Target time from supplied reference (s)", ylabel=noise_label, title=f"{trial}: out-of-fold, no landing selection")
        score = metrics(g.actual_dbfs.to_numpy(), g[selected_model].to_numpy())
        ax.text(.01,.97,f'RMSE {score["rmse_db"]:.3f} dB | MAE {score["mae_db"]:.3f} dB | R² {score["r2"]:.3f}',
                transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        ax.legend(loc='lower right'); fig.tight_layout(); fig.savefig(output/f"{trial}.png", dpi=120); plt.close(fig)
    a = ranks[(ranks.model == "speed_lowstate_hist") & (ranks.level == "family")].sort_values("mean_rmse_increase_db")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.barh(a.feature, a.mean_rmse_increase_db); ax.axvline(0, c="black", lw=.6)
    ax.set_xlabel("Held-out RMSE increase after within-trial block permutation (dB)")
    fig.tight_layout(); fig.savefig(output/"family_importance.png", dpi=120); plt.close(fig)

def report(summary, ranks, config, output):
    lines = ["# 보행 전체 LowState 기반 소음 예측", "",
             f"- {config['n_trials']}개 시퀀스, {config['n_windows']:,}개 창. 착지 조건으로 선택하지 않았습니다.",
             f"- 현재와 과거 {max(config['lags_ms']):g}ms의 상태로 {config['horizon_ms']:g}ms 이후의 소음을 예측합니다. 목표 정의: 20ms RMS noise in dBFS.",
             f"- 선정된 상태·파생 feature {config['n_raw_lowstate']}개와 실제 속도를 사용합니다. RPY는 CSV에 보존하되 중복 입력하지 않습니다.",
             "- 시각/tick/trial/명령속도/오디오는 입력하지 않습니다. 접촉 파생 모드의 발 속도 후보는 접촉 확정 후 사용 가능한 몸체 상대 z속도입니다.",
             ("- K=20 기준 모델과 K 자동 선정 모델을 비교합니다." if 'selected_k' in config else "- LowState 단독 모델과 실제 몸체 속도를 추가한 모델을 비교합니다."),
             "- 외부/내부 검증 모두 시퀀스 단위입니다. 학습 데이터에서만 정규화하고 내부 검증에서만 파라미터를 선택합니다.",
             "- 각 학습 시퀀스는 동일한 총 가중치를 가집니다.", "", "## 시퀀스 밖 예측 성능", "",
             "|모델|평균 시퀀스 RMSE (dB)|전체 RMSE (dB)|평균 시퀀스 R²|", "|---|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"|{r.model}|{r.mean_trial_rmse_db:.4f}|{r.pooled_rmse_db:.4f}|{r.mean_trial_r2:.4f}|")
    lines += ["", "R²의 시퀀스 평균은 평가 기준으로만 사용하고 예측에 넣지 않습니다. "
              "평균 소음 수준이 달라진 시퀀스에서는 R²가 음수가 될 수 있습니다.", "", "## 예측에 유용한 LowState", "",
              "검증 시퀀스/연속 구간 안에서 500ms 블록(설정 변경 가능)을 섞습니다. "
              "한 feature의 현재/과거값은 함께 이동합니다. 중요도는 RMSE 증가이며 인과 영향력이 아닙니다. "
              "자동 선정에서 개별 feature 중요도는 해당 feature가 선택된 fold의 검증 시퀀스만 집계합니다. n_trials도 함께 확인합니다.", "",
              "|Feature|평균 RMSE 증가 (dB)|성능이 나빠진 시퀀스 비율|", "|---|---:|---:|"]
    a = ranks[(ranks.model == "speed_lowstate_hist") & (ranks.level == "raw_feature")].sort_values("mean_rmse_increase_db", ascending=False).head(12)
    for r in a.itertuples():
        lines.append(f"|{r.feature}|{r.mean_rmse_increase_db:.4f}|{r.positive_trial_fraction:.0%}|")
    lines += ["", "## 해석 범위", "",
              "- move_start~move_stop 전체 유효 구간입니다. 시작/종료 가감속을 포함하며 정상속도 여부는 표시만 합니다.",
              "- 명령속도 0의 대조 기록은 기본 모델에서 제외합니다. 나머지는 순간 속도가 낮아도 유지합니다.",
              "- 과거 창과 목표는 시간 공백을 가로지르지 않습니다. coverage.csv에 시퀀스별 사용량이 있습니다.",
              "- 저장된 숫자 LowState를 모두 전처리합니다. 원본에 없는 motor temperature/ddq 등은 복원할 수 없습니다.",
              "- 로봇 리샘플링은 과거값 유지입니다. 선형보간을 명시적으로 허용한 실행은 회고적 분석으로만 해석합니다.",
              "- 소음은 미보정 dBFS이며 실제 음압 dB SPL이 아닙니다.",
              "- 중요도는 상관된 feature 사이에 분산될 수 있고 블록 교환으로 비현실적인 상태 조합이 생길 수 있습니다.",
              "- 중요도는 모델이 이용하는 정보입니다. 변수 값을 낮추면 소음이 감소한다거나 보상에 넣어야 한다는 결론이 아닙니다.",
              "- 같은 세션 trial은 공유 조건이 있습니다. 별도 날짜/바닥/마이크 조건에서 검증해야 일반화를 판단할 수 있습니다.",
              "- 저장 모델은 지정 테스트 시퀀스를 제외한 학습 데이터로 재학습했습니다. 보고된 성능은 외부 교차검증 성능입니다.",
              ("- K와 트리 크기는 내부 검증에서만 선정합니다. 별도 최종 독립 테스트 데이터는 필요합니다." if 'selected_k' in config else "- 외부 결과로 모델 종류를 고른 선정은 탐색적입니다. 독립 최종 테스트 데이터는 별도로 필요합니다."), ""]
    if 'selected_k' in config:
        lines += [f"최종 배포 모델 K={config['selected_k']}입니다. 위 외부 검증은 각 fold 내부에서 선정한 K를 사용하므로 최종 K 하나의 성능이 아닙니다. fold별 K는 상위 폴더의 fold_selection.csv에서 확인합니다.", ""]
    if not config.get("preprocessing_metadata", {}).get("delay_quality", {}).get("accepted", False):
        lines += ["오디오 지연 보정은 검증되지 않았습니다. 짧은 예측 시간의 성능/순위는 동기 오차의 영향을 받을 수 있습니다. "
                  "관측된 미래 시점 관계를 인과 순서로 해석하지 않습니다.", ""]
    if not config.get("no_plots",False): lines += figure_index_lines()
    (output/"report.md").write_text("\n".join(lines))

def save_plot(g, path, selected_model="speed_lowstate_hist", noise_label="RMS (dBFS)"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for i, (_, s) in enumerate(g.groupby("segment", sort=False)):
        for ax in axes:
            ax.plot(s.time_from_move_start, s.actual_dbfs, c="black", lw=.7, label="Measured" if i == 0 else None)
            ax.plot(s.time_from_move_start, s[selected_model], c="tab:orange", lw=.9,
                    label=model_label(selected_model) if i == 0 else None)
        axes[0].set(xlabel="Target time from supplied reference (s)", ylabel=noise_label)
    middle = float(g.time_from_move_start.median())
    axes[1].set(xlim=(middle-1, middle+1), xlabel="Target time: 2 second detail (s)", ylabel=noise_label)
    score = metrics(g.actual_dbfs.to_numpy(), g[selected_model].to_numpy())
    axes[0].text(.01,.97,f'{model_label(selected_model)} | RMSE {score["rmse_db"]:.3f} dB | MAE {score["mae_db"]:.3f} dB | R² {score["r2"]:.3f}',
                 transform=axes[0].transAxes,va='top',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    axes[0].legend(loc='lower right'); fig.suptitle(f"{g.trial.iloc[0]} — excluded from training and tuning")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def prediction_metrics(y, pred, meta, model, fold=None, loud_threshold=None):
    """Per-trial errors; a loud threshold must originate from training data."""
    rows = []
    groups = meta.trial.to_numpy()
    for trial in pd.unique(groups):
        mask = groups == trial
        row = dict(trial=trial, model=model, **metrics(y[mask], pred[mask]))
        if fold is not None:
            row.update(fold=fold, n_samples=int(mask.sum()))
        else:
            row.update(n_windows=int(mask.sum()))
        if loud_threshold is not None:
            error = pred[mask]-y[mask]
            loud = y[mask] >= loud_threshold
            row.update(mean_error_db=float(error.mean()),
                       absolute_error_p90_db=float(np.quantile(abs(error), .9)),
                       loud_threshold_train_dbfs=float(loud_threshold), n_loud_windows=int(loud.sum()),
                       loud_rmse_db=float(np.sqrt(np.mean(error[loud]**2))) if loud.any() else np.nan)
        rows.append(row)
    return rows


def validate_predictions(predictions, model_names):
    if predictions.empty or not model_names or len(model_names) != len(set(model_names)):
        raise ValueError('Nonempty predictions and unique model names required')
    required = ['trial', 'actual_dbfs'] + list(model_names)
    missing = set(required)-set(predictions.columns)
    if missing:
        raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    if predictions.trial.isna().any() or not np.isfinite(predictions[['actual_dbfs']+list(model_names)].to_numpy()).all():
        raise ValueError('Cannot evaluate missing/nonfinite targets or predictions')


def accuracy_summary(results, predictions):
    """Report equal-trial means and pooled metrics separately."""
    rows = []
    y = predictions.actual_dbfs.to_numpy()
    for name, group in results.groupby('model', sort=False):
        pooled = metrics(y, predictions[name].to_numpy())
        rows.append(dict(model=name, mean_trial_rmse_db=group.rmse_db.mean(),
                         mean_trial_mae_db=group.mae_db.mean(), mean_trial_r2=group.r2.mean(),
                         pooled_rmse_db=pooled['rmse_db'], pooled_mae_db=pooled['mae_db'],
                         pooled_r2=pooled['r2'], n_trials=len(group), n_samples=len(y)))
    return pd.DataFrame(rows).sort_values('mean_trial_rmse_db')


def model_label(name):
    return {'sparse_hist':'Sparse history / Boosting', 'history_hist':'Dense history / Boosting', 'history_tcn':'Causal TCN', 'mean_baseline':'Mean baseline', 'speed_ridge':'Speed / Ridge',
            'speed_hist':'Speed / Boosting', 'lowstate_ridge':'LowState / Ridge',
            'lowstate_hist':'LowState / Boosting', 'speed_lowstate_ridge':'LowState + speed / Ridge',
            'all_features_hist':'All candidates / Boosting', 'correlation_k_20_hist':'Correlation K=20 / Boosting', 'fixed_k_20_hist':'Fixed K=20 / Boosting', 'speed_lowstate_hist':'LowState + speed / Boosting'}.get(name, name)


def accuracy_plots(predictions, results, output, title, selected_model):
    """Accuracy figures for saved predictions; never choose a model on test scores."""
    output.mkdir(parents=True, exist_ok=True)
    summary = accuracy_summary(results, predictions)
    names = summary.model.tolist()
    colors = ['tab:orange' if n == selected_model else 'tab:blue' for n in names]
    labels = [model_label(n) for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, metric, caption in zip(axes, ['rmse_db','mae_db','r2'],
                                   ['RMSE (dB): lower is better','MAE (dB): lower is better','R²: higher is better']):
        values = summary[f'mean_trial_{metric}'].to_numpy()
        bars = ax.barh(np.arange(len(names)), values, color=colors, alpha=.85)
        ax.scatter(summary[f'pooled_{metric}'], np.arange(len(names)), c='black', marker='D', s=28, label='Pooled windows', zorder=3)
        pooled_values=summary[f'pooled_{metric}'].to_numpy()
        span=max(float(np.max(np.abs(np.r_[values,pooled_values]))),.1)
        for i,(value,pooled_value) in enumerate(zip(values,pooled_values)):
            ax.text(max(value,pooled_value)+span*.035,i,f'{value:.3f}',va='center',fontsize=9)
        ax.axvline(0, color='grey', lw=.6)
        ax.set_xlabel(caption); ax.grid(axis='x', alpha=.2)
        ax.margins(x=.22)
    axes[0].set_yticks(np.arange(len(names)), labels); axes[0].invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle(f'{title}: model accuracy\nBars = equal-trial mean; diamond = pooled; orange = model selected during training')
    fig.tight_layout(); fig.savefig(output/'model_accuracy.png', dpi=150); plt.close(fig)

    ncols = min(4, len(names)); nrows = int(np.ceil(len(names)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.4*nrows), squeeze=False, sharex=True, sharey=True)
    actual = predictions.actual_dbfs.to_numpy()
    low = min(float(actual.min()), float(predictions[names].min().min()))
    high = max(float(actual.max()), float(predictions[names].max().max()))
    pad = max((high-low)*.03, .1)
    for ax, name in zip(axes.flat, names):
        pred = predictions[name].to_numpy(); score = metrics(actual, pred)
        ax.hexbin(actual, pred, gridsize=45, bins='log', mincnt=1, cmap='viridis')
        ax.plot([low-pad,high+pad],[low-pad,high+pad], '--', color='tab:red', lw=1, label='Perfect prediction')
        ax.set(title=model_label(name), xlabel='Measured noise (dBFS)', ylabel='Predicted noise (dBFS)',
               xlim=(low-pad,high+pad), ylim=(low-pad,high+pad))
        ax.set_aspect('equal', adjustable='box')
        ax.text(.04,.96,f'Pooled RMSE {score["rmse_db"]:.3f} dB\nMAE {score["mae_db"]:.3f} dB\nR² {score["r2"]:.3f}',
                transform=ax.transAxes, va='top', fontsize=9, bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    for ax in list(axes.flat)[len(names):]: ax.set_visible(False)
    fig.suptitle(f'{title}: measured vs predicted ({len(actual):,} windows)\nRed diagonal = perfect prediction; hexagon color shows sample density')
    fig.tight_layout(); fig.savefig(output/'actual_vs_predicted.png', dpi=150); plt.close(fig)

    selected = names if 'history_tcn' in names else [n for n in ['speed_hist','lowstate_hist',selected_model] if n in names]
    selected = list(dict.fromkeys(selected)) or names[:3]
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    errors = {name:predictions[name].to_numpy()-actual for name in selected}
    minimum = min(float(e.min()) for e in errors.values()); maximum = max(float(e.max()) for e in errors.values())
    bins = np.linspace(minimum-.001,maximum+.001,65)
    for name,error in errors.items():
        color = 'tab:orange' if name==selected_model else {'speed_hist':'tab:blue','lowstate_hist':'tab:green','sparse_hist':'tab:blue','history_hist':'tab:green','history_tcn':'tab:orange'}.get(name,'tab:purple')
        axes[0].hist(error,bins=bins,density=True,histtype='step',lw=1.8,label=model_label(name),color=color)
        absolute = np.sort(abs(error))
        axes[1].plot(absolute,np.arange(1,len(absolute)+1)/len(absolute),label=model_label(name),color=color)
    axes[0].axvline(0,color='black',ls='--',lw=.8)
    axes[0].set(xlabel='Prediction - measurement (dB)',ylabel='Density',title='Signed errors: zero is ideal')
    axes[1].set(xlabel='Absolute prediction error (dB)',ylabel='Fraction of windows',ylim=(0,1),title='Absolute-error cumulative distribution')
    for ax in axes: ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.suptitle(f'{title}: error distributions');fig.tight_layout()
    fig.savefig(output/'error_distribution.png',dpi=150);plt.close(fig)

    matrix = results.pivot(index='trial',columns='model',values='rmse_db').reindex(columns=names)
    fig,ax=plt.subplots(figsize=(12,max(4, .32*len(matrix)+2)))
    im=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='YlOrRd',vmin=0)
    ax.set_xticks(np.arange(len(names)),labels,rotation=35,ha='right')
    ax.set_yticks(np.arange(len(matrix)),matrix.index)
    for i in range(len(matrix)):
        for j in range(len(names)):
            value=matrix.iloc[i,j]
            if np.isfinite(value):
                ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,
                        color='white' if value>float(matrix.max().max())*.6 else 'black')
    ax.set(title=f'{title}: RMSE per trial (lower is better)',xlabel='Model',ylabel='Held-out trial')
    fig.colorbar(im,ax=ax,label='RMSE (dB)');fig.tight_layout()
    fig.savefig(output/'trial_rmse_heatmap.png',dpi=150);plt.close(fig)


def figure_index_lines():
    return ['','## 정확도 시각화','',
            '회귀 모델은 RMSE·MAE가 낮고 R²가 높을수록 예측 성능이 좋습니다. 정확도를 백분율로 환산하지 않습니다.', '',
            '- [모델별 RMSE·MAE·R²](figures/model_accuracy.png): 막대는 시퀀스 동일 가중 평균, 검은 점은 전체 창 기준입니다.',
            '- [실측–예측 비교](figures/actual_vs_predicted.png): 대각선에 가까울수록 오차가 작습니다.',
            '- [오차 분포](figures/error_distribution.png): 과대·과소 예측과 절대 오차의 누적 분포입니다.',
            '- [시퀀스별 RMSE](figures/trial_rmse_heatmap.png): 특정 시퀀스에서 성능이 나빠지는지 확인합니다.',
            '- `figures/<trial>.png`: 실측과 예측의 시간 경향 및 RMSE·MAE·R²입니다.', '']


def summarize_validation(trials, predictions, imps, output):
    """Aggregate held-out metrics and trial-balanced permutation importance."""
    output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output/"trial_metrics.csv", index=False)
    summary = pd.DataFrame([dict(model=name, mean_trial_rmse_db=g.rmse_db.mean(), pooled_rmse_db=metrics(predictions.actual_dbfs.to_numpy(),predictions[name].to_numpy())["rmse_db"],
                                mean_trial_mae_db=g.mae_db.mean(), mean_trial_r2=g.r2.mean(), n_trials=len(g), n_samples=len(predictions)) for name,g in trials.groupby("model")]).sort_values("mean_trial_rmse_db")
    summary.to_csv(output/"model_summary.csv", index=False)
    comparison = trials.pivot(index="trial", columns="model", values="rmse_db")
    gains = [('lowstate_added_rmse_gain_db','speed_ridge','speed_lowstate_ridge'),
             ('nonlinear_rmse_gain_db','speed_lowstate_ridge','speed_lowstate_hist'),
             ('nonlinear_lowstate_added_rmse_gain_db','speed_hist','speed_lowstate_hist'),
             ('adaptive_k_rmse_gain_db','fixed_k_20_hist','speed_lowstate_hist')]
    for column, baseline, selected in gains:
        if baseline in comparison and selected in comparison:
            comparison[column] = comparison[baseline]-comparison[selected]
    comparison.to_csv(output/"trial_model_comparison.csv")
    imps.to_csv(output/"permutation_importance_trials.csv", index=False)
    average = imps.groupby(["model","level","feature","trial"], as_index=False).rmse_increase_db.mean()
    ranks = average.groupby(["model","level","feature"]).rmse_increase_db.agg(mean_rmse_increase_db="mean", sd_across_trials="std",
                       positive_trial_fraction=lambda a:float((a>0).mean()), n_trials="count").reset_index()
    ranks = ranks.sort_values(["model","level","mean_rmse_increase_db"], ascending=[True,True,False]); ranks.to_csv(output/"feature_importance.csv", index=False)
    return summary, ranks


def save_holdout_evaluation(results, predictions, run, output, no_plots=False):
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output/'test_metrics.csv', index=False)
    accuracy_summary(results,predictions).to_csv(output/'test_summary.csv',index=False)
    if not no_plots:
        (output/'figures').mkdir(parents=True,exist_ok=True)
        selected_model=run.get('selected_model','speed_lowstate_hist')
        for trial, g in predictions.groupby('trial'):
            save_plot(g, output/'figures'/f'{trial}.png',selected_model)
        accuracy_plots(predictions,results,output/'figures','Held-out test',selected_model)
    lines = [f"# {' / '.join(run['test_trials'])} 보행 전체 소음 테스트", '',
             f"학습 {len(run['train_trials'])}개 시퀀스 / 테스트 {len(run['test_trials'])}개 시퀀스. 테스트 시퀀스는 feature 분석, 학습과 파라미터 선택에서 제외했습니다.",
             '학습 코드가 저장한 모델로 예측했습니다. test.py에서는 재학습하거나 파라미터를 선택하지 않습니다.', '',
             '|시퀀스|모델|창 수|RMSE (dB)|MAE (dB)|R²|평균 오차 (dB)|', '|---|---|---:|---:|---:|---:|---:|']
    for row in results.itertuples():
        lines.append(f'|{row.trial}|{row.model}|{row.n_windows}|{row.rmse_db:.3f}|{row.mae_db:.3f}|{row.r2:.3f}|{row.mean_error_db:.3f}|')
    lines += ['', '평균 오차는 예측−측정입니다. 양수면 소음을 높게 예측한 것입니다.',
              '실제 소음의 높은 구간은 학습 데이터 90% 분위수로 정의하고 loud_rmse_db에 별도 기록했습니다.', '',
              '이전 연구 과정에 등장한 기록을 재분할했다면 완전히 새로 수집한 최종 독립 검증은 아닙니다.',
              f"목표는 {run['horizon_ms']:g}ms 이후입니다. 정의: 20ms RMS noise in dBFS.",
              '다른 날짜·바닥·마이크 조건으로의 일반화를 확정하지 않습니다.', '']
    if not no_plots: lines += figure_index_lines()
    (output/'report.md').write_text('\n'.join(lines))


def evaluate_saved_predictions(source, kind, output=None, no_plots=False):
    """Evaluate raw predictions independently of previously generated metric files."""
    output = source if output is None else output
    config = json.loads((source/'run.json').read_text())
    filename = 'out_of_fold_predictions.csv' if kind == 'cross-validation' else 'test_predictions.csv'
    predictions = pd.read_csv(source/filename)
    known = ['mean_baseline','speed_ridge','speed_hist','lowstate_ridge','lowstate_hist',
             'speed_lowstate_ridge','speed_lowstate_hist']
    names = config.get('model_names', [name for name in known if name in predictions])
    validate_predictions(predictions,names)
    selected_model = config.get('selected_model','speed_lowstate_hist')
    if selected_model not in names:
        raise ValueError('Training-selected model is absent from prediction columns')
    rows = []
    if kind == 'cross-validation':
        tuning = pd.read_csv(source/'fold_tuning.csv')
        observed = set()
        for fold, group in tuning.groupby('fold',sort=False):
            memberships = group.test_trials.unique()
            if len(memberships) != 1:
                raise ValueError('Models disagree on fold membership')
            trials = set(memberships[0].split(';'))
            if trials & observed:
                raise ValueError('Trial appears in more than one validation fold')
            observed |= trials
            selected = predictions[predictions.trial.isin(trials)]
            if set(selected.trial) != trials:
                raise ValueError('Missing validation trial predictions')
            for name in names:
                rows.extend(prediction_metrics(selected.actual_dbfs.to_numpy(),selected[name].to_numpy(),selected,name,fold=fold))
        if observed != set(predictions.trial):
            raise ValueError('Fold definitions do not cover all prediction trials')
        trials = pd.DataFrame(rows)
        summary,ranks = summarize_validation(trials,predictions,pd.read_csv(source/'permutation_importance_trials.csv'),output)
        summary = accuracy_summary(trials,predictions)
        summary.to_csv(output/'model_summary.csv',index=False)
        if not no_plots:
            plots(predictions,ranks,output/'figures',selected_model)
            accuracy_plots(predictions,trials,output/'figures','Trial-grouped cross-validation',selected_model)
        report(summary,ranks,dict(config,no_plots=no_plots),output)
    elif kind == 'holdout':
        if set(predictions.trial) != set(config['test_trials']):
            raise ValueError('Prediction trials differ from the recorded test split')
        if set(config['train_trials']) & set(config['test_trials']):
            raise ValueError('Test trials overlap training trials')
        threshold = config.get('loud_threshold_train_dbfs')
        if threshold is None or not np.isfinite(threshold):
            raise ValueError('Saved training-only loud threshold required in run.json; rerun test.py')
        for name in names:
            rows.extend(prediction_metrics(predictions.actual_dbfs.to_numpy(),predictions[name].to_numpy(),
                                           predictions,name,loud_threshold=threshold))
        save_holdout_evaluation(pd.DataFrame(rows),predictions,config,output,no_plots)
    else:
        raise ValueError(f'Unknown evaluation kind: {kind}')
    print(f'Saved {kind} evaluation -> {output}')
    return pd.DataFrame(rows)


def evaluate_optimization(source, output=None, no_plots=False):
    """Evaluate saved nested tuning results without using test scores to select K."""
    output = source if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads((source/'run.json').read_text())
    final = pd.read_csv(source/'final_k_summary.csv').sort_values('k')
    folds = pd.read_csv(source/'fold_selection.csv')
    outer = pd.read_csv(source/'k_outer_trial_metrics.csv')
    aggregate = outer.groupby('k', as_index=False).agg(
        mean_trial_rmse_db=('rmse_db','mean'), sd_trial_rmse_db=('rmse_db','std'),
        mean_trial_mae_db=('mae_db','mean'), mean_trial_r2=('r2','mean'), n_trials=('trial','nunique'))
    aggregate.to_csv(output/'k_outer_summary.csv', index=False)
    evaluate_saved_predictions(source/'cross_validation', 'cross-validation', output/'cross_validation', no_plots)
    cv = pd.read_csv(output/'cross_validation/model_summary.csv')
    stages = [('nested_outer_cv', cv)]
    if (source/'test/test_predictions.csv').exists():
        evaluate_saved_predictions(source/'test', 'holdout', output/'test', no_plots)
        stages.append(('heldout_test', pd.read_csv(output/'test/test_summary.csv')))
    selected = final[final.k == run['selected_k']].iloc[0]
    rows = [dict(stage='final_inner_selection', model='speed_lowstate_hist', k=run['selected_k'],
                 max_leaf_nodes=run['selected_param'], mean_trial_rmse_db=selected.mean_trial_rmse_db,
                 mean_trial_mae_db=selected.mean_trial_mae_db, mean_trial_r2=selected.mean_trial_r2,
                 n_trials=selected.n_trials, used_for_k_selection=True)]
    for stage, table in stages:
        for r in table.to_dict('records'):
            # Outer adaptive scores have fold-specific K; no single deployed K generated them.
            k = (20 if r['model'] in ('fixed_k_20_hist','correlation_k_20_hist') else len(run['candidate_lowstate_features']) if r['model']=='all_features_hist' else run['selected_k'] if stage=='heldout_test' and r['model']=='speed_lowstate_hist' else np.nan)
            rows.append(dict(r, stage=stage, k=k, used_for_k_selection=False))
    pd.DataFrame(rows).to_csv(output/'optimal_k_metrics.csv', index=False)
    if not no_plots:
        figures = output/'figures'
        figures.mkdir(parents=True, exist_ok=True)
        search = pd.read_csv(source/'final_k_search.csv')
        fig, ax = plt.subplots(figsize=(10,5))
        for leaf, table in search.groupby('max_leaf_nodes'):
            table=table.sort_values('k')
            ax.plot(table.k, table.mean_trial_rmse_db, 'o-', label=f'Leaves = {leaf}')
        ax.scatter([run['selected_k']], [run['final_inner_mean_trial_rmse_db']], marker='*', s=200, color='red', zorder=5, label='Selected by inner CV')
        if run['rmse_tolerance_db'] > 0:
            ax.axhline(search.mean_trial_rmse_db.min()+run['rmse_tolerance_db'], ls='--', color='gray', label='RMSE tolerance limit')
        ax.set(xticks=final.k, xlabel='K: selected state / derived features (speed 3 axes kept)', ylabel='Mean trial RMSE (dB)', title='Final K selection: training-only inner group CV')
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(figures/'k_selection_curve.png',dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1,3,figsize=(15,4))
        for ax, column, label in zip(axes,['mean_trial_rmse_db','mean_trial_mae_db','mean_trial_r2'],['RMSE (dB), lower is better','MAE (dB), lower is better','R², higher is better']):
            ax.plot(aggregate.k,aggregate[column],'o-'); ax.axvline(run['selected_k'],color='red',ls='--',label='Final selected K')
            ax.set(xticks=final.k,xlabel='K',ylabel=label); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Fixed-K outer validation: diagnostic comparison, not final K selection');fig.tight_layout();fig.savefig(figures/'k_outer_validation.png',dpi=150);plt.close(fig)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(folds.fold.astype(str),folds.selected_k)
        ax.axhline(run['selected_k'],color='red',ls='--',label=f'Final K = {run["selected_k"]}')
        ax.set(xlabel='Outer fold',ylabel='K selected using inner training trials',title='Feature-count selection across folds',yticks=final.k);ax.legend();fig.tight_layout();fig.savefig(figures/'fold_selected_k.png',dpi=150);plt.close(fig)
        fig, axes=plt.subplots(1,3,figsize=(13,4))
        for stage_index,(stage,table) in enumerate(stages):
            comparison_names=['correlation_k_20_hist','fixed_k_20_hist','speed_lowstate_hist','all_features_hist'] if run.get('selection_backend')=='extra-trees' else ['fixed_k_20_hist','speed_lowstate_hist']
            subset=table.set_index('model').reindex(comparison_names)
            for ax,column,label in zip(axes,['mean_trial_rmse_db','mean_trial_mae_db','mean_trial_r2'],['RMSE (dB)','MAE (dB)','R²']):
                bars=ax.bar(np.arange(len(subset))+(stage_index-.5)*.32,subset[column],width=.32,label=stage)
                ax.bar_label(bars,fmt='%.3f',fontsize=7,padding=2)
                labels=['Correlation 20','Model 20','Adaptive K','All candidates'] if len(subset)==4 else ['Fixed K=20','Adaptive K']
                ax.set(xticks=np.arange(len(subset)),xticklabels=labels,ylabel=label)
                ax.tick_params(axis='x',labelrotation=20)
                ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Adaptive selection accuracy: outer CV and reserved test');fig.tight_layout();fig.savefig(figures/'selected_k_accuracy.png',dpi=150);plt.close(fig)
    lines = ['# Top-K 자동 튜닝 결과', '',
             f"최종 선정: 상태·파생 변수 K={run['selected_k']}개 + 실제 속도 3축, lag 포함 {run['n_predictors']}차원. max_leaf_nodes={run['selected_param']}.",
             f"후보 K: {run['k_values']}; 트리 leaf 후보: {run['max_leaf_nodes']}; 외부 {run['folds']} / 내부 {run['inner_folds']}개 시퀀스 단위 fold.",
             f"허용 RMSE 차이: {run['rmse_tolerance_db']} dB. 가장 낮은 내부 시퀀스 평균 RMSE + 허용치 이내에서 가장 작은 K를 고릅니다.", '',
             '## 선정과 검증의 구분', '',
             ('1. 내부 학습 시퀀스에서만 전체 후보를 ExtraTrees로 학습하고, 모든 lag의 불순도 감소 중요도를 원변수별로 합산합니다. 내부 검증으로 K와 Boosting leaf 수를 함께 선택합니다.' if run.get('selection_backend')=='extra-trees' else '1. 내부 학습 시퀀스에서만 Spearman 순위를 계산하고, 내부 검증으로 K와 leaf 수를 함께 선택합니다. 원변수의 모든 lag를 함께 유지합니다.'),
             '2. 외부 학습 시퀀스로 다시 선정·학습한 모델을 외부 검증합니다. 외부 fold마다 K가 다를 수 있으며 이는 자동 선정 절차의 성능입니다.',
             '3. 최종 K는 전체 개발 학습 시퀀스의 내부 검증을 다시 실행해 선택합니다. 외부 K별 곡선과 테스트 결과를 선정에 사용하지 않습니다.',
             '4. 테스트는 저장 모델을 그대로 사용합니다. K=20 기준 모델도 동일한 내부 분할로 leaf 수를 선택합니다.', '',
             ('탐색 K와 ExtraTrees 순위 안에서의 선택이며 모든 변수 조합의 전역 최적값은 아닙니다. 트리 순위는 상호작용을 반영할 수 있지만 상관된 변수의 중요도 분산과 연속형 변수 선호 편향이 있습니다. 순위 자체를 인과 효과로 해석하지 않습니다.' if run.get('selection_backend')=='extra-trees' else '탐색 후보 안에서 선정한 K이며 모든 feature 조합에 대한 전역 최적값을 의미하지 않습니다. 순위는 단변량 상관이므로 변수 중복·상호작용까지 최적화하지 않습니다.'), '',
             '|평가 단계|모델|시퀀스 평균 RMSE (dB)|MAE (dB)|R²|', '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['stage']}|{row['model']}|{row['mean_trial_rmse_db']:.4f}|{row['mean_trial_mae_db']:.4f}|{row['mean_trial_r2']:.4f}|")
    lines += ['', '## 파일 안내', '',
              '- `final_k_search.csv` / `final_k_summary.csv`: 전체 학습 데이터의 내부 검증, K·leaf별 점수와 K별 최상 파라미터.',
              '- `k_search_summary.csv` / `inner_trial_metrics.csv`: 외부 fold와 최종 학습의 내부 탐색 지표·분할.',
              '- `k_outer_trial_metrics.csv` / `k_outer_summary.csv`: 고정 K별 외부 검증 원자료·요약. K 선정용이 아닙니다.',
              '- `fold_selection.csv`: 각 외부 fold가 선택한 K·leaf·학습/검증 시퀀스.',
              '- `optimal_k_metrics.csv`: 내부 선정, 외부 검증, 테스트를 구분한 지표.',
              '- `feature_rankings.csv` / `selected_features.csv/json`: 구간별 상관 순위와 최종 입력.',
              '- `models/`: 학습 모델 묶음과 최종 예측 모델. 기본 학습 모델을 덮어쓰지 않습니다.',
              '- `cross_validation/` / `test/`: 각 평가의 정확도 그림, 시계열 그림, 지표와 보고서.', '',
              '테스트 1.8_6/1.8_7은 이전 연구 과정에서도 관찰한 기록이므로 완전히 새로운 최종 독립 검증은 아닙니다. 오디오 시간 동기도 검증되지 않았습니다. 결과는 소음 예측 기여를 보여주며 인과 영향력을 증명하지 않습니다.', '']
    if not no_plots:
        lines += ['## 그림', '', '- [최종 K 선정 곡선](figures/k_selection_curve.png)', '- [K별 외부 검증](figures/k_outer_validation.png)',
                  '- [fold별 선정 K](figures/fold_selected_k.png)', '- [선정 절차와 K=20 정확도 비교](figures/selected_k_accuracy.png)', '']
    if run.get('selection_backend')=='extra-trees':
        lines += ['', f"외부 fold별 선택 K: {folds.selected_k.tolist()}. 서로 다르면 최적 개수의 분할 민감성이 있다는 뜻이며 최종 K를 보편적 최적값으로 해석하지 않습니다."]
        for stage, scores in stages:
            if stage=='heldout_test':
                scores=scores.set_index('model')
                before=float(scores.loc['correlation_k_20_hist','mean_trial_rmse_db'])
                after=float(scores.loc['speed_lowstate_hist','mean_trial_rmse_db'])
                full=float(scores.loc['all_features_hist','mean_trial_rmse_db'])
                lines += ['', f'Test 평균 시퀀스 RMSE: 상관 기반 20개 {before:.4f}dB, 모델 기반 자동 K {after:.4f}dB, 전체 후보 {full:.4f}dB. 자동 K의 기준 대비 변화는 {after-before:+.4f}dB이며 양수는 오차 증가입니다.']
        ranking=pd.read_csv(source/'feature_rankings.csv')
        final_rank=ranking[(ranking.context=='final') & (ranking.stage=='all_training_trials')].copy()
        final_rank['selected_input']=final_rank.feature.isin(run['selected_lowstate_features']+run['speed_features'])
        final_rank.to_csv(output/'final_feature_importance.csv',index=False)
        import joblib
        bundle=joblib.load(source/'models/noise_models.joblib')
        feature_sets={name:set(a['config']['selected_lowstate_features']) for name,a in bundle['models'].items()}
        comparison=[]
        for feature in run['candidate_lowstate_features']+run['speed_features']:
            comparison.append(dict(feature=feature,role='body_speed' if feature in run['speed_features'] else 'lowstate' if feature.startswith('ls_') else 'contact_derived',
                correlation_k20=feature in feature_sets['correlation_k_20_hist'],
                model_k20=feature in feature_sets['fixed_k_20_hist'],
                adaptive_k=feature in feature_sets['speed_lowstate_hist'],
                fixed_speed=feature in run['speed_features']))
        pd.DataFrame(comparison).to_csv(output/'input_feature_comparison.csv',index=False)
        lines += ['', '## 모델 기반 입력 선정', '',
                  f"전체 상태·파생 후보 {len(run['candidate_lowstate_features'])}개 + 고정 몸체 속도 3개. K는 속도를 제외한 변수 수이고, 유효성 마스크도 예산에 포함합니다.",
                  f"선정용 ExtraTrees 설정: {run['ranking_model_settings']}",
                  '선정 트리는 최종 예측 모델이 아닙니다. 최종 예측은 HistGradientBoosting이며 두 모델 모두 시퀀스 총 가중치를 동일하게 적용합니다.',
                  '상관 기반 20개 기준도 각 내부 학습 fold에서 다시 선정하며 leaf 수를 같은 내부 분할에서 튜닝합니다. 전체 후보 기준은 K=전체 후보 수입니다.',
                  '외부/최종 내부 검증으로 결정한 adaptive 모델을 그대로 test합니다. Test 결과로 K나 모델을 변경하지 않습니다.',
                  '동일 세션 test를 이전 실험에도 사용했으므로 새 세션의 최종 확인이 필요합니다.', '',
                  '모델 기반 순위: final_feature_importance.csv; 선정 목록: selected_features.csv; 분할별 선정 목록: feature_selection_by_split.csv.',
                  '상관 기반 기준의 내부 검증: correlation_baseline_inner_metrics.csv.',
                  '입력별 선정 방식 비교: input_feature_comparison.csv.']
        if not no_plots:
            top=final_rank[final_rank.feature.isin(run['candidate_lowstate_features'])].sort_values('model_importance').tail(30)
            fig,ax=plt.subplots(figsize=(11,9));ax.barh(top.feature,top.model_importance,color=['tab:orange' if a else 'tab:blue' for a in top.selected_input])
            ax.set(xlabel='ExtraTrees impurity importance summed across lags',title='Training-only model feature ranking: orange = selected')
            fig.tight_layout();fig.savefig(output/'figures/model_feature_ranking.png',dpi=150);plt.close(fig)
            lines += ['[모델 기반 순위 그림](figures/model_feature_ranking.png)']
    (output/'report.md').write_text('\n'.join(lines))
    print(f'Saved optimization evaluation -> {output}')


def match_noise_peaks(actual, predicted, times, prominence_db=6., distance_ms=100., tolerance_ms=40., sample_ms=20.):
    """One-to-one peak matching in a single contiguous segment, no time shifting."""
    from scipy.signal import find_peaks
    distance=max(1,round(distance_ms/sample_ms))
    true_idx=find_peaks(actual,prominence=prominence_db,distance=distance)[0]
    pred_idx=find_peaks(predicted,prominence=prominence_db,distance=distance)[0]
    candidates=[]
    for i in true_idx:
        for j in pred_idx:
            dt=abs(float(times[i]-times[j]))*1000
            if dt<=tolerance_ms+1e-7:
                candidates.append((dt,int(i),int(j)))
    matched=[]; used_true=set(); used_pred=set()
    for dt,i,j in sorted(candidates):
        if i not in used_true and j not in used_pred:
            used_true.add(i); used_pred.add(j)
            matched.append(dict(actual_time=float(times[i]),predicted_time=float(times[j]),
                actual_peak_dbfs=float(actual[i]),predicted_peak_dbfs=float(predicted[j]),
                peak_amplitude_error_db=float(predicted[j]-actual[i]),peak_time_error_ms=float((times[j]-times[i])*1000)))
    return len(true_idx),len(pred_idx),matched


def temporal_trial_metrics(predictions, model, threshold, config):
    rows=prediction_metrics(predictions.actual_dbfs.to_numpy(),predictions[model].to_numpy(),predictions,model,loud_threshold=threshold)
    if len(rows)!=1: raise ValueError('Pass one trial at a time')
    row=rows[0]; match_rows=[]; n_true=n_pred=0; change_errors=[]
    # Window drops can create gaps even inside an original segment.
    sorted_pred=predictions.sort_values('target_time')
    breaks=(sorted_pred.segment.ne(sorted_pred.segment.shift()) |
            sorted_pred.target_time.diff().gt(config['sample_ms']/1000*1.5)).cumsum()
    for segment, g in sorted_pred.groupby(breaks,sort=False):
        y=g.actual_dbfs.to_numpy(); p=g[model].to_numpy(); t=g.target_time.to_numpy()
        a,b,matches=match_noise_peaks(y,p,t,config['peak_prominence_db'],config['peak_distance_ms'],config['peak_match_tolerance_ms'],config['sample_ms'])
        n_true+=a; n_pred+=b
        match_rows.extend(dict(r,trial=row['trial'],model=model,segment=int(segment)) for r in matches)
        changes=np.r_[False,np.abs(np.diff(y))>=config['irregular_change_threshold_db']]
        change_errors.extend((p[changes]-y[changes]).tolist())
    n_match=len(match_rows)
    precision=n_match/n_pred if n_pred else np.nan
    recall=n_match/n_true if n_true else np.nan
    f1=2*n_match/(n_true+n_pred) if n_true+n_pred else np.nan
    row.update(n_actual_peaks=n_true,n_predicted_peaks=n_pred,n_matched_peaks=n_match,
        peak_precision=precision,peak_recall=recall,peak_f1=f1,
        peak_amplitude_mae_db=float(np.mean([abs(r['peak_amplitude_error_db']) for r in match_rows])) if match_rows else np.nan,
        peak_time_mae_ms=float(np.mean([abs(r['peak_time_error_ms']) for r in match_rows])) if match_rows else np.nan,
        n_irregular_windows=len(change_errors),irregular_rmse_db=float(np.sqrt(np.mean(np.square(change_errors)))) if change_errors else np.nan)
    return row,match_rows


def temporal_accuracy_summary(rows, predictions):
    summary=accuracy_summary(rows,predictions)
    extra=[]
    for name,g in rows.groupby('model'):
        actual=int(g.n_actual_peaks.sum()); predicted=int(g.n_predicted_peaks.sum()); matched=int(g.n_matched_peaks.sum())
        extra.append(dict(model=name,mean_trial_loud_rmse_db=g.loud_rmse_db.mean(),mean_trial_irregular_rmse_db=g.irregular_rmse_db.mean(),
            n_actual_peaks=actual,n_predicted_peaks=predicted,n_matched_peaks=matched,
            peak_precision=matched/predicted if predicted else np.nan,peak_recall=matched/actual if actual else np.nan,
            peak_f1=2*matched/(actual+predicted) if actual+predicted else np.nan,
            mean_trial_peak_amplitude_mae_db=g.peak_amplitude_mae_db.mean(),mean_trial_peak_time_mae_ms=g.peak_time_mae_ms.mean()))
    return summary.merge(pd.DataFrame(extra),on='model')


def temporal_comparison_plot(g, config, path):
    fig,axes=plt.subplots(2,1,figsize=(14,7))
    names=config['model_names']; colors=['tab:blue','tab:green','tab:orange']
    for i,(_,segment) in enumerate(g.groupby('segment',sort=False)):
        for ax in axes:
            ax.plot(segment.time_from_move_start,segment.actual_dbfs,color='black',lw=.9,label='Measured RMS' if i==0 else None)
        axes[0].plot(segment.time_from_move_start,segment[config['selected_model']],color='tab:red',lw=.8,label='Training-selected prediction' if i==0 else None)
        for name,color in zip(names,colors):
            axes[1].plot(segment.time_from_move_start,segment[name],color=color,lw=1,label=model_label(name) if i==0 else None)
    center=float(g.loc[g.actual_dbfs.idxmax(),'time_from_move_start'])
    center=np.clip(center,g.time_from_move_start.min()+1,g.time_from_move_start.max()-1)
    axes[1].set_xlim(center-1,center+1)
    for ax in axes:
        ax.set(xlabel='Target time from move start (s)',ylabel='20ms RMS (dBFS)');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower right')
    axes[1].set_title('Largest measured peak: same 2 second interval for all models')
    fig.suptitle(f'{g.trial.iloc[0]}: causal prediction, no post-hoc time shift');fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def benchmark_temporal_models(source, config, output):
    import joblib
    import platform
    import time
    from threadpoolctl import threadpool_limits
    from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import build_windows
    bundle=joblib.load(source/'models/noise_models.joblib')
    raw=source/'test/run.json'
    if not raw.exists(): return
    test_run=json.loads(raw.read_text());reference=next(iter(bundle['models'].values()))
    fields=['trial','time_abs','time_from_move_start','segment','command_vx']+reference['columns']
    df=pd.read_csv(test_run['test_input']) if config.get('contact_feature_config') else pd.read_csv(test_run['test_input'],usecols=fields)
    df=df[df.trial.isin(config['excluded_test_trials'])]
    x,_,_,_,_=build_windows(df,reference['columns'],config['lags_ms'],config['sample_ms'],config['horizon_ms'],with_targets=False,include_standing=config['include_standing'],contact_config=config.get('contact_feature_config'))
    rows=[]
    with threadpool_limits(limits=1):
        for name,a in bundle['models'].items():
            inputs=x[:,a['indices']]
            for i in range(20): a['model'].predict(inputs[i:i+1])
            durations=[]
            for i in range(300):
                j=i%len(inputs); start=time.perf_counter_ns();a['model'].predict(inputs[j:j+1]);durations.append((time.perf_counter_ns()-start)/1e6)
            settings=config['final_model_settings'][name]
            rows.append(dict(model=name,history_ms=settings['history_ms'],n_predictors=len(a['indices']),
                median_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)),p99_ms=float(np.quantile(durations,.99)),
                max_ms=max(durations),n_calls=len(durations),warmup_calls=20,model_threads=1,
                horizon_ms=config['horizon_ms'],cpu=platform.processor(),machine=platform.machine(),
                scope='model prediction incl TCN scaling; excludes window construction, ROS and IO'))
    pd.DataFrame(rows).to_csv(output/'inference_latency.csv',index=False)


def evaluate_temporal(source, output=None, no_plots=False):
    output=source if output is None else output; output.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'run.json').read_text());names=config['model_names'];summaries=[]
    stage_results={}
    for stage,folder,filename in [('nested_outer_cv','cross_validation','out_of_fold_predictions.csv'),('heldout_test','test','test_predictions.csv')]:
        pred_path=source/folder/filename
        if not pred_path.exists():
            if stage=='heldout_test': continue
            raise ValueError(f'Missing predictions: {pred_path}')
        pred=pd.read_csv(pred_path);validate_predictions(pred,names)
        if stage=='nested_outer_cv':
            tuning=pd.read_csv(source/folder/'fold_tuning.csv');thresholds=pd.read_csv(source/folder/'fold_thresholds.csv').set_index('fold')
            by_trial={}
            for fold,g in tuning.groupby('fold'):
                if len(g.test_trials.unique())!=1: raise ValueError('Models disagree on validation trials')
                for trial in g.test_trials.iloc[0].split(';'):
                    if trial in by_trial: raise ValueError('Duplicate outer validation trial')
                    by_trial[trial]=(fold,float(thresholds.loc[fold,'loud_threshold_train_dbfs']))
            if set(by_trial)!=set(pred.trial) or set(pred.trial)!=set(config['training_trials']):
                raise ValueError('Incomplete outer prediction coverage')
        else:
            test_run=json.loads((source/folder/'run.json').read_text())
            if set(pred.trial)!=set(config['excluded_test_trials']) or set(pred.trial)&set(config['training_trials']):
                raise ValueError('Test split mismatch')
            by_trial={trial:(None,test_run['loud_threshold_train_dbfs']) for trial in pred.trial.unique()}
        rows=[];matches=[]
        for trial,g in pred.groupby('trial',sort=False):
            fold,threshold=by_trial[trial]
            for name in names:
                row,pairs=temporal_trial_metrics(g,name,threshold,config);row['fold']=fold
                rows.append(row);matches.extend(pairs)
        results=pd.DataFrame(rows);summary=temporal_accuracy_summary(results,pred)
        dest=output/folder;dest.mkdir(parents=True,exist_ok=True)
        results.to_csv(dest/'trial_metrics.csv',index=False);summary.to_csv(dest/'model_summary.csv',index=False)
        pd.DataFrame(matches).to_csv(dest/'peak_matches.csv',index=False)
        summaries.append(summary.assign(stage=stage));stage_results[stage]=summary
        if not no_plots:
            figures=dest/'figures';figures.mkdir(parents=True,exist_ok=True)
            accuracy_plots(pred,results,figures,stage,config['selected_model'])
            for trial,g in pred.groupby('trial',sort=False): temporal_comparison_plot(g,config,figures/f'{trial}.png')
    pd.concat(summaries,ignore_index=True).to_csv(output/'temporal_metrics.csv',index=False)
    final=pd.read_csv(source/'final_history_search.csv')
    if not no_plots:
        figures=output/'figures';figures.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(10,5))
        for name in ['history_hist','history_tcn']:
            g=final[final.model==name].sort_values('history_ms')
            ax.plot(g.history_ms,g.inner_mean_trial_rmse_db,'o-',label=model_label(name))
        ax.axhline(final[final.model=='sparse_hist'].inner_mean_trial_rmse_db.iloc[0],color='gray',ls='--',label='Sparse Boosting baseline')
        ax.set(xticks=config['history_ms'],xlabel='Past history (ms)',ylabel='Inner mean trial RMSE (dB)',title='Final history selection: training trials only');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figures/'history_selection_curve.png',dpi=150);plt.close(fig)
        curves=pd.read_csv(source/'tcn_learning_curves.csv');curves=curves[curves.context=='final']
        fig,axes=plt.subplots(len(config['history_ms']),1,figsize=(10,3*len(config['history_ms'])),squeeze=False)
        for ax,h in zip(axes[:,0],config['history_ms']):
            g=curves[curves.history_ms==h]
            ax.plot(g.epoch,g.validation_mean_trial_rmse_db,color='tab:orange',label='Validation trial RMSE (dB)')
            other=ax.twinx();other.plot(g.epoch,g.weighted_normalized_mse,color='tab:blue',ls='--',label='Training normalized MSE')
            ax.set(title=f'TCN: {h}ms history',xlabel='Epoch',ylabel='Validation RMSE (dB)');other.set_ylabel('Training normalized MSE');ax.legend(loc='upper right');other.legend(loc='upper left')
        fig.tight_layout();fig.savefig(figures/'tcn_learning_curves.png',dpi=150);plt.close(fig)
        columns=['mean_trial_loud_rmse_db','peak_f1','mean_trial_peak_amplitude_mae_db','mean_trial_peak_time_mae_ms']
        labels=['High-noise RMSE (dB), lower','Peak F1, higher','Matched peak amplitude MAE (dB), lower','Matched peak timing MAE (ms), lower']
        fig,axes=plt.subplots(len(stage_results),4,figsize=(19,4*len(stage_results)),squeeze=False)
        for axes_row,(stage,table) in zip(axes,stage_results.items()):
            ordered=table.set_index('model').reindex(names)
            for ax,column,label in zip(axes_row,columns,labels):
                bars=ax.bar(np.arange(len(names)),ordered[column],color=['tab:blue','tab:green','tab:orange'])
                ax.bar_label(bars,fmt='%.3f',fontsize=8);ax.set(xticks=np.arange(len(names)),xticklabels=['Sparse HGB','History HGB','TCN'],ylabel=label,title=stage);ax.grid(axis='y',alpha=.2);ax.margins(y=.15)
        fig.tight_layout();fig.savefig(figures/'peak_accuracy.png',dpi=150);plt.close(fig)
    benchmark_temporal_models(source,config,output)
    if not no_plots and (output/'inference_latency.csv').exists():
        latency=pd.read_csv(output/'inference_latency.csv').set_index('model').reindex(names)
        fig,ax=plt.subplots(figsize=(10,5))
        for j,(column,label) in enumerate([('median_ms','Median'),('p95_ms','P95'),('p99_ms','P99')]):
            bars=ax.bar(np.arange(len(names))+(j-1)*.24,latency[column],width=.24,label=label)
            ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.set(xticks=np.arange(len(names)),xticklabels=[model_label(name) for name in names],ylabel='Warm single-window inference (ms)',title='CPU 1 thread: model prediction including TCN scaling')
        ax.legend();ax.grid(axis='y',alpha=.2);ax.margins(y=.2);fig.tight_layout();fig.savefig(output/'figures/inference_latency.png',dpi=150);plt.close(fig)
    lines=['# 과거 상태 창과 causal TCN 비교', '',
           f"학습 {config['n_trials']}개 시퀀스 / 공통 {config['n_windows']:,}개 창. 각 모델은 최대 {config['common_history_ms']}ms 과거가 유효한 동일 시점에서 비교합니다.",
           f"최종 모델은 내부 검증 RMSE로 선정한 {config['selected_model']}, 과거 {config['selected_history_ms']}ms입니다.",
           'LowState K=30(설정 변경 가능)의 순위는 내부 학습·외부 학습·최종 학습마다 별도로 계산합니다. 실제 속도 3축도 동일하게 사용합니다.',
           '기존 sparse 모델은 0·20·60·100ms, history 모델은 20ms 간격의 연속 과거 상태를 사용합니다. 모두 t+20ms 중심의 20ms RMS를 예측합니다.', '',
           '## 구조와 검증', '',
           'Boosting 구조는 기존과 같은 120개 트리, leaf 15, 학습률 0.08, L2 10, 최소 leaf 표본 40개입니다. 이번 탐색은 과거 길이에 집중합니다.',
           f"TCN은 채널 {config['tcn']['hidden']}, kernel 3, dilation 1·2·4·8·16의 residual causal convolution 5개와 마지막 시점의 선형 출력입니다. 수용 범위는 63시점입니다.",
           '입력·목표의 표준화와 학습 loss에서 각 시퀀스의 총 가중치를 동일하게 적용합니다. AdamW와 MSE를 사용하며 dropout 0.1, gradient clipping 1입니다.',
           f"외부 {config['folds']}개 GroupKFold 안에서 한 번의 GroupShuffleSplit로 학습 trial 20%를 내부 검증에 예약합니다. 동일 내부 분할에서 과거 길이와 TCN early stopping을 선정합니다.",
           f"TCN은 최대 {config['tcn']['max_epochs']} epoch, patience {config['tcn']['patience']}입니다. 선택 epoch가 상한이면 더 긴 학습의 가능성이 남아 있습니다.",
           '외부 모델은 선택한 epoch 수로 외부 학습 데이터를 다시 학습합니다. 최종 모델도 전체 개발 데이터의 내부 검증을 다시 실행해 선정하며 테스트 점수로 모델·시간 창을 바꾸지 않습니다.', '',
           '## 평가', '', '|단계|모델|평균 RMSE (dB)|큰 소음 RMSE (dB)|피크 F1|매칭 피크 크기 MAE (dB)|매칭 피크 시점 MAE (ms)|', '|---|---|---:|---:|---:|---:|---:|']
    for stage,table in stage_results.items():
        for r in table.itertuples(): lines.append(f'|{stage}|{r.model}|{r.mean_trial_rmse_db:.4f}|{r.mean_trial_loud_rmse_db:.4f}|{r.peak_f1:.4f}|{r.mean_trial_peak_amplitude_mae_db:.4f}|{r.mean_trial_peak_time_mae_ms:.2f}|')
    if 'heldout_test' in stage_results:
        observed=stage_results['heldout_test'].set_index('model')
        baseline=observed.loc['sparse_hist']; dense=observed.loc['history_hist']; tcn=observed.loc['history_tcn']
        lines += ['', '## 이번 실행의 결과 해석', '',
                  f"공통 평가 시점에서 sparse Boosting 테스트 평균 RMSE는 {baseline.mean_trial_rmse_db:.4f}dB, dense Boosting은 {dense.mean_trial_rmse_db:.4f}dB, TCN은 {tcn.mean_trial_rmse_db:.4f}dB입니다.",
                  f"TCN 큰 소음 RMSE는 {tcn.mean_trial_loud_rmse_db:.4f}dB이며 기준 모델은 {baseline.mean_trial_loud_rmse_db:.4f}dB입니다. 피크 F1은 TCN {tcn.peak_f1:.4f}, 기준 {baseline.peak_f1:.4f}입니다.",
                  '전체 정확도·큰 소음 오차·피크 검출은 별개의 지표입니다. 일부 지표 개선만으로 불규칙 피크 예측이 해결됐다고 판단하지 않습니다.',
                  f"저장된 선정 모델 {config['selected_model']}은 최종 내부 검증에서 골랐으며 테스트 점수로 변경하지 않았습니다. 기존 기본 모델은 유지합니다."]
    lines += ['', f"피크는 prominence {config['peak_prominence_db']}dB, 최소 간격 {config['peak_distance_ms']}ms로 검출하며 ±{config['peak_match_tolerance_ms']}ms 안에서 가까운 피크부터 1:1 매칭합니다. 테스트 확인 전에 설정한 기준입니다.",
              '크기·시간 오차는 매칭된 피크만의 값입니다. 놓친 피크는 recall, 가짜 피크는 precision/F1로 함께 확인해야 합니다. F1은 모든 trial의 피크 수를 합산해 계산합니다.',
              '큰 소음은 해당 학습 데이터의 90% 분위수 이상입니다. 외부 fold마다 외부 학습에서만 기준을 정하고, 테스트는 최종 학습 기준을 사용합니다.',
              f"irregular_rmse_db는 실측이 인접 유효 창 사이 {config['irregular_change_threshold_db']}dB 이상 바뀐 시점의 오차입니다. 착지 조건으로 학습 데이터를 거르지 않습니다.", '',
              '## 실시간 추론', '',
              '`inference_latency.csv`는 관측 테스트 상태 300개 호출, 20회 워밍업, CPU 1 thread에서 측정합니다. TCN 표준화를 포함하고 창 구성·ROS·통신·IO는 제외합니다.',
              '과거 창은 링 버퍼에 쌓으므로 매번 과거 길이만큼 기다리지 않습니다. 처음 시작할 때 그 길이의 이력이 필요합니다. 계산 지연과 예측 horizon 20ms는 구분합니다.',
              '목표는 미래 20ms 중심의 RMS이며 라벨의 미래 30ms까지 오디오가 필요하지만 모델 입력에는 오디오가 없습니다.', '',
              '## 해석 범위', '',
              '34개 기록의 인접 창이 겹칩니다. 행 수를 독립 표본 수로 해석하지 않습니다. K=30과 모델 구조는 이전 결과를 보고 고른 탐색적 설계입니다.',
              '과거 창을 바꿔도 모든 비교 모델에 같은 최대 과거 창 유효 시점을 적용했습니다. 이전 100ms 결과와의 직접 숫자 비교보다 이 실행의 sparse baseline과 비교합니다.',
              '모델 선택용 내부 holdout 한 번은 변동성이 있으므로 장기적인 성능 결론에는 반복 시퀀스 검증·다른 수집 세션이 필요합니다.',
              '1.8_6/1.8_7은 이전 연구 과정에서도 본 데이터로 새 독립 테스트가 아닙니다. 오디오 시간 동기는 검증되지 않았습니다. 예측 곡선을 사후 이동시키지 않습니다.',
              '소음 목표는 20ms RMS이고 입력 간격은 20ms입니다. 더 짧은 충격·측정되지 않은 바닥/진동 정보를 모델 구조 변경만으로 복원할 수는 없습니다.',
              '이번 구현은 GRU를 포함하지 않습니다. 작은 TCN의 가능성을 Boosting과 비교한 실험입니다.', '',
              '## 결과 파일', '', '- `history_search.csv`, `final_history_search.csv`: 내부 시간 창 탐색 점수.',
              '- `feature_rankings.csv`, `selected_features.csv/json`: 학습 구간별 상관 순위와 최종 입력.',
              '- `temporal_metrics.csv`: 외부 검증·테스트의 전체/큰 소음/피크 지표.',
              '- `cross_validation/`, `test/`: 원시 예측, 시퀀스별 지표, 피크 매칭과 정확도·시계열 그림.',
              '- `models/`: 모델 묶음과 내부 검증으로 선정한 단독 추론 모델.', '- `inference_latency.csv`: 모델별 추론 지연.', '']
    if not no_plots: lines += ['- [과거 길이 선정 곡선](figures/history_selection_curve.png)', '- [피크 정확도](figures/peak_accuracy.png)', '- [TCN 학습 곡선](figures/tcn_learning_curves.png)', '- [추론 지연](figures/inference_latency.png)', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(f'Saved temporal evaluation -> {output}')


def evaluate_contact_comparison(source=None, no_plots=False):
    """Compare raw/contact candidates on identical windows; select models by CV."""
    source = Path(source or EVALUATION_DIR/'contact_comparison')
    raw_root = source/'raw'
    contact_root = EVALUATION_DIR
    configurations = {'raw': json.loads((raw_root/'cross_validation/run.json').read_text()),
                      'contact': json.loads((contact_root/'cross_validation/run.json').read_text())}
    for key in ['top_k','lags_ms','sample_ms','horizon_ms','training_trials','test_trials','input_size','input_mtime_ns']:
        if configurations['raw'][key] != configurations['contact'][key]:
            raise ValueError(f'Raw/contact comparison setting differs: {key}; rebuild both runs')
    rows, selected_rows, trial_rows = [], [], []
    stored = {}
    for split, folder, filename, summary_file in [
        ('cv','cross_validation','out_of_fold_predictions.csv','model_summary.csv'),
        ('test','test','test_predictions.csv','test_summary.csv')]:
        frames, summaries = {}, {}
        for mode, root in [('raw',raw_root),('contact',contact_root)]:
            frames[mode] = pd.read_csv(root/folder/filename)
            summaries[mode] = pd.read_csv(root/folder/summary_file).set_index('model')
        keys = ['trial','segment','anchor_time','target_time','actual_dbfs']
        pd.testing.assert_frame_equal(frames['raw'][keys],frames['contact'][keys],check_exact=True)
        if split == 'cv':
            for mode, root in [('raw',raw_root),('contact',contact_root)]:
                tuning = pd.read_csv(root/folder/'fold_tuning.csv')
                stored[mode] = tuning[tuning.model=='speed_hist'][['fold','train_trials','test_trials']].reset_index(drop=True)
            pd.testing.assert_frame_equal(stored['raw'],stored['contact'])
        for model in summaries['raw'].index:
            before, after = summaries['raw'].loc[model], summaries['contact'].loc[model]
            rows.append(dict(split=split,model=model,raw_rmse_db=before.mean_trial_rmse_db,
                             contact_rmse_db=after.mean_trial_rmse_db,
                             rmse_gain_db=before.mean_trial_rmse_db-after.mean_trial_rmse_db,
                             raw_r2=before.mean_trial_r2,contact_r2=after.mean_trial_r2))
        chosen = {mode: configurations[mode]['selected_model'] for mode in configurations}
        for mode in chosen:
            score = summaries[mode].loc[chosen[mode]]
            selected_rows.append(dict(split=split,feature_mode=mode,selected_model=chosen[mode],
                n_windows=len(frames[mode]),mean_trial_rmse_db=score.mean_trial_rmse_db,
                mean_trial_mae_db=score.mean_trial_mae_db,mean_trial_r2=score.mean_trial_r2,
                pooled_rmse_db=score.pooled_rmse_db,pooled_r2=score.pooled_r2))
        if split == 'test':
            for trial, positions in frames['raw'].groupby('trial',sort=False).groups.items():
                a,b = frames['raw'].loc[positions],frames['contact'].loc[positions]
                y = a.actual_dbfs.to_numpy()
                before = metrics(y,a[chosen['raw']].to_numpy())
                after = metrics(y,b[chosen['contact']].to_numpy())
                threshold = configurations['raw']['loud_threshold_train_dbfs']
                if not np.isclose(threshold,configurations['contact']['loud_threshold_train_dbfs']):
                    raise ValueError('Comparison has mismatched training loudness thresholds')
                loud=y>=threshold
                trial_rows.append(dict(trial=trial,n_windows=len(y),raw_rmse_db=before['rmse_db'],
                    contact_rmse_db=after['rmse_db'],rmse_gain_db=before['rmse_db']-after['rmse_db'],
                    raw_r2=before['r2'],contact_r2=after['r2'],n_loud=int(loud.sum()),
                    raw_loud_rmse_db=metrics(y[loud],a[chosen['raw']].to_numpy()[loud])['rmse_db'] if loud.any() else np.nan,
                    contact_loud_rmse_db=metrics(y[loud],b[chosen['contact']].to_numpy()[loud])['rmse_db'] if loud.any() else np.nan))
            stored['test_frames']=frames
    table,selected,trials = pd.DataFrame(rows),pd.DataFrame(selected_rows),pd.DataFrame(trial_rows)
    source.mkdir(parents=True,exist_ok=True)
    table.to_csv(source/'all_model_comparison.csv',index=False)
    selected.to_csv(source/'selected_model_comparison.csv',index=False)
    trials.to_csv(source/'test_trial_comparison.csv',index=False)
    (source/'comparison_checks.json').write_text(json.dumps(dict(identical_windows_and_targets=True,
        identical_outer_folds=True,model_selection='training CV only',top_k=configurations['contact']['top_k'],
        raw_candidates=len(configurations['raw']['candidate_lowstate_features']),
        contact_candidates=len(configurations['contact']['candidate_lowstate_features']),
        extra_history_ms=configurations['contact']['contact_feature_config']['extra_history_ms']),indent=2))
    ranking = pd.read_csv(FEATURE_DIR/'feature_correlation_summary.csv')
    derived = ranking[ranking.feature.str.startswith(('impact_','contact_'))].dropna(subset=['mean_abs_rho'])
    derived.to_csv(FEATURE_DIR/'contact_feature_correlation_summary.csv',index=False)
    if not no_plots:
        figure_dir=source/'figures';figure_dir.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,metric,label in zip(axes,['mean_trial_rmse_db','mean_trial_mae_db','mean_trial_r2'],['RMSE (dB)','MAE (dB)','R2']):
            for i,mode in enumerate(['raw','contact']):
                group=selected[selected.feature_mode==mode].set_index('split').loc[['cv','test']]
                bars=ax.bar(np.arange(2)+(i-.5)*.32,group[metric],width=.32,label=mode)
                ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
            ax.set(xticks=[0,1],xticklabels=['CV','Test'],ylabel=label);ax.legend();ax.grid(axis='y',alpha=.2)
        fig.suptitle('Identical windows / Top-K=20 / Models chosen by training CV')
        fig.tight_layout();fig.savefig(figure_dir/'accuracy_comparison.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(trials.trial,trials.rmse_gain_db);ax.axhline(0,color='black',lw=.7)
        ax.set(ylabel='Raw RMSE - contact RMSE (dB)',title='Test sequences: positive means improvement');ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(figure_dir/'test_trial_gain.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(16,10))
        order=derived.iloc[::-1]
        axes[0].barh(order.feature,order.mean_abs_rho);axes[0].set(xlabel='Mean absolute Spearman rho')
        axes[1].barh(order.feature,order.mean_rho);axes[1].axvline(0,color='black',lw=.6);axes[1].set(xlabel='Mean signed Spearman rho')
        for ax in axes:ax.tick_params(axis='y',labelsize=7)
        fig.suptitle('Whole-walking training data: contact features / future 20ms RMS')
        fig.tight_layout();fig.savefig(FEATURE_DIR/'contact_feature_correlations.png',dpi=150);plt.close(fig)
        frames=stored['test_frames']
        for trial in trials.trial:
            a=frames['raw'][frames['raw'].trial==trial];b=frames['contact'][frames['contact'].trial==trial]
            fig,ax=plt.subplots(figsize=(14,4))
            for segment in a.segment.unique():
                left,right=a[a.segment==segment],b[b.segment==segment]
                ax.plot(left.time_from_move_start,left.actual_dbfs,color='black',lw=.7,label='Measured')
                ax.plot(left.time_from_move_start,left[configurations['raw']['selected_model']],alpha=.8,lw=.7,label='Raw')
                ax.plot(right.time_from_move_start,right[configurations['contact']['selected_model']],alpha=.8,lw=.7,label='Contact candidates')
            handles,labels=ax.get_legend_handles_labels();unique=dict(zip(labels,handles));ax.legend(unique.values(),unique.keys())
            ax.set(xlabel='Target time from move_start (s)',ylabel='20ms RMS (dBFS)',title=trial);ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(figure_dir/f'{trial}_prediction_comparison.png',dpi=150);plt.close(fig)
    lines=['# 전체 보행 시계열 접촉 feature 비교','',
        '접촉 이벤트로 행을 필터링하지 않았다. 동일한 원본, 유효 시간 창, target, 외부 fold, top-K 20 조건으로 비교했다.',
        'raw는 LowState 50개 후보, contact는 여기에 접촉 파생 36개를 추가했다. 몸체 속도 3개와 lag 0/20/60/100ms는 동일하다.',
        '모델 종류와 하이퍼파라미터는 각 방법의 train 교차검증으로 선정했다. Test로 최종 모델을 교체하지 않았다.',
        '기존 결과의 previous_raw는 준비 구간이 달라 직접 비교 기준으로 사용하지 않았다.', '',
        '|구간|후보|선택 모델|창 수|시퀀스 평균 RMSE(dB)|평균 R²|','|---|---|---|---:|---:|---:|']
    lines.extend(f'|{r.split}|{r.feature_mode}|{r.selected_model}|{r.n_windows}|{r.mean_trial_rmse_db:.4f}|{r.mean_trial_r2:.4f}|' for r in selected.itertuples())
    lines += ['',f'Test에서 RMSE가 감소한 시퀀스: {int((trials.rmse_gain_db>0).sum())}/{len(trials)}.',
        '모델 구조를 고정한 비교는 all_model_comparison.csv, 선택 모델 비교는 selected_model_comparison.csv, 시퀀스별·고소음 구간 비교는 test_trial_comparison.csv에 저장했다.',
        '추가 후보의 효과에는 기존 입력의 교체도 포함된다. 독립적인 인과 효과를 나타내는 실험은 아니다.',
        'Spearman 순위는 단조 관계만 평가한다. 상관이 작아도 상호작용·비선형 기여가 없다는 뜻은 아니다.',
        '힘 적분은 단위 미검증 대리지표이며 접촉 전 발 속도는 몸체 좌표계 상대 z속도다. 확정 전 접촉 정보는 입력에 넣지 않았다.',
        '보정 지연 0ms는 물리적 동기화가 검증됐다는 뜻이 아니다. Test 6개와 이전 분석에 사용한 동일 세션으로 일반화 범위가 제한된다.',
        '실제 RL 보상 채택 전에는 새 세션 검증과 제어 변수 변경 실험이 필요하다.', '',
        '[정확도 비교](figures/accuracy_comparison.png) · [시퀀스별 개선](figures/test_trial_gain.png)']
    (source/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved identical-window contact comparison -> {source}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['all','cross-validation','holdout','optimization','temporal','contact-comparison'],default='all')
    parser.add_argument('--input',type=Path,help='Saved prediction directory; requires one explicit --kind')
    parser.add_argument('--output',type=Path,help='Destination; defaults to input directory; for all, a parent containing both evaluation folders')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.kind=='contact-comparison':
        evaluate_contact_comparison(args.input,args.no_plots)
        return
    if args.kind=='temporal':
        source=args.input or EVALUATION_DIR/'model_evaluation_temporal'
        evaluate_temporal(source,args.output,args.no_plots)
        return
    if args.kind=='optimization':
        source=args.input or EVALUATION_DIR/'model_evaluation_op'
        evaluate_optimization(source,args.output,args.no_plots)
        return
    if args.kind=='all' and args.input is not None:
        parser.error('--input requires --kind cross-validation or --kind holdout')
    kinds=['cross-validation','holdout'] if args.kind=='all' else [args.kind]
    for kind in kinds:
        folder='cross_validation' if kind=='cross-validation' else 'test'
        source=args.input or EVALUATION_DIR/folder
        filename='out_of_fold_predictions.csv' if kind=='cross-validation' else 'test_predictions.csv'
        if not (source/filename).exists() or not (source/'run.json').exists():
            parser.error(f'Missing saved predictions/config in {source}; run train.py and test.py first')
    for kind in kinds:
        folder='cross_validation' if kind=='cross-validation' else 'test'
        source=args.input or EVALUATION_DIR/folder
        output=(args.output/folder if args.kind=='all' else args.output) if args.output else source
        evaluate_saved_predictions(source,kind,output,args.no_plots)


if __name__=='__main__':
    main()
