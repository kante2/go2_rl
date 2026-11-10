"""Consistent observed/predicted scatter and recording timelines for all models."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def observed_vs_predicted_figure(table, folder, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, split in zip(axes, ('train', 'train_oof', 'test')):
        data = table[table.split.eq(split)]
        ax.hexbin(data.y_true_dbfs, data.y_pred_dbfs, gridsize=60, mincnt=1, bins='log', cmap='viridis')
        limits = [min(data.y_true_dbfs.min(), data.y_pred_dbfs.min())-1,
                  max(data.y_true_dbfs.max(), data.y_pred_dbfs.max())+1]
        ax.plot(limits, limits, 'r--', lw=1)
        ax.set(xlim=limits, ylim=limits, xlabel='Observed noise (dBFS)', ylabel='Predicted noise (dBFS)',
               title={'train': 'Train: in-sample', 'train_oof': 'Train: grouped out-of-fold', 'test': 'Test'}[split])
        mse = mean_squared_error(data.y_true_dbfs, data.y_pred_dbfs)
        mae = mean_absolute_error(data.y_true_dbfs, data.y_pred_dbfs)
        r2 = r2_score(data.y_true_dbfs, data.y_pred_dbfs)
        ax.text(.04, .96,
                f'RMSE = {np.sqrt(mse):.3f} dB\nMAE = {mae:.3f} dB\nR² = {r2:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=.95))
        ax.grid(alpha=.2)
    fig.suptitle(f'{title} | causal forecast, t+20ms')
    save(fig, folder / 'observed_vs_predicted.png')


def prediction_figures(table, folder, title):
    observed_vs_predicted_figure(table, folder, title)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ('train_oof', 'test')):
        data = table[table.split.eq(split)]
        error = data.y_pred_dbfs-data.y_true_dbfs
        ax.hexbin(data.y_true_dbfs, error, gridsize=50, bins='log', mincnt=1)
        ax.axhline(0, color='red', ls='--')
        ax.set(xlabel='Observed noise (dBFS)', ylabel='Prediction - observation (dB)', title=split)
    save(fig, folder / 'residuals.png')
    for (split, trial), data in table.groupby(['split', 'trial'], sort=False):
        fig, ax = plt.subplots(figsize=(12, 3.5))
        for _, segment in data.groupby('segment', sort=False):
            t = segment.target_time.to_numpy()-data.anchor_time.min()
            ax.plot(t, segment.y_true_dbfs, color='black', lw=.8, label='Observed')
            ax.plot(t, segment.y_pred_dbfs, color='tab:blue', lw=.8, label='Predicted')
            if 'observation_std_db' in segment:
                std = segment.observation_std_db.to_numpy()
                mean = segment.y_pred_dbfs.to_numpy()
                ax.fill_between(t, mean-1.96*std, mean+1.96*std, alpha=.2, color='tab:blue', label='95% observation interval')
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=8)
        ax.set(xlabel='Target time from first anchor (s)', ylabel='Noise (dBFS)', title=f'{title} | {split}/{trial}')
        ax.grid(alpha=.2)
        save(fig, folder / split / f'{trial}_predictions.png')


def training_curve(history, folder, ylabel, xcolumn, ycolumn):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history[xcolumn], history[ycolumn])
    ax.set(xlabel=xcolumn, ylabel=ylabel, title='Training data only')
    ax.grid(alpha=.3)
    save(fig, folder / 'training_cost_curve.png')


def comparison_figure(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    keys = list(dict.fromkeys(metrics.model))
    x = np.arange(len(keys))
    for ax, metric, unit in zip(axes, ('rmse_db', 'mae_db', 'r2'), ('RMSE (dB)', 'MAE (dB)', 'R²')):
        for offset, split in [(-.18, 'train_oof'), (.18, 'test')]:
            rows = metrics[metrics.split.eq(split)].set_index('model').loc[keys]
            ax.bar(x+offset, rows[metric], width=.35, label=split)
        ax.set_xticks(x, ['Ridge', 'HGB', 'Sparse GPR'])
        ax.set_ylabel(unit)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend()
    fig.suptitle('PyTorch | identical 64 features and recording splits | causal forecast')
    save(fig, path)


def uncertainty_figure(table, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    from scipy.stats import norm
    levels = np.array([.5, .8, .9, .95, .99])
    for split in ('train_oof', 'test'):
        d = table[table.split.eq(split)]
        errors = abs(d.y_true_dbfs.to_numpy()-d.y_pred_dbfs.to_numpy())
        std = d.observation_std_db.to_numpy()
        coverages = [(errors <= norm.ppf((1+level)/2)*std).mean() for level in levels]
        axes[0].plot(levels, coverages, 'o-', label=split)
    axes[0].plot([.5, 1], [.5, 1], 'k--')
    axes[0].set(xlabel='Nominal observation interval coverage', ylabel='Empirical coverage')
    axes[0].legend()
    d = table[table.split.eq('test')]
    axes[1].hexbin(d.latent_std_db, abs(d.y_pred_dbfs-d.y_true_dbfs), gridsize=40, mincnt=1, bins='log')
    axes[1].set(xlabel='Latent function std (dB)', ylabel='Absolute test error (dB)')
    save(fig, folder / 'uncertainty_calibration.png')
