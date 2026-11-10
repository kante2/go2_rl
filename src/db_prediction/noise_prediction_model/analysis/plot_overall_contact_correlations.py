#!/usr/bin/env python3
"""Plot all audited landing events without speed bins or trial averaging."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import noise_prediction_model.analysis.landing_event_dynamics as ld

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'noise_prediction_model/results/analysis/landing_dynamics/alignment_impact_review'
STYLES = {'train': dict(color='#2674ac', marker='o', alpha=.18, s=9),
          'test': dict(color='#e47b26', marker='^', alpha=.35, s=13)}
FIELDS = {'force': 'Contact force vs noise', 'vz': 'Signed vertical foot velocity vs noise',
          'force_rate': 'Contact force rate (12ms) vs noise',
          'vz_rate': 'Vertical foot velocity rate (12ms) vs noise'}


def scatter(ax, points, xfield, yfield):
    for split, style in STYLES.items():
        g = points[points.split.eq(split)]
        good = np.isfinite(g[xfield]) & np.isfinite(g[yfield])
        ax.scatter(g.loc[good, xfield], g.loc[good, yfield], **style,
                   edgecolors='none', rasterized=True, label=f'{split}: {good.sum():,} events')
    ax.grid(alpha=.17);ax.set_axisbelow(True)


def legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False, markerscale=2)
    for handle in leg.legend_handles:handle.set_alpha(.8)


def main():
    sources = [OUT/'phase_event_features.csv',OUT/'noise_correlations_paired_by_event.csv',
               OUT/'fixed_shift_correlations_by_trial.csv']
    hashes = {str(p.relative_to(ROOT)):ld.digest(p) for p in sources}
    events = pd.read_csv(sources[0])
    pairs = pd.read_csv(sources[1])
    pairs = pairs[pairs.audio.eq('noise_5ms_dbfs')]
    if events.event_id.duplicated().any() or pairs.duplicated(['event_id','driver']).any():
        raise ValueError('Duplicate event identities')
    points = events[['event_id','split','trial','leg','contact_time_abs','measured_planar_speed_mps',
                     'synchronization_status','other_landings_in_extended_window','noise_post40_rms_dbfs',
                     'force_early_0_12_max','force_late_20_60_max',
                     'downward_minus2_mps','downward_pre20_mean_mps']].merge(
        pairs.pivot(index='event_id', columns='driver', values='rho_original').rename(
            columns={name:f'{name}_spearman_rho' for name in FIELDS}),
        left_on='event_id',right_index=True,how='left',validate='one_to_one')
    if len(points) != 8228 or points.filter(like='_spearman_rho').isna().any().any():
        raise ValueError('Unexpected event coverage or undefined coefficients')
    points.to_csv(OUT/'overall_correlation_points.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
    for ax,(driver,title) in zip(axes.flat,FIELDS.items()):
        scatter(ax,points,'measured_planar_speed_mps',f'{driver}_spearman_rho')
        ax.axhline(0,color='.35',lw=.8)
        ax.set(title=title,ylim=(-1.02,1.02),
               xlabel='Measured planar body speed (m/s)',ylabel='Within-event Spearman rho')
    legend(fig,axes[0,0])
    fig.suptitle('Overall noise correlations | all 8,228 detected landing events\n'
                 'Each dot is one event; rho uses 41 paired time samples in [-20,+60]ms\n'
                 'Original timestamps, lag = 0ms, centered 5ms audio RMS; no trial averaging or speed bins',fontsize=12)
    fig.savefig(OUT/'overall_spearman_scatter.png',dpi=180);plt.close(fig)

    settings=[('force_early_0_12_max','Early contact force: [0,+12]ms maximum','Force (sensor units)'),
              ('force_late_20_60_max','Later contact force: [+20,+60]ms maximum','Force (sensor units)'),
              ('downward_minus2_mps','Downward foot speed: last sample at -2ms','Downward speed (m/s)'),
              ('downward_pre20_mean_mps','Downward foot speed: pre-20ms signed-v_z mean, clipped','Downward speed (m/s)')]
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True,layout='constrained')
    for ax,(field,title,label) in zip(axes.flat,settings):
        scatter(ax,points,field,'noise_post40_rms_dbfs')
        ax.set(title=title,xlabel=label,ylabel='Noise: exact post-contact [0,+40)ms RMS (dBFS)')
    legend(fig,axes[0,0])
    fig.suptitle('Overall feature magnitude vs measured noise | one dot per event\n'
                 'Shared microphone; early/later force are separate features; original timestamps\n'
                 'All measured-speed conditions shown together; no fitted line or pooled correlation',fontsize=12)
    fig.savefig(OUT/'overall_feature_noise_scatter.png',dpi=180);plt.close(fig)

    curves=pd.read_csv(sources[2])
    # Equal-trial means of within-trial event-median correlations, separately
    # for train/test. This summary is explicitly distinct from the event scatter.
    summary=curves[curves.audio.eq('noise_5ms_dbfs')].groupby(
        ['split','driver','sample_shift_ms'],as_index=False).agg(
            rho=('median_rho','mean'),n_trials=('trial','nunique'))
    summary['noise_follows_driver_ms']=-summary.sample_shift_ms
    summary.to_csv(OUT/'overall_lag_sensitivity.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
    for ax,(driver,title) in zip(axes.flat,FIELDS.items()):
        for split,style in STYLES.items():
            g=summary[summary.driver.eq(driver)&summary.split.eq(split)].sort_values('noise_follows_driver_ms')
            ax.plot(g.noise_follows_driver_ms,g.rho,color=style['color'],marker=style['marker'],
                    ms=4,lw=1.5,label=split)
        ax.axhline(0,color='.6',lw=.6);ax.axvline(0,color='.35',ls='--',lw=.8)
        ax.set(title=title,xlabel='Noise time minus driver time (ms; positive = noise later)',
               ylabel='Equal-trial mean of event-median rho');ax.grid(alpha=.17)
    axes[0,0].legend()
    fig.suptitle('Overall sensitivity to signal / noise timing\n'
                 'Fixed shifts only; no offset chosen by audio correlation. 5ms RMS noise.',fontsize=12)
    fig.savefig(OUT/'overall_lag_sensitivity.png',dpi=180);plt.close(fig)
    for p,value in hashes.items():
        if ld.digest(ROOT/p)!=value:raise AssertionError('Source changed')
    metadata=dict(n_events=len(points),n_sequences=int(events.groupby(['split','trial']).ngroups),
        events_by_split=points.split.value_counts().to_dict(),
        main_scatter_x='median measured planar body speed within each event [-20,+60]ms',
        main_scatter_y='within-event same-time Spearman with centered 5ms audio RMS',
        main_scatter_aggregates_events=False,original_inputs_unchanged=True,
        source_sha256=hashes,script_sha256=ld.digest(__file__))
    (OUT/'overall_correlation_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (OUT/'overall_correlation_graphs.md').write_text(
        '# Overall correlation graphs\n\n'
        '- `overall_spearman_scatter.png`: every event as one dot; x is measured planar body speed; '
        'y is within-event force/velocity/rate vs noise Spearman over [-20,+60]ms at zero lag. '
        'The signed velocity (+up/-down) panel differs from a pre-impact downward speed magnitude. '
        'Blue circles are train, orange triangles test. Events are not averaged or binned.\n'
        '- `overall_feature_noise_scatter.png`: event feature magnitude vs exact original PCM post40ms noise. '
        'The pre20ms speed is max(-mean(v_z),0), distinct from mean(max(-v_z,0)). '
        'All speeds/feet/sequences are mixed for display; this is not controlled or causal correlation.\n'
        '- `overall_lag_sensitivity.png`: fixed-shift sensitivity, train/test separately. '
        'Positive lag means noise follows the driver. Unlike the scatter, these curves summarize '
        'within-trial event medians with equal trial weight, including low-count trials.\n\n'
        'All three use the existing force-threshold detections, not known physical touchdown times. '
        'The 3 sequences without detected events do not contribute points. '
        'Overlapping events share microphone data; points are not independent. No source timestamps were changed.\n\n'
        'Reproduce: `.venv/bin/python noise_prediction_model/analysis/plot_overall_contact_correlations.py`\n',encoding='utf-8')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
