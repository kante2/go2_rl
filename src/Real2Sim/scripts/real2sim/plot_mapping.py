#!/usr/bin/env python3
"""Plot already exported log mapping results. No simulation or parameter fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(directory):
    figure = directory / 'figures/mapping'
    figure.mkdir(parents=True, exist_ok=True)
    results = json.loads((directory / 'metrics/mapping_summary.json').read_text())
    feet = ('FR','FL','RR','RL')
    for result in results:
        name = result['trial']
        with np.load(directory / 'processed' / f'{name}_mapping.npz') as z:
            t = z['t']; mask = z['steady']
            # Zoom illustrates the same original signal with only its time shifted.
            begin = result['synchronization']['calibration']['window_s'][0] + 1.
            selected = (t >= begin) & (t <= begin + 2.4)
            metadata = json.loads((directory / 'metrics' / f'{name}.json').read_text())
            reference = metadata['before']['force']
            sync = result['synchronization']
            fig, axes = plt.subplots(4,2,figsize=(14,10),sharex=True)
            for foot, leg in enumerate(feet):
                real = z['real_foot_sensor_raw'][:,foot]
                sims = [z['sim_foot_fz_N_clock_only'][:,foot], z['sim_foot_fz_N_before'][:,foot]]
                lim_r = (float(real.min())-3, float(real.max())+3)
                lim_s = (-5, float(max(x.max() for x in sims))+5)
                for col, sim in enumerate(sims):
                    ax = axes[foot,col]; right=ax.twinx()
                    ax.plot(t[selected],real[selected],color='C0',lw=1,label='Real sensor')
                    right.plot(t[selected],sim[selected],color='C1',lw=1,label='Existing sim Fz')
                    ax.set_ylim(lim_r); right.set_ylim(lim_s)
                    ax.set_ylabel(f'{leg}: counts',color='C0'); right.set_ylabel('N',color='C1')
                    ax.grid(alpha=.2)
                    real_std, sim_std = reference['real_std'][foot], reference['sim_std'][foot]
                    rmse = np.nan
                    if min(real_std, sim_std) > 1e-10:
                        a = (real[mask] - reference['real_mean'][foot]) / real_std
                        b = (sim[mask] - reference['sim_mean'][foot]) / sim_std
                        rmse = np.sqrt(np.mean((a - b) ** 2))
                    label = 'Clock baseline: no time shift' if col == 0 else (
                        'After validated common time shift' if sync['passed'] else 'REVIEW: time shift not applied')
                    ax.set_title(label + f' | z-RMSE {rmse:.3f}', fontsize=9)
                    if foot==3: ax.set_xlabel('Real record time [s]')
            detail = (' | +/-500ms search; cycle selected early, checked late; phase shift, not hardware delay'
                      if sync.get('search_expansion') else ' | same samples, independent counts/N axes, shared standardization')
            fig.suptitle(f'{name}: foot load before/after time correction | applied shift {sync["applied_lag_s"]*1000:+.1f} ms\n{sync["status"]}{detail}', fontsize=12)
            fig.tight_layout();fig.savefig(figure/f'{name}_foot_force_before_after.png',dpi=150);plt.close(fig)
    # Units change under actor normalization: use separate axes, never overlay units.
    fig,axes=plt.subplots(2,3,figsize=(24,12))
    x=np.arange(len(results))
    panels=[(axes[0,0],'q','before','rad'),(axes[0,1],'q','after','dimensionless'),
        (axes[1,0],'dq','before','rad/s'),(axes[1,1],'dq','after','dimensionless'),
        (axes[0,2],'ddq','after','rad/s²'),(axes[1,2],'force','after','dimensionless')]
    for ax,q,mode,unit in panels:
        values=[r['metrics'][mode][q]['rmse'] if q!='force' else np.sqrt(np.mean(np.square(r['metrics'][mode]['force']))) for r in results]
        ax.bar(x,values,color='C0' if mode=='before' else 'C1')
        ax.set_xticks(x,[r['trial'] for r in results]);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2)
        plt.setp(ax.get_xticklabels(), rotation=65, ha='right', fontsize=7)
        ax.set_title(f"{q}: {'01 physical' if mode=='before' else '02 actor input'}" if q in ('q','dq') else f'{q}: unchanged (no deployment rule)')
    fig.suptitle('Same time-aligned logs: 01 physical / 02 deployed actor input\nCoordinate mapping changes units; this is not physical error reduction')
    fig.tight_layout();fig.savefig(figure/'mapping_rmse_summary.png',dpi=150);plt.close(fig)
    for r in results:
        name=r['trial']
        with np.load(directory/'processed'/f'{name}_mapping.npz') as z:
            t=z['t'];start=r['synchronization']['calibration']['window_s'][0]+1
            mask=(t>=start)&(t<=start+2.4)
            fig,axes=plt.subplots(3,2,figsize=(14,9),sharex=True)
            for row,q in enumerate(('q','dq','ddq')):
                for col,mode in enumerate(('before','after')):
                    ax=axes[row,col]
                    ax.plot(t[mask],z[f'real_{q}_{mode}'][mask,5],label='Real',lw=1)
                    ax.plot(t[mask],z[f'sim_{q}_{mode}'][mask,5],label='Existing sim',lw=1)
                    unit={'q':'rad','dq':'rad/s','ddq':'rad/s²'}[q] if mode=='before' or q=='ddq' else 'dimensionless'
                    ax.set_ylabel(f'{q} [{unit}]');ax.grid(alpha=.2);ax.legend()
                    ax.set_title('01: time-aligned physical logs' if mode=='before' else ('02: normalized deployed actor input' if q!='ddq' else '02: ddq unchanged; no deployment rule'))
            for ax in axes[-1]:ax.set_xlabel('Real record time [s]')
            fig.suptitle(f'{name} / FL calf: real and existing sim logs\nNo inverse normalization in 02; same joint names, time axis and samples')
            fig.tight_layout();fig.savefig(figure/f'{name}_joint_mapping.png',dpi=150);plt.close(fig)
    print(f'Saved {2*len(results)+1} figures to {figure}')



if __name__ == '__main__':
    import sys
    if __package__ in (None, ''):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        __package__ = 'real2sim'
    from .config import DEFAULT_DATASET, result_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    args = parser.parse_args()
    plot(args.result_dir or result_path(args.dataset))
