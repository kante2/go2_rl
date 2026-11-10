#!/usr/bin/env python3
"""Phase, search-window and correlation diagnostics for the alignment audit."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import noise_prediction_model.analysis.audit_contact_alignment_impact as audit
import noise_prediction_model.analysis.landing_event_dynamics as ld
import matplotlib.pyplot as plt

OUT, ROOT, REL = audit.OUT, audit.ROOT, audit.REL


def fractional_rho(x, noise, shift):
    """Explicit retrospective linear interpolation for sub-grid diagnostics."""
    shift = np.broadcast_to(shift, len(x))
    loc = audit.CORE[None,:] + shift[:,None]/2
    low = np.floor(loc).astype(int); high = np.ceil(loc).astype(int)
    if low.min()<0 or high.max()>=len(REL):raise ValueError('Shift outside support')
    weight = loc-low
    values = ((1-weight)*np.take_along_axis(x,low,axis=1)
              +weight*np.take_along_axis(x,high,axis=1))
    return ld.row_correlation(rankdata(values,axis=1),rankdata(noise[:,audit.CORE],axis=1))


def phase_features(e,a):
    e=e.copy();rows=[]
    for stop in (40,60,80):
        time,width,edge=audit.peak_time(a['force'],0,stop)
        e[f'force_peak_0_{stop}_ms']=time
        e[f'force_peak_0_{stop}_edge']=edge
    # F should be related to momentum change; F and speed magnitudes need not
    # peak together. A foot is not a free point mass, so this is only descriptive.
    for lo,hi in ((-12,20),(-20,60),(-20,80)):
        ft,fw,fe=audit.peak_time(a['force_rate'],lo,hi)
        vt,vw,ve=audit.peak_time(a['vz_rate'],lo,hi)
        for i in range(len(e)):
            rows.append(dict(event_id=e.event_id.iloc[i],split=e.split.iloc[i],trial=e.trial.iloc[i],
                             leg=e.leg.iloc[i],search_lo_ms=lo,search_hi_ms=hi,
                             force_rate_peak_ms=ft[i],vz_rate_peak_ms=vt[i],rate_delta_ms=vt[i]-ft[i],
                             peak_at_edge=bool(fe[i]|ve[i]),tie_width_ms=max(fw[i],vw[i])))
    e['force_peak_minus_initial_vz_rate_peak_ms']=e.force_peak_0_60_ms-e.vz_rate_peak_ms
    early=(REL>=0)&(REL<=12);late=(REL>=20)&(REL<=60)
    pre=(REL>=-20)&(REL<=-10)
    e['force_early_0_12_max']=np.max(a['force'][:,early],axis=1)
    e['force_late_20_60_max']=np.max(a['force'][:,late],axis=1)
    e['force_late_to_early_ratio']=e.force_late_20_60_max/np.maximum(e.force_early_0_12_max,1.)
    e['late_peak_after20ms']=e.force_peak_0_60_ms>20
    e['late_exceeds_early_20percent']=e.force_late_to_early_ratio>1.2
    e['noise_peak_0_60_ms']=audit.peak_time(a['noise_5ms_dbfs'],0,60)[0]
    # Measured speed is sampled independently of force/audio extrema.
    e['downward_minus2_over_pre20max']=e.downward_minus2_mps/np.maximum(e.downward_pre20_max_mps,1e-9)
    return e,pd.DataFrame(rows)


def additional_correlations(e,a):
    shifts=pd.read_csv(OUT/'other_train_timing_shifts.csv').set_index(['split','trial','leg'])
    delta=np.array([shifts.loc[(r.split,r.trial,r.leg),'timing_delta_ms'] for r in e.itertuples()])
    rows=[]
    for driver in audit.DRIVERS:
        sign=1 if driver in ('vz','vz_rate') else -1
        for audio in ('noise_5ms_dbfs','noise_20ms_dbfs'):
            orig=fractional_rho(a[driver],a[audio],0)
            # Per-event variant is a sensitivity check using THIS event's two
            # waveforms. It is neither clock calibration nor a valid deployed fix.
            for name,shift in [('other_train_fractional',sign*np.clip(delta,-20,20)),
                                ('this_event_retrospective',sign*np.clip(e.rate_delta_ms,-20,20))]:
                shifted=fractional_rho(a[driver],a[audio],shift)
                for (split,trial),idx in e.groupby(['split','trial']).groups.items():
                    idx=np.asarray(idx);valid=~e.iloc[idx].rate_peak_at_edge.to_numpy()
                    idx=idx[valid]
                    rows.append(dict(split=split,trial=trial,driver=driver,audio=audio,strategy=name,
                        n_events=len(idx),rho_original=np.nanmedian(orig[idx]),rho_shifted=np.nanmedian(shifted[idx]),
                        median_paired_delta_rho=np.nanmedian(shifted[idx]-orig[idx]),
                        mean_abs_paired_delta_rho=np.nanmean(abs(shifted[idx]-orig[idx])),
                        fraction_abs_delta_gt0p05=np.mean(abs(shifted[idx]-orig[idx])>.05)))
    amp=[]
    for (split,trial,leg),g in e.groupby(['split','trial','leg']):
        for feature in ('force_early_0_12_max','force_late_20_60_max','force_peak_0_40',
                        'downward_minus2_mps','downward_pre20_max_mps','downward_pre20_mean_mps'):
            for target in ('noise_post40_rms_dbfs','noise_phase_0_12_dbfs','noise_phase_12_30_dbfs','noise_phase_30_60_dbfs'):
                good=np.isfinite(g[feature])&np.isfinite(g[target]);x,y=g.loc[good,feature],g.loc[good,target]
                rho=spearmanr(x,y).statistic if len(x)>=10 and x.nunique()>1 and y.nunique()>1 else np.nan
                amp.append(dict(split=split,trial=trial,leg=leg,feature=feature,target=target,n_events=len(x),rho=rho))
    return pd.DataFrame(rows),pd.DataFrame(amp)


def force_estimator_crosscheck(e):
    signals={k:np.full((len(e),len(REL)),np.nan) for k in ('sensor_matched20','dynamic_fz')}
    checked=[];count_pcm=0
    for (split,trial),idx in e.groupby(['split','trial']).groups.items():
        idx=np.asarray(idx);local=e.iloc[idx]
        rawpath=ROOT/f'data/1007/{split}/{trial}.npz'
        checked.append(rawpath)
        with np.load(rawpath,allow_pickle=False) as raw:
            csv=pd.read_csv(ROOT/f'preprocessed/1007/{split}/{trial}.csv',usecols=['audio_delay_s'])
            delays=csv.audio_delay_s.unique()
            if len(delays)!=1:raise ValueError('One audio delay required')
            t0=float(raw['audio_t0'])-float(delays[0]);slope=float(raw['audio_slope']);audio=raw['audio']
            for lo,hi in ((0,12),(12,30),(30,60)):
                starts=local.contact_time_abs.to_numpy()+lo/1000
                stops=local.contact_time_abs.to_numpy()+hi/1000
                values,valid=ld.pcm_rms(audio,slope,t0,starts,stops)
                e.loc[idx,f'noise_phase_{lo}_{hi}_dbfs']=values
                # Validate direct PCM windows for every trial and phase.
                for pos in np.unique(np.linspace(0,len(idx)-1,min(3,len(idx)),dtype=int)):
                    if not valid[pos]:continue
                    left=int(np.ceil((starts[pos]-t0)/slope-1));right=int(np.ceil((stops[pos]-t0)/slope-1))
                    x=audio[left:right].astype(float);x-=x.mean()
                    db=20*np.log10(max(np.sqrt(np.mean(x*x)),1e-12)/32768)
                    np.testing.assert_allclose(values[pos],db,atol=1e-7,rtol=0);count_pcm+=1
        path=ROOT/f'comparison/results/timeseries/{split}/{trial}.csv.gz'
        if not path.exists():continue
        checked.append(path)
        frame=pd.read_csv(path,float_precision='round_trip')
        t=frame.time_abs.to_numpy()
        tt=local.contact_time_abs.to_numpy()[:,None]+REL[None,:]/1000
        right=np.searchsorted(t,tt);left=right-1
        valid=(left>=0)&(right<len(t));left=np.clip(left,0,len(t)-1);right=np.clip(right,0,len(t)-1)
        valid&=(t[right]-t[left])<=.006
        for leg in ld.LEGS:
            use=local.leg.eq(leg).to_numpy()
            for name,col in [('sensor_matched20',f'{leg.lower()}_sensor_smooth20_units'),
                             ('dynamic_fz',f'{leg.lower()}_dynamic_fz_N')]:
                values=frame[col].to_numpy()
                # Linear native-to-grid mapping is diagnostic, not retiming.
                xx=np.interp(tt[use].ravel(),t,values).reshape(use.sum(),len(REL))
                xx[~valid[use]]=np.nan
                signals[name][idx[use]]=xx
        print(f'Phase/torque crosscheck {split}/{trial}',flush=True)
    for name,values in signals.items():
        use=np.isfinite(values[:,(REL>=0)&(REL<=60)]).all(axis=1)
        e[f'{name}_peak_ms']=np.nan
        e.loc[use,f'{name}_peak_ms']=audit.peak_time(values[use],0,60)[0]
    return e,signals,checked,count_pcm


def phase_table(e):
    return e.groupby(['split','trial','leg'],as_index=False).agg(n_events=('event_id','size'),
        sensor_peak_0_60_median_ms=('force_peak_0_60_ms','median'),
        sensor_peak_0_80_median_ms=('force_peak_0_80_ms','median'),
        fraction_peak_after20ms=('late_peak_after20ms','mean'),
        late_to_early_ratio_median=('force_late_to_early_ratio','median'),
        fraction_late_exceeds_early20percent=('late_exceeds_early_20percent','mean'),
        initial_deceleration_peak_median_ms=('vz_rate_peak_ms','median'),
        matched20_sensor_peak_median_ms=('sensor_matched20_peak_ms','median'),
        dynamic_force_peak_median_ms=('dynamic_fz_peak_ms','median'),
        median_downward_minus2_over_pre20max=('downward_minus2_over_pre20max','median'))


def individual_plot(e,a,est,trial='1.5_3',split='train'):
    whole=e[e.trial.eq(trial)&e.split.eq(split)]
    center=(whole.contact_time_from_move_start.min()+whole.contact_time_from_move_start.max())/2
    for leg in ld.LEGS:
        local=whole[whole.leg.eq(leg)]
        if len(local)<1:continue
        # Consecutive events around recording midpoint, chosen without audio.
        pos=int(np.argmin(abs(local.contact_time_from_move_start.to_numpy()-center)))
        ids=local.iloc[max(0,pos-1):pos+3].index.to_numpy()
        fig,axes=plt.subplots(4,len(ids),figsize=(4.5*len(ids),11),squeeze=False,sharex=True,layout='constrained')
        for col,idx in enumerate(ids):
            r=e.iloc[idx]
            axes[0,col].set_title(f'{r.contact_time_from_move_start:.3f}s | {leg}\nRate delta {r.rate_delta_ms:+.1f}ms')
            axes[0,col].plot(REL,a['noise_5ms_dbfs'][idx],c='black',label='5ms RMS')
            axes[0,col].plot(REL,a['noise_20ms_dbfs'][idx],c='.5',ls='--',label='20ms RMS')
            # Other foot onsets explain shared-microphone ambiguity.
            neighbors=whole[(whole.contact_time_abs>=r.contact_time_abs-.03)&(whole.contact_time_abs<=r.contact_time_abs+.08)&(whole.event_id!=r.event_id)]
            for rr in neighbors.itertuples():
                x=(rr.contact_time_abs-r.contact_time_abs)*1000
                axes[0,col].axvline(x,c=audit.COLORS[rr.leg],ls=':',lw=1)
                axes[0,col].text(x,.98,rr.leg,transform=axes[0,col].get_xaxis_transform(),ha='center',va='top',fontsize=8)
            ax=axes[1,col];right=ax.twinx()
            ax.plot(REL,a['force'][idx],c=audit.COLORS[leg],label='Recorded force')
            right.plot(REL,a['vz'][idx],c='crimson',label='Estimated v_z')
            ax.axvline(r.force_peak_0_60_ms,c=audit.COLORS[leg],ls=':',lw=1)
            right.axhline(0,c='.5',lw=.5);right.set_ylabel('v_z (m/s)',color='crimson')
            axes[2,col].plot(REL,est['sensor_matched20'][idx],c=audit.COLORS[leg],label='Sensor, matched 20ms')
            axr=axes[2,col].twinx();axr.plot(REL,est['dynamic_fz'][idx],c='tab:orange',label='Torque/model Fz')
            axr.set_ylabel('Estimated Fz (N)',color='tab:orange')
            for key,c in [('force_rate','tab:orange'),('vz_rate','tab:purple')]:
                values=a[key][idx];scale=np.max(abs(values[(REL>=-20)&(REL<=60)]))
                axes[3,col].plot(REL,values/max(scale,1e-12),c=c,label=key)
            for ax in axes[:,col]:
                ax.axvline(0,c='black',ls='--',lw=.8);ax.grid(alpha=.15);ax.set_xlim(-30,80)
            axes[3,col].set_xlabel('Time from force-threshold onset (ms)')
            axes[0,col].legend(fontsize=7,loc='lower right');axes[3,col].legend(fontsize=7)
        for row,label in enumerate(('Shared noise (dBFS)','Recorded force (sensor units)',
                                   'Matched sensor force (sensor units)','12ms rates / own maximum')):
            axes[row,0].set_ylabel(label)
        fig.suptitle(f'{split}/{trial} {leg}: consecutive midpoint events; no peak-based event selection\n'
                     'Black dashed: original force onset. Colored labels in audio row: OTHER feet onsets.\n'
                     'Torque/model force is an imperfect alternative, not ground truth. Separate axes have different units.',fontsize=11)
        fig.savefig(OUT/f'{trial}_{leg}_individual_events.png',dpi=150);plt.close(fig)


def plots(e,phases,search,extra,amp):
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for leg in ld.LEGS:
        g=e[e.trial.eq('1.5_3')&e.leg.eq(leg)]
        axes[0,0].scatter(g.contact_time_from_move_start,g.force_peak_0_60_ms,s=18,c=audit.COLORS[leg],label=leg)
    axes[0,0].axhline(20,c='.4',ls='--');axes[0,0].set(title='1.5_3: every force peak within [0,60]ms',
        xlabel='Time from move start (s)',ylabel='Peak time from original onset (ms)');axes[0,0].legend(ncol=4,fontsize=8)
    for leg in ld.LEGS:
        g=e[e.leg.eq(leg)]
        axes[0,1].scatter(g.measured_planar_speed_mps,g.force_peak_0_60_ms,s=8,
                          c=audit.COLORS[leg],label=leg,alpha=.12)
    axes[0,1].set(title='All events: force peak timing versus measured body speed',
                   xlabel='Measured planar body speed (m/s)',
                   ylabel='Peak time from original onset (ms)');axes[0,1].legend(fontsize=8)
    for hi,c in [(20,'tab:blue'),(60,'tab:orange'),(80,'tab:green')]:
        g=search[search.search_hi_ms.eq(hi)]
        axes[1,0].hist(g.rate_delta_ms,bins=np.arange(-102,104,2),alpha=.4,label=f'search to +{hi}ms',color=c)
    axes[1,0].set(title='Rate timing: all events, no trial averaging',
                   xlabel='dv/dt peak - dF/dt peak (ms)',ylabel='Events');axes[1,0].legend(fontsize=8)
    for feature,c in [('force_peak_0_40','tab:blue'),('force_early_0_12_max','tab:orange'),('force_late_20_60_max','tab:green')]:
        g=amp[amp.feature.eq(feature)&amp.target.eq('noise_post40_rms_dbfs')&amp.split.eq('train')]
        axes[1,1].hist(g.rho.dropna(),bins=np.linspace(-1,1,21),alpha=.4,label=feature,color=c)
    axes[1,1].set(title='Within trial/foot event amplitude association',xlabel='Spearman rho with SAME post40ms noise',ylabel='Trial-foot pairs');axes[1,1].legend(fontsize=7)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.savefig(OUT/'two_stage_force_and_noise.png',dpi=165);plt.close(fig)


def main():
    inputs=[OUT/'event_timing_and_features.csv',OUT/'audited_waveforms.npz',OUT/'other_train_timing_shifts.csv']
    input_hashes={str(p.relative_to(ROOT)):ld.digest(p) for p in inputs}
    e=pd.read_csv(inputs[0]);a=dict(np.load(inputs[1],allow_pickle=False))
    np.testing.assert_array_equal(e.event_id.to_numpy(str),a['event_id'])
    e,search=phase_features(e,a)
    e,est,checked,npcm=force_estimator_crosscheck(e)
    extra,amp=additional_correlations(e,a)
    phases=phase_table(e)
    e.to_csv(OUT/'phase_event_features.csv',index=False)
    phases.to_csv(OUT/'phase_timing_by_trial_leg.csv',index=False)
    search.to_csv(OUT/'rate_search_window_sensitivity.csv',index=False)
    extra.to_csv(OUT/'fractional_and_event_shift_correlations.csv',index=False)
    amp.to_csv(OUT/'phase_amplitude_noise_correlations.csv',index=False)
    np.savez_compressed(OUT/'force_crosscheck_waveforms.npz',event_id=e.event_id.to_numpy(str),**est)
    individual_plot(e,a,est)
    plots(e,phases,search,extra,amp)
    for path,value in input_hashes.items():
        if ld.digest(ROOT/path)!=value:raise AssertionError('Analysis input changed')
    verification=dict(passed=True,n_events=len(e),direct_pcm_windows_verified=npcm,
        source_comparison_files=int(sum(p.suffix=='.gz' for p in checked)),
        waveform_event_order_verified=True,phase_windows_use_original_audio_clock=True,
        no_original_signals_shifted=True,script_sha256=ld.digest(__file__),
        input_sha256=input_hashes,additional_source_sha256={str(p.relative_to(ROOT)):ld.digest(p) for p in checked})
    (OUT/'phase_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if 'sha256' not in k},indent=2),flush=True)


if __name__=='__main__':main()
