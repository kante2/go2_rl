#!/usr/bin/env python3
"""Export per-foot contact-event features and the exact shaded sample windows.

Foot velocity: R_WB * (v_body_B + omega_B x p_foot_B + J(q)*dq).
Default windows are [-20,+40]ms; pre-velocity is the last row strictly BEFORE
the force-threshold onset. Test records always remain in the test split.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/low_noise_matplotlib')
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from common_time import TrialReader
from go2_kinematics import Go2Kinematics,add_foot_kinematics,MODEL_PATH,MODEL_SOURCE
from contact_detection import detect_contact
from contact_noise_visualization import intervals,LEGS

ROOT=Path(__file__).resolve().parents[1]
BEFORE_MS=20.
AFTER_MS=40.
GENERATED_CSVS={'all_trials.csv','quality.csv','contact_event_features.csv','contact_event_samples.csv'}


def continuous_segments(frame):
    t=frame.time_abs.to_numpy(float)
    if len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('at least two finite increasing times are required')
    dt=float(np.median(np.diff(t)))
    breaks=np.r_[False,np.diff(t)>1.5*dt+1e-9]
    if 'segment' in frame:breaks|=np.r_[False,np.diff(frame.segment.to_numpy())!=0]
    return np.cumsum(breaks),dt


def raw_context(path,frame,model):
    """Timestamp/geometry audits; no fitting to audio, force or model outcomes."""
    reader=TrialReader(path)
    ls=reader.load('ls_t');sp=reader.load('sp_t');time=frame.time_abs.to_numpy(float)
    context=dict(reader=reader,clock={},frame_audit={},geometry={})
    for prefix,source in [('ls',ls),('sp',sp)]:
        idx=np.searchsorted(source,time,side='right')-1
        good=(idx>=0)&(time<=source[-1]);safe=np.clip(idx,0,len(source)-1)
        context[f'{prefix}_source_time']=np.where(good,source[safe],np.nan)
        context[f'{prefix}_age_ms']=np.where(good,(time-source[safe])*1000,np.nan)
    # Validate body-vs-odom convention against independent position differences.
    st=reader.load('sp_t');sv=reader.load('sp_vel');position=reader.load('sp_pos')
    li=np.clip(np.searchsorted(ls,st,side='right')-1,0,len(ls)-1)
    quat=reader.load('ls_quaternion')[li]
    rotation=Rotation.from_quat(quat[:,[1,2,3,0]])
    rotated=rotation.apply(sv)
    right=np.searchsorted(st,st+.2,side='left')
    left=np.flatnonzero(right<len(st));right=right[left]
    elapsed=st[right]-st[left]
    ev=dict(zip(reader.load('ev_label').astype(str),reader.load('ev_t')))
    ok=(elapsed>=.18)&(elapsed<=.25)&(st[left]>=ev['move_start']+.5)&(st[right]<=ev['move_stop']-.5)
    if np.any(ok):
        a,b=left[ok],right[ok]
        observed=(position[b]-position[a])/(st[b]-st[a])[:,None]
        body=(rotated[a]+rotated[b])/2
        odom=(sv[a]+sv[b])/2
        context['frame_audit']=dict(body_frame_position_velocity_mae_xy_mps=float(np.mean(abs(observed[:,:2]-body[:,:2]))),
            odom_frame_position_velocity_mae_xy_mps=float(np.mean(abs(observed[:,:2]-odom[:,:2]))))
    idx=np.clip(np.searchsorted(sp,time,side='right')-1,0,len(sp)-1)
    sport_positions=reader.load('sp_foot_pos')[idx]
    for leg in ('fr','fl','rr','rl'):
        offset=('fr','fl','rr','rl').index(leg)*3
        predicted=frame[[f'foot_position_body_{leg}_{a}_m' for a in 'xyz']].to_numpy(float)
        context['geometry'][leg]=float(np.mean(np.linalg.norm(predicted-sport_positions[:,offset:offset+3],axis=1)))
    context['audio']=reader.load('audio')
    context['audio_slope']=float(reader.load('audio_slope'))
    context['audio_t0']=float(reader.load('audio_t0'))
    return context


def pcm_window_statistics(context,start,stop,delay):
    a=context['audio_slope'];b=context['audio_t0']-delay;audio=context['audio']
    left=int(np.ceil((start-b)/a-1));right=int(np.ceil((stop-b)/a-1))
    if left<0 or right>len(audio) or right<=left:return np.nan,np.nan,False
    x=audio[left:right].astype(float);x-=x.mean()
    rms=float(np.sqrt(np.mean(x*x)));peak=float(np.max(abs(x)))
    return 20*np.log10(max(rms,1e-12)/32768.),20*np.log10(max(peak,1e-12)/32768.),True


def analyze_frame(frame,trial_name,on=10.,off=5.,confirm_ms=6.,before_ms=BEFORE_MS,
                  after_ms=AFTER_MS,raw_path=None,split='train',sync_summary=None,model=None,velocity_frame='body'):
    if min(before_ms,after_ms)<=0 or not np.isfinite([before_ms,after_ms]).all():
        raise ValueError('positive finite pre/post windows required')
    frame=frame.sort_values('time_abs').reset_index(drop=True)
    frame=add_foot_kinematics(frame,model,velocity_frame)
    segments,dt=continuous_segments(frame);t=frame.time_abs.to_numpy(float)
    states={};causal={};event_rows=[];summary=[]
    thresholds=(sync_summary or {}).get('contact_thresholds',{})
    for leg in LEGS:
        leg_on=thresholds.get(leg,{}).get('on',on)
        leg_off=thresholds.get(leg,{}).get('off',off)
        states[leg],causal[leg],items=detect_contact(t,frame[f'ls_ff_{leg}'],segments,leg_on,leg_off,confirm_ms,max_gap_s=1.5*dt)
        for item in items:
            i=int(np.searchsorted(t,item['onset_time_abs']));j=int(np.searchsorted(t,item['confirmed_time_abs']))
            event_rows.append(dict(trial=trial_name,leg=leg.upper(),**item,
                onset_time_from_move_start=float(frame.time_from_move_start.iloc[i]),
                confirmed_time_from_move_start=float(frame.time_from_move_start.iloc[j])))
        summary.append(dict(trial=trial_name,leg=leg.upper(),contact_on_count=sum(x['event']=='contact_on' for x in items),
            contact_off_count=sum(x['event']=='contact_off' for x in items),estimated_contact_fraction=float(np.mean(states[leg]==1)),
            unknown_fraction=float(np.mean(states[leg]==-1)),on_threshold=leg_on,off_threshold=leg_off,confirmation_ms=confirm_ms))
    event_columns=['trial','leg','event','segment','onset_time_abs','confirmed_time_abs','confirmation_delay_ms',
        'onset_time_from_move_start','confirmed_time_from_move_start']
    events=pd.DataFrame(event_rows,columns=event_columns)
    starts=events[events.event=='contact_on'].sort_values(['onset_time_abs','leg']).reset_index(drop=True)
    context=raw_context(raw_path,frame,model) if raw_path is not None else None
    sync=sync_summary or {};sync_status=sync.get('status','NOT_AUDITED')
    delay=float(frame.audio_delay_s.iloc[0]) if 'audio_delay_s' in frame else 0.
    if 'audio_delay_s' in frame and not np.allclose(frame.audio_delay_s,delay):
        raise ValueError('one recorded audio delay per trial required')
    features=[];samples=[]
    for ordinal,event in enumerate(starts.to_dict('records')):
        leg=event['leg'].lower();onset=event['onset_time_abs'];index=int(np.searchsorted(t,onset))
        leg_on=thresholds.get(leg,{}).get('on',on);leg_off=thresholds.get(leg,{}).get('off',off)
        segment=segments[index];start=onset-before_ms/1000;stop=onset+after_ms/1000
        window=(segments==segment)&(t>=start-1e-9)&(t<=stop+1e-9)
        indexes=np.flatnonzero(window);post=indexes[t[indexes]>=onset-1e-9];pre=indexes[t[indexes]<onset-1e-9]
        before=int(pre[-1]) if len(pre) else None
        complete=(len(indexes)>0 and t[indexes[0]]<=start+dt*.51 and t[indexes[-1]]>=stop-dt*.51)
        same_pre=before is not None and onset-t[before]<=1.5*dt+1e-9
        force=frame[f'ls_ff_{leg}'].to_numpy(float)
        world=frame[[f'foot_velocity_world_{leg}_{a}_mps' for a in 'xyz']].to_numpy(float)
        neighbor=starts[(starts.onset_time_abs>=start-1e-9)&(starts.onset_time_abs<=stop+1e-9)]
        other=neighbor[neighbor.leg!=leg.upper()]
        pre_valid=same_pre and np.isfinite(world[before]).all()
        source_age_ok=True
        if context is not None and before is not None:
            ages=[context[f'{p}_age_ms'][before] for p in ('ls','sp')]
            source_age_ok=bool(np.isfinite(ages).all() and min(ages)>=-1e-6 and max(ages)<=20+1e-6)
        pre_valid=bool(pre_valid and source_age_ok)
        event_id=f'{trial_name}:{leg.upper()}:{ordinal:05d}'
        row=dict(event_id=event_id,trial=trial_name,split=split,leg=leg.upper(),segment=int(segment),
            contact_on_threshold=leg_on,contact_off_threshold=leg_off,
            command_vx=float(frame.command_vx.iloc[index]) if 'command_vx' in frame else np.nan,
            command_vy=float(frame.command_vy.iloc[index]) if 'command_vy' in frame else np.nan,
            command_vyaw=float(frame.command_vyaw.iloc[index]) if 'command_vyaw' in frame else np.nan,
            contact_time_abs=onset,contact_time_from_move_start=event['onset_time_from_move_start'],
            confirmed_time_abs=event['confirmed_time_abs'],confirmation_delay_ms=event['confirmation_delay_ms'],
            window_start_abs=start,window_stop_abs=stop,window_before_ms=before_ms,window_after_ms=after_ms,
            window_complete=bool(complete),window_sample_count=len(indexes),pre_sample_count=len(pre),post_sample_count=len(post),
            pre_velocity_time_abs=t[before] if before is not None else np.nan,
            pre_velocity_offset_ms=(t[before]-onset)*1000 if before is not None else np.nan,
            force_at_contact_sensor_units=float(force[index]),
            force_post_peak_sensor_units=float(np.max(force[post])) if complete else np.nan,
            force_post_mean_sensor_units=float(np.mean(force[post])) if complete else np.nan,
            force_post_integral_sensor_unit_s=float(np.trapz(force[post],t[post])) if complete else np.nan,
            force_post_max_rise_sensor_units_per_s=float(np.max(np.diff(force[post])/np.diff(t[post]))) if complete and len(post)>1 else np.nan,
            pre_speed_world_mps=float(np.linalg.norm(world[before])) if pre_valid else np.nan,
            pre_downward_speed_mps=float(max(-world[before,2],0)) if pre_valid else np.nan,
            pre_speed_world_mean_mps=float(frame[f'foot_speed_world_{leg}_mps'].iloc[pre].mean()) if complete else np.nan,
            noise_at_contact_rms_20ms_dbfs=float(frame.audio_dbfs_20ms.iloc[index]),
            noise_post_max_rms_20ms_dbfs=float(frame.audio_dbfs_20ms.iloc[post].max()) if complete else np.nan,
            audio_delay_s=delay,synchronization_status=sync_status,
            synchronization_review_required=sync_status!='CONSISTENT_CANDIDATE',
            other_leg_contact_count_in_window=len(other),other_legs_in_window=','.join(sorted(set(other.leg))),
            isolated_foot_event=not len(other),body_velocity_frame=velocity_frame,
            pre_velocity_valid=pre_valid)
        for axis,j in zip('xyz',range(3)):
            row[f'pre_velocity_world_{axis}_mps']=float(world[before,j]) if pre_valid else np.nan
        for prefix in ('ls','sp'):
            row[f'pre_{prefix}_source_time_abs']=context[f'{prefix}_source_time'][before] if context is not None and before is not None else np.nan
            row[f'pre_{prefix}_source_age_ms']=context[f'{prefix}_age_ms'][before] if context is not None and before is not None else np.nan
        noise_valid=False
        row['noise_window_rms_dbfs']=row['noise_window_peak_dbfs']=row['noise_post_rms_dbfs']=row['noise_post_peak_dbfs']=np.nan
        if context is not None:
            rms,peak,aok=pcm_window_statistics(context,start,stop,delay)
            prms,ppeak,bok=pcm_window_statistics(context,onset,stop,delay)
            row.update(noise_window_rms_dbfs=rms,noise_window_peak_dbfs=peak,noise_post_rms_dbfs=prms,noise_post_peak_dbfs=ppeak)
            noise_valid=aok and bok
        numeric_ok=np.isfinite(force[indexes]).all() and np.isfinite(world[indexes]).all()
        row['feature_valid']=bool(complete and pre_valid and numeric_ok and noise_valid)
        row['quality_ok']=bool(row['feature_valid'] and not row['synchronization_review_required'])
        features.append(row)
        sample=pd.DataFrame(dict(event_id=event_id,trial=trial_name,split=split,leg=leg.upper(),segment=int(segment),
            time_abs=t[indexes],time_from_move_start=frame.time_from_move_start.iloc[indexes].to_numpy(),
            time_from_contact_ms=(t[indexes]-onset)*1000,force_sensor_units=force[indexes],
            foot_speed_world_mps=frame[f'foot_speed_world_{leg}_mps'].iloc[indexes].to_numpy(),
            foot_downward_speed_mps=frame[f'foot_downward_speed_{leg}_mps'].iloc[indexes].to_numpy(),
            audio_rms_20ms=frame.audio_rms_20ms.iloc[indexes].to_numpy() if 'audio_rms_20ms' in frame else np.nan,
            audio_dbfs_20ms=frame.audio_dbfs_20ms.iloc[indexes].to_numpy(),
            audio_peak_2ms=frame.audio_peak_2ms.iloc[indexes].to_numpy() if 'audio_peak_2ms' in frame else np.nan,
            feature_valid=row['feature_valid'],synchronization_status=sync_status))
        for axis,j in zip('xyz',range(3)):sample[f'foot_velocity_world_{axis}_mps']=world[indexes,j]
        samples.append(sample)
    feature_table=pd.DataFrame(features)
    if feature_table.empty:feature_table=pd.DataFrame(columns=['event_id','trial','split','leg','feature_valid','quality_ok'])
    sample_table=pd.concat(samples,ignore_index=True) if samples else pd.DataFrame(columns=['event_id','trial','split','leg','time_abs'])
    settings=dict(window_before_ms=before_ms,window_after_ms=after_ms,body_velocity_frame=velocity_frame,
        on_threshold_sensor_units=on,off_threshold_sensor_units=off,confirmation_ms=confirm_ms)
    return dict(frame=frame,segments=segments,dt=dt,states=states,causal=causal,events=events,
        features=feature_table,samples=sample_table,summary=summary,context=context,settings=settings)


def central_limits(frame,segments,zoom_seconds):
    times=frame.time_from_move_start.to_numpy(float);middle=float((times[0]+times[-1])/2)
    candidates=[]
    for segment in np.unique(segments):
        local=times[segments==segment]
        if local[-1]-local[0]>=zoom_seconds:
            candidates.append((abs((local[0]+local[-1])/2-middle),local[0],local[-1]))
    if candidates:
        _,left,right=min(candidates);middle=float(np.clip(middle,left+zoom_seconds/2,right-zoom_seconds/2))
    return middle-zoom_seconds/2,middle+zoom_seconds/2


def draw_contact_noise_figure(result,trial_name,output,on=10.,off=5.,confirm_ms=6.,zoom_seconds=1.5):
    """Original view: noise and contact-state bands, full record and detail."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    f=result['frame'];segments=result['segments'];events=result['events']
    times=f.time_from_move_start.to_numpy(float);dt=result['dt']
    lo,hi=result.get('display_zoom_limits_s',central_limits(f,segments,zoom_seconds))
    colors={'fl':'tab:blue','fr':'tab:orange','rl':'tab:green','rr':'tab:purple'}
    fig,axes=plt.subplots(4,1,figsize=(15,11),gridspec_kw={'height_ratios':[2,1.25,2,1.25]})
    limits=[(times[0],times[-1]+dt),(lo,hi)]
    for level,(noise_ax,gait_ax) in enumerate(zip(axes[::2],axes[1::2])):
        left,right=limits[level]
        for segment in np.unique(segments):
            m=(segments==segment)&(times>=left)&(times<=right)
            if np.any(m):noise_ax.plot(times[m],f.audio_dbfs_20ms.to_numpy()[m],color='black',lw=.7)
        for row,leg in enumerate(LEGS):
            spans=intervals(times,result['states'][leg],segments)
            for state,color in [(1,'#0075ad'),(0,'white'),(-1,'#c8c8c8')]:
                selected=[(a,b-a) for a,b,s in spans if s==state and b>=left and a<=right]
                if selected:gait_ax.broken_barh(selected,(row-.33,.66),facecolors=color,edgecolors='none')
            starts=events[(events.leg==leg.upper())&(events.event=='contact_on')]
            x=starts.onset_time_from_move_start.to_numpy();x=x[(x>=left)&(x<=right)]
            gait_ax.scatter(x,np.full(len(x),row-.42),marker='v',s=20,c='black',zorder=4)
            ymin,ymax=f.audio_dbfs_20ms.min(),f.audio_dbfs_20ms.max();height=max(float(ymax-ymin),1.)
            noise_ax.scatter(x,np.full(len(x),ymax+height*(.04+.045*row)),marker='v',s=13,c=colors[leg],label=leg.upper())
        for i in np.flatnonzero(np.r_[False,np.diff(segments)!=0]):
            for ax in (noise_ax,gait_ax):ax.axvspan(times[i-1]+dt,times[i],color='#c8c8c8',alpha=.65)
        noise_ax.set(xlim=(left,right),ylabel='Measured 20ms RMS (dBFS)',
            title='Full walking interval' if level==0 else result.get('display_zoom_title',f'{zoom_seconds:g}s central detail (chosen without audio)'))
        if level == 0 and 'display_zoom_limits_s' in result:
            for ax in (noise_ax,gait_ax):ax.axvspan(lo,hi,color='gold',alpha=.25,zorder=3)
        noise_ax.legend(title='Estimated contact onset',ncol=4,fontsize=8,loc='upper right');noise_ax.grid(alpha=.2)
        gait_ax.set(xlim=(left,right),ylim=(3.65,-.65),yticks=range(4),yticklabels=[leg.upper() for leg in LEGS],
            xlabel='Time from move_start (s)');gait_ax.grid(axis='x',alpha=.2)
    fig.legend(handles=[Patch(facecolor='#0075ad',label='Estimated contact'),Patch(facecolor='white',edgecolor='gray',label='Below threshold'),
        Patch(facecolor='#c8c8c8',label='Unknown / gap'),Line2D([],[],color='black',marker='v',linestyle='none',label='Threshold onset, retrospectively confirmed')],
        loc='upper center',ncol=4,bbox_to_anchor=(.5,.955),fontsize=9)
    fig.suptitle(f'{trial_name}: force-only contact estimates and measured noise\nPer-foot thresholds in contact_detection.json (sensor units); confirmation {confirm_ms:g}ms; timing is not hardware-validated',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(Path(output)/'contact_noise.png',dpi=150);plt.close(fig)
    return [float(lo),float(hi)]


def draw_contact_force_velocity_figure(result,trial_name,output,on=10.,off=5.,confirm_ms=6.,zoom_seconds=1.5,with_noise=False):
    """Continuous signed vertical velocity; shading is the exported event window."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.collections import PolyCollection
    f=result['frame'];segments=result['segments'];features=result['features']
    times=f.time_from_move_start.to_numpy(float);lo,hi=result.get('display_zoom_limits_s',central_limits(f,segments,zoom_seconds))
    colors={'fl':'tab:blue','fr':'tab:orange','rl':'tab:green','rr':'tab:purple'}
    offset=int(with_noise)
    fig,axes=plt.subplots(4+offset,2,figsize=(20,14 if with_noise else 11),sharex='col' if with_noise else False)
    views=[(times[0],times[-1]+result['dt']),(lo,hi)]
    for col,(left,right) in enumerate(views):
        axes[0,col].set_title('Full walking interval' if col==0 else result.get('display_zoom_title',f'{zoom_seconds:g}s central detail (chosen without audio)'))
        if col == 0 and 'display_zoom_limits_s' in result:
            for ax in axes[:,col]:ax.axvspan(lo,hi,color='gold',alpha=.25)
        if with_noise:
            noise_ax=axes[0,col]
            for segment in np.unique(segments):
                m=(segments==segment)&(times>=left)&(times<=right)
                if np.any(m):noise_ax.plot(times[m],f.audio_dbfs_20ms.to_numpy()[m],color='black',lw=.8)
            noise_ax.set_ylabel('Measured noise\n20ms RMS (dBFS)');noise_ax.grid(alpha=.2)
        for row,leg in enumerate(LEGS):
            ax=axes[row+offset,col];twin=ax.twinx()
            for segment in np.unique(segments):
                m=(segments==segment)&(times>=left)&(times<=right)
                if not np.any(m):continue
                ax.plot(times[m],f[f'ls_ff_{leg}'].to_numpy()[m],color=colors[leg],lw=.8)
                twin.plot(times[m],f[f'foot_velocity_world_{leg}_z_mps'].to_numpy()[m],color='crimson',alpha=.8,lw=.8)
            ax.set_ylabel(f'{leg.upper()} force\n(sensor units)',color=colors[leg])
            twin.set_ylabel('Vertical velocity (m/s)\n+ up / - down',color='crimson')
            limit=max(float(np.nanmax(np.abs(f[f'foot_velocity_world_{leg}_z_mps']))),.05)*1.05
            twin.set_ylim(-limit,limit)
            twin.axhline(0,color='crimson',ls=':',alpha=.35,lw=.7)
            ax.axhline(next(x['on_threshold'] for x in result['summary'] if x['leg']==leg.upper()),color=colors[leg],ls=':',alpha=.5,lw=.7)
            local=features[features.leg==leg.upper()]
            if len(local):
                onset=local.contact_time_from_move_start
                start=onset-local.window_before_ms/1000;stop=onset+local.window_after_ms/1000
                local=local[(stop>=left)&(start<=right)]
            if len(local):
                onset=local.contact_time_from_move_start.to_numpy()
                start=onset-local.window_before_ms.to_numpy()/1000;stop=onset+local.window_after_ms.to_numpy()/1000
                spans=[[(a,0),(a,1),(b,1),(b,0)] for a,b in zip(start,stop)]
                ax.add_collection(PolyCollection(spans,facecolors=colors[leg],edgecolors='none',alpha=.15,
                    transform=ax.get_xaxis_transform()),autolim=False)
                if with_noise:
                    noise_ax.add_collection(PolyCollection(spans,facecolors=colors[leg],edgecolors='none',alpha=.08,
                        transform=noise_ax.get_xaxis_transform()),autolim=False)
                ax.vlines(onset,0,1,transform=ax.get_xaxis_transform(),colors=colors[leg],linestyles='--',alpha=.45,lw=.6)
                ax.scatter(onset,local.force_at_contact_sensor_units,c=colors[leg],s=10,zorder=5)
                valid=local[local.pre_velocity_valid]
                pre=valid.contact_time_from_move_start+valid.pre_velocity_offset_ms/1000
                twin.scatter(pre,valid.pre_velocity_world_z_mps,c='crimson',marker='v',s=18,zorder=6)
            ax.set_xlim(left,right);ax.grid(alpha=.2)
        axes[-1,col].set_xlabel('Time from move_start (s)')
    before=result['settings']['window_before_ms'];after=result['settings']['window_after_ms']
    subject='measured noise, per-foot contact force and continuous vertical velocity' if with_noise else 'per-foot contact force and continuous vertical velocity'
    fig.suptitle(f'{trial_name}: {subject}\nPer-foot thresholds in contact_detection.json; confirm {confirm_ms:g}ms; shaded [-{before:g},+{after:g}]ms; positive = up, negative = down',fontsize=13)
    handles=[Patch(color='gray',alpha=.2,label='Exported event window'),
        Line2D([],[],color='crimson',label='Estimated vertical velocity (+ up / - down)'),
        Line2D([],[],color='crimson',marker='v',linestyle='none',label='Last pre-contact velocity')]
    if with_noise:handles.insert(0,Line2D([],[],color='black',label='Shared microphone noise (20ms RMS)'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=len(handles),fontsize=9)
    filename='contact_force_velocity_with_noise.png' if with_noise else 'contact_force_velocity.png'
    fig.tight_layout(rect=(0,0,1,.915));fig.savefig(Path(output)/filename,dpi=140);plt.close(fig)


def draw_contact_force_velocity_with_noise_figure(result,trial_name,output,on=10.,off=5.,confirm_ms=6.,zoom_seconds=1.5):
    draw_contact_force_velocity_figure(result,trial_name,output,on,off,confirm_ms,zoom_seconds,with_noise=True)


def draw_contact_figure(result,trial_name,output,on=10.,off=5.,confirm_ms=6.,zoom_seconds=1.5):
    zoom=draw_contact_noise_figure(result,trial_name,output,on,off,confirm_ms,zoom_seconds)
    draw_contact_force_velocity_with_noise_figure(result,trial_name,output,on,off,confirm_ms,zoom_seconds)
    return zoom


def atomic_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.csv.tmp');frame.to_csv(temporary,index=False,float_format='%.17g');temporary.replace(path)


def save_trial_result(result,trial_name,output,on,off,confirm_ms,make_plot=True,zoom_seconds=1.5):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    f=result['frame'];state_table=f[['time_abs','time_from_move_start']].copy();state_table['segment']=result['segments']
    for leg in LEGS:
        state_table[f'{leg}_estimated_contact']=result['states'][leg];state_table[f'{leg}_causal_contact']=result['causal'][leg]
    atomic_csv(state_table,output/'contact_states.csv');atomic_csv(result['events'],output/'contact_events.csv')
    atomic_csv(result['features'],output/'contact_event_features.csv');atomic_csv(result['samples'],output/'contact_event_samples.csv')
    zoom=draw_contact_figure(result,trial_name,output,on,off,confirm_ms,zoom_seconds) if make_plot else None
    context=result['context'];config=dict(trial=trial_name,on_threshold_sensor_units=on,off_threshold_sensor_units=off,
        confirmation_ms=confirm_ms,timestamp_convention='first sustained force threshold crossing; confirmed later',
        estimated_state_is_retrospective=True,causal_state_column_suffix='_causal_contact',no_audio_used_for_detection=True,
        no_additional_time_shift=True,zoom_limits_s=zoom,legs=result['summary'],
        velocity_formula='R_WB * (v_body_B + omega_B cross p_foot_B + J(q)*dq)',
        velocity_model_source=MODEL_SOURCE,urdf_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        velocity_point='URDF foot frame origin; rigid point proxy, not deformable sole contact point',
        force_units='uncalibrated recorded units',noise='shared microphone, not isolated per-foot sound',
        body_velocity_frame_audit=context['frame_audit'] if context else {},
        foot_position_mae_m=context['geometry'] if context else {})
    config.update(result['settings'])
    (output/'contact_detection.json').write_text(json.dumps(config,indent=2))


def collect_event_tables(input_dir,figure_dir):
    input_dir=Path(input_dir);figure_dir=Path(figure_dir)
    splits=[p for p in (input_dir/'train',input_dir/'test') if p.is_dir()]
    if not splits:splits=[input_dir]
    result={}
    for split in splits:
        for filename in ('contact_event_features.csv','contact_event_samples.csv'):
            tables=[]
            for path in sorted(split.glob('*.csv')):
                if path.name in GENERATED_CSVS:continue
                artifact=figure_dir/path.stem/filename
                if artifact.exists():tables.append(pd.read_csv(artifact,float_precision='round_trip'))
            if tables:
                table=pd.concat(tables,ignore_index=True)
                if filename=='contact_event_features.csv' and table.event_id.duplicated().any():raise ValueError('duplicate event IDs')
                if filename=='contact_event_features.csv':
                    keys=['window_before_ms','window_after_ms','body_velocity_frame']
                    settings={key:sorted(table[key].dropna().unique().tolist()) for key in keys if key in table}
                    if any(len(value)>1 for value in settings.values()):raise ValueError('mixed extraction settings; rerun all trials with the same options')
                    source_paths=[p for p in sorted(split.glob('*.csv')) if p.name not in GENERATED_CSVS and p.stem in set(table.trial)]
                    metadata=dict(schema_version=1,split=split.name,settings=settings,events=len(table),
                        trials=sorted(set(table.trial)),feature_valid_count=int(table.feature_valid.sum()) if 'feature_valid' in table else None,
                        quality_ok_count=int(table.quality_ok.sum()) if 'quality_ok' in table else None,
                        force_units='uncalibrated sensor units',velocity_units='m/s',velocity_frame='world/odom',
                        contact_threshold_policy='per-event thresholds retained in contact_on_threshold/contact_off_threshold; per-trial sensor-only calibration when available',
                        velocity_method='LowState URDF Jacobian + gyro cross foot position + Sport body linear velocity; quaternion rotation',
                        velocity_model_source=MODEL_SOURCE,urdf_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                        noise_method='original int16 PCM; recorded audio clock minus stored audio_delay_s; DC removed; window [start,stop)',
                        noise_attribution='shared microphone; overlapping foot events share sound; not per-foot isolated sound',
                        synchronization_policy='preserve stored preprocessing delay; carry diagnostic status; no additional alignment',
                        input_csvs=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_paths])
                    (split/'contact_event_features.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
                atomic_csv(table,split/filename);result[f'{split.name}/{filename}']=len(table)
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'preprocessed/1104')
    parser.add_argument('--figure-dir',type=Path,default=ROOT/'preprocessed/1104/figure')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/1104')
    parser.add_argument('--trials',nargs='+')
    parser.add_argument('--on-threshold',type=float,default=10.)
    parser.add_argument('--off-threshold',type=float,default=5.)
    parser.add_argument('--confirm-ms',type=float,default=6.)
    parser.add_argument('--before-ms',type=float,default=BEFORE_MS)
    parser.add_argument('--after-ms',type=float,default=AFTER_MS)
    parser.add_argument('--body-velocity-frame',choices=['body','odom'],default='body')
    parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--zoom-seconds',type=float,default=1.5)
    args=parser.parse_args()
    if args.trials:
        expected=dict(window_before_ms=[args.before_ms],window_after_ms=[args.after_ms],
            body_velocity_frame=[args.body_velocity_frame],contact_on_threshold=[args.on_threshold],contact_off_threshold=[args.off_threshold])
        for split in ('train','test'):
            meta=args.input_dir/split/'contact_event_features.metadata.json'
            if meta.exists():
                stored=json.loads(meta.read_text()).get('settings',{})
                if any(stored.get(key,value)!=value for key,value in expected.items()):
                    parser.error('partial updates must keep stored extraction settings; rerun all trials to change them')
    files=[p for split in ('train','test') for p in sorted((args.input_dir/split).glob('*.csv')) if p.name not in GENERATED_CSVS]
    if args.trials:files=[p for p in files if p.stem in args.trials]
    if not files or len({p.stem for p in files})!=len(files):parser.error('nonempty unique trial CSVs required')
    if args.trials and set(args.trials)!={p.stem for p in files}:parser.error('requested trial not found')
    model=Go2Kinematics();summaries=[]
    for path in files:
        f=pd.read_csv(path,float_precision='round_trip')
        syncpath=args.figure_dir/path.stem/'synchronization.json'
        sync=json.loads(syncpath.read_text()) if syncpath.exists() else {}
        result=analyze_frame(f,path.stem,args.on_threshold,args.off_threshold,args.confirm_ms,args.before_ms,args.after_ms,
            args.data_dir/path.parent.name/f'{path.stem}.npz',path.parent.name,sync,model,args.body_velocity_frame)
        save_trial_result(result,path.stem,args.figure_dir/path.stem,args.on_threshold,args.off_threshold,args.confirm_ms,
            not args.no_plot,args.zoom_seconds)
        summaries.extend(result['summary'])
        print(f"{path.parent.name}/{path.stem}: {len(result['features'])} events / {len(result['samples'])} samples",flush=True)
    from contact_noise_visualization import save_contact_summary
    save_contact_summary(args.figure_dir,summaries)
    counts=collect_event_tables(args.input_dir,args.figure_dir)
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
