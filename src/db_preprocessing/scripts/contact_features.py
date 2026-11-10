"""Past-only contact features for whole-walking noise prediction.

Force has unverified units: its integral is a proxy, not calibrated N*s.
Pre-contact velocity is the last body-frame foot vz before threshold onset,
available only once the contact is confirmed. It is not ground-relative speed.
"""
from collections import deque
import numpy as np
from contact_detection import detect_contact

LEGS=('fr','fl','rr','rl')
PREFIXES=('impact_rate6_','impact_rate_max20_','impact_integral20_',
          'contact_on_','contact_known_','contact_age_','contact_landing_valid_',
          'contact_pre_vz_body_','contact_pre_valid_')
DEFAULT_CONFIG=dict(version=1,on=10.,off=5.,confirm_ms=6.,rate_ms=6.,window_ms=20.,
                    age_cap_ms=100.,sampling_hz=500.,extra_history_ms=26.)


def feature_names():
    return [prefix+leg for prefix in PREFIXES for leg in LEGS]


def feature_dependency(name):
    if name.startswith('contact_pre_vz_body_'):
        return 'contact_pre_valid_'+name.removeprefix('contact_pre_vz_body_')
    if name.startswith('contact_age_'):
        return 'contact_landing_valid_'+name.removeprefix('contact_age_')
    return None


def add_contact_features(frame,config=None,requested=None):
    config=dict(DEFAULT_CONFIG if config is None else config)
    if config.get('version')!=1:raise ValueError('Unsupported contact feature version')
    names=feature_names() if requested is None else list(requested)
    if set(names)-set(feature_names()):raise ValueError('Unknown derived contact feature')
    result=frame.copy();gap_limit=1.5/float(config['sampling_hz'])
    for trial,group in frame.groupby('trial',sort=False):
        group=group.sort_values('time_abs');ix=group.index.to_numpy();t=group.time_abs.to_numpy(float)
        if len(t)<2 or np.any(np.diff(t)<=0):raise ValueError(f'{trial}: increasing timestamps required')
        breaks=np.r_[True,np.diff(t)>gap_limit+1e-9]
        if 'segment' in group:breaks|=np.r_[False,np.diff(group.segment)!=0]
        segments=np.cumsum(breaks)
        for leg in LEGS:
            local=[name for name in names if name.endswith('_'+leg)]
            if not local:continue
            force_key='ls_ff_'+leg
            if force_key not in group:raise ValueError(f'Missing {force_key}')
            f=group[force_key].to_numpy(float)
            _,causal,events=detect_contact(t,f,segments,config['on'],config['off'],config['confirm_ms'],max_gap_s=gap_limit)
            values={name:np.zeros(len(t),float) for name in [prefix+leg for prefix in PREFIXES]}
            values['contact_on_'+leg]=(causal==1).astype(float)
            values['contact_known_'+leg]=(causal!=-1).astype(float)
            for name in ('impact_rate6_','impact_rate_max20_','impact_integral20_'):values[name+leg][:]=np.nan
            physical_breaks = breaks | ~np.isfinite(f) | np.r_[False, ~np.isfinite(f[:-1])]
            physical_segments = np.cumsum(physical_breaks)
            for segment in np.unique(physical_segments):
                pos=np.flatnonzero(physical_segments==segment);tt=t[pos];ff=f[pos]
                if not np.isfinite(ff).all(): continue
                past=np.searchsorted(tt,tt-config['rate_ms']/1000+1e-9,side='right')-1
                good=past>=0;rate=np.full(len(pos),np.nan)
                rate[good]=np.maximum((ff[good]-ff[past[good]])/(tt[good]-tt[past[good]]),0.)
                values['impact_rate6_'+leg][pos]=rate
                integral=np.r_[0.,np.cumsum((ff[1:]+ff[:-1])*.5*np.diff(tt))]
                past=np.searchsorted(tt,tt-config['window_ms']/1000+1e-9,side='right')-1
                good=past>=0
                values['impact_integral20_'+leg][pos[good]]=integral[good]-integral[past[good]]
                queue=deque()
                for j,value in enumerate(rate):
                    while queue and tt[j]-tt[queue[0]]>config['window_ms']/1000+1e-9:queue.popleft()
                    if np.isfinite(value):
                        while queue and rate[queue[-1]]<=value:queue.pop()
                        queue.append(j)
                    if queue and (tt[j]-tt[0])*1000>=config['extra_history_ms']-1e-6:
                        values['impact_rate_max20_'+leg][pos[j]]=rate[queue[0]]
            needs_velocity=any(name.startswith(('contact_pre_vz_body_','contact_pre_valid_')) for name in local)
            velocity_key=f'sp_foot_vel_{leg}_z'
            if needs_velocity and velocity_key not in group:raise ValueError(f'Missing {velocity_key} for pre-contact velocity')
            velocity=group[velocity_key].to_numpy(float) if needs_velocity else np.full(len(t),np.nan)
            confirmed={int(np.searchsorted(t,e['confirmed_time_abs']-1e-9)):e for e in events if e['event']=='contact_on'}
            onset=None;pre=0.;pre_valid=False
            for i in range(len(t)):
                if breaks[i] or causal[i]!=1:onset=None;pre=0.;pre_valid=False
                if i in confirmed:
                    event=confirmed[i];onset=event['onset_time_abs']
                    before=int(np.searchsorted(t,onset-1e-9,side='left'))-1
                    pre_valid=before>=0 and segments[before]==segments[i] and np.isfinite(velocity[before])
                    pre=float(velocity[before]) if pre_valid else 0.
                if onset is not None and causal[i]==1:
                    values['contact_age_'+leg][i]=min((t[i]-onset)*1000,config['age_cap_ms'])
                    values['contact_landing_valid_'+leg][i]=1.
                    values['contact_pre_vz_body_'+leg][i]=pre
                    values['contact_pre_valid_'+leg][i]=float(pre_valid)
            for name in local:result.loc[ix,name]=values[name]
    return result
