"""Force-only loaded-contact detection shared by synchronization and analysis."""
import numpy as np

DEFAULT_ON=10.
DEFAULT_OFF=5.
DEFAULT_CONFIRM_MS=6.

def detect_contact(times,force,segments,on=DEFAULT_ON,off=DEFAULT_OFF,confirm_ms=DEFAULT_CONFIRM_MS,max_gap_s=None):
    """Hysteresis with persistent evidence. Reset at missing data/segment gaps."""
    if not np.isfinite([on,off,confirm_ms]).all() or not 0<=off<on or confirm_ms<0:
        raise ValueError('Require finite thresholds 0 <= off < on and confirmation >= 0ms')
    times=np.asarray(times,float);force=np.asarray(force,float);segments=np.asarray(segments)
    if len(times)!=len(force) or len(times)!=len(segments):raise ValueError('Array lengths differ')
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Times must be finite and increasing')
    if not len(times):return np.array([],dtype=np.int8),np.array([],dtype=np.int8),[]
    typical=float(np.median(np.diff(times))) if len(times)>1 else .002
    gap_limit=1.5*typical if max_gap_s is None else float(max_gap_s)
    estimated=np.full(len(times),-1,dtype=np.int8);causal=estimated.copy();events=[]
    state=-1;pending=None;start=None
    for i,(t,f) in enumerate(zip(times,force)):
        if i==0 or segments[i]!=segments[i-1] or t-times[i-1]>gap_limit+1e-9:
            state=-1;pending=None;start=None
        if not np.isfinite(f):
            state=-1;pending=None;start=None;continue
        wanted=(1 if f>=on else (0 if f<=off else None))
        if wanted is None or wanted==state:
            pending=None;start=None
        else:
            if pending!=wanted:pending=wanted;start=i
            if (t-times[start])*1000+1e-6>=confirm_ms:
                previous=state;state=wanted
                estimated[start:i+1]=state
                if previous!=-1:
                    events.append(dict(event='contact_on' if state else 'contact_off',
                        segment=int(segments[i]),onset_time_abs=float(times[start]),
                        confirmed_time_abs=float(t),confirmation_delay_ms=float((t-times[start])*1000)))
                pending=None;start=None
        estimated[i]=state;causal[i]=state
    return estimated,causal,events
