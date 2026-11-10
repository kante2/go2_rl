"""Check exported event windows against source CSVs and original PCM."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from go2_kinematics import Go2Kinematics

ROOT=Path(__file__).resolve().parents[1]


def verify(input_dir,raw_dir,figure_dir,protected_manifest=None):
    counts={};trials=[];raw_windows=0;velocity_checks=0;model=Go2Kinematics()
    for split in ('train','test'):
        events=pd.read_csv(input_dir/split/'contact_event_features.csv',float_precision='round_trip')
        samples=pd.read_csv(input_dir/split/'contact_event_samples.csv',float_precision='round_trip')
        assert events.event_id.is_unique
        assert set(events.split)=={split} and set(samples.split)=={split}
        assert set(samples.event_id)==set(events.event_id)
        assert (events.loc[events.pre_velocity_valid,'pre_velocity_time_abs']<events.loc[events.pre_velocity_valid,'contact_time_abs']).all()
        for name,g in events.groupby('trial'):
            source=pd.read_csv(input_dir/split/f'{name}.csv',float_precision='round_trip')
            assert name not in trials;trials.append(name)
            times=source.time_abs.to_numpy()
            part=samples[samples.trial==name]
            sample_index=np.searchsorted(times,part.time_abs.to_numpy())
            assert np.array_equal(times[sample_index],part.time_abs.to_numpy())
            np.testing.assert_allclose(source.audio_dbfs_20ms.iloc[sample_index],part.audio_dbfs_20ms,atol=1e-10)
            sample_groups={key:value for key,value in part.groupby('event_id',sort=False)}
            for event in g.to_dict('records'):
                sample=sample_groups[event['event_id']]
                assert sample.segment.nunique()==1
                assert np.all(np.diff(sample.time_abs)>0)
                assert np.all(np.diff(sample.time_abs)<.003+1e-9)
                assert sample.time_from_contact_ms.min()>=-event['window_before_ms']-1e-5
                assert sample.time_from_contact_ms.max()<=event['window_after_ms']+1e-5
                if event['window_complete']:
                    dt=float(np.median(np.diff(times)))
                    expected_count=int(round((event['window_before_ms']+event['window_after_ms'])/1000/dt))+1
                    assert len(sample)==expected_count
                at=int(np.searchsorted(times,event['contact_time_abs']))
                leg=event['leg'].lower()
                assert source[f'ls_ff_{leg}'].iloc[at]==event['force_at_contact_sensor_units']
                idx=np.searchsorted(times,sample.time_abs)
                np.testing.assert_array_equal(source[f'ls_ff_{leg}'].iloc[idx],sample.force_sensor_units)
            candidates=g[g.pre_velocity_valid]
            selected=candidates.iloc[np.linspace(0,len(candidates)-1,min(8,len(candidates)),dtype=int)] if len(candidates) else candidates
            with np.load(raw_dir/split/f'{name}.npz',allow_pickle=False) as raw:
                audio=raw['audio'];audio_t=float(raw['audio_t0'])+float(raw['audio_slope'])*np.arange(1,len(audio)+1)
                for event in selected.to_dict('records'):
                    i=int(np.searchsorted(times,event['pre_velocity_time_abs']));leg=event['leg'].lower()
                    q=source[[f'ls_q_{leg}_{j}' for j in ('hip','thigh','calf')]].iloc[i].to_numpy(float)[None,:]
                    dq=source[[f'ls_dq_{leg}_{j}' for j in ('hip','thigh','calf')]].iloc[i].to_numpy(float)
                    p,jac=model.position_jacobian(leg,q)
                    omega=source[[f'ls_gyro_{a}' for a in 'xyz']].iloc[i].to_numpy(float)
                    v=source[[f'sp_vel_{a}' for a in 'xyz']].iloc[i].to_numpy(float)
                    quat=source[[f'ls_quaternion_{a}' for a in ('x','y','z','w')]].iloc[i].to_numpy(float)
                    rotation=Rotation.from_quat(quat)
                    expected=rotation.apply(jac[0]@dq+np.cross(omega,p[0])+v)
                    np.testing.assert_allclose([event[f'pre_velocity_world_{a}_mps'] for a in 'xyz'],expected,atol=1e-10)
                    velocity_checks+=1
                    corrected=audio_t-event['audio_delay_s']
                    for left,right,col in [(event['window_start_abs'],event['window_stop_abs'],'noise_window_rms_dbfs'),
                        (event['contact_time_abs'],event['window_stop_abs'],'noise_post_rms_dbfs')]:
                        a,b=np.searchsorted(corrected,[left,right],side='left')
                        x=audio[a:b].astype(float)
                        expected=20*np.log10(max(float(np.std(x)),1e-12)/32768.)
                        np.testing.assert_allclose(event[col],expected,atol=1e-8)
                        raw_windows+=1
        counts[split]=dict(trials=int(events.trial.nunique()),events=len(events),samples=len(samples),
            feature_valid=int(events.feature_valid.sum()),quality_ok=int(events.quality_ok.sum()))
    protected_count=0
    if protected_manifest:
        manifest=json.loads(Path(protected_manifest).read_text())
        for path,expected in manifest.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,f'Source changed: {path}'
        protected_count=len(manifest)
    result=dict(passed=True,splits=counts,velocity_vectors_checked=velocity_checks,raw_pcm_windows_checked=raw_windows,
        protected_existing_artifacts_unchanged=protected_count)
    (figure_dir/'contact_event_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'preprocessed/1104')
    p.add_argument('--raw-dir',type=Path,default=ROOT/'data/1104')
    p.add_argument('--figure-dir',type=Path,default=ROOT/'preprocessed/1104/figure')
    p.add_argument('--protected-manifest',type=Path)
    args=p.parse_args();verify(args.input_dir,args.raw_dir,args.figure_dir,args.protected_manifest)
