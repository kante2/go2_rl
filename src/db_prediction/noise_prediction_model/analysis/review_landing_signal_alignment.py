#!/usr/bin/env python3
"""Audit displayed audio units and force/velocity timing, without retiming data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'noise_prediction_model/results/analysis/landing_dynamics'
REL = np.arange(-50, 102, 2)
SEARCH = np.flatnonzero((REL >= -12) & (REL <= 20))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rate(values, half_steps):
    result = np.full(values.shape, np.nan)
    result[:, half_steps:-half_steps] = (values[:, 2*half_steps:] - values[:, :-2*half_steps]) / (.004*half_steps)
    return result


def plateau_peak(values):
    """Midpoint of maximum plateau rather than earliest maximum sample."""
    local = values[:, SEARCH]
    maximum = np.max(local, axis=1)
    equal = np.isclose(local, maximum[:, None], rtol=1e-9, atol=1e-9)
    times = np.broadcast_to(REL[SEARCH], equal.shape)
    first = np.min(np.where(equal, times, np.inf), axis=1)
    last = np.max(np.where(equal, times, -np.inf), axis=1)
    center = (first + last) / 2
    invalid = maximum <= 0
    center[invalid] = np.nan
    return center, first, last, equal.sum(axis=1), equal[:, 0] | equal[:, -1], maximum


def source_clock_check(events):
    rows = []
    legs = ('fr', 'fl', 'rr', 'rl')
    for trial, group in events.groupby('trial'):
        fields = ['time_abs', *[f'ls_ff_{leg}' for leg in legs],
                  *[f'ls_{field}_{leg}_{joint}' for field in ('q', 'dq') for leg in legs for joint in ('hip', 'thigh', 'calf')],
                  *[f'sp_vel_{axis}' for axis in 'xyz']]
        csv = pd.read_csv(ROOT / f'preprocessed/1007/train/{trial}.csv', usecols=fields)
        times = csv.time_abs.to_numpy()
        onsets = group.contact_time_abs.to_numpy()
        centers = np.searchsorted(times, onsets - 1e-8)
        if not np.allclose(times[centers], onsets, atol=1e-7, rtol=0):
            raise AssertionError('CSV onset mapping failed')
        with np.load(ROOT / f'data/1007/train/{trial}.npz', allow_pickle=False) as raw:
            lt, st = raw['ls_t'], raw['sp_t']
            li = np.searchsorted(lt, onsets, side='right') - 1
            si = np.searchsorted(st, onsets, side='right') - 1
            if min(li.min(), si.min()) < 0:
                raise AssertionError('Source mapping outside recorded data')
            for field in ('q', 'dq'):
                expected = raw[f'ls_{field}'][li]
                actual = csv.iloc[centers][[f'ls_{field}_{leg}_{joint}' for leg in legs for joint in ('hip', 'thigh', 'calf')]].to_numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-10)
            np.testing.assert_allclose(csv.iloc[centers][[f'ls_ff_{leg}' for leg in legs]], raw['ls_ff'][li], atol=1e-10)
            np.testing.assert_allclose(csv.iloc[centers][[f'sp_vel_{axis}' for axis in 'xyz']], raw['sp_vel'][si], atol=1e-10)
        ls_age, sp_age = (onsets-lt[li])*1000, (onsets-st[si])*1000
        rows.append(dict(trial=trial, n_events=len(group), raw_force_q_dq_mapping_verified=True,
                         raw_sport_velocity_mapping_verified=True,
                         ls_age_median_ms=np.median(ls_age), ls_age_max_ms=ls_age.max(),
                         sp_age_median_ms=np.median(sp_age), sp_age_max_ms=sp_age.max(),
                         source_timestamp_abs_difference_median_ms=np.median(abs(lt[li]-st[si])*1000)))
    return pd.DataFrame(rows)


def audit(events, arrays):
    # All compared derivatives are recomputed with matching centered operators.
    rows, sensitivity = [], []
    for half_steps in (1, 2, 3, 5):
        df, dv = rate(arrays['force'], half_steps), rate(arrays['vz'], half_steps)
        ft, first, last, count, fedge, fm = plateau_peak(df)
        vt, _, _, _, vedge, vm = plateau_peak(dv)
        difference = vt-ft
        sensitivity.append(dict(span_ms=half_steps*4, n_events=len(events),
                                median_difference_ms=float(np.nanmedian(difference)),
                                q25_difference_ms=float(np.nanpercentile(difference, 25)),
                                q75_difference_ms=float(np.nanpercentile(difference, 75)),
                                fraction_abs_difference_le4ms=float(np.mean(abs(difference)<=4)),
                                fraction_abs_difference_le6ms=float(np.mean(abs(difference)<=6))))
        if half_steps == 3:
            result = events[['event_id','trial','leg','command_vx','synchronization_status','contact_time_abs']].copy()
            result['force_rate_peak_center_ms']=ft
            result['force_rate_first_max_ms']=first
            result['force_rate_last_max_ms']=last
            result['force_rate_peak_tie_count']=count
            result['velocity_rate_peak_center_ms']=vt
            result['rate_peak_difference_ms']=difference
            result['peak_at_search_edge']=fedge|vedge
            result['force_rate_peak_sensor_units_per_s']=fm
            result['velocity_rate_peak_mps2']=vm
            result['force_rate_peak_sensor_units_per_ms']=fm/1000
    # A second timing measure does not rely on derivative maxima.
    before=(REL>=-20)&(REL<=-10); late=(REL>=60)&(REL<=100)
    f0=np.median(arrays['force'][:,before],axis=1)
    fp=np.max(arrays['force'][:,(REL>=0)&(REL<=20)],axis=1)
    v0=np.median(arrays['vz'][:,before],axis=1);vs=np.median(arrays['vz'][:,late],axis=1)
    for name,values,threshold in [('force',arrays['force'],(f0+fp)/2),('velocity',arrays['vz'],(v0+vs)/2)]:
        hit=values[:,SEARCH]>=threshold[:,None]
        result[f'{name}_half_transition_ms']=np.where(hit.any(axis=1),REL[SEARCH][hit.argmax(axis=1)],np.nan)
    result['half_transition_difference_ms']=result.velocity_half_transition_ms-result.force_half_transition_ms
    result['velocity_transition_amplitude_mps']=vs-v0
    result['force_transition_amplitude_sensor_units']=fp-f0
    post=(REL>=0)&(REL<=40)
    hit=abs(arrays['vz'][:,post])<=.1
    result['first_near_zero_velocity_ms']=np.where(hit.any(axis=1),REL[post][hit.argmax(axis=1)],np.nan)
    return result,pd.DataFrame(sensitivity)


def aggregate(result,keys):
    return result.groupby(keys,as_index=False).agg(
        n_events=('event_id','size'), force_rate_peak_center_ms=('force_rate_peak_center_ms','median'),
        velocity_rate_peak_center_ms=('velocity_rate_peak_center_ms','median'),
        median_peak_difference_ms=('rate_peak_difference_ms','median'),
        peak_difference_q25_ms=('rate_peak_difference_ms',lambda x:x.quantile(.25)),
        peak_difference_q75_ms=('rate_peak_difference_ms',lambda x:x.quantile(.75)),
        fraction_abs_difference_gt6ms=('rate_peak_difference_ms',lambda x:x.abs().gt(6).mean()),
        median_half_transition_difference_ms=('half_transition_difference_ms','median'),
        n_search_edge=('peak_at_search_edge','sum'))


def overview(result,sensitivity,output):
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    d=result.rate_peak_difference_ms.dropna()
    axes[0].hist(d,bins=np.arange(-33,34,2),color='tab:blue',edgecolor='white')
    axes[0].axvline(0,color='black',ls='--',lw=1)
    axes[0].set(xlabel='Velocity-rate peak minus force-rate peak (ms)',ylabel='Number of events',
                title='12ms centered difference | plateau midpoint timing')
    axes[0].text(.02,.95,f'Median = {d.median():+.1f}ms\nWithin ±4ms = {100*d.abs().le(4).mean():.1f}%\n'
                         f'Within ±6ms = {100*d.abs().le(6).mean():.1f}%',
                 transform=axes[0].transAxes,va='top',bbox=dict(facecolor='white',alpha=.8,edgecolor='.8'))
    y=sensitivity.median_difference_ms.to_numpy()
    axes[1].errorbar(sensitivity.span_ms,y,
                     yerr=np.vstack([y-sensitivity.q25_difference_ms,sensitivity.q75_difference_ms-y]),
                     fmt='o-',color='tab:purple',capsize=5)
    axes[1].axhline(0,color='.5',ls='--',lw=1)
    axes[1].set(xlabel='Centered difference span (ms)',ylabel='Peak-time difference (ms)',
                title='Sensitivity to the differentiation window\nMedian and event IQR; not clock calibration',xticks=[4,8,12,20])
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Force / velocity timing review | no source timestamps shifted',fontsize=13)
    fig.savefig(output/'rate_alignment_overview.png',dpi=165);plt.close(fig)


def detail(events,arrays,output,trial='0.3_2'):
    if not events.trial.eq(trial).any():return
    fig,axes=plt.subplots(6,4,figsize=(18,17),sharex=True,layout='constrained')
    for col,leg in enumerate(('FL','FR','RL','RR')):
        mask=(events.trial.eq(trial)&events.leg.eq(leg)).to_numpy()
        profiles={k:np.median(v[mask],axis=0) for k,v in arrays.items()}
        ft=plateau_peak(profiles['force_rate'][None,:])[0][0]
        vt=plateau_peak(profiles['vz_rate'][None,:])[0][0]
        axes[0,col].set_title(f'{leg}: n={mask.sum()}\nMedian-profile peak Δt={vt-ft:+.1f}ms')
        settings=[('noise_5ms_dbfs',1,'5ms RMS level\n(dBFS)','black'),
                  ('force',1,'Contact force\n(sensor units)','tab:blue'),
                  ('vz',1,'Vertical velocity\n(+up / -down, m/s)','crimson'),
                  ('force_rate',.001,'Force rate over 12ms\n(sensor units/ms)','tab:orange'),
                  ('vz_rate',1,'Velocity rate over 12ms\n(m/s², estimated)','tab:purple')]
        for row,(name,scale,label,color) in enumerate(settings):
            data=arrays[name][mask]*scale;q=np.percentile(data,[25,50,75],axis=0)
            axes[row,col].fill_between(REL,q[0],q[2],color=color,alpha=.18)
            axes[row,col].plot(REL,q[1],color=color,lw=1.7)
            if col==0:axes[row,col].set_ylabel(label)
        for key,color in [('force_rate','tab:orange'),('vz_rate','tab:purple')]:
            peak=np.max(profiles[key][SEARCH])
            axes[5,col].plot(REL,profiles[key]/peak,color=color,lw=1.7,label=key)
        axes[5,col].legend(fontsize=7,loc='upper right')
        axes[5,col].set_xlabel('Time from force-threshold onset (ms)')
        if col==0:axes[5,col].set_ylabel('Median-profile rates\nscaled by own positive peak')
        for ax in axes[:,col]:
            ax.axvline(0,color='black',ls='--',lw=.8)
            ax.axvline(ft,color='tab:orange',ls=':',lw=.8)
            ax.axvline(vt,color='tab:purple',ls=':',lw=.8)
            ax.axhline(0,color='.7',lw=.5);ax.grid(alpha=.15);ax.set_xlim(-50,100)
    fig.suptitle(f'{trial}: audio units and force/velocity timing review\n'
                 'Absolute 5ms RMS dBFS; force rate displayed per ms; rate overlay is normalized for timing only\n'
                 'Dashed black = force onset; dotted orange/purple = each median-profile rate peak; no time shifting',fontsize=12)
    fig.savefig(output/f'{trial}_alignment_detail.png',dpi=150);plt.close(fig)


def main():
    output=BASE/'force_velocity_sync_review';output.mkdir(exist_ok=True)
    sources=[BASE/'landing_aligned_samples.csv',BASE/'landing_event_metrics.csv']
    hashes={str(p):digest(p) for p in sources}
    samples=pd.read_csv(sources[0]);events=pd.read_csv(sources[1])
    if not events.split.eq('train').all() or events.event_id.duplicated().any():raise ValueError('Train events required')
    ids=samples.drop_duplicates('event_id').event_id
    events=events.set_index('event_id').loc[ids].reset_index()
    times=samples.relative_time_ms.to_numpy().reshape(-1,len(REL))
    np.testing.assert_array_equal(times,np.broadcast_to(REL,times.shape))
    arrays={k:samples[k].to_numpy(float).reshape(-1,len(REL)) for k in
            ('force','vz','force_rate','vz_rate','noise_5ms_dbfs','noise_change_db')}
    # Recompute the original 12ms derivatives from the stored undifferentiated signals.
    np.testing.assert_allclose(rate(arrays['force'],3)[:,3:-3],arrays['force_rate'][:,3:-3],atol=1e-6)
    np.testing.assert_allclose(rate(arrays['vz'],3)[:,3:-3],arrays['vz_rate'][:,3:-3],atol=1e-7)
    expected_noise=arrays['noise_5ms_dbfs']-events.noise_baseline_rms_dbfs.to_numpy()[:,None]
    np.testing.assert_allclose(expected_noise,arrays['noise_change_db'],atol=1e-8)
    result,sensitivity=audit(events,arrays)
    result.to_csv(output/'rate_timing_events.csv',index=False)
    aggregate(result,['trial']).to_csv(output/'rate_timing_by_trial.csv',index=False)
    aggregate(result,['trial','leg']).to_csv(output/'rate_timing_by_trial_leg.csv',index=False)
    sensitivity.to_csv(output/'difference_window_sensitivity.csv',index=False)
    result[result.rate_peak_difference_ms.abs().gt(6)|result.peak_at_search_edge].to_csv(output/'events_to_review.csv',index=False)
    clocks=source_clock_check(events);clocks.to_csv(output/'source_clock_checks.csv',index=False)
    overview(result,sensitivity,output);detail(events,arrays,output)
    if any(digest(p)!=h for p,h in hashes.items()):raise AssertionError('Analysis inputs changed')
    summary=dict(n_events=len(events),n_trials=int(events.trial.nunique()),
                 median_rate_peak_difference_ms=float(result.rate_peak_difference_ms.median()),
                 fraction_abs_difference_le4ms=float(result.rate_peak_difference_ms.abs().le(4).mean()),
                 fraction_abs_difference_le6ms=float(result.rate_peak_difference_ms.abs().le(6).mean()),
                 n_abs_difference_gt6ms=int(result.rate_peak_difference_ms.abs().gt(6).sum()),
                 n_search_edge=int(result.peak_at_search_edge.sum()),
                 median_half_transition_difference_ms=float(result.half_transition_difference_ms.median()),
                 raw_source_mapping_verified=True,original_derivatives_recomputed=True,
                 baseline_db_subtraction_verified=True,no_retiming=True,input_sha256=hashes,script_sha256=digest(__file__))
    (output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 소음 단위와 접촉력·발 속도 시각 검토','',
           '## 1. RMS와 dBFS','',
           'RMS는 창 안의 신호 진폭을 요약하는 계산이고 dBFS는 그 값을 PCM full scale에 대해 표현한 단위다. '
           '기존 코드도 DC 제거 RMS를 구한 뒤 20 log10(RMS/32768)로 dBFS를 계산한다. '
           '원시 파형의 양·음 샘플을 단순 평균하면 상쇄되므로 소음 크기의 요약에 RMS를 사용했다.','',
           '기존 trials/*_landing_profiles.png의 1행은 절대 dBFS가 아니라 '
           '5ms RMS dBFS에서 해당 착지 [−50,−20)ms RMS dBFS를 뺀 ΔdB다. '
           '예: −20dBFS − (−40dBFS) = +20dB. 새 상세 그림은 혼동을 줄이기 위해 절대 dBFS를 표시한다.','',
           '## 2. 힘 변화율이 수천인 이유','',
           '기존 힘 변화율 = [F(t+6ms)−F(t−6ms)]/0.012s. '
           '센서값 60이 12ms 동안 증가하면 5000 sensor units/s, 같은 값을 ms당 쓰면 5 sensor units/ms다. '
           '12ms는 계산 구간이고 /s가 결과 단위다. 값이 1000배 잘못된 것은 아니다. '
           '힘 단위가 보정되지 않았으므로 N/s로 해석하지 않는다. 새 상세 그림만 /ms로 환산했다.','',
           '## 3. 전체 시각 검토','',
           f'{events.trial.nunique()}개 train trial의 {len(events):,}개 착지를 조사했다. 힘 변화율과 속도 원값의 영점이 같은 시점일 필요는 없다. '
           '여기서는 힘 변화율과 속도 변화율의 최대 시점, 별도로 힘·속도 변화의 50% 도달 시점을 비교했다.','',
           '주 peak 검색 구간은 힘 검출 시작 [−12,+20]ms다. '
           '평평한 최대 구간은 첫 최대 표본 대신 최대 plateau의 시작·끝 중간 시각으로 대표했다. '
           '기존 peak_time 컬럼은 np.argmax의 첫 최대 시각이므로 새 지표와 정의가 다르다. '
           '원래 값은 덮어쓰지 않았다. 서로 떨어진 동률 최대가 있으면 이 중간값도 불확실하므로 tie 폭을 함께 저장했다.','',
           f'- 12ms 계산 기준 peak 시간차(속도 변화율 − 힘 변화율) 중앙값: {summary["median_rate_peak_difference_ms"]:+.1f}ms.',
           f'- 절댓값 4ms 이내: {100*summary["fraction_abs_difference_le4ms"]:.1f}%.',
           f'- 절댓값 6ms 이내: {100*summary["fraction_abs_difference_le6ms"]:.1f}%.',
           f'- 6ms 초과: {summary["n_abs_difference_gt6ms"]}개. 검색 경계 최대: {summary["n_search_edge"]}개. 중복 가능.',
           f'- 힘/속도 50% 변화 시점 차이 중앙값: {summary["median_half_transition_difference_ms"]:+.1f}ms.','',
           '|차분 구간|peak 차이 중앙값|25~75% 범위|±4ms 이내|','|---|---:|---:|---:|']
    for r in sensitivity.itertuples():
        lines.append(f'|{r.span_ms}ms|{r.median_difference_ms:+.1f}ms|{r.q25_difference_ms:+.1f}~{r.q75_difference_ms:+.1f}ms|{100*r.fraction_abs_difference_le4ms:.1f}%|')
    lines+=['','차분 창에 따라 peak 시점이 바뀌므로 하나의 peak 차이를 물리적 시계 오차로 보정해서는 안 된다. '
            '12ms centered 차분은 원신호가 변하기 최대 6ms 전부터 반응할 수 있다. '
            '힘과 속도에 같은 centered 연산을 적용한 것은 확인했지만, 서로 다른 파형의 최대 시각이 같아야 한다는 뜻은 아니다.','',
            '## 4. 원본 메시지 대응','',
            '모든 착지 0ms에서 CSV force·12개 q·12개 dq가 동일한 원본 LowState 수신 시각의 previous sample과 일치함을 확인했다. '
            'Sport 몸체 속도도 원본 수신 시각 기준 previous sample과 대조했다. '
            '상관과 속도를 만들기 위해 force와 관절 데이터 중 하나를 별도로 이동시킨 코드상 오프셋은 발견하지 못했다. '
            '하지만 같은 메시지에 담겼다는 것이 센서 내부 획득 시각과 필터 지연까지 같다는 보증은 아니다.','',
            '현재 속도는 URDF 기준점의 강체 운동학·몸체 상태 추정으로 계산한 값이며 실제 발바닥 접촉점 속도의 독립 측정이 아니다. '
            '오디오 동기화 문제와 force–velocity 관계는 별개다. '
            '5ms centered RMS의 ±2.5ms 평활화는 오디오 비교에 영향을 주며, force–velocity 시각 비교를 설명하는 보정값은 아니다.','',
            '## 파일과 재현','',
            '- rate_alignment_overview.png: 전체 peak 차이 분포 및 차분 구간 민감도.',
            '- 0.3_2_alignment_detail.png: 절대 소음 dBFS, /ms 힘 변화율, 두 변화율의 정규화 겹침 그림.',
            '- rate_timing_events.csv / rate_timing_by_trial.csv / rate_timing_by_trial_leg.csv: 모든 이벤트와 trial·발별 결과.',
            '- events_to_review.csv: 6ms 초과 또는 검색 경계 최대 이벤트. 모두 오류라는 뜻은 아니다.',
            '- source_clock_checks.csv: raw 메시지 대응 검증과 수신 표본 age.',
            '- difference_window_sensitivity.csv: 4/8/12/20ms의 결과.',
            '- verification.json: 입력 해시 및 계산 검증.','',
            '원래 시각·속도·변화율·기존 trial 그림은 변경하지 않았다. '
            '세로선에 맞추려고 속도를 0으로 강제하거나 신호를 이동하지 않았다.','',
            '재실행: `.venv/bin/python noise_prediction_model/analysis/review_landing_signal_alignment.py`','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Saved review ->',output)


if __name__=='__main__':main()
