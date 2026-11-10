"""Auditable natural-footfall synchronization reports and raw clock diagnostics."""
from pathlib import Path
import json
import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def clock_quality(trial,data):
    keys=set(trial.keys)
    slope=float(trial.load('audio_slope')); nominal=float(trial.load('fs')) if 'fs' in keys else np.nan
    stats=dict(audio_fs_estimated_hz=1/slope,audio_fs_nominal_hz=nominal,
               audio_fs_difference_hz=1/slope-nominal,audio_callback_count=0,
               audio_arrival_residual_median_ms=np.nan,audio_arrival_residual_p95_ms=np.nan,
               audio_arrival_residual_max_ms=np.nan,ls_gap_count=0,sp_gap_count=0,
               ls_max_interval_ms=np.nan,sp_max_interval_ms=np.nan)
    series={}
    if {'audio_cb_t','audio_n_end'}<=keys:
        callbacks=np.asarray(trial.load('audio_cb_t'),dtype=float)
        counts=np.asarray(trial.load('audio_n_end'),dtype=float)
        if len(callbacks)!=len(counts) or not np.isfinite(callbacks).all() or not np.isfinite(counts).all():
            raise ValueError(f'{trial.name}: invalid audio callback timing arrays')
        residual=(callbacks-(float(trial.load('audio_t0'))+slope*counts))*1000
        if len(callbacks):
            stats.update(audio_callback_count=len(callbacks),audio_arrival_residual_median_ms=float(np.median(residual)),
                audio_arrival_residual_p95_ms=float(np.quantile(residual,.95)),audio_arrival_residual_max_ms=float(np.max(residual)))
            series.update(audio_callback_time=callbacks,audio_arrival_residual_ms=residual)
    target=data['time_abs'];max_gap=float(data.get('max_gap_s',.02))
    for key in ['ls','sp']:
        t=np.asarray(trial.load(f'{key}_t'),dtype=float);dt=np.diff(t)
        stats[f'{key}_gap_count']=int(np.sum(dt>max_gap))
        stats[f'{key}_max_interval_ms']=float(np.max(dt)*1000)
        idx=np.searchsorted(t,target,side='right')-1
        age=np.where(idx>=0,(target-t[np.clip(idx,0,len(t)-1)])*1000,np.nan)
        series[f'{key}_time']=t[1:];series[f'{key}_interval_ms']=dt*1000;series[f'{key}_age_ms']=age
    if 'ls_tick' in keys:
        ticks=np.asarray(trial.load('ls_tick'),dtype=float).reshape(-1)
        if len(ticks)!=len(series['ls_time'])+1:
            raise ValueError(f'{trial.name}: ls_tick and ls_t lengths differ')
        delta=np.diff(ticks);positive=delta[np.isfinite(delta)&(delta>0)]
        typical=float(np.median(positive)) if len(positive) else np.nan
        gaps=series['ls_interval_ms']>max_gap*1000
        stats.update(ls_tick_typical_increment=typical,
            ls_tick_nonpositive_increment_count=int(np.sum(delta<=0)),
            ls_receive_gap_with_small_tick_increment_count=int(np.sum(gaps&(delta>0)&(delta<=2*typical))))
        series['ls_tick_increment_relative']=delta/typical if np.isfinite(typical) else np.full_like(delta,np.nan)
    return stats,series



def save_trial_diagnostics(trial, data, summary, events, audio, output, plots=True):
    import pandas as pd
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats, series = clock_quality(trial, data)
    summary['clock_quality'] = stats
    write_json(output/'synchronization.json', summary)
    write_json(output/'footfall_matches.json', events)
    rows = [{k: v for k, v in item.items() if k not in {'contacts', 'candidates'}} for item in events]
    pd.DataFrame(rows).to_csv(output/'footfall_matches.csv', index=False)
    if not plots:
        return
    from synchronization_figures import plot_trial_figures, load_middle_checks
    middle = load_middle_checks(output.parent).get(trial.name)
    plot_trial_figures(trial, summary, events, audio, output, middle)


def save_run_summary(output, summaries, run, plots=True):
    import pandas as pd
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in summaries:
        rows.append(dict(trial=s['trial'], split=s['split'], status=s['status'], passed=s['passed'],
                         applied_audio_delay_ms=1000*s['applied_audio_delay_s'],
                         start_events=s['calibration']['events'], start_inliers=s['calibration']['inliers'],
                         end_events=s['validation']['events'], end_inliers=s['validation']['inliers'],
                         start_ambiguous=s['calibration']['ambiguous'], end_ambiguous=s['validation']['ambiguous'],
                         end_residual_ms=s['end_residual_ms'], end_abs_error_p90_ms=s['end_abs_error_p90_ms']))
    frame = pd.DataFrame(rows); frame.to_csv(output/'synchronization_summary.csv', index=False)
    write_json(output/'run.json', run)
    passed = sum(s['passed'] for s in summaries)
    lines = ['# 1104 착지 기반 시간 동기화', '', f'검증 통과 {passed}/{len(summaries)}개.', '',
             '시작 5초 착지로 시퀀스별 고정 지연을 추정하고, 종료 착지는 별도로 검증합니다.',
             '첫/마지막 단일 착지는 전체 매칭 파일에 기록하며 주변 착지의 일관성으로 판단합니다.',
             '종료 중앙 잔차와 inlier 절대 잔차 P90 모두 10ms 이하여야 통과합니다.',
             '모호한 지연 모드·부족한 매칭은 검토 대상으로 남깁니다. 임계값은 발별 시작 힘 분포에서만 계산합니다.',
             'all_trials.csv에는 동기화 검증과 최소 출력 길이를 통과한 trial만 포함합니다.',
             '개별 trial CSV와 접촉 분석에는 검토 대상도 보존하고 상태를 표시합니다.',
             '접촉 임계값 통과와 음향 상승의 정렬이며 하드웨어 취득 시각의 동기 검증은 아닙니다.', '',
             '|split|trial|상태|보정(ms)|종료 잔차(ms)|', '|---|---|---|---:|---:|']
    for s in summaries:
        residual = s['end_residual_ms']
        lines.append(f"|{s['split']}|{s['trial']}|{s['status']}|{1000*s['applied_audio_delay_s']:.3f}|{'' if residual is None else f'{residual:.3f}'}|")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    if plots:
        from synchronization_figures import plot_overview, load_middle_checks
        plot_overview(summaries, output, load_middle_checks(output))
